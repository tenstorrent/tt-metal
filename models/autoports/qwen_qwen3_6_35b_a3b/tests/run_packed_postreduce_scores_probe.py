# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact bits for scalar-page versus packed per-token MoE post-reduce scores.

CPU mode validates fixtures only. Acceptance mode uses the existing native op
on a 1x4 mesh, with production BF16 scores and 32x8x2048 combined carriers.
Run serialized under an external timeout; this is not a serving benchmark.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def fixtures(valid_token_rows=0):
    """Distinct per-rank payloads; nonlocal combine cells obey init_zeros=True."""
    random = torch.Generator().manual_seed(270117)
    table = torch.full((4, 256), -1, dtype=torch.int32)
    for rank in range(4):
        table[rank, rank * 64 : (rank + 1) * 64] = 0
    cases = []
    for number, (name, rows, slots) in enumerate(
        (
            ("none", (), ()),
            ("b2_first", range(2), (0,)),
            ("b2_last", range(2), (7,)),
            ("b4_mixed", range(4), (0, 3, 7)),
            ("high31", (31,), (1, 6)),
            ("all32_local", range(32), range(8)),
            ("equal_scores", range(32), (0, 2, 4, 6)),
            ("signed_zero_scores", range(4), range(8)),
            ("cancellation", range(32), range(8)),
            ("tiny_values", range(2), (0, 7)),
            ("none_after_active", (), ()),
            ("changed_after_none", range(2), (1, 4, 7)),
        )
    ):
        ids = torch.empty((4, 32, 8), dtype=torch.int32)
        combined = torch.zeros((4, 1, 32, 8, 2048), dtype=torch.bfloat16)
        scores = torch.softmax(torch.randn((4, 1, 32, 8), generator=random), dim=-1).to(torch.bfloat16)
        if name == "equal_scores":
            scores.fill_(0.125)
        if name == "signed_zero_scores":
            scores[..., ::2] = 0.0
            scores[..., 1::2] = -0.0
        for rank in range(4):
            # Distinct global IDs, including IDs across the 63/64 ownership edge.
            ids[rank] = (((rank + 1) % 4) * 64 + torch.arange(8)).expand(32, -1)
            for row in rows:
                for slot in slots:
                    ids[rank, row, slot] = rank * 64 + (row + 7 * slot + number) % 64
                    value = torch.randn((2048,), generator=random) * (rank + 1) / 8
                    if name == "cancellation":
                        value = torch.full((2048,), (-1.0 if slot % 2 else 1.0) * 64)
                        value[slot::8] += 0.5
                    if name == "tiny_values":
                        value = torch.tensor([0.0, -0.0, 2.0**-126, -(2.0**-126)]).repeat(512)
                    combined[rank, 0, row, slot] = value.to(torch.bfloat16)
        if valid_token_rows:
            combined[:, :, valid_token_rows:] = 0
            for rank in range(4):
                ids[rank, valid_token_rows:] = ((rank + 1) % 4) * 64 + torch.arange(8)
        cases.append((name, dict(combined=combined, scores=scores, ids=ids, table=table.clone())))
    return cases


def validate_fixtures(cases):
    for name, values in cases:
        combined, scores, ids, table = (values[key] for key in ("combined", "scores", "ids", "table"))
        assert combined.shape == (4, 1, 32, 8, 2048)
        assert scores.shape == (4, 1, 32, 8)
        assert ids.shape == (4, 32, 8) and 0 <= ids.min() <= ids.max() < 256
        assert combined.dtype == scores.dtype == torch.bfloat16
        assert torch.isfinite(combined).all() and torch.isfinite(scores).all()
        for rank in range(4):
            assert all(len(set(row.tolist())) == 8 for row in ids[rank]), name
            local = table[rank][ids[rank].long()] != -1
            assert torch.count_nonzero(combined[rank, 0][~local]) == 0, name
        # Reshape is purely representational on CPU; this is not a device claim.
        assert torch.equal(
            scores.reshape(4, 1, 32, 8, 1).view(torch.int16).flatten(), scores.view(torch.int16).flatten()
        )


def poison_omitted_rows(values, valid_token_rows):
    """Only the candidate sees these forbidden-to-read omitted inputs."""
    assert 1 <= valid_token_rows < 32
    poisoned = {key: value.clone() for key, value in values.items()}
    pattern = torch.tensor([float("nan"), float("inf"), -float("inf"), 19.0, -7.0, -0.0, 1.0, 0.0])
    poisoned["combined"][:, :, valid_token_rows:] = pattern.to(torch.bfloat16).repeat(256)
    poisoned["scores"][:, :, valid_token_rows:] = pattern.to(torch.bfloat16)
    for rank in range(4):
        # Local tail IDs ensure accidental execution cannot be masked by the
        # ordinary nonlocal-expert skip. Every ID remains legal and distinct.
        poisoned["ids"][rank, valid_token_rows:] = rank * 64 + torch.arange(8)
    for key in ("combined", "scores"):
        assert torch.equal(
            poisoned[key][:, :, :valid_token_rows].view(torch.int16),
            values[key][:, :, :valid_token_rows].view(torch.int16),
        )
    assert torch.equal(poisoned["ids"][:, :valid_token_rows], values["ids"][:, :valid_token_rows])
    assert torch.equal(poisoned["table"], values["table"])
    assert not torch.isfinite(poisoned["combined"][:, :, valid_token_rows:]).all()
    return poisoned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "acceptance"), default="acceptance")
    parser.add_argument(
        "--valid-token-rows",
        type=int,
        default=0,
        choices=range(33),
        help="0: packed weights only; 1..32: additionally specialize candidate prefix rows (requires native API)",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/packed_postreduce_scores/result.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh --output path to preserve prior evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    native = REPO / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce"
    sources = [
        Path(__file__),
        ROOT / "tt/model.py",
        ROOT / "tt/multichip_decoder.py",
        *sorted(path for path in native.rglob("*") if path.suffix in (".cpp", ".hpp")),
    ]
    result = dict(
        status="RUNNING",
        command=sys.argv,
        mode=args.mode,
        valid_token_rows=args.valid_token_rows,
        hardware_verified=False,
        mesh=[1, 4],
        tracking=True,
        skip_program_cache=False,
        scope="Exact post-reduce component bits only; no generator, sampling, serving or latency claim",
        local_shapes=dict(combined=[1, 1, 32, 8, 2048], scores_tile=[1, 1, 32, 8], ids=[1, 32, 8], table=[1, 256]),
        weights=dict(reference=[1, 1, 32, 8, 1], candidate=[1, 1, 32, 8]),
        source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        cases=[],
        invalid_parameters=[],
        captures=0,
        releases=0,
        cleanup_errors=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh, traces, outputs = None, {}, {}
    failure = None
    try:
        torch.set_num_threads(4)
        cases = fixtures(args.valid_token_rows)
        validate_fixtures(cases)
        if 0 < args.valid_token_rows < 32:
            for _, values in cases:
                poison_omitted_rows(values, args.valid_token_rows)
        result["fixture_names"] = [name for name, _ in cases]
        if args.mode == "cpu":
            result.update(status="PASS", tracking="not_exercised", skip_program_cache="not_exercised")
            return

        os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
        os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
        assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
        assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        from ttnn.tools import trace_allocation_tracker

        import ttnn as t

        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)

        def upload(key, value, device=False):
            kwargs = dict(
                dtype=t.uint16 if key == "ids" else t.int32 if key == "table" else t.bfloat16,
                layout=t.TILE_LAYOUT if key == "scores" else t.ROW_MAJOR_LAYOUT,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=0),
            )
            if device:
                kwargs.update(device=mesh, memory_config=t.DRAM_MEMORY_CONFIG)
            return t.from_torch(value.contiguous(), **kwargs)

        inputs = {key: upload(key, value, True) for key, value in cases[0][1].items()}

        def call(name):
            shape = (1, 1, 32, 8, 1) if name == "reference" else (1, 1, 32, 8)
            weights = t.reshape(t.to_layout(inputs["scores"], t.ROW_MAJOR_LAYOUT), shape)
            kwargs = {}
            if name == "candidate" and args.valid_token_rows:
                kwargs["valid_token_rows"] = args.valid_token_rows
            return t.experimental.deepseek_prefill.post_combine_reduce(
                inputs["combined"],
                weights,
                inputs["ids"],
                inputs["table"],
                expert_dim=3,
                output_memory_config=t.DRAM_MEMORY_CONFIG,
                **kwargs,
            )

        def host_bits(value):
            parts = [t.to_torch(part).contiguous() for part in t.get_device_tensors(value)]
            assert len(parts) == 4
            assert all(part.dtype == torch.bfloat16 and tuple(part.shape) == (1, 1, 32, 2048) for part in parts)
            return [part.view(torch.int16).clone() for part in parts]

        def exact(label, actual, expected):
            for rank, (left, right) in enumerate(zip(actual, expected)):
                assert left.shape == right.shape
                different = int((left != right).sum())
                assert different == 0, (label, rank, different)

        def source_bits():
            return {
                key: [t.to_torch(part).contiguous().view(torch.uint8).clone() for part in t.get_device_tensors(value)]
                for key, value in inputs.items()
            }

        def unchanged_sources(expected):
            for key, actual in source_bits().items():
                exact(f"source-preservation/{key}", actual, expected[key])

        def zero_tail(bits):
            if args.valid_token_rows:
                assert all(
                    torch.count_nonzero(value[:, :, args.valid_token_rows :]) == 0 for value in bits
                ), "Omitted output rows must be explicit positive zero bits"

        def refresh(values):
            for key, value in values.items():
                t.copy_host_to_device_tensor(upload(key, value), inputs[key])

        def check_invalid_parameters():
            # Match conftest.expect_error's triage markers in this standalone
            # executable; catch only the expected type and exact fatal text.
            def reject(label, message, operation):
                print(f'[EXPECTED_ERROR BEGIN] RuntimeError message="{message}"', flush=True)
                try:
                    try:
                        unexpected = operation()
                    except RuntimeError as error:
                        if message not in str(error):
                            raise
                        result["invalid_parameters"].append(dict(case=label, expected_message=message, rejected=True))
                    else:
                        t.deallocate(unexpected)
                        raise AssertionError(f"Invalid post-reduce input accepted: {label}")
                finally:
                    print(f'[EXPECTED_ERROR END] RuntimeError message="{message}"', flush=True)

            weights = t.reshape(t.to_layout(inputs["scores"], t.ROW_MAJOR_LAYOUT), (1, 1, 32, 8))
            op = t.experimental.deepseek_prefill.post_combine_reduce
            common = dict(expert_dim=3, output_memory_config=t.DRAM_MEMORY_CONFIG)
            reject(
                "rows33",
                "valid_token_rows must be in [1, 32] (0 disables)",
                lambda: op(inputs["combined"], weights, inputs["ids"], inputs["table"], valid_token_rows=33, **common),
            )
            reject(
                "no_dispatch_table",
                "valid_token_rows requires the dispatch-table expert-skip path",
                lambda: op(inputs["combined"], weights, valid_token_rows=2, **common),
            )
            multi = {
                key: upload(key, torch.cat((value, value), dim=1 if key == "ids" else 2), True)
                for key, value in cases[0][1].items()
                if key != "table"
            }
            multi_weights = t.reshape(t.to_layout(multi["scores"], t.ROW_MAJOR_LAYOUT), (1, 1, 64, 8))
            reject(
                "multiple_chunks",
                "valid_token_rows requires exactly one physical 32-row chunk",
                lambda: op(
                    multi["combined"], multi_weights, multi["ids"], inputs["table"], valid_token_rows=2, **common
                ),
            )
            row_index = t.from_torch(
                torch.arange(256, dtype=torch.int32).reshape(1, 1, 32, 8).repeat(4, 1, 1, 1),
                dtype=t.uint32,
                layout=t.ROW_MAJOR_LAYOUT,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=0),
                device=mesh,
                memory_config=t.DRAM_MEMORY_CONFIG,
            )
            row_table = t.reshape(inputs["combined"], (256, 2048))
            reject(
                "row_indexed",
                "valid_token_rows does not support combine_row_index",
                lambda: op(
                    row_table,
                    weights,
                    inputs["ids"],
                    inputs["table"],
                    combine_row_index=row_index,
                    valid_token_rows=2,
                    **common,
                ),
            )

        if args.valid_token_rows:
            check_invalid_parameters()
            gc.collect()
            save()

        # Warm both layout transformations, native signatures and reads before
        # capture. No reference layout temporary survives this call boundary.
        for name in ("reference", "candidate"):
            value = call(name)
            host_bits(value)
            t.deallocate(value)
            del value
        refresh(cases[0][1])
        t.synchronize_device(mesh)
        gc.collect()
        result["input_addresses"] = {key: int(value.buffer_address()) for key, value in inputs.items()}
        for name in ("reference", "candidate"):
            trace = t.begin_trace_capture(mesh, cq_id=0)
            traces[name] = trace
            try:
                outputs[name] = call(name)
            finally:
                t.end_trace_capture(mesh, trace, cq_id=0)
            trace_allocation_tracker.acknowledge_corruptible(outputs[name])
            result["captures"] += 1
        entries = mesh.num_program_cache_entries()
        result["acknowledged"] = {
            name: dict(trace_id=int(traces[name]), output_buffer_id=int(value.buffer_unique_id()))
            for name, value in outputs.items()
        }
        result["lifetime"] = (
            "Only two distinct fully written captured outputs are acknowledged and retained through trace release. "
            "Each replay is consumed into a CPU bit snapshot before another replay. The shared combined values, "
            "TILE scores, IDs and ownership table predate capture, retain addresses and are never acknowledged. "
            "Reference/candidate score layout temporaries die inside call; no broad scope or cache exemption."
        )
        for fixture, values in cases:
            refresh(values)
            original_sources = source_bits() if args.valid_token_rows else None
            actual = {}
            for name in traces:
                value = call(name)
                actual[name] = host_bits(value)
                t.deallocate(value)
                del value
            exact(f"{fixture}/eager", actual["candidate"], actual["reference"])
            zero_tail(actual["candidate"])
            if original_sources is not None:
                unchanged_sources(original_sources)
            result["cases"].append(dict(fixture=fixture, mode="eager", exact_bits=True))
            for repeat in range(2):
                # Repeat 1 has no refresh, so both changed and unchanged input
                # replays use the same trace IDs and persistent buffer addresses.
                for name, trace in traces.items():
                    gc.collect()
                    unsafe = t._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)
                    assert not unsafe, (name, list(unsafe))
                    t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    snapshot = host_bits(outputs[name])
                    exact(f"{fixture}/trace/{repeat}/{name}", snapshot, actual["reference"])
                    zero_tail(snapshot)
                    if original_sources is not None:
                        unchanged_sources(original_sources)
                assert {key: int(value.buffer_address()) for key, value in inputs.items()} == result["input_addresses"]
                assert mesh.num_program_cache_entries() == entries
                result["cases"].append(dict(fixture=fixture, mode="trace", unchanged=bool(repeat), exact_bits=True))
            if 0 < args.valid_token_rows < 32:
                # First establish the exact baseline above. Then expose only
                # the candidate to stale nonfinite payload, weights and local
                # IDs in omitted rows. Never run the reference on poisoned data.
                refresh(poison_omitted_rows(values, args.valid_token_rows))
                poisoned_sources = source_bits()
                try:
                    value = call("candidate")
                    snapshot = host_bits(value)
                    t.deallocate(value)
                    del value
                    exact(f"{fixture}/stale-tail/eager", snapshot, actual["reference"])
                    zero_tail(snapshot)
                    unchanged_sources(poisoned_sources)
                    result["cases"].append(dict(fixture=fixture, mode="stale_tail_eager", exact_bits=True))
                    for repeat in range(2):
                        gc.collect()
                        trace = traces["candidate"]
                        unsafe = t._ttnn.operations.trace.get_unsafe_tracked_ids(mesh, trace)
                        assert not unsafe, ("candidate", list(unsafe))
                        t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        snapshot = host_bits(outputs["candidate"])
                        exact(f"{fixture}/stale-tail/trace/{repeat}", snapshot, actual["reference"])
                        zero_tail(snapshot)
                        unchanged_sources(poisoned_sources)
                        result["cases"].append(
                            dict(fixture=fixture, mode="stale_tail_trace", unchanged=bool(repeat), exact_bits=True)
                        )
                    assert {key: int(value.buffer_address()) for key, value in inputs.items()} == result[
                        "input_addresses"
                    ]
                    assert mesh.num_program_cache_entries() == entries
                finally:
                    refresh(values)
                unchanged_sources(original_sources)
            save()
            print(f"PACKED_POSTREDUCE_PASS {fixture}", flush=True)
        result.update(status="PASS", hardware_verified=True, program_cache_entries=entries)
    except BaseException as error:
        failure = error
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
    finally:
        if mesh is not None:
            for trace in traces.values():
                try:
                    t.release_trace(mesh, trace)
                    result["releases"] += 1
                except BaseException as error:
                    result["cleanup_errors"].append(repr(error))
            try:
                t.close_mesh_device(mesh)
            except BaseException as error:
                result["cleanup_errors"].append(repr(error))
        if result["cleanup_errors"]:
            result["status"] = "FAIL"
        save()
        print(
            json.dumps(
                dict(status=result["status"], hardware_verified=result["hardware_verified"], cases=len(result["cases"]))
            )
        )
    if failure is not None:
        raise failure
    if result["status"] != "PASS":
        raise RuntimeError(f"Packed post-reduce cleanup failed: {result['cleanup_errors']}")


if __name__ == "__main__":
    main()
