# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU mapping audit or parent-serialized exact packed epilogue device probe.

No TTNN/Torch imports unless --device is explicitly selected. The device path
compares both existing TTNN chains, every rank, original/changed inputs, eager
and trace replay, with optional one-op BF16 CB-roundtrip control. This is a
pointwise probe, not a full-decoder result or default selection.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tt import packed_swiglu_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
SOURCE_PATHS = (
    ROOT / "tt/packed_swiglu_candidate.py",
    Path(__file__),
    REPO / "ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp",
    REPO / "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.cpp",
    REPO / "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_program_factory.cpp",
    REPO / "ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_sfpu.hpp",
    REPO / "tt_metal/impl/buffers/tensor_accessor_args.cpp",
    REPO / "tt_metal/hw/inc/api/tensor/tensor_accessor.h",
    REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h",
    REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h",
)


def hashes():
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS}


def cpu_check():
    cases = 0
    for rows in (*range(1, 33), 33, 65, 513):
        for cores in (1, 8, 32, 64, 110):
            for width in (8704, 9216):
                p = candidate.tile_plan(rows, input_width=width, cores=cores)
                covered = [t for start, count in p["spans"] for t in range(start, start + count)]
                assert covered == list(range(p["output_tiles"]))
                seen = set()
                for t in covered:
                    row, col = divmod(t, 136)
                    gate, up = row * p["input_stride_tiles"] + col, row * p["input_stride_tiles"] + 136 + col
                    assert gate // p["input_stride_tiles"] == up // p["input_stride_tiles"] == row
                    assert gate % p["input_stride_tiles"] < 136 <= up % p["input_stride_tiles"] < 272
                    assert gate not in seen and up not in seen
                    seen.update((gate, up))
                assert len(seen) == 2 * p["output_tiles"]
                cases += 1
    shard_cases = 0
    for rows in (1, 17, 32, 33, 65):
        for storage in (4, 8, 16):
            height, width = candidate.output_shard_shape(rows, storage)
            assert height == (rows + 31) // 32 * 32
            assert width % 32 == 0 and width * storage >= 4352
            # Global output tile pages map to unique (storage core, local page),
            # including a partial last shard when136 tiles do not divide cores.
            pages = set()
            for row in range(height // 32):
                for col in range(136):
                    page = (col // (width // 32), row * (width // 32) + col % (width // 32))
                    assert page not in pages and page[0] < storage
                    pages.add(page)
            assert len(pages) == (height // 32) * 136
            shard_cases += 1
    assert candidate.output_shard_shape(32, 8) == (32, 544)
    tree = ast.parse((ROOT / "tt/packed_swiglu_candidate.py").read_text())
    banned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(n.name == "torch" for n in node.names):
            banned.append("torch import")
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name.startswith("torch.") or name in (
                "ttnn.to_torch",
                "ttnn.from_torch",
                "ttnn.to_layout",
                "ttnn.to_memory_config",
            ):
                banned.append(name)
    assert not banned, banned
    assert "silu_tile<false>(0)" in candidate._REGISTER_COMPUTE
    assert "mul_binary_tile<false>(0,1,0)" in candidate._REGISTER_COMPUTE
    assert "pack_tile(0,3)" in candidate._ROUNDTRIP_COMPUTE
    silu = SOURCE_PATHS[-2].read_text()
    multiply = SOURCE_PATHS[-1].read_text()
    dispatch = SOURCE_PATHS[3].read_text()
    assert "if constexpr (!is_fp32_dest_acc_en)" in silu and "RoundMode::Nearest" in silu
    assert "result = float32_to_bf16_rne(result)" in multiply
    assert "fast_and_approximate_mode.value_or(false)" in dispatch
    return {
        "pass": True,
        "scope": "CPU tile mapping and static source only; native compile/device equality untested",
        "mapping_cases": cases,
        "width_shard_mapping_cases": shard_cases,
        "runtime_host_conversions_or_layout_fallbacks": banned,
        "source_sha256": hashes(),
        "decode_b32_plan": candidate.tile_plan(32),
    }


def device_probe(args):
    from unittest.mock import patch

    import torch

    import ttnn

    torch.set_num_threads(2)
    rows_values = list(range(1, 33)) if args.rows == "all" else [int(s) for s in args.rows.split(",")]
    if not rows_values or any(r < 1 for r in rows_values) or args.repeats < 1 or args.replays < 2:
        raise ValueError("Positive rows/repeats and at least two changed-input replays are required")
    implementations = ["register", "roundtrip"] if args.implementation == "both" else [args.implementation]
    env = {
        k: os.environ.get(k, "0")
        for k in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING")
    }
    report = {
        "status": "running",
        "pass": False,
        "kind": "packed_swiglu_pointwise_probe",
        "scope": "same packed BF16 activation; no matmul/weight-layout change or full-decoder claim",
        "mesh": [1, args.chips],
        "rows": rows_values,
        "cores": args.cores,
        "input_placement": args.memory,
        "storage_cores": args.storage_cores if args.memory == "l1_width" else None,
        "environment": env,
        "source_sha256": hashes(),
        "cases": [],
        "timing_eligible": all(v in ("", "0") for v in env.values()),
        "runtime_conversion_guard": "pending",
        "trace_lifetime": "all group outputs, reference reads, copies and implementation warmups precede first capture",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def fail_conversion(*unused, **kw):
        raise RuntimeError("Unexpected host conversion inside pointwise execution/capture")

    @contextmanager
    def guarded():
        with patch.object(ttnn, "from_torch", fail_conversion), patch.object(ttnn, "to_torch", fail_conversion):
            yield

    def baseline(x, rounded, output=None):
        gate, up = x[:, :, :4352], x[:, :, 4352:]
        options = {} if output is None else {"output_tensor": output}
        if args.memory != "dram":
            options["memory_config"] = candidate.output_memory_config(x)
        return (
            ttnn.multiply(ttnn.silu(gate), up, **options)
            if rounded
            else ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], **options)
        )

    def host_shards(x):
        return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(x)]

    def compare(actual, expected):
        checks = []
        if len(actual) != args.chips or len(expected) != args.chips:
            raise AssertionError("Missing mesh-rank comparison")
        for rank, (a, b) in enumerate(zip(actual, expected)):
            if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
                raise AssertionError("Reference/result shape or dtype differs")
            mismatches = int((a.view(torch.int16) != b.view(torch.int16)).sum())
            af, bf = a.float().flatten(), b.float().flatten()
            finite = bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
            pcc = 1.0 if mismatches == 0 else float(torch.corrcoef(torch.stack((af, bf)))[0, 1])
            checks.append(
                {
                    "rank": rank,
                    "bit_equal": mismatches == 0,
                    "mismatched_elements": mismatches,
                    "finite": finite,
                    "pcc": pcc if math.isfinite(pcc) else None,
                    "max_abs_error": float((af - bf).abs().max()) if finite else None,
                }
            )
        return checks

    def input_values(rows, variant):
        generator = torch.Generator().manual_seed(771 + rows * 7 + variant)
        values = []
        edges = torch.tensor(
            [-100.0, -32.0, -8.0, -1.0, -0.5, -0.0, 0.0, 1e-30, -1e-30, 0.5, 1.0, 8.0, 32.0, 100.0],
            dtype=torch.bfloat16,
        )
        for rank in range(args.chips):
            x = (torch.randn((1, rows, 8704), generator=generator) * (1.0 + rank + 3.0 * variant)).bfloat16()
            x[:, :, : len(edges)] = edges.roll(rank + variant)
            x[:, :, 4352 : 4352 + len(edges)] = edges.roll(3 + rank + variant)
            values.append(x)
        return values

    mesh = None
    active_traces = []
    cache_frozen = False
    save()
    try:
        with guarded():
            try:
                ttnn.to_torch(None)
            except RuntimeError as exc:
                assert "Unexpected host conversion" in str(exc)
                report["conversion_guard_positive_control"] = "pass"
            else:
                raise AssertionError("Host conversion sentinel did not reject its positive control")
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, args.chips),
            physical_device_ids=[0] if args.chips == 1 else [],
            trace_region_size=32 * 1024 * 1024,
        )
        if mesh.arch() != ttnn.device.Arch.BLACKHOLE:
            raise ValueError("Probe requires Blackhole")
        for rows in rows_values:
            input_memory = ttnn.DRAM_MEMORY_CONFIG if args.memory == "dram" else ttnn.L1_MEMORY_CONFIG
            if args.memory == "l1_width":
                storage = ttnn.num_cores_to_corerangeset(
                    args.storage_cores, mesh.compute_with_storage_grid_size(), row_wise=True
                )
                input_memory = ttnn.create_sharded_memory_config(
                    ((rows + 31) // 32 * 32, math.ceil(272 / args.storage_cores) * 32),
                    storage,
                    ttnn.ShardStrategy.WIDTH,
                    use_height_and_width_as_shard_shape=True,
                )
            banks = [
                ttnn.from_torch(
                    torch.cat(input_values(rows, variant), -1),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=input_memory,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                )
                for variant in (0, 1)
            ]
            packed = ttnn.clone(banks[0])
            for rounded in (True, False):
                refs = []
                for bank in banks:
                    with guarded():
                        native = baseline(bank, rounded)
                    refs.append(host_shards(native))
                    ttnn.deallocate(native)
                    del native
                # CopyDeviceOperation also owns persistent program-cache storage.
                # Warm every replay-copy signature before any trace is alive.
                with guarded():
                    for bank in banks:
                        ttnn.copy(bank, packed)
                    ttnn.copy(banks[0], packed)
                    native_output = baseline(packed, rounded)
                    for _ in range(2):
                        warm = baseline(packed, rounded, output=native_output)
                        del warm
                prepared = {}
                for implementation in implementations:
                    with guarded():
                        out = candidate.packed_swiglu(
                            packed, round_silu=rounded, cores=args.cores, implementation=implementation
                        )
                        assert out.memory_config() == candidate.output_memory_config(packed)
                        candidate.packed_swiglu(
                            packed, round_silu=rounded, output=out, cores=args.cores, implementation=implementation
                        )
                    prepared[implementation] = {"output": out, "eager": compare(host_shards(out), refs[0])}
                ttnn.synchronize_device(mesh)
                warmed_entries = mesh.num_program_cache_entries()
                # All implementations must be ready, including the roundtrip
                # GenericOp program. Never skip tracking or acknowledge buffers.
                mesh.set_program_cache_misses_allowed(False)
                cache_frozen = True
                with guarded():
                    baseline_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    captured_native = baseline(packed, rounded, output=native_output)
                    ttnn.end_trace_capture(mesh, baseline_trace, cq_id=0)
                active_traces.append(baseline_trace)
                assert captured_native.buffer_address() == native_output.buffer_address()
                del captured_native
                for implementation in implementations:
                    out = prepared[implementation]["output"]
                    eager = prepared[implementation]["eager"]
                    with guarded():
                        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        captured = candidate.packed_swiglu(
                            packed, round_silu=rounded, output=out, cores=args.cores, implementation=implementation
                        )
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    active_traces.append(trace)
                    assert captured is out
                    replays = []
                    for step in range(args.replays):
                        variant = step % 2
                        with guarded():
                            ttnn.copy(banks[variant], packed)
                            ttnn.execute_trace(mesh, baseline_trace, cq_id=0, blocking=True)
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                        replays.append(
                            {
                                "variant": variant,
                                "candidate": compare(host_shards(out), refs[variant]),
                                "baseline_trace": compare(host_shards(native_output), refs[variant]),
                            }
                        )
                    with guarded():
                        ttnn.copy(banks[0], packed)
                        # Alternate order over three timing rounds to reduce order bias.
                        timings = {"baseline": [], "candidate": []}
                        for repetition in range(3):
                            order = (("baseline", baseline_trace), ("candidate", trace))
                            for label, trace_id in (order if repetition % 2 == 0 else reversed(order)):
                                ttnn.synchronize_device(mesh)
                                start = time.perf_counter()
                                for _ in range(args.repeats):
                                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                timings[label].append((time.perf_counter() - start) * 1000 / args.repeats)
                    checks = eager + [c for step in replays for k in ("candidate", "baseline_trace") for c in step[k]]
                    medians = {k: statistics.median(v) for k, v in timings.items()}
                    record = {
                        "rows": rows,
                        "round_silu": rounded,
                        "reference_chain": "silu_then_multiply" if rounded else "multiply_input_silu",
                        "implementation": implementation,
                        "pass": all(c["bit_equal"] and c["finite"] for c in checks),
                        "eager": eager,
                        "replays": replays,
                        "persistent_output_identity": captured is out,
                        "input_memory_config": str(packed.memory_config()),
                        "output_memory_config": str(out.memory_config()),
                        "default_placement_preserved": out.memory_config() == candidate.output_memory_config(packed),
                        "baseline_output_preallocated": True,
                        "program_cache_entries_after_all_warmups": warmed_entries,
                        "program_cache_misses_forbidden_during_capture_replay": True,
                        "timing_ms": medians,
                        "timing_rounds_ms": timings,
                        "repeats_per_round": args.repeats,
                        "candidate_change_pct": (medians["candidate"] / medians["baseline"] - 1) * 100,
                        "decision": "provisional_pointwise_only",
                        "traffic_plan": candidate.tile_plan(rows),
                    }
                    report["cases"].append(record)
                    save()
                    ttnn.release_trace(mesh, trace)
                    active_traces.remove(trace)
                    del captured, out
                ttnn.release_trace(mesh, baseline_trace)
                active_traces.remove(baseline_trace)
                assert mesh.num_program_cache_entries() == warmed_entries
                mesh.set_program_cache_misses_allowed(True)
                cache_frozen = False
                # Keep every candidate output alive until all group traces end.
                for item in prepared.values():
                    ttnn.deallocate(item["output"])
                ttnn.deallocate(native_output)
                del prepared
                del native_output
            for tensor in [packed, *banks]:
                ttnn.deallocate(tensor)
            del packed, banks
        report["runtime_conversion_guard"] = "pass_for_candidate_and_reference_calls_capture_replay"
        report["pass"] = bool(report["cases"]) and all(c["pass"] for c in report["cases"])
        report["status"] = "pass" if report["pass"] else "fail"
        save()
        if not report["pass"]:
            raise AssertionError("Packed epilogue failed bitwise comparison; inspect JSON")
    except BaseException:
        report["status"], report["pass"] = "fail", False
        report["failure"] = traceback.format_exc()
        save()
        raise
    finally:
        if mesh is not None:
            for trace_id in active_traces:
                ttnn.release_trace(mesh, trace_id)
            if cache_frozen:
                mesh.set_program_cache_misses_allowed(True)
            ttnn.close_mesh_device(mesh)
    print(json.dumps({"output": str(args.output), "pass": report["pass"], "cases": len(report["cases"])}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--cpu-check", action="store_true")
    action.add_argument("--device", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--chips", type=int, choices=(1, 4), default=4)
    p.add_argument(
        "--rows", default="1,2,3,17,31,32", help="Comma-separated compact rows or all for1..32; >32 also supported"
    )
    p.add_argument("--cores", type=int)
    p.add_argument("--memory", choices=("dram", "l1", "l1_width"), default="dram")
    p.add_argument("--storage-cores", type=int, choices=(4, 8, 16), default=8)
    p.add_argument("--implementation", choices=("register", "roundtrip", "both"), default="both")
    p.add_argument("--replays", type=int, default=4)
    p.add_argument("--repeats", type=int, default=100)
    args = p.parse_args()
    if args.cpu_check:
        result = cpu_check()
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        if not args.output:
            p.error("--device requires --output for failure and provenance evidence")
        device_probe(args)


if __name__ == "__main__":
    main()
