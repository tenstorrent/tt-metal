# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-input real-weight optimized-vs-TP4 decoder comparison."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from tracy import signpost
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder


@contextmanager
def runtime_device_only():
    """Guard public host tensor transfers around forward, excluding setup/readback."""

    def forbidden(*args, **kwargs):
        raise AssertionError("host tensor conversion reached runtime forward")

    with ExitStack() as stack:
        for name in ("from_torch", "to_torch", "as_tensor", "to_device", "from_device", "copy_host_to_device_tensor"):
            if hasattr(ttnn, name):
                stack.enter_context(patch.object(ttnn, name, forbidden))
        yield


def remap_cache_pages(page_table, cache_shards):
    """Reverse physical IDs and data together; logical history is unchanged."""
    blocks = cache_shards[0].shape[0]
    assert all(value.shape[0] == blocks for value in cache_shards)
    assert page_table.min() >= 0 and page_table.max() < blocks
    permutation = torch.arange(blocks - 1, -1, -1)
    return permutation[page_table.long()].to(torch.int32), [value[permutation].contiguous() for value in cache_shards]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--stress", type=int, default=0)
    parser.add_argument("--profile-drain", action="store_true")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=33)
    parser.add_argument("--layout", default="sharded")
    parser.add_argument("--legacy-decode-layout", action="store_true")
    parser.add_argument(
        "--decoder-source", type=Path, help="Test-only immutable stage4 TP4 timing control (B1 or B32 S33/S513)"
    )
    parser.add_argument("--policy-json", default="{}")
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--activation-file", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.policy_file:
        args.policy_json = args.policy_file.read_text()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    root = Path(__file__).parents[1]
    selected_source = root / ("tt/optimized_decoder.py" if args.baseline else "tt/multichip_decoder.py")
    decoder_cls = OptimizedDecoder if args.baseline else MultichipDecoder
    if args.decoder_source:
        assert not args.baseline and not args.legacy_decode_layout and args.layout == "sharded"
        assert args.batch == 1 or (
            args.batch == 32 and args.length in (33, 513) and args.stress == 0
        ), "Stage4 control supports B1 or the frozen B32 S33/S513 performance workloads"
        assert args.policy_file is None and json.loads(args.policy_json) == {}, "Stage4 control cannot override policy"
        selected_source = args.decoder_source.resolve()
        assert (
            hashlib.sha256(selected_source.read_bytes()).hexdigest()
            == "082c82faa6b9500f654dcc3f0e26f1c4fab656e987d3fd079bc1c79ec19a3c44"
        )
        assert (
            hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest()
            == "315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22"
        )
        name = "models.autoports.qwen_qwen3_6_27b.tt._stage4_control_multichip_decoder"
        spec = importlib.util.spec_from_file_location(name, selected_source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        decoder_cls = module.MultichipDecoder
    config = Qwen3_5TextConfig(**json.loads((root / "doc/hf_config.json").read_text())["text_config"])
    weights = load_layer(args.layer)
    x = torch.randn(args.batch, args.length, 5120, dtype=torch.bfloat16) * 0.1
    d = torch.randn(args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
    if args.activation_file:
        recorded = torch.load(args.activation_file, weights_only=True)
        x, d = recorded["x"], recorded["d"]
        assert tuple(x.shape) == (args.batch, args.length, 5120) and tuple(d.shape) == (args.batch, 1, 5120)
    pages = (args.length + max(33, args.stress) + 31) // 32
    table = torch.randperm(args.batch * pages).reshape(args.batch, pages)

    def tensor_hash(value):
        return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

    metadata = {
        "layer": args.layer,
        "batch": args.batch,
        "length": args.length,
        "input_sha256": tensor_hash(x),
        "decode_input_sha256": tensor_hash(d),
        "page_table_sha256": tensor_hash(table),
        "optimized_source_sha256": hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
        "hf_revision": "8f33b98937c6031bb49e713dee3e8e185aafd26f",
    }
    if args.activation_file:
        metadata["activation_file_sha256"] = hashlib.sha256(args.activation_file.read_bytes()).hexdigest()
    if args.stress:
        metadata["stress_steps"] = args.stress
    rope = Qwen3_5TextRotaryEmbedding(config)
    rotations = [
        rope(t, p)
        for t, p in [
            (x, torch.arange(args.length).expand(args.batch, -1)),
            (d, torch.full((args.batch, 1), args.length)),
        ]
    ]
    if not args.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if args.baseline else 4),
        physical_device_ids=[0] if args.baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )
    try:
        model = decoder_cls.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            **({} if args.baseline else {"policy": {"residual_layout": args.layout, **json.loads(args.policy_json)}}),
        )
        if args.diagnose:

            def inspect_value(label, value):
                if isinstance(value, (tuple, list)):
                    for i, v in enumerate(value):
                        inspect_value(f"{label}[{i}]", v)
                elif isinstance(value, ttnn.Tensor):
                    records = []
                    for tensor in ttnn.get_device_tensors(value):
                        host = ttnn.to_torch(tensor).float()
                        bad = ~torch.isfinite(host)
                        records.append(
                            {
                                "bad": int(bad.sum()),
                                "locations": bad.nonzero()[:4].tolist(),
                                "max": float(host.nan_to_num().abs().max()),
                            }
                        )
                    print("FINITE", label, list(value.shape), records, flush=True)

            def instrument(owner, name):
                operation = getattr(owner, name)

                def wrapped(*arguments, **keywords):
                    value = operation(*arguments, **keywords)
                    label = name + (":" + arguments[1] if len(arguments) > 1 and isinstance(arguments[1], str) else "")
                    inspect_value(label, value)
                    return value

                setattr(owner, name, wrapped)

            for name in ["_norm", "_linear", "_gdn", "_collective", "_finish"]:
                if hasattr(model, name):
                    instrument(model, name)
            instrument(ttnn.transformer, "chunk_gated_delta_rule")
            instrument(ttnn.transformer, "concatenate_heads")
        upload = model._upload if args.baseline else model.upload_residual
        # The immutable stage4 control consumes its original public [B,1,H]
        # input; the current decoder consumes [1,B,H]. Both upload outside the
        # timing window, with no adapter inserted into either measured forward.
        upload_decode = (
            upload
            if args.baseline or args.legacy_decode_layout or args.decoder_source
            else model.upload_decode_residual
        )
        tx, td = upload(x), upload_decode(d)
        state = (
            model.create_recurrent_state(args.batch)
            if model.kind == "linear_attention"
            else model.create_paged_state(table)
        )
        tensors = [state.conv, state.recurrent] if model.kind == "linear_attention" else [state.key, state.value]

        def read(t, residual=False):
            shards = ttnn.get_device_tensors(t)
            vals = [ttnn.to_torch(v) for v in shards]
            if args.baseline:
                return vals[0]
            if residual:
                value = torch.cat(vals, -1) if args.layout == "sharded" else vals[0]
                return value.reshape(args.batch, -1, 5120)
            if model.kind == "linear_attention" and len(t.shape) == 3:
                # Conv channels are rank-local Q,K,V, not contiguous global channels.
                return torch.cat(
                    [torch.cat([v[..., a:z] for v in vals], -1) for a, z in [(0, 512), (512, 1024), (1024, 2560)]], -1
                )
            return torch.cat(vals, 1)

        pk, dk = {}, {}
        if model.kind == "full_attention":
            pk = {"rope": tuple(model._upload(t.unsqueeze(1)) for t in rotations[0])}
            dk = {
                "rope": tuple(model._upload(t.unsqueeze(1)) for t in rotations[1]),
                "current_pos": ttnn.from_torch(
                    torch.full((args.batch,), args.length, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                ),
            }
        empty_state = [ttnn.clone(t) for t in tensors]
        initial_bulk_before = getattr(model, "_prefill_cache_bulk_calls", 0)
        initial_ordered_before = getattr(model, "_prefill_cache_ordered_calls", 0)
        with runtime_device_only():
            output = model.prefill_forward(tx, state=state, **pk)
        initial_cache_dispatch = {
            "bulk_calls": getattr(model, "_prefill_cache_bulk_calls", 0) - initial_bulk_before,
            "ordered_calls": getattr(model, "_prefill_cache_ordered_calls", 0) - initial_ordered_before,
        }
        result = {"prefill": read(output, True), "prefill_state": [read(t) for t in tensors]}
        prefilled_state = [ttnn.clone(t) for t in tensors]
        print("PREFILL_DONE", flush=True)
        with runtime_device_only():
            output = model.decode_forward(td, state=state, **dk)
        result.update(decode=read(output, True), decode_state=[read(t) for t in tensors])
        print("DECODE_DONE", flush=True)
        if args.baseline:
            report = {"baseline": True}
        else:
            ref = torch.load(args.reference, weights_only=True)
            assert ref.get("metadata") == metadata, "Stale or unmatched single-chip reference; regenerate it"
            report = {phase: pcc(ref[phase], result[phase]) for phase in ["prefill", "decode"]}
            report["contribution"] = {
                phase: pcc(ref[phase] - inp, result[phase] - inp) for phase, inp in [("prefill", x), ("decode", d)]
            }
            report["per_user_pcc"] = {
                phase: [pcc(ref[phase][i], result[phase][i]) for i in range(args.batch)]
                for phase in ["prefill", "decode"]
            }
            report["state"] = {
                phase: [pcc(a, b) for a, b in zip(ref[phase], result[phase])]
                for phase in ["prefill_state", "decode_state"]
            }
            report["pass"] = (
                min(
                    *[report[p] for p in ["prefill", "decode"]],
                    *report["contribution"].values(),
                    *[v for values in report["per_user_pcc"].values() for v in values],
                    *[v for values in report["state"].values() for v in values],
                )
                >= 0.995
            )
        report["initial_prefill_cache_dispatch"] = initial_cache_dispatch
        if (
            not args.baseline
            and model.kind == "full_attention"
            and model.policy.get("batched_prefill_cache", False)
            and args.batch > 1
            and args.length > 1
        ):
            assert initial_cache_dispatch["bulk_calls"] > 0, "Initial PCC did not exercise bulk prefill"
        report["finite"] = all(
            torch.isfinite(value).all().item()
            for phase in ["prefill", "decode", "prefill_state", "decode_state"]
            for value in (result[phase] if isinstance(result[phase], list) else [result[phase]])
        )
        assert report["finite"], "Non-finite output/cache in initial forward"
        # Preserve actual allocations/configs even when the first PCC gate fails.
        # Requested policy alone is not proof that weight precision reached setup.
        report["matrix_allocations"] = {
            key: {
                "logical_shape": list(value.shape),
                "padded_shape": list(value.padded_shape),
                "dtype": str(value.dtype),
                "memory_config": str(value.memory_config()),
            }
            for key, value in model.weights.items()
            if len(value.shape) == 2
        }
        report["matmul_compute_configs"] = {key: repr(value) for key, value in model.matmul_configs.items()}
        if not args.baseline and model.kind == "full_attention":
            report["decode_attention_plan"] = {
                "source": "resolved model configuration; lowered-op profile is separate evidence",
                "grid": model.policy.get("decode_sdpa_grid", [8, 8] if args.batch > 1 else [8, 4]),
                "q_chunk_size": 32,
                "k_chunk_size": model._decode_attention_chunk(state, decode=True),
                "key_dtype": str(state.key.dtype),
                "value_dtype": str(state.value.dtype),
                "page_table_shape": list(
                    (
                        model._attention_page_table(state)
                        if hasattr(model, "_attention_page_table")
                        else state.page_table
                    ).shape
                ),
            }
        if not args.baseline and not report["pass"]:
            report.update(
                status="initial_pcc_failed",
                layer=args.layer,
                batch=args.batch,
                length=args.length,
                selected_policy=dict(model.policy),
                reference_metadata=metadata,
                source_sha256=hashlib.sha256(selected_source.read_bytes()).hexdigest(),
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            raise AssertionError(report)
        if args.trace:

            def restore(snapshot):
                for src, dst in zip(snapshot, tensors):
                    ttnn.copy(src, dst)

            original, changed = upload_decode(d), upload_decode(-d)
            cases = [
                {"name": "original", "input": original, "state": prefilled_state},
                {"name": "changed_activation", "input": changed, "state": prefilled_state},
            ]
            if model.kind == "full_attention":

                def host_replicated(value, dtype, layout):
                    return ttnn.from_torch(
                        value.contiguous(), dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
                    )

                original_positions = torch.full((args.batch,), args.length, dtype=torch.int32)
                mixed_positions = (args.length - 1 - torch.arange(args.batch) % 3).clamp_min(0).to(torch.int32)
                physical_table = ttnn.to_torch(ttnn.get_device_tensors(state.page_table)[0])
                original_metadata = {
                    "positions": host_replicated(original_positions, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    "rope": [host_replicated(t.unsqueeze(1), ttnn.bfloat16, ttnn.TILE_LAYOUT) for t in rotations[1]],
                    "table": host_replicated(physical_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    "position_values": original_positions.tolist(),
                    "page_mapping": "original",
                }
                for case in cases:
                    case.update(original_metadata)
                mixed = rope(d, mixed_positions.reshape(args.batch, 1))
                cases.append(
                    {
                        **cases[0],
                        "name": "changed_positions_rope",
                        "positions": host_replicated(mixed_positions, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                        "rope": [host_replicated(t.unsqueeze(1), ttnn.bfloat16, ttnn.TILE_LAYOUT) for t in mixed],
                        "position_values": mixed_positions.tolist(),
                    }
                )
                # Change physical addresses while retaining exactly the same
                # logical history. new_cache[permutation[old]] = old_cache[old].
                # Reversal is its own inverse. Every rank keeps its local heads.
                remapped = []
                for saved in prefilled_state:
                    values = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(saved)]
                    remapped_table, values = remap_cache_pages(physical_table, values)
                    remapped.append(
                        model._upload(values[0], saved.dtype)
                        if args.baseline
                        else model._upload_shards(values, saved.dtype)
                    )
                cases.append(
                    {
                        **cases[0],
                        "name": "changed_page_mapping",
                        "state": remapped,
                        "table": host_replicated(remapped_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                        "page_mapping": "reverse_physical_ids_with_cache",
                    }
                )

            def prepare(case):
                restore(case["state"])
                ttnn.copy(case["input"], td)
                if model.kind == "full_attention":
                    ttnn.copy_host_to_device_tensor(case["positions"], dk["current_pos"])
                    for host, device in zip(case["rope"], dk["rope"]):
                        ttnn.copy_host_to_device_tensor(host, device)
                    ttnn.copy_host_to_device_tensor(case["table"], state.page_table)

            result["trace_cases"] = {}
            # Compile every restore/copy/forward variant before the only capture.
            for _ in range(2):
                prepare(cases[0])
                restore(empty_state)
                warm = model.prefill_forward(tx, state=state, **pk)
                del warm
                for case in cases:
                    prepare(case)
                    warm = model.decode_forward(td, state=state, **dk)
                    result["trace_cases"][case["name"]] = {
                        "output": read(warm, True),
                        "state": [read(t) for t in tensors],
                    }
                    del warm
            assert not torch.equal(result["trace_cases"]["changed_activation"]["output"], result["decode"])
            if model.kind == "full_attention":
                assert torch.equal(
                    result["trace_cases"]["changed_page_mapping"]["output"], result["decode"]
                ), "Physical page remapping changed logical output"
                assert not torch.equal(
                    result["trace_cases"]["changed_positions_rope"]["output"], result["decode"]
                ), "Position/RoPE negative control"
            prepare(cases[0])
            if args.profile_drain:
                ttnn.ReadDeviceProfiler(mesh)
            # Public table inspection/mutation above revoked bulk eligibility.
            # Re-arm using a fresh private upload, outside forward/capture.
            cache_policy = not args.baseline and model.policy.get("batched_prefill_cache", False)
            if cache_policy and model.kind == "full_attention" and args.batch > 1:
                # Decode trace capture occurs below, after this setup change.
                model.replace_paged_table_from_host(state, physical_table)
            cache_bulk_before = getattr(model, "_prefill_cache_bulk_calls", 0)
            cache_ordered_before = getattr(model, "_prefill_cache_ordered_calls", 0)
            # Timed eager prefill includes dispatch and completion; resets excluded.
            prefill_ms = []
            for _ in range(3):
                restore(empty_state)
                ttnn.synchronize_device(mesh)
                signpost("PERF_PREFILL")
                start = time.perf_counter()
                warm = model.prefill_forward(tx, state=state, **pk)
                ttnn.synchronize_device(mesh)
                prefill_ms.append((time.perf_counter() - start) * 1000)
                signpost("PERF_PREFILL_END")
                # Readback is outside the measured interval. Every measured
                # sample must reproduce the initially PCC-checked forward.
                assert torch.equal(read(warm, True), result["prefill"]), "Timed prefill output changed"
                assert all(
                    torch.equal(read(t), expected) for t, expected in zip(tensors, result["prefill_state"])
                ), "Timed prefill cache changed"
                del warm
            report["prefill_cache_dispatch"] = {
                "bulk_calls_during_timed_prefill": getattr(model, "_prefill_cache_bulk_calls", 0) - cache_bulk_before,
                "ordered_calls_during_timed_prefill": getattr(model, "_prefill_cache_ordered_calls", 0)
                - cache_ordered_before,
                "public_exposure_revokes_certification": True,
                "captured_prefill_bulk_enabled": False,
            }
            if cache_policy and model.kind == "full_attention" and args.batch > 1 and args.length > 1:
                assert report["prefill_cache_dispatch"]["bulk_calls_during_timed_prefill"] > 0, "Bulk prefill unused"
            restore(prefilled_state)
            for _ in range(2):
                warm = model.decode_forward(td, state=state, **dk)
                del warm
                restore(prefilled_state)
            ttnn.synchronize_device(mesh)

            # Sentinel covers the real executed path, including inherited methods.
            with runtime_device_only():
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced = model.decode_forward(td, state=state, **dk)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            tracked = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
            positive = False
            if tracked:
                unsafe = ttnn.empty((1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                try:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                except RuntimeError as exc:
                    if "unsafe" not in str(exc).lower() and "allocation" not in str(exc).lower():
                        raise
                    positive = True
                finally:
                    ttnn.deallocate(unsafe)
                    del unsafe
                assert positive, "allocation tracker positive control did not reject"
            replay_pcc, deterministic, replay_records = [], True, []
            for repeat in range(2):
                for case in cases:
                    prepare(case)
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    actual = read(traced, True)
                    actual_state = [read(t) for t in tensors]
                    expected = result["trace_cases"][case["name"]]
                    replay_pcc.append(pcc(actual, expected["output"]))
                    output_equal = torch.equal(actual, expected["output"])
                    state_equal = all(torch.equal(a, b) for a, b in zip(actual_state, expected["state"]))
                    deterministic &= output_equal and state_equal
                    entry = {
                        "case": case["name"],
                        "repeat": repeat,
                        "output_bit_equal_eager": output_equal,
                        "state_bit_equal_eager": state_equal,
                        "positions": case.get("position_values"),
                        "page_mapping": case.get("page_mapping"),
                    }
                    if not args.baseline:
                        baseline_case = ref["trace_cases"][case["name"]]
                        entry["baseline_output_pcc"] = pcc(actual, baseline_case["output"])
                        entry["baseline_state_pcc"] = [pcc(a, b) for a, b in zip(actual_state, baseline_case["state"])]
                        assert min(entry["baseline_output_pcc"], *entry["baseline_state_pcc"]) >= 0.995, entry
                    replay_records.append(entry)
            assert min(replay_pcc) >= 0.9999 and deterministic, replay_records
            prepare(cases[0])
            ttnn.synchronize_device(mesh)
            if args.profile_drain:
                ttnn.ReadDeviceProfiler(mesh)
            start = time.perf_counter()
            for _ in range(args.repeats):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            trace_ms = (time.perf_counter() - start) * 1000 / args.repeats
            if args.profile_drain:
                ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE")
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            signpost("PERF_DECODE_END")
            report["trace"] = {
                "id": str(tid),
                "captures": 1,
                "replays": len(cases) * 2 + args.repeats + 1,
                "cases": replay_records,
                "changed_positions_rope": model.kind == "full_attention",
                "changed_page_mapping": model.kind == "full_attention",
                "replay_pcc": replay_pcc,
                "bit_equal_eager": deterministic,
                "changed_input_control": True,
                "allocation_tracking": tracked,
                "skip_program_cache": os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
                "positive_control_rejected": positive,
                "acknowledged_buffers": [],
                "runtime_conversion_sentinels": "pass",
                "runtime_conversion_sentinel_scope": "eager prefill, eager decode, decode capture; six public transfer APIs",
            }
            report["timing"] = {
                "prefill_ms": sum(prefill_ms) / len(prefill_ms),
                "prefill_samples_ms": prefill_ms,
                "prefill_estimator": "arithmetic_mean_of_three_warmed_calls",
                "decode_trace_ms": trace_ms,
                "regime": "serial",
                "tracking_enabled": tracked,
                "decode_blocking": False,
            }
            ttnn.release_trace(mesh, tid)
        if args.stress:
            assert args.trace, "Stress reuses prepared trace metadata"
            prepare(cases[0])
            stress_original, stress_changed = cases[0]["input"], cases[1]["input"]
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            stress_output = model.decode_forward(td, state=state, **dk)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            runs = []
            for repetition in range(2):
                prepare(cases[0])
                snapshots = []
                for step in range(args.stress):
                    ttnn.copy(stress_original if step % 2 == 0 else stress_changed, td)
                    if model.kind == "full_attention":
                        positions = torch.full((args.batch,), args.length + step, dtype=torch.int32)
                        ttnn.copy_host_to_device_tensor(
                            host_replicated(positions, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), dk["current_pos"]
                        )
                        angles = rope(d, positions.reshape(args.batch, 1))
                        for value, target in zip(angles, dk["rope"]):
                            ttnn.copy_host_to_device_tensor(
                                host_replicated(value.unsqueeze(1), ttnn.bfloat16, ttnn.TILE_LAYOUT), target
                            )
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                    if step % 16 == 0 or step == args.stress - 1:
                        value = read(stress_output, True)
                        assert torch.isfinite(value).all(), (repetition, step)
                        snapshots.append(value)
                runs.append({"output": torch.stack(snapshots), "state": [read(t) for t in tensors]})
            ttnn.release_trace(mesh, tid)
            assert torch.equal(runs[0]["output"], runs[1]["output"])
            assert all(torch.equal(x, y) for x, y in zip(runs[0]["state"], runs[1]["state"]))
            result["stress"] = runs[0]
            report["stress"] = {
                "steps": args.stress,
                "runs": 2,
                "bit_equal": True,
                "changed_inputs": True,
                "increasing_positions": model.kind == "full_attention",
            }
            if not args.baseline:
                report["stress"]["output_pcc"] = pcc(ref["stress"]["output"], runs[0]["output"])
                report["stress"]["state_pcc"] = [pcc(x, y) for x, y in zip(ref["stress"]["state"], runs[0]["state"])]
                assert min(report["stress"]["output_pcc"], *report["stress"]["state_pcc"]) >= 0.995
        if args.baseline:
            result["metadata"] = metadata
            torch.save(result, args.reference)
        report["reference_metadata"] = metadata
        report.update(
            layer=args.layer,
            batch=args.batch,
            length=args.length,
            layout=args.layout,
            candidate_policy=json.loads(args.policy_json),
            source_sha256=hashlib.sha256(selected_source.read_bytes()).hexdigest(),
            optimized_baseline_sha256=hashlib.sha256((root / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
        )
        report["selected_policy"] = dict(model.policy)
        report["decoder_source_override"] = str(selected_source) if args.decoder_source else None
        report["provenance"] = {
            "decoder_source_sha256": report["source_sha256"],
            "optimized_baseline_sha256": report["optimized_baseline_sha256"],
            "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mesh": [1, 1 if args.baseline else 4],
            "hardware": "Blackhole P300c",
            "regime": "serial",
            "ccl_num_links": 0 if args.baseline else model.ccl.num_links,
            "semaphore_grid": None if args.baseline else [11, 10],
            "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
            "watcher": os.environ.get("TT_METAL_WATCHER", "0"),
            "timed_trace_replays": args.repeats,
            "decode_input_shape_per_rank": list(td.shape),
            "legacy_decode_layout": args.legacy_decode_layout,
        }
        if args.profile_drain:
            ttnn.ReadDeviceProfiler(mesh)
        from models.autoports.qwen_qwen3_6_27b.tests.check_optimized_multichip_evidence import loaded_native_snapshot

        report["native_loaded"] = loaded_native_snapshot(root.parents[2])
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        if not args.baseline:
            assert report["pass"], report
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
