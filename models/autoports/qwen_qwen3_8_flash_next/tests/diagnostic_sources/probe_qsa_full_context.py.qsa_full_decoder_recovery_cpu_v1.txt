# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full layer-3 decode bridge at the last advertised context position.

Prefix inputs are the existing 2560-wide attention fixture, not full decoder
inputs. TT prefix state is produced by attention prefill from empty state, or
restored from a provenance-checked export of that exact TT computation. No HF
cache or intermediate is injected into TT. Each final 10240-wide input passes
through public FunctionalDecoder.decode_forward, including hyper-connections,
QSA, router, real experts, and shared expert. The independent reference builds
its prefix using HF projections, then calls the unchanged full HF decoder.

This proves full-decoder continuation/decode at the advertised context; it is
explicitly not evidence of full-decoder long-prefill on 10240-wide inputs.
Two changed final inputs branch from the same prefix because no further token
fits after position262143. Every warmup/capture/replay restores all TT state.
The explicit recovery-state option accepts separately validated reduced prefix
state evidence and retains its original failed all-output gate in this report.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import ReferenceCache
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_decoder import load_reference, metric_passes, metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.qsa_state_fixture import (
    STATE_TENSORS,
    load_prefix,
    runtime_source_hashes,
)


@torch.inference_mode()
def prepare_reference(args, reference, rope, source, provenance):
    key = {
        "version": 1,
        "prefix": args.prefix,
        "capacity": 262144,
        "seed": 1733 + args.prefix,
        "source_fixture_key": source["key"],
        "provenance": provenance,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if args.fixture_cache.exists():
        payload = torch.load(args.fixture_cache, weights_only=True, map_location="cpu")
        if payload["key"] != key:
            raise ValueError("CPU bridge fixture provenance changed; choose a new fixture-cache path")
        return payload, True
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    start_time = time.monotonic()
    prefix_k = torch.empty(1, 2, args.prefix, 256, dtype=torch.bfloat16)
    prefix_v = torch.empty_like(prefix_k)
    prefix_raw = torch.empty(1, args.prefix, 128, dtype=torch.bfloat16)
    attention = reference.self_attn
    for start in range(0, args.prefix, 32):
        end = min(start + 32, args.prefix)
        hidden = source["hidden"][:, start:end]
        cos, sin = rope(hidden, torch.arange(start, end).reshape(1, -1))
        key_states = attention.k_norm(attention.k_proj(hidden).reshape(1, end - start, 2, 256)).transpose(1, 2)
        prefix_k[:, :, start:end] = apply_rotary_pos_emb(key_states, cos=cos, sin=sin)
        prefix_v[:, :, start:end] = attention.v_proj(hidden).reshape(1, end - start, 2, 256).transpose(1, 2)
        prefix_raw[:, start:end] = attention.indexer.index_qk_proj(hidden)[..., 512:]
        if end % 32768 == 0 or end == args.prefix:
            print(json.dumps({"HF_prefix_rows": end, "seconds": time.monotonic() - start_time}), flush=True)
    generator = torch.Generator().manual_seed(key["seed"])
    inputs = [(torch.randn(1, 1, 10240, generator=generator) * 0.02).bfloat16() for _ in range(2)]
    assert not torch.equal(inputs[0], inputs[1])
    embeddings = rope(inputs[0], torch.arange(args.prefix + 1).reshape(1, -1))
    expected = []
    for branch, value in enumerate(inputs):
        cache = ReferenceCache(raw=prefix_raw, k=prefix_k, v=prefix_v)
        output = reference(
            value,
            position_embeddings=embeddings,
            attention_mask=torch.zeros(1, 1, 1, args.prefix + 1, dtype=torch.bfloat16),
            past_key_values=cache,
        )
        assert output.shape == value.shape and torch.isfinite(output).all()
        assert cache.k.shape[-2] == cache.raw.shape[1] == args.prefix + 1
        expected.append(output)
        print(json.dumps({"HF_full_decoder_branch": branch, "seconds": time.monotonic() - start_time}), flush=True)
    payload = {
        "key": key,
        "inputs": inputs,
        "expected": expected,
        "preparation_seconds": time.monotonic() - start_time,
        "reference": "Unmodified installed HF Qwen4ExpTextDecoderLayer layer3 on independently projected prefix",
    }
    args.fixture_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.fixture_cache.with_name(args.fixture_cache.name + f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, args.fixture_cache)
    finally:
        temporary.unlink(missing_ok=True)
    return payload, False


def run_device(args, config, weights, source, fixture, report):
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    if args.tt_recovery_state:
        from models.autoports.qwen_qwen3_8_flash_next.tests.qsa_recovery_fixture import load_recovery_prefix

        saved = load_recovery_prefix(
            args.tt_recovery_state,
            recovery_report=args.recovery_report,
            recovery_provenance=args.recovery_provenance,
            source_path=args.source_cache,
            source_key=source["key"],
            prefix=args.prefix,
            capacity=262144,
        )
    else:
        saved = (
            load_prefix(args.tt_prefix_state, source_key=source["key"], prefix=args.prefix, capacity=262144)
            if args.tt_prefix_state
            else None
        )
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill import output_metrics
    from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        model = FunctionalDecoder.from_state_dict(weights, hf_config=config, layer_idx=3, mesh_device=mesh)
        if saved:
            permutation = saved["tensors"]["page_table"]
        else:
            permutation = torch.randperm(8192, generator=torch.Generator().manual_seed(1980)).reshape(1, -1).int()
            if permutation[0, 0] == 0:
                permutation = permutation.roll(1, dims=1)
        state = model.create_state(capacity=262144, page_table=permutation)
        attention = state.attention

        def host(value, target=None):
            return ttnn.from_torch(
                value.contiguous(),
                dtype=target.dtype if target is not None else ttnn.bfloat16,
                layout=target.layout if target is not None else ttnn.TILE_LAYOUT,
            )

        def position():
            return int(ttnn.to_torch(attention.position).reshape(-1)[0])

        if saved:
            for name in STATE_TENSORS:
                target = getattr(attention, name)
                assert tuple(saved["tensors"][name].shape) == tuple(target.shape), name
                ttnn.copy_host_to_device_tensor(host(saved["tensors"][name], target), target)
            report["prefix"] = {
                "origin": saved["origin"],
                "export": str(args.tt_recovery_state or args.tt_prefix_state),
                "all_output_gate": False if args.tt_recovery_state else saved["prefix_metric"],
                "runtime_hashes_verified": True,
            }
            if args.tt_recovery_state:
                report["prefix"].update(
                    strict_full_prefill_passed=False,
                    original_full_prefill=saved["original_full_prefill"],
                    verified_recovery=saved["verified_recovery"],
                    boundary_evidence=saved["boundary_evidence"],
                )
        else:
            inputs = ttnn.to_device(host(source["hidden"][:, : args.prefix].unsqueeze(0)), mesh)
            with runtime_audit():
                output = model.chunked_attn.prefill_forward(inputs, state=attention)
            ttnn.synchronize_device(mesh)
            actual = ttnn.to_torch(output).reshape(1, args.prefix, 2560)
            measured = output_metrics(source["expected"][:, : args.prefix], actual)
            report["prefix"] = {
                "origin": "Attention prefix computed from initially empty TT state in this run",
                "all_output_gate": measured,
            }
            assert measured["passed"], "Attention prefix output gate failed"
            ttnn.deallocate(inputs)
            ttnn.deallocate(output)
        assert position() == args.prefix
        names = ("k_cache", "v_cache", "raw_history", "compressed_keys", "position")
        mutable = {name: getattr(attention, name) for name in names}
        prefix_host = {name: host(ttnn.to_torch(tensor), tensor) for name, tensor in mutable.items()}

        def restore():
            for name, tensor in mutable.items():
                ttnn.copy_host_to_device_tensor(prefix_host[name], tensor)

        hidden_host = [host(value.unsqueeze(0)) for value in fixture["inputs"]]
        hidden_device = ttnn.to_device(hidden_host[0], mesh)
        with runtime_audit():
            warmed = model.decode_forward(hidden_device, state=state)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warmed)
        restore()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            output = model.decode_forward(hidden_device, state=state)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(output)
        first_outputs, first_states = [], []
        report["decode"] = []
        repeated_output, repeated_state = [], []
        for repetition in range(2):
            for branch in range(2):
                restore()
                assert position() == args.prefix
                ttnn.copy_host_to_device_tensor(hidden_host[branch], hidden_device)
                start = time.perf_counter()
                with runtime_audit():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                seconds = time.perf_counter() - start
                actual = ttnn.to_torch(output).reshape(1, 1, 10240)
                final_state = {name: ttnn.to_torch(tensor).clone() for name, tensor in mutable.items()}
                assert position() == args.prefix + 1
                if repetition == 0:
                    first_outputs.append(actual.clone())
                    first_states.append(final_state)
                    report["decode"].append(
                        {
                            "branch": branch,
                            "position": args.prefix,
                            "position_after": position(),
                            "traced_host_wall_seconds": seconds,
                            **metrics(fixture["expected"][branch], actual, fixture["inputs"][branch]),
                        }
                    )
                else:
                    repeated_output.append(torch.equal(actual, first_outputs[branch]))
                    repeated_state.append(
                        {name: torch.equal(final_state[name], first_states[branch][name]) for name in names}
                    )
        report.update(
            captures=1,
            replays=4,
            changed_final_inputs=2,
            repeat_output_equal=repeated_output,
            repeat_state_equal=repeated_state,
            runtime_fallback_audit="clean",
            allocation_tracking=True,
            distinct_input_outputs_differ=not torch.equal(first_outputs[0], first_outputs[1]),
        )
        report["passed"] = (
            all(metric_passes(row) for row in report["decode"])
            and all(repeated_output)
            and all(all(row.values()) for row in repeated_state)
            and report["distinct_input_outputs_differ"]
        )
        ttnn.release_trace(mesh, trace)
        trace = None
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument(
        "--source-cache", type=Path, default=Path("/home/ttuser/qwen38-flash/qsa_cpu_prefill_262144.pt")
    )
    state_group = parser.add_mutually_exclusive_group()
    state_group.add_argument("--tt-prefix-state", type=Path, help="Reuse a passing version-1 TT-produced prefix export")
    state_group.add_argument(
        "--tt-recovery-state", type=Path, help="Reuse a separately validated version-2 boundary export"
    )
    parser.add_argument("--recovery-report", type=Path)
    parser.add_argument("--recovery-provenance", type=Path)
    parser.add_argument("--prefix", type=int, default=262143)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.prefix < 262144:
        parser.error("Prefix must leave one final token inside advertised context")
    if args.tt_recovery_state:
        if not args.recovery_report or not args.recovery_provenance or args.prefix != 262143:
            parser.error("Recovery state requires its closed report/provenance and prefix262143")
    elif args.recovery_report or args.recovery_provenance:
        parser.error("Recovery evidence requires --tt-recovery-state")
    torch.set_num_threads(8)
    config, reference, rope, weights, provenance = load_reference(args.snapshot)
    source = torch.load(args.source_cache, weights_only=True, mmap=True, map_location="cpu")
    for field in ("config_sha256", "reference_sha256", "transformers"):
        assert source["key"][field] == provenance[field], field
    assert source["key"]["snapshot"] == str(args.snapshot.resolve())
    assert source["key"]["length"] == source["key"]["capacity"] == config.max_position_embeddings == 262144
    assert source["key"]["decode_steps"] == 0 and source["key"]["seed"] == 1979
    fixture, hit = prepare_reference(args, reference, rope, source, provenance)
    report = {
        "scope": __doc__,
        "layer_idx": 3,
        "advertised_context": 262144,
        "physical_capacity": 262144,
        "decode_context": args.prefix + 1,
        "threshold": 0.995,
        "stage_acceptance": "not_evaluated",
        "provenance": provenance,
        "CPU_fixture": {
            "path": str(args.fixture_cache),
            "hit": hit,
            "key": fixture["key"],
            "reference": fixture["reference"],
            "preparation_seconds": fixture["preparation_seconds"],
        },
        "device_executed": not args.cpu_only,
        "runtime_source_sha256": runtime_source_hashes(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not args.cpu_only:
            run_device(args, config, weights, source, fixture, report)
    except Exception as exc:
        report["passed"] = False
        report["error"] = f"{type(exc).__name__}: {str(exc).split('backtrace:')[0]}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not args.cpu_only and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
