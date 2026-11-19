# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Matched warmed latency near capacity using a verified streamed HF oracle.

This supplemental benchmark does not replace fresh stress or capacity gates.
Cache resets, uploads, planning, readbacks and correctness checks are outside
timing. Each timed prefill starts at ordinary zero state; decode replay restores
the resulting prefix state outside timing, as in the standard short benchmark.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import runpy
import statistics
import sys
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, pcc, weights


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    assert not args.output.exists(), "Preserve previous results"
    provenance_path = Path(str(args.output) + ".provenance.json")
    assert not provenance_path.exists(), "Preserve previous provenance"
    assert "ttnn" not in sys.modules, "TTNN must not be imported before host preflight/environment checks"
    harness_sha = sha(__file__)
    dependency_sha = {
        str(Path(__file__).with_name(name)): sha(Path(__file__).with_name(name))
        for name in ("reference.py", "long_context.py", "provenance.py", "runtime_audit.py")
    }
    assert sha(args.oracle_cache) == args.expect_oracle_sha256, "Frozen oracle cache content mismatch"
    assert sha(args.snapshot) == args.expect_sha256
    assert args.prefill_repeats >= 3 and args.decode_repeats >= 60
    for name in ("TT_VISIBLE_DEVICES", "TT_METAL_WATCHER"):
        assert name not in os.environ, (name, "Unset this variable for clean serial chip0 timing")
    for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_STREAMING_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
        assert os.environ.get(name) in (None, "", "0"), (name, "Use a clean serial timing environment")
    assert os.environ.get("QWEN_DECODER") == "fused"
    torch.set_num_threads(8)
    torch.manual_seed(154)
    c = config()
    sd = weights(args.layer)
    input_rng_state = torch.random.get_rng_state().clone()
    saved = torch.load(args.oracle_cache, map_location="cpu", weights_only=True)
    assert sha(args.oracle_cache) == args.expect_oracle_sha256, "Oracle changed during loading"
    assert set(saved) == {"identity", "expected"}, "Only reference outputs and identity are allowed"
    total = saved["identity"]["length"]
    assert 0 < args.length < total <= c.max_position_embeddings
    identity = {
        "layer": args.layer,
        "length": total,
        "seed": 154,
        "config": c.to_dict(),
        "versions": {v: str(importlib.metadata.version(v)) for v in ("torch", "transformers")},
        "reference_source": {name: sha(Path(__file__).with_name(name)) for name in ("reference.py", "long_context.py")},
        "weight_sha256": {
            name: hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, value in sd.items()
        },
    }
    assert saved["identity"] == identity, "HF oracle identity mismatch"
    expected = saved["expected"]
    assert expected.shape == (1, total, 2048) and expected.dtype == torch.bfloat16
    assert bool(torch.isfinite(expected).all())
    assert torch.equal(torch.random.get_rng_state(), input_rng_state), "Oracle validation consumed input RNG state"
    x = torch.randn(1, total, 2048, dtype=torch.bfloat16) * 0.05
    prefix, ref = x[:, : args.length], expected[:, : args.length]
    token, ref_d = x[:, args.length : args.length + 1], expected[:, args.length : args.length + 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    provenance = json.loads(provenance_path.read_text())
    provenance.update(
        measured_decoder_source=str(args.snapshot.resolve()),
        measured_decoder_source_sha256=args.expect_sha256,
        execution="TTNN device" if args.run_device else "host-only oracle validation",
        device=provenance["device"] if args.run_device else None,
        supplemental_benchmark=True,
        oracle_cache_sha256=args.expect_oracle_sha256,
        benchmark_harness_sha256=harness_sha,
        benchmark_dependency_sha256=dependency_sha,
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    result = dict(
        status="host_setup_pass",
        decoder="FusedDecoder",
        source_sha256=args.expect_sha256,
        harness_sha256=harness_sha,
        dependency_sha256=dependency_sha,
        layer=args.layer,
        length=args.length,
        context=total,
        allocated_context=total,
        decode_context=args.length + 1,
        decode_position=args.length,
        seed=154,
        input_rng_sha256=hashlib.sha256(input_rng_state.numpy().tobytes()).hexdigest(),
        oracle_cache=str(args.oracle_cache.resolve()),
        oracle_cache_sha256=args.expect_oracle_sha256,
        oracle_identity=identity,
        reference="Unchanged long_context.py streamed HF oracle; no reference state enters runtime",
        device_executed=False,
        acceptance_eligible=False,
        timing_policy="Serial; zero prefix state each pass; setup/reset/readback excluded; synchronized warmed prefill and blocking trace replay",
        prefill_checks=[],
        prefill_samples_ms=[],
        decode_samples_ms=[],
        decode_checks=[],
        device_closed=False,
        timing_sample_policy={
            "prefill_warmup": 1,
            "prefill_timed": args.prefill_repeats,
            "decode_correctness_warmups": 3,
            "decode_timed": args.decode_repeats,
        },
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def stability():
        checks = {
            "snapshot": sha(args.snapshot) == args.expect_sha256,
            "oracle_cache": sha(args.oracle_cache) == args.expect_oracle_sha256,
            "harness": sha(__file__) == harness_sha,
            **{name: sha(name) == expected for name, expected in dependency_sha.items()},
        }
        return checks

    def quality(reference, actual, residual):
        check = dict(pcc=pcc(reference, actual), contribution_pcc=pcc(reference - residual, actual - residual))
        assert all(math.isfinite(value) and 0.995 <= value <= 1.000001 for value in check.values()), check
        return check

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules, "Host-only preflight unexpectedly imported TTNN"
        result["ttnn_imported"] = False
        result["source_stability"] = stability()
        assert all(result["source_stability"].values()), result["source_stability"]
        save()
        print("LONG_BENCHMARK_HOST_SETUP_PASS", args.expect_sha256, flush=True)
        return

    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    assert all(stability().values()), "Inputs changed before device execution"
    decoder = runpy.run_path(str(args.snapshot))["FusedDecoder"]
    mesh, trace = None, None
    try:
        result["device_open_attempted"] = True
        save()
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
        result["device_executed"] = True
        dec = decoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        result["kind"] = dec.kind
        state = (
            dec.new_recurrent_state()
            if dec.kind == "linear_attention"
            else dec.new_kv_state(context_length=total, page_order=torch.randperm((total + 31) // 32))
        )
        keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")
        zeros = {key: ttnn.zeros_like(state[key]) for key in keys}
        tx = ttnn.from_torch(prefix, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        td = ttnn.from_torch(token, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        plan = dec.prepare_prefill(seq_len=args.length)
        positions = dec.prepare_positions(start_pos=args.length, seq_len=1) if dec.kind == "full_attention" else None
        first = None
        for iteration in range(args.prefill_repeats + 1):
            for key in keys:
                ttnn.copy(zeros[key], state[key])
            ttnn.synchronize_device(mesh)
            with device_only():
                start = time.perf_counter()
                out = dec.prefill_forward(tx, state=state, plan=plan)
                ttnn.synchronize_device(mesh)
                elapsed_ms = (time.perf_counter() - start) * 1000
            assert math.isfinite(elapsed_ms) and elapsed_ms > 0
            got = ttnn.to_torch(out).reshape(prefix.shape)
            check = dict(
                iteration=iteration,
                warmup=iteration == 0,
                **quality(ref, got, prefix),
                deterministic=first is None or torch.equal(first, got),
            )
            if first is None:
                first = got.clone()
            result["prefill_checks"].append(check)
            if iteration:
                result["prefill_samples_ms"].append(elapsed_ms)
            save()
            print("LONG_WARMED_PREFILL", iteration, elapsed_ms, check, flush=True)
            assert check["pcc"] >= 0.995 and check["contribution_pcc"] >= 0.995 and check["deterministic"]
            del out, got
        del first, zeros
        result["prefill_warmed_ms"] = statistics.median(result["prefill_samples_ms"])
        snapshot = {key: ttnn.clone(state[key]) for key in keys}

        def restore():
            for key in keys:
                ttnn.copy(snapshot[key], state[key])
            ttnn.synchronize_device(mesh)

        with device_only():
            out = dec.decode_forward(td, state=state, positions=positions)
        got = ttnn.to_torch(out).reshape(token.shape)
        eager_check = quality(ref_d, got, token)
        result["decode_eager_pcc"] = eager_check["pcc"]
        result["decode_eager_contribution_pcc"] = eager_check["contribution_pcc"]
        del out, got
        restore()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = dec.decode_forward(td, state=state, positions=positions)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(out)
        first = None
        for repeat in range(3):
            restore()
            with device_only():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            got = ttnn.to_torch(out).reshape(token.shape)
            check = dict(repeat=repeat, **quality(ref_d, got, token))
            check["deterministic"] = first is None or torch.equal(first, got)
            result["decode_checks"].append(check)
            assert check["deterministic"]
            if first is None:
                first = got.clone()
            else:
                assert torch.equal(first, got)
        for _ in range(args.decode_repeats):
            restore()
            with device_only():
                start = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                elapsed_ms = (time.perf_counter() - start) * 1000
            assert math.isfinite(elapsed_ms) and elapsed_ms > 0
            result["decode_samples_ms"].append(elapsed_ms)
        got = ttnn.to_torch(out).reshape(token.shape)
        assert torch.equal(first, got)
        final_check = quality(ref_d, got, token)
        result.update(
            status="supplemental_long_benchmark_pass",
            device_executed=True,
            deterministic=True,
            decode_traced_pcc=final_check["pcc"],
            decode_traced_contribution_pcc=final_check["contribution_pcc"],
            decode_traced_warmed_ms=statistics.median(result["decode_samples_ms"]),
            runtime_fallback_audit="pass",
        )
        save()
    except BaseException as error:
        result.update(status="supplemental_long_benchmark_failed", error=repr(error))
        save()
        raise
    finally:
        cleanup_errors = []
        if mesh is not None:
            if trace is not None:
                try:
                    ttnn.release_trace(mesh, trace)
                except BaseException as error:
                    cleanup_errors.append("release_trace: " + repr(error))
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except BaseException as error:
                cleanup_errors.append("close_mesh_device: " + repr(error))
        try:
            result["source_stability"] = stability()
            result["source_unchanged"] = all(result["source_stability"].values())
        except BaseException as error:
            result["source_unchanged"] = False
            cleanup_errors.append("source_stability: " + repr(error))
        result["cleanup_errors"] = cleanup_errors
        if cleanup_errors or not result["source_unchanged"]:
            result["status"] = "supplemental_long_benchmark_failed"
        save()
        assert not cleanup_errors, cleanup_errors
        assert result["source_unchanged"], result.get("source_stability")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--expect-sha256", required=True)
    parser.add_argument("--oracle-cache", type=Path, required=True)
    parser.add_argument("--expect-oracle-sha256", required=True, help="Frozen streamed-HF cache content hash")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=262143)
    parser.add_argument("--prefill-repeats", type=int, default=3)
    parser.add_argument("--decode-repeats", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-device", action="store_true")
    run(parser.parse_args())
