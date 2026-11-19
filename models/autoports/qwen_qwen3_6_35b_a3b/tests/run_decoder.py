# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real-weight layer validation CLI; all Torch and conversions stay at boundaries."""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.decoder_factory import Decoder as FunctionalDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    if args.output:
        record(args.output)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    c = config()
    sd = weights(args.layer)
    hf = layer(sd, args.layer)
    x = torch.randn(1, args.length, c.hidden_size, dtype=torch.bfloat16) * 0.05
    xd = torch.randn(1, 1, c.hidden_size, dtype=torch.bfloat16) * 0.05
    activation_file = getattr(args, "activation_file", None)
    if activation_file:
        payload = torch.load(activation_file, weights_only=True)
        assert payload["metadata"]["checkpoint_revision"] == "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
        x = payload["layers"][str(args.layer)]["x"]
        xd = payload["layers"][str(args.layer)]["xd"]
        assert x.shape == (1, args.length, c.hidden_size) and xd.shape == (1, 1, c.hidden_size)
    hc = DynamicCache(config=c)
    ref = hf_forward(hf, x, hc, 0)
    hf_prefill_state = (
        hc.layers[args.layer].recurrent_states.clone() if c.layer_types[args.layer] == "linear_attention" else None
    )
    ref_d = hf_forward(hf, xd, hc, args.length)
    print("HF_DONE", args.layer, args.length, flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        dec = FunctionalDecoder.from_state_dict(
            sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh, **json.loads(getattr(args, "build_kwargs", "{}"))
        )
        state = (
            dec.new_recurrent_state()
            if dec.kind == "linear_attention"
            else dec.new_kv_state(context_length=args.context, page_order=torch.randperm((args.context + 31) // 32))
        )
        chunk = getattr(args, "chunk_size", None)
        plan = dec.prepare_prefill(seq_len=args.length, **({"chunk_size": chunk} if chunk is not None else {}))
        positions = dec.prepare_positions(start_pos=args.length, seq_len=1) if dec.kind == "full_attention" else None

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        tx, td = upload(x), upload(xd)
        t0 = time.perf_counter()
        with device_only():
            result = dec.prefill_forward(tx, state=state, plan=plan)
        got = ttnn.to_torch(result).reshape(x.shape)
        print("PREFILL", pcc(ref, got), "wall", time.perf_counter() - t0, flush=True)
        scores = {
            "decoder": type(dec).__name__,
            "activation_source": str(activation_file) if activation_file else "seeded random BF16, std0.05",
            "precision_policy": getattr(dec, "precision_policy", None),
            "layer": args.layer,
            "kind": dec.kind,
            "length": args.length,
            "context": args.context,
            "prefill_chunk_size": max(part[1] for part in plan),
            "requested_chunk_size": chunk,
            "prefill_pcc": pcc(ref, got),
            "prefill_contribution_pcc": pcc(ref - x, got - x),
        }
        assert scores["prefill_pcc"] >= 0.995, scores
        if hf_prefill_state is not None:
            host_state = ttnn.to_torch(state["recurrent"]).reshape(hf_prefill_state.shape)
            scores["hf_prefill_state_pcc"] = pcc(hf_prefill_state, host_state)
            scores["hf_prefill_state_relative_l2"] = float(
                (hf_prefill_state - host_state).norm() / hf_prefill_state.norm()
            )
            assert scores["hf_prefill_state_pcc"] >= 0.995
        del result
        if args.profile:
            from tracy import signpost

            for key in ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v"):
                ttnn.copy(ttnn.zeros_like(state[key]), state[key])
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_PREFILL")
            with device_only():
                result = dec.prefill_forward(tx, state=state, plan=plan)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            ttnn.ReadDeviceProfiler(mesh)
            assert pcc(ref, ttnn.to_torch(result).reshape(x.shape)) >= 0.995
            del result
        if getattr(args, "compare_functional", False):
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import FunctionalDecoder as Baseline

            baseline = Baseline.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
            control_state = (
                baseline.new_recurrent_state()
                if dec.kind == "linear_attention"
                else baseline.new_kv_state(context_length=args.context)
            )
            control_plan = baseline.prepare_prefill(seq_len=args.length)
            with device_only():
                control = baseline.prefill_forward(tx, state=control_state, plan=control_plan)
            control_host = ttnn.to_torch(control).reshape(x.shape)
            scores["unfused_prefill_pcc"] = pcc(control_host, got)
            if dec.kind == "linear_attention":
                control_recurrent = ttnn.to_torch(control_state["recurrent"])
                scores["unfused_prefill_state_pcc"] = pcc(control_recurrent, ttnn.to_torch(state["recurrent"]))
                assert scores["unfused_prefill_state_pcc"] >= 0.995
            scores["unfused_prefill_contribution_pcc"] = pcc(control_host - x, got - x)
            with device_only():
                control_d = baseline.decode_forward(td, state=control_state, positions=positions)
            control_decode_host = ttnn.to_torch(control_d).reshape(xd.shape).clone()
            del control, control_d, baseline, control_state
            assert scores["unfused_prefill_pcc"] >= 0.995
            assert scores["unfused_prefill_contribution_pcc"] >= 0.995
        if not args.profile:
            times = []
            for _ in range(getattr(args, "prefill_repeats", 3)):
                for key in ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v"):
                    ttnn.copy(ttnn.zeros_like(state[key]), state[key])
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                with device_only():
                    result = dec.prefill_forward(tx, state=state, plan=plan)
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - begin) * 1000)
                assert pcc(ref, ttnn.to_torch(result).reshape(x.shape)) >= 0.995
                del result
            scores["prefill_warmed_ms"] = statistics.median(times)
            scores["prefill_samples_ms"] = times
        state_keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")
        snapshot = {key: ttnn.clone(state[key]) for key in state_keys}

        def restore():
            for key in state_keys:
                ttnn.copy(snapshot[key], state[key])

        with device_only():
            warm = dec.decode_forward(td, state=state, positions=positions)
        scores["decode_eager_pcc"] = pcc(ref_d, ttnn.to_torch(warm).reshape(xd.shape))
        print("DECODE_EAGER", scores["decode_eager_pcc"], flush=True)
        del warm
        restore()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = dec.decode_forward(td, state=state, positions=positions)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(out)
        outputs = []
        for repeat in range(3):
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            outputs.append(ttnn.to_torch(out).reshape(xd.shape).clone())
        scores["decode_traced_pcc"] = pcc(ref_d, outputs[0])
        scores["deterministic"] = all(torch.equal(outputs[0], a) for a in outputs[1:])
        print("TRACED", scores, flush=True)
        assert scores["decode_traced_pcc"] >= 0.995, scores
        assert scores["deterministic"]
        if getattr(args, "compare_functional", False):
            scores["unfused_decode_pcc"] = pcc(control_decode_host, outputs[0])
            scores["unfused_decode_contribution_pcc"] = pcc(control_decode_host - xd, outputs[0] - xd)
            assert scores["unfused_decode_pcc"] >= 0.995
            assert scores["unfused_decode_contribution_pcc"] >= 0.995
        if not args.profile:
            times = []
            for _ in range(getattr(args, "decode_repeats", 12)):
                restore()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                times.append((time.perf_counter() - begin) * 1000)
            scores["decode_traced_warmed_ms"] = statistics.median(times)
            scores["decode_samples_ms"] = times
            scores["latency_tracking_enabled"] = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
        if args.profile:
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE")
            profile_times = []
            for _ in range(16):
                begin = time.perf_counter()
                with device_only():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                profile_times.append((time.perf_counter() - begin) * 1000)
                # Measurement boundary, never part of a decoder pass. Drain
                # each completed replay to avoid dropping profiler markers.
                ttnn.ReadDeviceProfiler(mesh)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE_END")
            scores["profile_decode_host_samples_ms"] = profile_times
            scores["profile_decode_host_median_ms"] = statistics.median(profile_times)
            scores["profile_decode_replays"] = 16
            scores["profile_regime"] = "serial; physical chip 0; allocation tracking disabled"
            scores["profile_drain"] = "before measured windows and between complete decode replays"
        scores["runtime_fallback_audit"] = "pass"
        ttnn.release_trace(mesh, trace)
        if args.output:
            Path(args.output).write_text(json.dumps(scores, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-kwargs", default="{}")
    p.add_argument("--activation-file")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output")
    p.add_argument("--compare-functional", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--prefill-repeats", type=int, default=3)
    p.add_argument("--decode-repeats", type=int, default=12)
    run(p.parse_args())
