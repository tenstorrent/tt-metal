# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile one complete first-call/cold correctness prefill and sixteen measured decode traces.

This bounded-capture diagnostic does not report warmed prefill latency. Use a separate
clean benchmark for warmed latency. All original correctness and decode checks remain.
No native or decoder arithmetic is changed. Hardware execution is parent-owned.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
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
    hc = DynamicCache(config=c)
    ref = hf_forward(hf, x, hc, 0)
    hf_prefill_state = (
        hc.layers[args.layer].recurrent_states.clone() if c.layer_types[args.layer] == "linear_attention" else None
    )
    ref_d = hf_forward(hf, xd, hc, args.length)
    print("HF_DONE", args.layer, args.length, flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        dec = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        state = (
            dec.new_recurrent_state()
            if dec.kind == "linear_attention"
            else dec.new_kv_state(context_length=args.context, page_order=torch.randperm((args.context + 31) // 32))
        )
        chunk = getattr(args, "chunk_size", None)
        plan = dec.prepare_prefill(seq_len=args.length, **{"chunk_size": chunk} if chunk is not None else {})
        positions = dec.prepare_positions(start_pos=args.length, seq_len=1) if dec.kind == "full_attention" else None

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        tx, td = (upload(x), upload(xd))
        t0 = time.perf_counter()
        if args.profile:
            from tracy import signpost

            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_PREFILL")
        with device_only():
            result = dec.prefill_forward(tx, state=state, plan=plan)
        if args.profile:
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            ttnn.ReadDeviceProfiler(mesh)
        got = ttnn.to_torch(result).reshape(x.shape)
        print("PREFILL", pcc(ref, got), "wall", time.perf_counter() - t0, flush=True)
        scores = {
            "decoder": type(dec).__name__,
            "layer": args.layer,
            "kind": dec.kind,
            "length": args.length,
            "context": args.context,
            "prefill_chunk_size": max((part[1] for part in plan)),
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
            assert pcc(ref, got) >= 0.995
            scores[
                "profile_prefill_regime"
            ] = "first-call/cold, single complete correctness prefill; clean warmed latency separately measured"
            scores["profile_prefill_passes"] = 1
            scores["profile_warmed_latency_claim"] = False
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
        scores["deterministic"] = all((torch.equal(outputs[0], a) for a in outputs[1:]))
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
            for _ in range(16):
                with device_only():
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                ttnn.ReadDeviceProfiler(mesh)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE_END")
            scores["profile_decode_replays"] = 16
            scores["profile_regime"] = "serial; physical chip 0; allocation tracking disabled"
            scores["profile_drain"] = "before measured windows and between complete decode replays"
        scores["runtime_fallback_audit"] = "pass"
        ttnn.release_trace(mesh, trace)
        if args.output:
            Path(args.output).write_text(json.dumps(scores, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


def main():
    global FunctionalDecoder
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--prefill-repeats", type=int, default=10)
    parser.add_argument("--decode-repeats", type=int, default=60)
    args = parser.parse_args()
    source = args.snapshot.resolve()
    assert source.is_file() and (not Path(args.output).exists()), "Use a snapshot and a fresh output path"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    loader = importlib.machinery.SourceFileLoader("profiled_fused_candidate", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    candidate = importlib.util.module_from_spec(spec)
    loader.exec_module(candidate)
    from models.autoports.qwen_qwen3_6_35b_a3b.tests import provenance

    original_forward = candidate.FusedDecoder._forward

    def forward_with_profile_drain(self, *call_args, **kwargs):
        result = original_forward(self, *call_args, **kwargs)
        if not kwargs["decode"]:
            import ttnn

            ttnn.synchronize_device(self.mesh_device)
            ttnn.ReadDeviceProfiler(self.mesh_device)
        return result

    candidate.FusedDecoder._forward = forward_with_profile_drain
    FunctionalDecoder = candidate.FusedDecoder
    identity = dict(
        measured_decoder_source=str(source),
        measured_decoder_source_sha256=digest,
        candidate_selection="Explicit diagnostic snapshot; not production selection",
        acceptance_eligible=False,
        profiling=True,
        profile_event_drain="after every existing prefill _forward chunk; additional standard phase/replay drains retained",
        profile_wrapper_source=str(Path(__file__).resolve()),
        profile_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        profile_prefill_regime="first-call/cold; no second warm prefill; clean warmed latency separately measured",
        profile_prefill_passes=1,
        profile_warmed_latency_claim=False,
        original_harness_source="models/autoports/qwen_qwen3_6_35b_a3b/tests/run_decoder.py",
        original_harness_sha256="b8242ff98ce14e7465f9aae093f1f85767a73bc4d0905eab5066c472f200bbeb",
        profile_decode_contract="unchanged: 3 correctness + 1 warm + 16 measured trace replays",
    )
    original_record = provenance.record

    def record(path):
        original_record(path)
        target = Path(str(path) + ".provenance.json")
        value = json.loads(target.read_text())
        value.update(identity)
        target.write_text(json.dumps(value, indent=2) + "\n")

    provenance.record = record
    try:
        run(
            argparse.Namespace(
                layer=args.layer,
                length=args.length,
                context=args.context,
                seed=42,
                chunk_size=None,
                output=args.output,
                compare_functional=False,
                profile=True,
                prefill_repeats=args.prefill_repeats,
                decode_repeats=args.decode_repeats,
            )
        )
    finally:
        provenance.record = original_record
        candidate.FusedDecoder._forward = original_forward
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
        if Path(args.output).exists():
            path = Path(args.output)
            value = json.loads(path.read_text())
            value.update(identity)
            path.write_text(json.dumps(value, indent=2) + "\n")


if __name__ == "__main__":
    main()
