# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import json
import os
import time

import torch
from tracy import signpost

import ttnn

from .optimized_coverage import ROOT, STD, Harness


def run(layer, baseline=False, tokens=128, repetitions=20, public=False, chunk_size=None):
    assert repetitions > 0
    torch.set_num_threads(8)
    torch.manual_seed(789 + layer)
    from . import optimized_coverage

    if baseline:
        from ..tt.fused_decoder import FusedDecoder

        optimized_coverage.OptimizedDecoder = FusedDecoder
        optimized_coverage.decoder_options = lambda: {}
        os.environ["OPT_CACHE"] = "bfloat16"
    h = Harness(layer, capacity=max(1024, (tokens + 512) // 512 * 512), allocation_tracking=False)
    try:
        x = (torch.randn(1, tokens + 1, 2560) * STD).bfloat16()
        if os.environ.get("OPT_REAL_INPUT") == "1":
            recorded = torch.load(ROOT / f"doc/optimized_decoder/recorded_inputs/layer_{layer}.pt", weights_only=True)
            x = recorded[:, torch.arange(tokens + 1) % recorded.shape[1]].clone()
            assert x.shape[1] == tokens + 1
        expected = h.ref(x[:, :tokens])
        expected_decode = h.ref(x[:, tokens:], start=tokens)
        pages = torch.randperm(h.blocks, dtype=torch.int32)[None]
        table = h.integer(pages)
        c, s = h.angles(0, tokens)
        inp = h.tt(x[:, :tokens][None].contiguous())
        kw = dict(
            kv_cache=h.cache,
            page_table=table,
            chunk_page_table=h.integer(pages[:, : (tokens + 31) // 32]),
            chunk_start=h.integer(torch.tensor([0], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        dec_in = h.tt(x[:, tokens:].reshape(1, 1, 1, 2560))
        c, s = h.angles(tokens, 1)
        dec_kw = dict(
            kv_cache=h.cache,
            page_table=table,
            current_pos=h.integer(torch.tensor([tokens], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        if public:
            if chunk_size is None:
                chunk_size = getattr(getattr(h.model, "policy", None), "prefill_chunk_size", 128)
            plan = h.model.prepare_prefill(page_table_host=pages, seq_len=tokens, chunk_size=chunk_size)

            def prefill():
                return h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)

        else:

            def prefill():
                return h.model.prefill_chunk_forward(inp, **kw)

        for _ in range(2):
            pre = prefill()
            dec = h.model.decode_forward(dec_in, **dec_kw)
            del pre, dec
            if os.environ.get("OPT_PROFILE_DRAIN") == "1":
                ttnn.ReadDeviceProfiler(h.mesh)
        gc.collect()
        ttnn.synchronize_device(h.mesh)
        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
        try:
            dec = h.model.decode_forward(dec_in, **dec_kw)
        finally:
            ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
        try:
            if os.environ.get("OPT_PROFILE_DRAIN") == "1":
                ttnn.ReadDeviceProfiler(h.mesh)
            # Outputs from eager prefill are ephemeral and gone before replay.
            ttnn.synchronize_device(h.mesh)
            signpost("PERF_PREFILL")
            start = time.monotonic()
            for iteration in range(repetitions):
                pre = prefill()
                if iteration < repetitions - 1:
                    del pre
            ttnn.synchronize_device(h.mesh)
            pre_ms = (time.monotonic() - start) * 1000 / repetitions
            signpost("PERF_PREFILL_END")
            if os.environ.get("OPT_PROFILE_DRAIN") == "1":
                ttnn.ReadDeviceProfiler(h.mesh)
            h.check("profile_prefill", expected, ttnn.to_torch(pre))
            pre_cpu = ttnn.to_torch(pre)
            del pre  # Eager outputs must be dead before replaying the live trace.
            # Warm replay before entering the measured replay window.
            ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
            signpost("PERF_DECODE")
            start = time.monotonic()
            for _ in range(repetitions):
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(h.mesh)
            dec_ms = (time.monotonic() - start) * 1000 / repetitions
            signpost("PERF_DECODE_END")
            h.check("profile_decode_replay", expected_decode, ttnn.to_torch(dec))
            result = dict(
                provenance=h.provenance,
                layer=layer,
                repetitions=repetitions,
                prefill_tokens=tokens,
                public_api=public,
                logical_chunk_size=chunk_size if public else tokens,
                physical_prefill_chunks=[
                    dict(offset=c[0], valid=c[1], physical=c[2], decode=c[3]) for c in plan["slots"][0]
                ]
                if public
                else [dict(offset=0, valid=tokens, physical=tokens, decode=False)],
                decode_batch=1,
                decode_position=tokens,
                host_elapsed_ms=dict(prefill=pre_ms, decode=dec_ms),
                pcc=h.rows,
            )
            directory = ROOT / "doc/optimized_decoder"
            if baseline:
                directory /= "baseline"
            directory /= os.environ.get("OPT_TAG", "initial")
            directory.mkdir(parents=True, exist_ok=True)
            baseline_path = ROOT / f"doc/optimized_decoder/baseline/reference/outputs_{layer}.pt"
            if not baseline and tokens == 128 and baseline_path.exists() and os.environ.get("OPT_REAL_INPUT") != "1":
                from .run_decoder import pcc

                base = torch.load(baseline_path, weights_only=True)
                result["unfused_pcc"] = {
                    "prefill": pcc(base["prefill"], pre_cpu),
                    "decode": pcc(base["decode"], ttnn.to_torch(dec)),
                }
                assert min(result["unfused_pcc"].values()) >= 0.995
            result["policy"] = os.environ.get("OPT_POLICY", "{}")
            result["input_source"] = (
                "recorded_checkpoint" if os.environ.get("OPT_REAL_INPUT") == "1" else "synthetic_scaled"
            )
            torch.save({"prefill": pre_cpu, "decode": ttnn.to_torch(dec)}, directory / f"outputs_{layer}.pt")
            result["runtime_diagnostics"] = "disabled; separate audited correctness runs"
            (directory / f"profile_{layer}.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            ttnn.release_trace(h.mesh, tid)
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--baseline", action="store_true", help="Profile completed fused decoder")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=20)
    p.add_argument("--public", action="store_true")
    p.add_argument("--chunk-size", type=int)
    a = p.parse_args()
    run(a.layer, a.baseline, a.tokens, a.repetitions, a.public, a.chunk_size)
