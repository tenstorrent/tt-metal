# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import json
import os
import time

import torch
from tracy import signpost

import ttnn

from .fused_coverage import ROOT, STD, Harness


def run(layer, baseline=False, tokens=128, repetitions=20):
    assert repetitions > 0
    torch.set_num_threads(8)
    torch.manual_seed(789 + layer)
    if baseline:
        from ..tt.functional_decoder import FunctionalDecoder
        from . import fused_coverage

        fused_coverage.FusedDecoder = FunctionalDecoder
    from . import fused_coverage
    from .fusion_candidates import FusedDecoder

    if os.environ.get("FUSIONS") and not baseline:
        fused_coverage.FusedDecoder = FusedDecoder
    for flag in os.environ.get("FUSIONS", "").split(","):
        if flag:
            setattr(FusedDecoder, flag, True)
    if os.environ.get("FUSION_IMPL") == "expert":
        from . import fused_coverage
        from .fused_expert_candidate import ExpertCandidate

        fused_coverage.FusedDecoder = ExpertCandidate
    if os.environ.get("FUSION_IMPL") in ("mask", "mask_rm", "mask_rm128", "minimal"):
        from .fused_final_candidates import FinalGraphCandidate

        fused_coverage.FusedDecoder = FinalGraphCandidate
    if os.environ.get("FUSION_IMPL") == "pad_rm":
        from .fused_final_candidates import RowMajorPadCandidate

        fused_coverage.FusedDecoder = RowMajorPadCandidate
    h = Harness(layer, allocation_tracking=False)
    try:
        x = (torch.randn(1, tokens + 1, 2560) * STD).bfloat16()
        expected = h.ref(x[:, :tokens])
        expected_decode = h.ref(x[:, tokens:], start=tokens)
        pages = torch.randperm(h.blocks, dtype=torch.int32)[None]
        table = h.integer(pages)
        c, s = h.angles(0, tokens)
        inp = h.tt(x[:, :tokens][None].contiguous())
        kw = dict(
            kv_cache=h.cache,
            page_table=table,
            chunk_page_table=h.integer(pages[:, : tokens // 32]),
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
        for _ in range(2):
            pre = h.model.prefill_chunk_forward(inp, **kw)
            dec = h.model.decode_forward(dec_in, **dec_kw)
            del pre, dec
        gc.collect()
        ttnn.synchronize_device(h.mesh)
        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
        try:
            dec = h.model.decode_forward(dec_in, **dec_kw)
        finally:
            ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
        try:
            # Outputs from eager prefill are ephemeral and gone before replay.
            ttnn.synchronize_device(h.mesh)
            signpost("PERF_PREFILL")
            start = time.monotonic()
            for iteration in range(repetitions):
                pre = h.model.prefill_chunk_forward(inp, **kw)
                if iteration < repetitions - 1:
                    del pre
            ttnn.synchronize_device(h.mesh)
            pre_ms = (time.monotonic() - start) * 1000 / repetitions
            signpost("PERF_PREFILL_END")
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
                decode_batch=1,
                decode_position=tokens,
                host_elapsed_ms=dict(prefill=pre_ms, decode=dec_ms),
                pcc=h.rows,
            )
            directory = ROOT / "doc/fused_decoder"
            if baseline:
                directory /= "baseline"
            directory /= os.environ.get("FUSION_TAG", "initial")
            directory.mkdir(parents=True, exist_ok=True)
            baseline_path = ROOT / f"doc/fused_decoder/baseline/reference/outputs_{layer}.pt"
            if not baseline and tokens == 128 and baseline_path.exists():
                from .run_decoder import pcc

                base = torch.load(baseline_path, weights_only=True)
                result["unfused_pcc"] = {
                    "prefill": pcc(base["prefill"], pre_cpu),
                    "decode": pcc(base["decode"], ttnn.to_torch(dec)),
                }
                assert min(result["unfused_pcc"].values()) >= 0.995
            result["fusions"] = os.environ.get("FUSIONS", "")
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
    p.add_argument("--baseline", action="store_true", help="Profile completed functional decoder")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--repetitions", type=int, default=20)
    a = p.parse_args()
    run(a.layer, a.baseline, a.tokens, a.repetitions)
