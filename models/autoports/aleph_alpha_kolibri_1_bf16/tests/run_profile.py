# SPDX-License-Identifier: Apache-2.0
import argparse
import gc
import hashlib
import json
import time

import torch
from tracy import signpost

import ttnn

from .run_coverage import ROOT, STD, Harness


def run(layer, baseline=False):
    torch.set_num_threads(8)
    torch.manual_seed(789 + layer)
    if baseline:
        from . import diagnostic_baseline
    h = Harness(layer, allocation_tracking=False)
    if baseline:
        h.provenance["executed_decoder"] = dict(
            path=str(diagnostic_baseline.path.relative_to(ROOT)),
            sha256=hashlib.sha256(diagnostic_baseline.path.read_bytes()).hexdigest(),
        )
    try:
        x = (torch.randn(1, 129, 2560) * STD).bfloat16()
        expected = h.ref(x[:, :128])
        expected_decode = h.ref(x[:, 128:], start=128)
        pages = torch.randperm(h.blocks, dtype=torch.int32)[None]
        table = h.integer(pages)
        c, s = h.angles(0, 128)
        inp = h.tt(x[:, :128][None].contiguous())
        kw = dict(
            kv_cache=h.cache,
            page_table=table,
            chunk_page_table=h.integer(pages[:, :4]),
            chunk_start=h.integer(torch.tensor([0], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        dec_in = h.tt(x[:, 128:].reshape(1, 1, 1, 2560))
        c, s = h.angles(128, 1)
        dec_kw = dict(
            kv_cache=h.cache,
            page_table=table,
            current_pos=h.integer(torch.tensor([128], dtype=torch.int32)),
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
            for iteration in range(3):
                pre = h.model.prefill_chunk_forward(inp, **kw)
                if iteration < 2:
                    del pre
            ttnn.synchronize_device(h.mesh)
            pre_ms = (time.monotonic() - start) * 1000 / 3
            signpost("PERF_PREFILL_END")
            h.check("profile_prefill", expected, ttnn.to_torch(pre))
            del pre  # Eager outputs must be dead before replaying the live trace.
            # Warm replay before entering the measured replay window.
            ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
            signpost("PERF_DECODE")
            start = time.monotonic()
            for _ in range(3):
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(h.mesh)
            dec_ms = (time.monotonic() - start) * 1000 / 3
            signpost("PERF_DECODE_END")
            h.check("profile_decode_replay", expected_decode, ttnn.to_torch(dec))
            result = dict(
                provenance=h.provenance,
                layer=layer,
                repetitions=3,
                prefill_tokens=128,
                decode_batch=1,
                decode_position=128,
                host_elapsed_ms=dict(prefill=pre_ms, decode=dec_ms),
                pcc=h.rows,
            )
            directory = ROOT / "doc/functional_decoder"
            if baseline:
                directory /= "before_numerical_fix/untracked_perf"
            directory.mkdir(parents=True, exist_ok=True)
            result["runtime_diagnostics"] = "disabled; separate audited correctness runs"
            (directory / f"profile_{layer}.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            ttnn.release_trace(h.mesh, tid)
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--baseline", action="store_true", help="Profile retained original source for repair cost")
    a = p.parse_args()
    run(a.layer, a.baseline)
