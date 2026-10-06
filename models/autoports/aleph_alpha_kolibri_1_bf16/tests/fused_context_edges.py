# SPDX-License-Identifier: Apache-2.0
"""Fused context boundary continuation checks using an HF-seeded cache prefix.

The unchanged paged-cache/context contract inherits complete streaming coverage
from the functional stage. This check allocates the full capacity with the new
weight/temporary footprint, seeds all-prefix reference K/V, and executes the
fused decoder at selected boundary chunks and traced decode positions. It does
not claim to execute every fused prefill token.
"""

import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F

import ttnn

from .fused_coverage import ROOT, STD, Harness, runtime_audit
from .reference import norm, rope


def run(layer, capacity):
    torch.set_num_threads(8)
    torch.manual_seed(654 + layer)
    h = Harness(layer, capacity)
    try:
        h.model.config.max_position_embeddings = capacity
        x = (torch.randn(1, capacity, 2560) * STD).bfloat16()
        # Independent reference keys/values for every real input token.
        rk = torch.empty(1, 4, capacity, 128, dtype=torch.bfloat16)
        rv = torch.empty_like(rk)
        w = h.weights
        for off in range(0, capacity, 4096):
            count = min(4096, capacity - off)
            z = norm(x[:, off : off + count], w["input_layernorm.weight"])
            k = F.linear(z, w["self_attn.k_proj.weight"]).view(1, count, 4, 128).transpose(1, 2)
            k = norm(k, w["self_attn.k_norm.weight"])
            if h.ref.sliding:
                k = rope(k, torch.arange(off, off + count)[None])
            rk[:, :, off : off + count] = k
            rv[:, :, off : off + count] = (
                F.linear(z, w["self_attn.v_proj.weight"]).view(1, count, 4, 128).transpose(1, 2)
            )
        print("FULL_REFERENCE_KV_READY", capacity, flush=True)
        pages = torch.randperm(h.blocks, dtype=torch.int32)[None]
        table = h.integer(pages)
        n = 128
        c, s = h.angles(0, n)
        inp = h.tt(torch.zeros(1, 1, n, 2560, dtype=torch.bfloat16))
        kw = dict(
            kv_cache=h.cache,
            page_table=table,
            chunk_page_table=h.integer(pages[:, :4]),
            chunk_start=h.integer(torch.tensor([0], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        with runtime_audit():
            warm = h.model.prefill_chunk_forward(inp, **kw)
        del warm
        dec_in = h.tt(torch.zeros(1, 1, 1, 2560, dtype=torch.bfloat16))
        c, s = h.angles(0, 1)
        dec_kw = dict(
            kv_cache=h.cache,
            page_table=table,
            current_pos=h.integer(torch.tensor([0], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        with runtime_audit():
            warm = h.model.decode_forward(dec_in, **dec_kw)
        del warm
        gc.collect()
        ttnn.synchronize_device(h.mesh)
        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
        try:
            dec_out = h.model.decode_forward(dec_in, **dec_kw)
        finally:
            ttnn.end_trace_capture(h.mesh, tid, cq_id=0)

        def reference(pos, value):
            h.ref.cache = (rk, rv)
            return h.ref(value, start=pos)

        def decode(pos, label="full_context_traced_decode"):
            value = (torch.randn(1, 1, 2560) * STD).bfloat16()
            h.copy(value.reshape(1, 1, 1, 2560), dec_in)
            h.copy(torch.tensor([pos], dtype=torch.int32), dec_kw["current_pos"])
            c, s = h.angles(pos, 1)
            h.copy(c, dec_kw["cos"])
            h.copy(s, dec_kw["sin"])
            expected = reference(pos, value)
            ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
            actual = ttnn.to_torch(dec_out)
            h.check(label, expected, actual, position=pos, context=pos + 1, trace_id=int(tid))

        checkpoints = {128, 1024, 8192, 65536, 262144, capacity}
        start_time = time.monotonic()
        try:
            if True:
                # Diagnose the longest SDPA/decode path before the expensive
                # prefill sweep. This control uses reference K/V only; it is
                # explicitly excluded from complete-prefill evidence below.
                for values, destination in zip((rk, rv), h.cache):
                    logical = values.reshape(1, 4, h.blocks, 32, 128)[0].permute(1, 0, 2, 3).contiguous()
                    physical = torch.empty_like(logical)
                    physical[pages[0].long()] = logical
                    h.copy(physical, destination)
                decode(capacity - 1, label="fused_decode_with_reference_prefix")
                del physical, logical
            offsets = sorted({0, 128, 8192 - 128, 65536 - 128, 262144 - 128, capacity - 256, capacity - 128})
            for off in offsets:
                value = x[:, off : off + 128][None].contiguous()
                c, s = h.angles(off, 128)
                h.copy(value, inp)
                h.copy(c, kw["cos"])
                h.copy(s, kw["sin"])
                h.copy(pages[:, off // 32 : off // 32 + 4], kw["chunk_page_table"])
                h.copy(torch.tensor([off], dtype=torch.int32), kw["chunk_start"])
                # First complete the awkward tail at context-17. Future padding
                # is zero; decode consumes exactly that logical prompt.
                if off + 128 in {262144, capacity}:
                    padded = value.clone()
                    padded[:, :, 111:] = 0
                    h.copy(padded, inp)
                    out = h.model.prefill_chunk_forward(inp, **kw)
                    actual = ttnn.to_torch(out)[0, 0, 103:111]
                    del out
                    expected = reference(off + 103, x[:, off + 103 : off + 111])
                    h.check(
                        "long_nonaligned_prefill",
                        expected,
                        actual,
                        logical_length=off + 111,
                        sampled_rows=[off + 103, off + 110],
                    )
                    decode(off + 111)
                    # Restore the final chunk; all real prefix rows match x.
                    h.copy(value, inp)
                out = h.model.prefill_chunk_forward(inp, **kw)
                if off + 128 in checkpoints:
                    actual = ttnn.to_torch(out)[0, 0, -8:]
                    expected = reference(off + 120, x[:, off + 120 : off + 128])
                    h.check(
                        "fused_prefill_with_reference_prefix",
                        expected,
                        actual,
                        logical_length=off + 128,
                        sampled_rows=[off + 120, off + 127],
                    )
                del out
                if off + 128 in {262144, capacity}:
                    decode(off + 127)
                    # Decode changed the final row: refill it before continuing.
                    out = h.model.prefill_chunk_forward(inp, **kw)
                    del out
                if (off + 128) % 8192 == 0:
                    ttnn.synchronize_device(h.mesh)
                    print("PROGRESS", off + 128, "SECONDS", round(time.monotonic() - start_time, 2), flush=True)
            result = dict(
                provenance=h.provenance,
                layer=layer,
                capacity=capacity,
                tested_prefill_end_position=capacity,
                tested_decode_context=capacity,
                rows=h.rows,
                selected_chunk_offsets=offsets,
                complete_streaming_baseline=f"functional_decoder/context_{layer}.json",
                prefix_source="HF-projected all-prefix K/V; selected chunks overwritten by fused path",
                reference="all real K/V; eight query rows per checkpoint",
                trace_captures=1,
                trace_request_releases=0,
                allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
                skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "1",
                seconds=time.monotonic() - start_time,
            )
            (ROOT / f"doc/fused_decoder/context_edges_{layer}.json").write_text(json.dumps(result, indent=2) + "\n")
        finally:
            ttnn.release_trace(h.mesh, tid)
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--capacity", type=int, default=1048576)
    a = p.parse_args()
    run(a.layer, a.capacity)
