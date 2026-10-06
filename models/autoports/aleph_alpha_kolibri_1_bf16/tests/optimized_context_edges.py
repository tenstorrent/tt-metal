# SPDX-License-Identifier: Apache-2.0
"""Optimized context boundary continuation checks with a complete optimized KV prefix.

The unchanged paged-cache/context contract inherits complete streaming coverage
from the functional stage. This check allocates the full capacity with the new
weight/temporary footprint, computes every prefix K/V with the optimized QKV path, and executes the
optimized decoder at selected boundary chunks and traced decode positions. It does
not claim to execute every optimized prefill token.
"""

import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F

import ttnn

from .optimized_coverage import ROOT, Harness, runtime_audit
from .reference import norm, rope


def run(layer, capacity, decode_only=False, output=None):
    torch.set_num_threads(8)
    torch.manual_seed(654 + layer)
    h = Harness(layer, capacity)
    try:
        h.model.config.max_position_embeddings = capacity
        x = h.inputs(capacity)
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
        seed_tokens = max(1024, h.model.policy.prefill_chunk_size)
        assert capacity % seed_tokens == 0
        seed_in = h.tt(torch.zeros(1, 1, seed_tokens, 2560, dtype=torch.bfloat16))
        c, s = h.angles(0, seed_tokens)
        seed_cos, seed_sin = h.tt(c), h.tt(s)
        seed_pages = h.integer(pages[:, : seed_tokens // 32])
        seed_start = h.integer(torch.tensor([0], dtype=torch.int32))
        # Prove the final large-chunk temporary footprint fits with the full
        # cache allocated, and prepare every program before a live trace.
        warm = h.model.prefill_chunk_forward(
            seed_in,
            kv_cache=h.cache,
            page_table=table,
            chunk_page_table=seed_pages,
            chunk_start=seed_start,
            cos=seed_cos,
            sin=seed_sin,
            chunk_start_alignment=0,
        )
        del warm

        def seed_chunk():
            q, k, v = h.model._qkv(seed_in, seed_cos, seed_sin)
            ttnn.experimental.paged_fill_cache(h.cache[0], ttnn.typecast(k, h.cache[0].dtype), seed_pages)
            ttnn.experimental.paged_fill_cache(h.cache[1], ttnn.typecast(v, h.cache[1].dtype), seed_pages)
            del q, k, v

        seed_chunk()
        boundary_plans = []
        if not decode_only:
            for pos in (65536 - 128, 65536):
                if pos + 128 <= capacity:
                    plan = h.model.prepare_prefill(page_table_host=pages, seq_len=128, start_pos=pos)
                    warm = h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)
                    del warm
                    boundary_plans.append((pos, plan))
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
            value = h.inputs(1)
            h.copy(value.reshape(1, 1, 1, 2560), dec_in)
            h.copy(torch.tensor([pos], dtype=torch.int32), dec_kw["current_pos"])
            c, s = h.angles(pos, 1)
            h.copy(c, dec_kw["cos"])
            h.copy(s, dec_kw["sin"])
            expected = reference(pos, value)
            ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
            start = time.monotonic()
            for _ in range(10):
                ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(h.mesh)
            latency_ms = (time.monotonic() - start) * 100
            actual = ttnn.to_torch(dec_out)
            h.check(
                label, expected, actual, position=pos, context=pos + 1, trace_id=int(tid), traced_warmed_ms=latency_ms
            )

        checkpoints = {128, 1024, 8192, 65536, 262144, capacity}
        start_time = time.monotonic()
        try:
            for off in range(0, capacity, seed_tokens):
                h.copy(x[:, off : off + seed_tokens][None], seed_in)
                c, s = h.angles(off, seed_tokens)
                h.copy(c, seed_cos)
                h.copy(s, seed_sin)
                h.copy(pages[:, off // 32 : (off + seed_tokens) // 32], seed_pages)
                with runtime_audit():
                    seed_chunk()
            print("OPTIMIZED_FULL_KV_PREFIX_READY", capacity, flush=True)
            decode(capacity - 1, label="optimized_decode_with_optimized_prefix")
            for pos, plan in boundary_plans:
                h.copy(x[:, pos : pos + 128][None].contiguous(), inp)
                with runtime_audit():
                    boundary_output = h.model.prefill_forward(inp, kv_cache=h.cache, plan=plan)
                actual = ttnn.to_torch(boundary_output)[0, 0, -8:]
                del boundary_output
                expected = reference(pos + 120, x[:, pos + 120 : pos + 128])
                h.check("public_precision_boundary", expected, actual, logical_length=pos + 128)
            offsets = sorted({0, 128, 8192 - 128, 65536 - 128, 262144 - 128, capacity - 256, capacity - 128})
            if decode_only:
                for pos in (128, 8192, 65536):
                    if pos < capacity:
                        decode(pos, label="context_tuning_decode")
                offsets = []

            def timed_prefill():
                warm = h.model.prefill_chunk_forward(inp, **kw)
                del warm
                ttnn.synchronize_device(h.mesh)
                start = time.monotonic()
                result = h.model.prefill_chunk_forward(inp, **kw)
                ttnn.synchronize_device(h.mesh)
                return result, (time.monotonic() - start) * 1000

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
                    out, prefill_ms = timed_prefill()
                    actual = ttnn.to_torch(out)[0, 0, 103:111]
                    del out
                    expected = reference(off + 103, x[:, off + 103 : off + 111])
                    h.check(
                        "long_nonaligned_prefill",
                        expected,
                        actual,
                        logical_length=off + 111,
                        sampled_rows=[off + 103, off + 110],
                        warmed_prefill_ms=prefill_ms,
                    )
                    decode(off + 111)
                    # Restore the final chunk; all real prefix rows match x.
                    h.copy(value, inp)
                out, prefill_ms = timed_prefill()
                if off + 128 in checkpoints:
                    actual = ttnn.to_torch(out)[0, 0, -8:]
                    expected = reference(off + 120, x[:, off + 120 : off + 128])
                    h.check(
                        "optimized_prefill_with_optimized_prefix",
                        expected,
                        actual,
                        logical_length=off + 128,
                        sampled_rows=[off + 120, off + 127],
                        warmed_prefill_ms=prefill_ms,
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
                tested_prefill_end_position=None if decode_only else capacity,
                tested_decode_context=capacity,
                rows=h.rows,
                selected_chunk_offsets=offsets,
                complete_streaming_baseline=f"functional_decoder/context_{layer}.json",
                prefix_source="Optimized QKV/norm/RoPE and paged_fill_cache for every prefix row; independent HF K/V reference",
                largest_physical_chunk_warmed=seed_tokens,
                reference="all real K/V; eight query rows per checkpoint",
                trace_captures=1,
                trace_request_releases=0,
                allocation_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
                skip_program_cache=os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "1",
                seconds=time.monotonic() - start_time,
            )
            name = output or f"context_edges_{layer}.json"
            (ROOT / "doc/optimized_decoder" / name).write_text(json.dumps(result, indent=2) + "\n")
        finally:
            ttnn.release_trace(h.mesh, tid)
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--capacity", type=int, default=1048576)
    p.add_argument("--decode-only", action="store_true")
    p.add_argument("--output")
    a = p.parse_args()
    run(a.layer, a.capacity, a.decode_only, a.output)
