# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Targeted LM-head weight-dtype probe (optimized_full_model).

The reduced tt-perf-report shows the column-sharded LM head (32 x 2048 x 25088) is the largest
terminal op: DRAM-bound at ~73.8% DRAM util, 98 cores, LoFi BF16 x BF16. This probe measures whether
a reduced-precision LM-head weight (BFP8) is (a) faster and (b) preserves the greedy top-k(k=1) token,
using the reduced one-of-each-kind model + a real decode hidden. Broad dtype frontier selection is
owned by $datatype-sweep; this is an evidence-backed feasibility/latency check for the LM-head lever.
"""
from __future__ import annotations

import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator


def _time_matmul(mesh, fn, iters=50):
    fn()  # warm
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t) / iters * 1e6  # us


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    try:
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=2048, num_layers=[0, 1, 4])
        m = gen.model
        P = 128
        torch.manual_seed(0)
        prompt = torch.randint(0, gen.vocab, (P,), dtype=torch.int64).tolist()
        gen._ensure_cache(1, P + 64)
        kv, pt = gen._kv_cache, gen._page_table
        h = m.prefill_layers(
            m.embed_prefill(gen._tokens_to_device(torch.tensor(prompt))), kv, pt, user_id=0, start_pos=0
        )
        last = ttnn.slice(h, [0, P - 1, 0], [1, P, gen.hidden])
        hid = m.final_norm(ttnn.reshape(last, (1, 1, 1, gen.hidden)))  # normed decode hidden [1,1,1,H]

        w_bf16 = m.lm_head_w
        w_bfp8 = ttnn.typecast(w_bf16, ttnn.bfloat8_b)

        def mm(w):
            return ttnn.linear(hid, w, compute_kernel_config=m._lm_ck)

        # latency
        t_bf16 = _time_matmul(mesh, lambda: mm(w_bf16))
        t_bfp8 = _time_matmul(mesh, lambda: mm(w_bfp8))

        # greedy token equality: gather full vocab, argmax
        def argmax_tok(w):
            shards = mm(w)
            full = ttnn.to_torch(shards, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).reshape(-1)[: gen.vocab]
            return int(torch.argmax(full)), full

        tok_bf16, lg_bf16 = argmax_tok(w_bf16)
        tok_bfp8, lg_bfp8 = argmax_tok(w_bfp8)
        # PCC of logits
        a = lg_bf16.float()
        b = lg_bfp8.float()
        pcc = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        print(
            "LMHEAD_PROBE",
            {
                "bf16_us": round(t_bf16, 1),
                "bfp8_us": round(t_bfp8, 1),
                "speedup_pct": round(100 * (t_bf16 - t_bfp8) / t_bf16, 1),
                "greedy_tok_bf16": tok_bf16,
                "greedy_tok_bfp8": tok_bfp8,
                "greedy_tok_match": tok_bf16 == tok_bfp8,
                "logits_pcc": round(pcc, 6),
            },
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
