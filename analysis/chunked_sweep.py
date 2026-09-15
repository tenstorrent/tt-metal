# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only CHUNKED/paged PREFILL sweep (TEN-4716 P3 validation).

Runs ttnn.transformer.chunked_scaled_dot_product_attention: a Q chunk of length Sq at absolute
position chunk_start_idx attends the paged KV cache [0 : chunk_start_idx+Sq] (dense prefix + causal
ramp). Captures per-engine counters (under tracy multipass) to validate predict()'s chunked K_eff
= chunk_start_idx/k_chunk + (Sq/k_chunk + 1)/2. Identity page table (contiguous) isolates compute
from paged-scatter BW. No torch golden.

Env: CK_S (cache len), CK_SQ (chunk query len), CK_START (chunk start idx), CK_NH, CK_D, CK_ITERS.
"""
from __future__ import annotations
import os
import pytest


@pytest.mark.parametrize("cstart", [int(x) for x in os.environ.get("CK_STARTS", "4096").split(",")])
def test_chunked_prefill(device, cstart):
    import torch
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    b = 1
    nh = int(os.environ.get("CK_NH", "16"))
    nkv = int(os.environ.get("CK_NKV", "16"))
    d = int(os.environ.get("CK_D", "128"))
    block_size = 128
    s = int(os.environ.get("CK_S", "8192"))  # total paged cache length
    Sq = int(os.environ.get("CK_SQ", "2048"))  # this chunk's query length
    assert (cstart + Sq) <= s, "chunk must fit in the cache"
    max_nb = s // block_size

    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    def to_paged(c):
        return c.reshape(b, nkv, max_nb, block_size, d).transpose(1, 2).reshape(b * max_nb, nkv, block_size, d)

    paged_k, paged_v = to_paged(K), to_paged(V)
    page_table = torch.arange(b * max_nb, dtype=torch.int32).reshape(b, max_nb)  # identity mapping
    q = fa_rand(b, nh, Sq, d)

    dram = ttnn.DRAM_MEMORY_CONFIG
    tt_q = ttnn.from_torch(
        q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram, pad_value=0.0
    )
    tt_k = ttnn.from_torch(
        paged_k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram, pad_value=0.0
    )
    tt_v = ttnn.from_torch(
        paged_v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram, pad_value=0.0
    )
    tt_pt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=block_size,
        exp_approx_mode=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    print(f"\n[chunked] cache={s} Sq={Sq} start={cstart} nh={nh} nkv={nkv} d={d}", flush=True)
    for _ in range(int(os.environ.get("CK_ITERS", "2"))):
        o = ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=tt_q,
            input_tensor_k=tt_k,
            input_tensor_v=tt_v,
            page_table_tensor=tt_pt,
            chunk_start_idx=cstart,
            compute_kernel_config=ck,
            program_config=pc,
        )
        ttnn.synchronize_device(device)
        o.deallocate()
    for t in (tt_q, tt_k, tt_v, tt_pt):
        t.deallocate()
    print(f"[chunked] OK cache={s} Sq={Sq} start={cstart}", flush=True)
