# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""paged_scaled_dot_product_attention_decode with OLMo-3.1-32B's decode settings and sliding window (4096), at positions on
both sides of the window, against a torch reference.

OLMo's 48 sliding-attention layers use a 4096-token window, so decode past position 4096 depends on the window handling.
OLMo's decode op settings: automatic chunking (q/k chunk size 0), HiFi3 with fp32 accumulation, bf16 KV cache.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    create_sliding_window_mask,
    fa_rand,
    nearest_n,
    nearest_pow_2,
    num_to_corerange,
)

POSITIONS = [1000, 3000, 3500, 4000, 4090, 4095, 4096, 4097, 4200, 4500, 5000, 5500, 6000, 7000]


def _run(
    device,
    window,
    auto_chunk,
    hifi3_fp32,
    sharded_out,
    positions=POSITIONS,
    b=8,
    nh=8,
    nkv=1,
    s=8192,
    d=128,
    single_active=False,
):
    block_size = 64
    blocks_per_seq = s // block_size
    torch.manual_seed(1234)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    perm = torch.randperm(b * blocks_per_seq)
    page_table = torch.argsort(perm).reshape(b, blocks_per_seq).to(torch.int32)

    def to_paged(c):
        return (
            c.reshape(b, nkv, blocks_per_seq, block_size, d)
            .transpose(1, 2)
            .reshape(b * blocks_per_seq, nkv, block_size, d)
        )

    paged_k, paged_v = to_paged(K)[perm], to_paged(V)[perm]
    dram = ttnn.DRAM_MEMORY_CONFIG
    tt_K = ttnn.as_tensor(paged_k, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V = ttnn.as_tensor(paged_v, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_pt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    padded_heads = nearest_pow_2(nearest_n(nh, n=32))
    grid = ttnn.CoreRangeSet({num_to_corerange(b)})
    spec = ttnn.ShardSpec(grid, (padded_heads, d), ttnn.ShardOrientation.ROW_MAJOR)
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    out_cfg = sharded if sharded_out else dram

    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi3 if hifi3_fp32 else ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=hifi3_fp32,
        packer_l1_acc=False,
    )
    results = {}
    for pos in positions:
        if pos >= s:
            continue
        # single_active: a batch-1 run decodes one user per column; the other slots carry position -1
        start_indices = [pos] + [-1] * (b - 1) if single_active else [pos + i for i in range(b)]
        chunk = 0 if auto_chunk else 128
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=0 if auto_chunk else padded_heads,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )
        Q = fa_rand(1, b, nh, d)
        tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=sharded)
        cp = ttnn.Tensor(torch.tensor(start_indices), ttnn.int32).to(device)
        tt_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            tt_pt,
            cur_pos_tensor=cp,
            scale=d**-0.5,
            sliding_window_size=window,
            program_config=program_config,
            compute_kernel_config=compute,
            memory_config=out_cfg,
        )
        got = ttnn.to_torch(tt_out)[:, :, :nh, :]
        ref_idx = [max(p_, 0) for p_ in start_indices]
        if window is not None:
            mask = create_sliding_window_mask(b, nh, s, ref_idx, window)
        else:
            mask = torch.zeros((b, nh, 1, s))
            for i in range(b):
                mask[i, :, :, ref_idx[i] + 1 :] = torch.finfo(torch.float32).min
        Qs = Q[:, :, :nh, :].permute(1, 2, 0, 3)
        Ks = K.repeat(1, nh // nkv, 1, 1)
        Vs = V.repeat(1, nh // nkv, 1, 1)
        ref = torch.nn.functional.scaled_dot_product_attention(Qs, Ks, Vs, mask, scale=d**-0.5, is_causal=False)
        ref = ref.squeeze(2).unsqueeze(0)
        active = [i for i, p_ in enumerate(start_indices) if p_ >= 0]
        ok, pcc = comp_pcc(ref[:, active], got[:, active], 0.99)
        results[pos] = (ok, pcc)
        logger.info(
            f"window={window} auto_chunk={auto_chunk} hifi3_fp32={hifi3_fp32} sharded_out={sharded_out} pos={pos}: {pcc} {'OK' if ok else 'BAD'}"
        )
    return results


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("window", [None, 4096], ids=["full", "window4096"])
@pytest.mark.parametrize("auto_chunk", [True, False], ids=["autochunk", "chunk128"])
@pytest.mark.parametrize("hifi3_fp32", [True, False], ids=["hifi3fp32", "hifi4"])
@pytest.mark.parametrize("sharded_out", [True, False], ids=["shardout", "dramout"])
@pytest.mark.parametrize("single_active", [False, True], ids=["all_active", "single_active"])
def test_paged_sdpa_decode_olmo_sliding_window(device, window, auto_chunk, hifi3_fp32, sharded_out, single_active):
    results = _run(device, window, auto_chunk, hifi3_fp32, sharded_out, single_active=single_active)
    bad = {p: pcc for p, (ok, pcc) in results.items() if not ok}
    assert not bad, f"PCC below 0.99 at positions {bad}"
