# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared checks for SDPAProgramConfig(fp32_accumulation=True) on SDPA decode (post-commit and nightly suites).

The default decode kernel rounds the running softmax sum and output accumulator through BF16 at every K chunk and
tree merge, so its error grows with context length. Each check runs the op with and without the option, compares
both against an FP64 reference and requires the option to stay under a relative-L2 bound and to beat the default by
a given factor.
"""

import math

import torch
from loguru import logger

import ttnn


def rel_l2(actual, expected):
    actual, expected = actual.double().reshape(-1), expected.double().reshape(-1)
    return float((actual - expected).norm() / expected.norm())


def compute_config(device, fp32_dest_acc_en=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )


def program_config(device, q_chunk_size, k_chunk_size, fp32_accumulation, max_cores_per_head_batch=16):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
        max_cores_per_head_batch=max_cores_per_head_batch,
        fp32_accumulation=fp32_accumulation,
    )


def reference_attention(q, k, v, ends):
    """FP64 attention of q [h, rows, d] over k/v [kv_heads, s, d]; query row r sees keys [0, ends[r])."""
    heads, kv_heads, d = q.shape[0], k.shape[0], q.shape[-1]
    k, v = k.double().repeat_interleave(heads // kv_heads, 0), v.double().repeat_interleave(heads // kv_heads, 0)
    out = torch.empty(q.shape, dtype=torch.float64)
    for r, end in enumerate(ends):
        scores = torch.einsum("hd,hsd->hs", q[:, r].double(), k[:, :end]) / math.sqrt(d)
        out[:, r] = torch.einsum("hs,hsd->hd", scores.softmax(-1), v[:, :end])
    return out


def paged_kv(device, batch, kv_heads, capacity, block_size, dtype, seed):
    """Random K/V in a shuffled paged cache. Returns device tensors, the page table and the stored values."""
    torch.manual_seed(seed)
    blocks = batch * capacity // block_size
    k = torch.randn(blocks, kv_heads, block_size, 128)
    v = torch.randn(blocks, kv_heads, block_size, 128)
    page_table = torch.randperm(blocks, dtype=torch.int32).reshape(batch, capacity // block_size)
    tt_k = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_page_table = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # The reference reads back what was stored (block-float formats round on upload).
    k, v = ttnn.to_torch(tt_k).float(), ttnn.to_torch(tt_v).float()

    def linear(cache, row):
        pages = cache[page_table[row].long()]  # [pages, kv_heads, block, d]
        return pages.permute(1, 0, 2, 3).reshape(kv_heads, -1, 128)

    return tt_k, tt_v, tt_page_table, [(linear(k, b), linear(v, b)) for b in range(batch)]


def check(errors, bound, factor):
    logger.info(f"relative L2 vs FP64: default {errors[False]:.5f}, fp32_accumulation {errors[True]:.5f}")
    assert errors[True] < bound, errors
    assert errors[True] * factor < errors[False], errors


def run_paged_decode(
    device, batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype, bound, factor
):
    tt_k, tt_v, tt_page_table, linear = paged_kv(device, batch, kv_heads, capacity, block_size, kv_dtype, 2026)
    torch.manual_seed(1014)
    q = torch.randn(1, batch, heads, 128)
    positions = torch.tensor([capacity - 1 - 37 * b for b in range(batch)], dtype=torch.int32)
    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_pos = ttnn.from_torch(positions, dtype=ttnn.int32, device=device)
    expected = torch.stack(
        [reference_attention(q[0, b].unsqueeze(1), *linear[b], [int(positions[b]) + 1])[:, 0] for b in range(batch)]
    )
    errors = {}
    for fp32_accumulation in (False, True):
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            page_table_tensor=tt_page_table,
            cur_pos_tensor=tt_pos,
            compute_kernel_config=compute_config(device),
            program_config=program_config(device, 0, k_chunk_size, fp32_accumulation, max_cores),
        )
        errors[fp32_accumulation] = rel_l2(ttnn.to_torch(out)[0, :, :heads], expected)
    check(errors, bound, factor)


def run_unpaged_decode(device, batch, heads, kv_heads, seq, k_chunk_size, bound, factor):
    torch.manual_seed(7)
    q = torch.randn(1, batch, heads, 128)
    k = torch.randn(batch, kv_heads, seq, 128)
    v = torch.randn(batch, kv_heads, seq, 128)
    positions = [seq - 1 - 101 * b for b in range(batch)]
    tt_q, tt_k, tt_v = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v)
    )
    tt_pos = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32, device=device)
    k, v = ttnn.to_torch(tt_k).float(), ttnn.to_torch(tt_v).float()
    expected = torch.stack(
        [reference_attention(q[0, b].unsqueeze(1), k[b], v[b], [positions[b] + 1])[:, 0] for b in range(batch)]
    )
    errors = {}
    for fp32_accumulation in (False, True):
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            cur_pos_tensor=tt_pos,
            compute_kernel_config=compute_config(device),
            program_config=program_config(device, 0, k_chunk_size, fp32_accumulation),
        )
        errors[fp32_accumulation] = rel_l2(ttnn.to_torch(out)[0, :, :heads], expected)
    check(errors, bound, factor)
