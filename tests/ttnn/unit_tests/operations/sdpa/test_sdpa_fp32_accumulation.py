# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""SDPAProgramConfig(fp32_accumulation=True): FP32 online-softmax state for SDPA decode and prefill.

The default kernels round the running softmax sum and output accumulator through BF16 (decode) or 19-bit source
registers (prefill) on every K chunk, so their error grows with context length. These tests compare the option with
an FP64 reference at long contexts, require it to beat the default on the same configuration, and check that
configurations outside the supported path are rejected.
"""

import math

import pytest
import torch

import ttnn

# Relative L2 error bound against the FP64 reference. The option measures 0.18-0.33% on these shapes; the default
# path measures 2-2.7% (decode) and 4.6-59% (prefill at 64K-512K starts).
FP32_ACCUMULATION_REL_L2 = 5e-3


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


def reference_attention(q, k, v, ends, sink=None):
    """FP64 attention of q [h, rows, d] over k/v [kv_heads, s, d]; query row r sees keys [0, ends[r])."""
    heads, kv_heads, d = q.shape[0], k.shape[0], q.shape[-1]
    k, v = k.double().repeat_interleave(heads // kv_heads, 0), v.double().repeat_interleave(heads // kv_heads, 0)
    out = torch.empty(q.shape, dtype=torch.float64)
    for r, end in enumerate(ends):
        scores = torch.einsum("hd,hsd->hs", q[:, r].double(), k[:, :end]) / math.sqrt(d)
        if sink is not None:
            scores = torch.cat([scores, sink.double().reshape(heads, 1) / math.sqrt(d)], dim=-1)
        probs = scores.softmax(-1)[:, :end]
        out[:, r] = torch.einsum("hs,hsd->hd", probs, v[:, :end])
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


@pytest.mark.parametrize(
    "batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype",
    [
        (1, 8, 2, 131072, 32, 256, 64, ttnn.bfloat8_b),  # long context, all cores on one head
        (8, 8, 2, 16384, 32, 128, 16, ttnn.bfloat16),  # batched, ragged positions
        (4, 16, 4, 8192, 64, 128, 8, ttnn.bfloat8_b),  # 64-token pages, more Q heads
    ],
    ids=["b1_131k", "b8_16k", "b4_8k_block64"],
)
def test_paged_decode_fp32_accumulation(
    device, batch, heads, kv_heads, capacity, block_size, k_chunk_size, max_cores, kv_dtype
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
    assert errors[True] < FP32_ACCUMULATION_REL_L2, errors
    assert errors[True] < errors[False] / 4, errors


def test_unpaged_decode_fp32_accumulation(device):
    torch.manual_seed(7)
    batch, heads, kv_heads, seq = 2, 8, 2, 8192
    q = torch.randn(1, batch, heads, 128)
    k = torch.randn(batch, kv_heads, seq, 128)
    v = torch.randn(batch, kv_heads, seq, 128)
    positions = [seq - 1, seq // 2 + 5]
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
            program_config=program_config(device, 0, 256, fp32_accumulation),
        )
        errors[fp32_accumulation] = rel_l2(ttnn.to_torch(out)[0, :, :heads], expected)
    assert errors[True] < FP32_ACCUMULATION_REL_L2, errors
    assert errors[True] < errors[False] / 4, errors


@pytest.mark.parametrize("start_as_tensor", [False, True], ids=["scalar_start", "tensor_start"])
@pytest.mark.parametrize(
    "start, chunk, q_chunk_size, k_chunk_size",
    [(61440, 2048, 64, 512), (126976, 4096, 128, 256)],
    ids=["60k", "124k"],
)
def test_chunked_prefill_fp32_accumulation(device, start, chunk, q_chunk_size, k_chunk_size, start_as_tensor):
    heads, kv_heads, block_size = 8, 2, 32
    capacity = 1 << (start + chunk - 1).bit_length()
    tt_k, tt_v, tt_page_table, linear = paged_kv(device, 1, kv_heads, capacity, block_size, ttnn.bfloat8_b, 61)
    torch.manual_seed(62)
    q = torch.randn(1, heads, chunk, 128)
    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    start_args = (
        {"chunk_start_idx_tensor": ttnn.from_torch(torch.tensor([start], dtype=torch.int32), device=device)}
        if start_as_tensor
        else {"chunk_start_idx": start}
    )
    rows = torch.unique(torch.cat([torch.linspace(0, chunk - 1, 24).long(), torch.tensor([chunk - 1])]))
    expected = reference_attention(q[0][:, rows], *linear[0], [start + int(r) + 1 for r in rows])

    errors = {}
    for fp32_accumulation in (False, True):
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            tt_page_table,
            compute_kernel_config=compute_config(device),
            program_config=program_config(device, q_chunk_size, k_chunk_size, fp32_accumulation),
            **start_args,
        )
        errors[fp32_accumulation] = rel_l2(ttnn.to_torch(out)[0][:, rows], expected)
    assert errors[True] < FP32_ACCUMULATION_REL_L2, errors
    assert errors[True] < errors[False] / 4, errors


@pytest.mark.parametrize("use_sink", [False, True], ids=["no_sink", "sink"])
def test_prefill_fp32_accumulation(device, use_sink):
    torch.manual_seed(11)
    heads, kv_heads, seq = 8, 2, 8192
    q = torch.randn(1, heads, seq, 128)
    k = torch.randn(1, kv_heads, seq, 128)
    v = torch.randn(1, kv_heads, seq, 128)
    sink = torch.rand(1, heads, 1, 1) * 4.0 if use_sink else None
    tt_q, tt_k, tt_v = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v)
    )
    tt_sink = ttnn.from_torch(sink, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if use_sink else None
    k, v = ttnn.to_torch(tt_k).float(), ttnn.to_torch(tt_v).float()
    rows = torch.unique(torch.cat([torch.linspace(0, seq - 1, 24).long(), torch.tensor([seq - 1])]))
    stored_sink = ttnn.to_torch(tt_sink).float() if use_sink else None
    expected = reference_attention(q[0][:, rows], k[0], v[0], [int(r) + 1 for r in rows], stored_sink)

    errors = {}
    for fp32_accumulation in (False, True):
        out = ttnn.transformer.scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            is_causal=True,
            attention_sink=tt_sink,
            compute_kernel_config=compute_config(device),
            program_config=program_config(device, 128, 256, fp32_accumulation),
        )
        errors[fp32_accumulation] = rel_l2(ttnn.to_torch(out)[0][:, rows], expected)
    assert errors[True] < FP32_ACCUMULATION_REL_L2, errors
    assert errors[True] < errors[False], errors


def test_fp32_accumulation_rejects_unsupported_configs(device, expect_error):
    torch.manual_seed(0)
    q = ttnn.from_torch(torch.randn(1, 1, 8, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kv = ttnn.from_torch(torch.randn(1, 2, 1024, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos = ttnn.from_torch(torch.tensor([1000], dtype=torch.int32), dtype=ttnn.int32, device=device)
    config = program_config(device, 0, 128, True)

    with expect_error(RuntimeError, "fp32_accumulation requires fp32_dest_acc_en"):
        ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            kv,
            kv,
            cur_pos_tensor=pos,
            compute_kernel_config=compute_config(device, fp32_dest_acc_en=False),
            program_config=config,
        )
    with expect_error(RuntimeError, "fp32_accumulation supports causal decode only"):
        ttnn.transformer.scaled_dot_product_attention_decode(
            q, kv, kv, is_causal=False, compute_kernel_config=compute_config(device), program_config=config
        )

    prefill_q = ttnn.from_torch(
        torch.randn(1, 8, 1024, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "fp32_accumulation requires fp32_dest_acc_en"):
        ttnn.transformer.scaled_dot_product_attention(
            prefill_q,
            kv,
            kv,
            is_causal=True,
            compute_kernel_config=compute_config(device, fp32_dest_acc_en=False),
            program_config=program_config(device, 128, 128, True),
        )
