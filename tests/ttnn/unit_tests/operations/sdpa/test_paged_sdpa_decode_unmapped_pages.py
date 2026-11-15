# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Paged decode SDPA must not dereference unallocated (-1) page-table entries.

The reader fetches every tile row of a user's last K chunk, including rows past
``cur_pos`` that the causal mask then drops. When the chunk reaches past the
user's allocation, those rows map to page-table entries of -1 (the sentinel
``paged_fill_cache`` already skips). Reading them dereferenced block 0xFFFFFFFF:
the tile id wrapped and the read landed at an unrelated DRAM address. Its bytes
(possibly non-finite) poisoned the softmax even though the positions were
masked, so the output depended on whatever buffer happened to sit there.

The reader now zero-fills rows mapped by -1 instead of reading them. The oracle
is exact: a -1-padded page table must produce the same bytes as one padded with
an explicitly zeroed, never-written block. Both runs feed identical (zero) K/V
tiles to compute. The stale content of the rest of the pool is set to a large
finite magnitude so that a wild read through the pool is visible.
"""

import pytest
import torch

import ttnn

BLOCK = 64
HEAD_DIM = 128


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _reference(q, k_cache, v_cache, page_table, cur_pos, scale):
    """fp32 causal attention over positions [0, cur_pos] gathered through the page table."""
    B, nq = q.shape[1], q.shape[2]
    nkv = k_cache.shape[1]
    out = torch.zeros(1, B, nq, HEAD_DIM)
    for b in range(B):
        pos = torch.arange(cur_pos[b] + 1)
        blocks = torch.tensor([page_table[b][p // BLOCK] for p in pos.tolist()])
        k = k_cache[blocks, :, pos % BLOCK, :].permute(1, 0, 2).float().repeat_interleave(nq // nkv, 0)
        v = v_cache[blocks, :, pos % BLOCK, :].permute(1, 0, 2).float().repeat_interleave(nq // nkv, 0)
        w = torch.softmax(torch.einsum("hd,hsd->hs", q[0, b].float(), k) * scale, -1)
        out[0, b] = torch.einsum("hs,hsd->hd", w, v)
    return out


@pytest.mark.parametrize("kv_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["bfp8", "bf16"])
@pytest.mark.parametrize("k_chunk_size", [128, 256])
def test_paged_decode_sdpa_zero_fills_unmapped_pages(device, kv_dtype, k_chunk_size):
    torch.manual_seed(0)
    batch, nkv, nq, width = 32, 2, 8, 64
    # Positions whose last k-chunk reaches past the user's allocation (p % k_chunk < k_chunk - BLOCK),
    # mixed with fully mapped ones; every user owns ceil((p + 1) / BLOCK) blocks.
    positions = [1 + 33 * i for i in range(batch)]
    tables, cursor = [], 0
    for p in positions:
        n = (p + 1 + BLOCK - 1) // BLOCK
        tables.append(list(range(cursor, cursor + n)))
        cursor += n
    null_block = cursor
    num_blocks = cursor + 1
    assert any(p % k_chunk_size < k_chunk_size - BLOCK for p in positions)

    # Live rows hold O(1) values; every never-written row (stale content) holds 1e30.
    k_cache = torch.full((num_blocks, nkv, BLOCK, HEAD_DIM), 1e30)
    v_cache = torch.full((num_blocks, nkv, BLOCK, HEAD_DIM), 1e30)
    for b, p in enumerate(positions):
        for t in range(p + 1):
            k_cache[tables[b][t // BLOCK], :, t % BLOCK] = torch.randn(nkv, HEAD_DIM)
            v_cache[tables[b][t // BLOCK], :, t % BLOCK] = torch.randn(nkv, HEAD_DIM)
    k_cache[null_block] = 0
    v_cache[null_block] = 0

    def padded(pad):
        return torch.tensor([row + [pad] * (width - len(row)) for row in tables], dtype=torch.int32)

    kt = ttnn.from_torch(k_cache, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v_cache, dtype=kv_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    k_host = ttnn.to_torch(kt).float()
    v_host = ttnn.to_torch(vt).float()
    q = torch.randn(1, batch, nq, HEAD_DIM)
    qt = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cur = ttnn.from_torch(torch.tensor(positions, dtype=torch.int32), dtype=ttnn.int32, device=device)
    grid = device.compute_with_storage_grid_size()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=k_chunk_size, exp_approx_mode=False
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    scale = HEAD_DIM**-0.5

    def run(page_table):
        pt = ttnn.from_torch(page_table, dtype=ttnn.int32, device=device)
        out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            qt,
            kt,
            vt,
            page_table_tensor=pt,
            cur_pos_tensor=cur,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_config,
        )
        return ttnn.to_torch(out)[:, :, :nq, :]

    out_unmapped = run(padded(-1))
    out_null = run(padded(null_block))

    reference = _reference(q, k_host, v_host, tables, positions, scale)
    assert torch.isfinite(out_unmapped.float()).all()
    for b in range(batch):
        assert _pcc(reference[0, b], out_null[0, b].float()) > 0.999, f"user {b} (null-block padding)"
        assert _pcc(reference[0, b], out_unmapped[0, b].float()) > 0.999, f"user {b} (-1 padding)"
    # Unmapped rows are zero-filled, so -1 padding is bit-identical to padding with a zeroed block.
    assert torch.equal(out_unmapped, out_null)
