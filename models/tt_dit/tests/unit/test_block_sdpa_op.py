# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Phase-2b device validation: the block-permuted fused SDPA matches the strided fused SDPA.

Both compute the same 3-D neighborhood attention; the block path just runs the queries in block order
(so the kernel's box stays compact) and K/V stay strided. So un-permuting the block output must match
the strided output. Isolated at the op level -- no decoder, no W-SP -- to validate the Phase-2 kernel
(neighborhood_box_block + block_query_coord) end to end on device."""
import pytest
import torch

import ttnn
from models.tt_dit.layers.block_permute import from_block_order_tt, to_block_order_tt


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _fused_sdpa(device, q, k, v, grid, kernel, *, block=None):
    """Fused neighborhood SDPA on a PLAIN-STRIDED (t,h,w) grid. K/V are wrow-paged (T*H, W*HD)."""
    T, H, W = grid
    kt, kh, kw = kernel
    B, NH, S, HD = tuple(q.shape)
    k_rm = ttnn.reshape(ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT), (B, NH, T * H, W * HD))
    v_rm = ttnn.reshape(ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT), (B, NH, T * H, W * HD))
    off = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    q_chunk = block[0] * block[1] * block[2] if block else 64
    g = device.compute_with_storage_grid_size()
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=q_chunk, k_chunk_size=32
    )
    return ttnn.transformer.scaled_dot_product_attention(
        q,
        k_rm,
        v_rm,
        is_causal=False,
        neighborhood_3d=(T, H, W, kt, kh, kw),
        neighborhood_gather=True,
        neighborhood_block=block,
        scale=HD**-0.5,
        windowed_q_token_offset=0,
        windowed_q_token_offset_tensor=off,
        program_config=pc,
    )


@pytest.mark.parametrize(
    "grid,kernel,block",
    [((8, 8, 8), (5, 5, 5), (4, 4, 4)), ((8, 16, 16), (5, 5, 5), (4, 8, 8))],
    ids=["8x8x8", "8x16x16"],
)
def test_block_sdpa_matches_strided(*, device, grid, kernel, block):
    T, H, W = grid
    HD, S = 64, T * H * W
    torch.manual_seed(0)
    mk = lambda: ttnn.from_torch(
        torch.randn(1, 1, S, HD, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    q, k, v = mk(), mk(), mk()

    ref = ttnn.to_torch(_fused_sdpa(device, q, k, v, grid, kernel))  # strided reference, strided order

    q_block = to_block_order_tt(q, grid, block)
    out = _fused_sdpa(device, q_block, k, v, grid, kernel, block=block)  # block order out
    out = ttnn.to_torch(from_block_order_tt(out, grid, block))  # back to strided order

    pcc = _pcc(out.float(), ref.float())
    print(f"\n  grid={grid} block={block}: block-vs-strided PCC = {pcc * 100:.4f} %")
    assert pcc > 0.99, f"block SDPA != strided SDPA (PCC {pcc})"
