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


def test_block_sdpa_w_origin(*, device):
    """W-SP composition: a shard's block-Q over a W-band [w_origin, w_origin+w_local) with FULL-W K/V and
    a nonzero w_origin (on the per-device offset tensor) must match the full-grid strided reference's band.
    This exercises exactly what each W-SP shard does, on a single chip."""
    T, H, W, HD, kernel = 8, 16, 16, 64, (5, 5, 5)
    SP, block = 2, (4, 8, 4)  # 2 W-shards; block divides (T,H,w_local=8), vol 128
    W_LOCAL = W // SP
    torch.manual_seed(0)
    qf, kf, vf = (torch.randn(1, 1, T * H * W, HD, dtype=torch.bfloat16) for _ in range(3))
    dev = lambda t, l=ttnn.TILE_LAYOUT: ttnn.from_torch(t, device=device, layout=l, dtype=ttnn.bfloat16)
    k, v = dev(kf), dev(vf)
    ref = ttnn.to_torch(_fused_sdpa(device, dev(qf), k, v, (T, H, W), kernel)).float().reshape(T, H, W, HD)

    for shard in range(SP):
        w0 = shard * W_LOCAL
        q_band = qf.reshape(T, H, W, HD)[:, :, w0 : w0 + W_LOCAL, :].reshape(1, 1, T * H * W_LOCAL, HD)
        qb = to_block_order_tt(dev(q_band), (T, H, W_LOCAL), block)
        off = ttnn.from_torch(
            torch.tensor([w0], dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        g = device.compute_with_storage_grid_size()
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(g.x, g.y),
            exp_approx_mode=False,
            q_chunk_size=block[0] * block[1] * block[2],
            k_chunk_size=32,
        )
        o = ttnn.transformer.scaled_dot_product_attention(
            qb,
            ttnn.reshape(ttnn.to_layout(k, ttnn.ROW_MAJOR_LAYOUT), (1, 1, T * H, W * HD)),
            ttnn.reshape(ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT), (1, 1, T * H, W * HD)),
            is_causal=False,
            neighborhood_3d=(T, H, W, *kernel),
            neighborhood_gather=True,
            neighborhood_block=block,
            neighborhood_w_shard=(W, 0),
            scale=HD**-0.5,
            windowed_q_token_offset=0,
            windowed_q_token_offset_tensor=off,
            program_config=pc,
        )
        out = ttnn.to_torch(from_block_order_tt(o, (T, H, W_LOCAL), block)).float().reshape(T, H, W_LOCAL, HD)
        ref_band = ref[:, :, w0 : w0 + W_LOCAL, :]
        pcc = _pcc(out, ref_band)
        print(f"\n  shard {shard} (w_origin={w0}): block-band-vs-full PCC = {pcc * 100:.4f} %")
        assert pcc > 0.99, f"W-SP shard {shard} block band != full reference (PCC {pcc})"
