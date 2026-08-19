# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-in-the-loop gate for the GNA fused kernel (P4/P5). Runs the SDPA op with neighborhood_gna=True and
checks it matches the strided (exact) reference. Baseline: the flag is a no-op today (== block path), so this
passes trivially; as the region-reuse reader/compute are wired in, this gate stays the correctness contract.

Run tight-loop:  pytest models/tt_dit/tests/unit/test_gna_device_kernel.py -x -q -s"""
import pytest
import torch

import ttnn
from models.tt_dit.layers.block_permute import from_block_order_tt, to_block_order_tt


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _sdpa(device, q, k, v, grid, kernel, *, block=None, gna=False):
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
        compute_with_storage_grid_size=(g.x, g.y), exp_approx_mode=False, q_chunk_size=q_chunk, k_chunk_size=256
    )
    return ttnn.transformer.scaled_dot_product_attention(
        q,
        k_rm,
        v_rm,
        is_causal=False,
        neighborhood_3d=(T, H, W, kt, kh, kw),
        neighborhood_gather=True,
        neighborhood_block=block,
        neighborhood_gna=gna,
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
def test_gna_matches_strided(*, device, grid, kernel, block):
    T, H, W = grid
    HD, S = 64, T * H * W
    torch.manual_seed(0)
    mk = lambda: ttnn.from_torch(
        torch.randn(1, 1, S, HD, dtype=torch.bfloat16), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    q, k, v = mk(), mk(), mk()
    ref = ttnn.to_torch(_sdpa(device, q, k, v, grid, kernel))  # strided exact reference
    qb = to_block_order_tt(q, grid, block)
    out = ttnn.to_torch(from_block_order_tt(_sdpa(device, qb, k, v, grid, kernel, block=block, gna=True), grid, block))
    pcc = _pcc(out.float(), ref.float())
    print(f"\n  grid={grid} block={block}: GNA-vs-strided PCC = {pcc * 100:.4f} %")
    assert pcc > 0.99, f"GNA path != strided (PCC {pcc})"
