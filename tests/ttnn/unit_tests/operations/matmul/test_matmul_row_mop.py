# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The Blackhole matmul row MOP (MM_ROW_MOP: one math MOP per reuse row of an output subblock instead of one per
output tile) must reproduce the per-tile MOP bit for bit. The 2D-multicast factory enables it by default on
Blackhole for full 32x32 tiles; TT_MM_ROW_MOP=0 keeps the per-tile MOP."""

import pytest
import torch
import ttnn

from tt_py_test_utils_common.utility_functions import is_blackhole


def _run_linear(device, torch_a, torch_b, out_subblock_h, out_subblock_w, math_fidelity, fp32_dest_acc_en):
    grid = ttnn.CoreCoord(2, 2)
    m_tiles, n_tiles = torch_a.shape[-2] // 32, torch_b.shape[-1] // 32
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=2,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=m_tiles // grid.y,
        per_core_N=n_tiles // grid.x,
        transpose_mcast=False,
        fused_activation=None,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=math_fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )
    a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.linear(
        a,
        b,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        dtype=ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16,
    )
    return ttnn.to_torch(out)


@pytest.mark.skipif(not is_blackhole(), reason="The matmul row MOP is Blackhole only")
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4])
@pytest.mark.parametrize(
    "out_subblock_h, out_subblock_w, fp32_dest_acc_en",
    [
        (1, 4, True),  # reuse along a row (the KDA projection shape)
        (1, 3, True),
        (2, 2, True),
        (2, 1, True),  # reuse down a column, DEST step of one tile
        (4, 1, True),
        (4, 2, False),  # reuse down a column, DEST step of two tiles
    ],
)
def test_matmul_row_mop_matches_per_tile_mop(
    device, monkeypatch, out_subblock_h, out_subblock_w, math_fidelity, fp32_dest_acc_en
):
    torch.manual_seed(0)
    # Per core: 4 x 12 output tiles, 16 K tiles in 8 blocks of 2.
    torch_a = torch.randn(1, 1, 256, 512, dtype=torch.bfloat16)
    torch_b = torch.randn(1, 1, 512, 768, dtype=torch.bfloat16)

    monkeypatch.setenv("TT_MM_ROW_MOP", "0")
    device.disable_and_clear_program_cache()
    per_tile = _run_linear(device, torch_a, torch_b, out_subblock_h, out_subblock_w, math_fidelity, fp32_dest_acc_en)

    monkeypatch.delenv("TT_MM_ROW_MOP")
    device.disable_and_clear_program_cache()
    row = _run_linear(device, torch_a, torch_b, out_subblock_h, out_subblock_w, math_fidelity, fp32_dest_acc_en)

    assert torch.equal(per_tile, row), "the row MOP changed the matmul result"
