# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# A NoC read's DRAM-side offset must land on this boundary. It is not the same on every arch.
NOC_DRAM_ALIGN = 64 if is_blackhole() else 32


def _reference_msda(value: torch.Tensor, grid: torch.Tensor, attn: torch.Tensor, align_corners: bool) -> torch.Tensor:
    """Pure PyTorch reference matching the device op's contract.

    Inputs:
      value: (N, h, w, D)   float32
      grid:  (N, Q*P, 1, 2) float32, normalized to [-1, 1]
      attn:  (N, Q, P)      float32
      align_corners: bilinear pixel-coord mapping (see device-op docstring)

    Output:
      (N, Q, D) float32
    """
    N, h, w, D = value.shape
    Q, P = attn.shape[1], attn.shape[2]

    value_nchw = value.permute(0, 3, 1, 2).contiguous()  # (N, D, h, w)
    grid_for_gs = grid.reshape(N, Q * P, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        value_nchw, grid_for_gs, mode="bilinear", padding_mode="zeros", align_corners=align_corners
    )  # (N, D, Q*P, 1)
    sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (N, Q*P, D)
    sampled = sampled.reshape(N, Q, P, D)
    out = (sampled * attn.unsqueeze(-1)).sum(dim=2)  # (N, Q, D)
    return out


@pytest.mark.parametrize("N", [1, 4])
@pytest.mark.parametrize("h_in,w_in", [(10, 10), (32, 32)])
@pytest.mark.parametrize("D", [16, 32, 48, 64])
@pytest.mark.parametrize("Q", [16, 64])
@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("align_corners", [False, True])
def test_msda_correctness(device, N, h_in, w_in, D, Q, P, align_corners):
    torch.manual_seed(0)
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(value, grid, attn, align_corners=align_corners)

    value_bf = value.to(torch.bfloat16)
    grid_bf = grid.to(torch.bfloat16)
    attn_bf = attn.to(torch.bfloat16)

    value_t = ttnn.from_torch(value_bf, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid_bf, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    attn_t = ttnn.from_torch(attn_bf, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    out_t = ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t, align_corners=align_corners)
    out = ttnn.to_torch(out_t)

    assert_with_pcc(ref, out.to(torch.float32), pcc=0.99)


@pytest.mark.parametrize("D", [8, 24, 40])
def test_msda_rejects_non_multiple_of_16(device, expect_error, D):
    """D values that are not multiples of 16 must be rejected at validation
    with an actionable error, not fail deep in the kernels."""
    N, h_in, w_in, Q, P = 1, 10, 10, 16, 4
    value_t = ttnn.from_torch(
        torch.randn(N, h_in, w_in, D).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    grid_t = ttnn.from_torch(
        (torch.rand(N, Q * P, 1, 2) * 2.0 - 1.0).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    attn_t = ttnn.from_torch(
        torch.softmax(torch.randn(N, Q, P), dim=-1).to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    with expect_error(RuntimeError, "multiple of 16"):
        ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t)


@pytest.mark.parametrize("P", [4, 8])
@pytest.mark.parametrize("pts_per_page", [1, 2, 4])
def test_msda_grid_point_folding(device, P, pts_per_page):
    """Every folding of the point axis into the grid's last dimension gives the same answer.

    A ROW_MAJOR page is the last dimension, so (N, Q, 1, P*2) is one NoC read per query where
    (N, Q*P, 1, 2) is P reads of four bytes.
    """
    N, h_in, w_in, D, Q = 2, 16, 16, 32, 32
    torch.manual_seed(0)
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(value, grid, attn, align_corners=False)

    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(value.to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(attn.to(torch.bfloat16), **rm)
    folded = grid.reshape(N, Q * P // pts_per_page, 1, 2 * pts_per_page)
    grid_t = ttnn.from_torch(folded.to(torch.bfloat16), **rm)

    out = ttnn.to_torch(ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t))
    assert_with_pcc(ref, out.to(torch.float32), pcc=0.99)


def test_msda_rejects_grid_width_not_dividing_p(device, expect_error):
    """A grid page must hold a divisor of P points; the reader indexes points modulo it."""
    # 4 points per page against P = 6: the page count still matches Q*P, only the stride does not.
    N, h_in, w_in, D, Q, P = 1, 10, 10, 16, 16, 6
    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(torch.randn(N, h_in, w_in, D).to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch((torch.rand(N, Q * P // 4, 1, 8) * 2.0 - 1.0).to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(torch.softmax(torch.randn(N, Q, P), dim=-1).to(torch.bfloat16), **rm)

    with expect_error(RuntimeError, "divisor of P"):
        ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t)


@pytest.mark.parametrize("num_heads", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [16, 32])
def test_msda_packed_heads(device, num_heads, head_dim):
    """Heads packed into value's last dimension match the head-major form they replace.

    The reader picks head n % num_heads out of the stick by byte offset, so the caller never
    materialises (B*num_heads, h, w, D).
    """
    if num_heads > 1 and (head_dim * 2) % NOC_DRAM_ALIGN:
        pytest.skip(f"head stride {head_dim * 2} B is under this arch's {NOC_DRAM_ALIGN} B alignment")
    B, h_in, w_in, Q, P = 2, 16, 16, 32, 4
    N = B * num_heads
    torch.manual_seed(0)
    # (B, h, w, num_heads*D) — heads packed; head-major view is the same data permuted.
    packed = torch.randn(B, h_in, w_in, num_heads * head_dim, dtype=torch.float32)
    major = packed.reshape(B, h_in, w_in, num_heads, head_dim).permute(0, 3, 1, 2, 4)
    major = major.reshape(N, h_in, w_in, head_dim).contiguous()
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(major, grid, attn, align_corners=False)

    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    grid_t = ttnn.from_torch(grid.to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(attn.to(torch.bfloat16), **rm)
    value_t = ttnn.from_torch(packed.to(torch.bfloat16), **rm)

    out = ttnn.to_torch(ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t, num_heads=num_heads))
    assert_with_pcc(ref, out.to(torch.float32), pcc=0.99)


def test_msda_rejects_num_heads_not_dividing_channels(device, expect_error):
    """value's last dim must split evenly into num_heads slices of a 16-multiple."""
    B, h_in, w_in, Q, P = 1, 10, 10, 16, 4
    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(torch.randn(B, h_in, w_in, 48).to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch((torch.rand(B * 2, Q * P, 1, 2) * 2.0 - 1.0).to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(torch.softmax(torch.randn(B * 2, Q, P), dim=-1).to(torch.bfloat16), **rm)

    with expect_error(RuntimeError, "multiple of 16"):
        ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t, num_heads=2)


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("num_levels,level", [(1, 0), (4, 0), (4, 2), (4, 3)])
def test_msda_packed_attn(device, num_heads, num_levels, level):
    """attn packing (head, level, point) into one row matches the (N, Q, P) slice it replaces.

    A head's run starts at h*num_levels*P and this call reads P points from level*P into it,
    so neither the head-major permute nor the spelled-out (num_levels, P) is ever built.
    num_heads also selects packed value, so both inputs are in the packed form here.
    """
    B, h_in, w_in, D, Q, P = 2, 16, 16, 32, 32, 4
    N = B * num_heads
    torch.manual_seed(0)
    packed_value = torch.randn(B, h_in, w_in, num_heads * D, dtype=torch.float32)
    major_value = packed_value.reshape(B, h_in, w_in, num_heads, D).permute(0, 3, 1, 2, 4)
    major_value = major_value.reshape(N, h_in, w_in, D).contiguous()
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    # (B, Q, num_heads, num_levels, P) is the logical layout the packed row flattens.
    full = torch.softmax(torch.randn(B, Q, num_heads, num_levels * P, dtype=torch.float32), dim=-1)
    full = full.reshape(B, Q, num_heads, num_levels, P)
    # The head-major (N, Q, P) form for this level, which the packed row must reproduce.
    sliced = full[:, :, :, level, :].permute(0, 2, 1, 3).reshape(N, Q, P).contiguous()

    ref = _reference_msda(major_value, grid, sliced, align_corners=False)

    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(packed_value.to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch(grid.to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(full.reshape(B, Q, num_heads * num_levels * P).to(torch.bfloat16), **rm)

    out = ttnn.to_torch(
        ttnn.experimental.multi_scale_deformable_attn(
            value_t, grid_t, attn_t, num_heads=num_heads, num_points=P, point_offset=level * P
        )
    )
    assert_with_pcc(ref, out.to(torch.float32), pcc=0.99)


def test_msda_rejects_point_offset_past_the_head_run(device, expect_error):
    """point_offset + P must stay inside a head's run, or the read walks into the next head."""
    # D is sized to clear the arch's DRAM alignment so the head-stride guard does not fire first.
    B, h_in, w_in, D, Q, P, num_heads, num_levels = 1, 10, 10, NOC_DRAM_ALIGN // 2, 16, 4, 2, 2
    N = B * num_heads
    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(torch.randn(B, h_in, w_in, num_heads * D).to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch((torch.rand(N, Q * P, 1, 2) * 2.0 - 1.0).to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(torch.rand(B, Q, num_heads * num_levels * P).to(torch.bfloat16), **rm)

    with expect_error(RuntimeError, "overruns attn's per-head run"):
        ttnn.experimental.multi_scale_deformable_attn(
            value_t, grid_t, attn_t, num_heads=num_heads, num_points=P, point_offset=num_levels * P
        )


@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("num_levels,level", [(1, 0), (4, 0), (4, 2), (4, 3)])
def test_msda_packed_grid(device, num_heads, num_levels, level):
    """A rank-3 grid packing (head, level, point, (x, y)) matches the (N, Q*P, 1, 2) form.

    The offsets here are 4 bytes per point rather than 2, so levels 2 and 3 land off the
    32-byte NoC boundary just as they do for attn.
    """
    B, h_in, w_in, D, Q, P = 2, 16, 16, 32, 32, 4
    N = B * num_heads
    torch.manual_seed(0)
    packed_value = torch.randn(B, h_in, w_in, num_heads * D, dtype=torch.float32)
    major_value = packed_value.reshape(B, h_in, w_in, num_heads, D).permute(0, 3, 1, 2, 4)
    major_value = major_value.reshape(N, h_in, w_in, D).contiguous()

    # (B, Q, num_heads, num_levels, P, 2) is the logical layout the packed row flattens.
    full_grid = torch.rand(B, Q, num_heads, num_levels, P, 2, dtype=torch.float32) * 2.0 - 1.0
    sliced_grid = full_grid[:, :, :, level, :, :].permute(0, 2, 1, 3, 4)
    sliced_grid = sliced_grid.reshape(N, Q * P, 1, 2).contiguous()

    full_attn = torch.softmax(torch.randn(B, Q, num_heads, num_levels * P, dtype=torch.float32), dim=-1)
    full_attn = full_attn.reshape(B, Q, num_heads, num_levels, P)
    sliced_attn = full_attn[:, :, :, level, :].permute(0, 2, 1, 3).reshape(N, Q, P).contiguous()

    ref = _reference_msda(major_value, sliced_grid, sliced_attn, align_corners=False)

    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(packed_value.to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch(full_grid.reshape(B, Q, num_heads * num_levels * P * 2).to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(full_attn.reshape(B, Q, num_heads * num_levels * P).to(torch.bfloat16), **rm)

    out = ttnn.to_torch(
        ttnn.experimental.multi_scale_deformable_attn(
            value_t, grid_t, attn_t, num_heads=num_heads, num_points=P, point_offset=level * P
        )
    )
    assert_with_pcc(ref, out.to(torch.float32), pcc=0.99)


def test_msda_rejects_packed_grid_point_offset_overrun(device, expect_error):
    """point_offset + P must stay inside the grid's per-head run as well as attn's."""
    # D is sized to clear the arch's DRAM alignment so the head-stride guard does not fire first.
    B, h_in, w_in, D, Q, P, num_heads, num_levels = 1, 10, 10, NOC_DRAM_ALIGN // 2, 16, 4, 2, 2
    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(torch.randn(B, h_in, w_in, num_heads * D).to(torch.bfloat16), **rm)
    # attn's run is long enough, so only the grid's bound can fire.
    attn_t = ttnn.from_torch(torch.rand(B, Q, num_heads * (num_levels + 1) * P).to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch((torch.rand(B, Q, num_heads * num_levels * P * 2) * 2.0 - 1.0).to(torch.bfloat16), **rm)

    with expect_error(RuntimeError, "overruns the grid's per-head run"):
        ttnn.experimental.multi_scale_deformable_attn(
            value_t, grid_t, attn_t, num_heads=num_heads, num_points=P, point_offset=num_levels * P
        )


def test_msda_rejects_head_stride_under_noc_alignment(device, expect_error):
    """A head's slice is read with no rounding, so its stride must clear the arch's DRAM alignment."""
    head_dim = NOC_DRAM_ALIGN // 2 - 16  # one 16-multiple short of the boundary
    if head_dim <= 0:
        pytest.skip("no legal 16-multiple below this arch's alignment")
    B, h_in, w_in, Q, P, num_heads = 1, 10, 10, 16, 4, 2
    N = B * num_heads
    rm = dict(device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    value_t = ttnn.from_torch(torch.randn(B, h_in, w_in, num_heads * head_dim).to(torch.bfloat16), **rm)
    grid_t = ttnn.from_torch((torch.rand(N, Q * P, 1, 2) * 2.0 - 1.0).to(torch.bfloat16), **rm)
    attn_t = ttnn.from_torch(torch.softmax(torch.randn(N, Q, P), dim=-1).to(torch.bfloat16), **rm)

    with expect_error(RuntimeError, "DRAM alignment"):
        ttnn.experimental.multi_scale_deformable_attn(value_t, grid_t, attn_t, num_heads=num_heads)
