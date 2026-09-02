# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


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


def _reference_msda_from_bf16(value, grid, attn, align_corners=False):
    """Reference computed from the bf16 values the op actually receives.

    Comparing a bf16-fed op against an fp32-input reference measures the input
    quantisation, not the op. That is harmless on the small maps above, but a
    bf16 grid coordinate resolves to ~0.0020, which is a third of a pixel across
    a 334-wide feature map -- enough to move the sample to different neighbours
    and drag PCC to ~0.97 against random values. Round first, then compare.
    """
    return _reference_msda(
        value.to(torch.bfloat16).to(torch.float32),
        grid.to(torch.bfloat16).to(torch.float32),
        attn.to(torch.bfloat16).to(torch.float32),
        align_corners=align_corners,
    )


def _run_msda(device, value, grid, attn, align_corners=False):
    """Run the op on bf16 copies of float32 inputs and return a float32 result."""
    to_dev = lambda t: ttnn.from_torch(t.to(torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
    out = ttnn.experimental.multi_scale_deformable_attn(
        to_dev(value), to_dev(grid), to_dev(attn), align_corners=align_corners
    )
    return ttnn.to_torch(out).to(torch.float32)


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("coord", [-1.0, 1.0, -0.999, 0.999, -1.001, 1.001])
def test_msda_grid_at_the_border(device, coord, align_corners):
    """Sampling exactly on, just inside and just outside the [-1, 1] border.

    The reader derives the pixel coordinate and its interpolation fraction with
    integer arithmetic, so the border is where an off-by-one in the floor or in
    the in-bounds test would show up.
    """
    torch.manual_seed(0)
    N, Q, P, D, h_in, w_in = 2, 32, 4, 32, 10, 10
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.full((N, Q * P, 1, 2), coord, dtype=torch.float32)
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(value, grid, attn, align_corners=align_corners)
    out = _run_msda(device, value, grid, attn, align_corners=align_corners)

    assert torch.isfinite(out).all(), "border sampling produced non-finite values"
    assert_with_pcc(ref, out, pcc=0.99)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_msda_non_finite_grid_samples_as_out_of_bounds(device, bad):
    """A non-finite sampling location contributes zero and poisons nothing.

    This is a deliberate divergence from torch's grid_sample, which propagates
    NaN out of such a coordinate. Callers do hand this op non-finite locations
    (see #47512), and a single bad point must not take the whole query -- or its
    neighbours in the same output tile -- with it. The reference below replaces
    the non-finite coordinate with a finite far-out-of-range one, which is
    exactly the behaviour being asserted.
    """
    torch.manual_seed(0)
    N, Q, P, D, h_in, w_in = 2, 64, 4, 32, 10, 10
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    # Poison every other query outright, so the test also covers a bad query
    # sitting next to good ones inside one 32-query output tile.
    poisoned = torch.zeros(Q, dtype=torch.bool)
    poisoned[::2] = True
    rows = poisoned.repeat_interleave(P)
    grid_bad = grid.clone()
    grid_bad[:, rows] = bad
    grid_ref = grid.clone()
    grid_ref[:, rows] = 8.0  # finite, and far outside [-1, 1]

    ref = _reference_msda(value, grid_ref, attn, align_corners=False)
    out = _run_msda(device, value, grid_bad, attn)

    assert torch.isfinite(out).all(), f"grid {bad} propagated a non-finite value"
    assert (out[:, poisoned] == 0).all(), f"queries sampled at {bad} should contribute nothing"
    assert_with_pcc(ref, out, pcc=0.99)


@pytest.mark.parametrize("P", [1, 2])
def test_msda_few_points(device, P):
    """P below the 4 that the multi-scale callers use."""
    torch.manual_seed(0)
    N, Q, D, h_in, w_in = 2, 64, 32, 16, 16
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(value, grid, attn, align_corners=False)
    assert_with_pcc(ref, _run_msda(device, value, grid, attn), pcc=0.99)


@pytest.mark.parametrize("Q", [1, 31, 33, 100])
def test_msda_query_count_not_tile_aligned(device, Q):
    """Q that does not divide 32, so the last output tile per batch is partial.

    The writer only emits the rows a partial tile actually carries, and the
    reader leaves the rest of that tile holding whatever the previous iteration
    wrote -- the zeroed scalar lane is what keeps the stale data out of the sum.
    """
    torch.manual_seed(0)
    N, P, D, h_in, w_in = 2, 4, 32, 16, 16
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 2.0 - 1.0
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda(value, grid, attn, align_corners=False)
    assert_with_pcc(ref, _run_msda(device, value, grid, attn), pcc=0.99)


def test_msda_at_encoder_scale(device):
    """A feature map and query count of the size a real caller uses.

    The existing shape sweep tops out at 32x32 with 64 queries, which fits in a
    handful of output tiles on a couple of cores. DINO-5scale's first level is
    200x334 with tens of thousands of queries, so this covers the work-split
    across the whole core grid and the DRAM-resident value tensor.
    """
    torch.manual_seed(0)
    N, Q, P, D, h_in, w_in = 8, 2048, 4, 32, 200, 334
    value = torch.randn(N, h_in, w_in, D, dtype=torch.float32)
    grid = torch.rand(N, Q * P, 1, 2, dtype=torch.float32) * 1.8 - 0.9
    attn = torch.softmax(torch.randn(N, Q, P, dtype=torch.float32), dim=-1)

    ref = _reference_msda_from_bf16(value, grid, attn)
    assert_with_pcc(ref, _run_msda(device, value, grid, attn), pcc=0.999)
