# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""clamp with tensor bounds, over the cases the removed branch distinguished."""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("case", ["ordinary", "min_is_zero", "min_above_max"])
def test_clamp_tensor_bounds(device, dtype, case):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)

    x = ((torch.rand((1, 1, 32, 32)) * 20) - 10).to(torch_dtype)
    lo = ((torch.rand((1, 1, 32, 32)) * 4) - 2).to(torch_dtype)
    hi = ((torch.rand((1, 1, 32, 32)) * 4) + 1).to(torch_dtype)
    if case == "min_is_zero":
        lo = torch.zeros_like(lo)  # the branch the removed relu served
    elif case == "min_above_max":
        lo, hi = hi, lo  # degenerate, and the gt guard that stays is what answers it

    t = [ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for v in (x, lo, hi)]
    got = ttnn.to_torch(ttnn.clamp(t[0], t[1], t[2]))

    want = torch.clamp(x.float(), lo.float(), hi.float()).to(torch_dtype)
    assert torch.equal(got, want), f"{int((got != want).sum())} of {want.numel()} differ"


@pytest.mark.parametrize("sign", [1.0, -1.0], ids=["positive_nan", "negative_nan"])
@pytest.mark.parametrize("bounds", [(0.0, 0.0), (0.0, 5.0), (-1.0, 1.0), (2.0, -2.0)])
def test_clamp_nan_input_fp32(device, sign, bounds):
    """A NaN input is NaN out, whatever the bounds and whichever sign the NaN has.

    The sign matters: minimum and maximum are one SFPSWAP and order by bit
    pattern, so a NaN with the sign bit set survives one of them and not the other.
    """
    nan = (torch.tensor(float("nan")) * sign).item()
    x = torch.full((1, 1, 32, 32), nan, dtype=torch.float32)
    lo = torch.full((1, 1, 32, 32), bounds[0], dtype=torch.float32)
    hi = torch.full((1, 1, 32, 32), bounds[1], dtype=torch.float32)

    t = [ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for v in (x, lo, hi)]
    got = ttnn.to_torch(ttnn.clamp(t[0], t[1], t[2]))

    assert (
        got.isnan().all()
    ), f"{int((~got.isnan()).sum())} of {got.numel()} came back not NaN, first {got.flatten()[0]}"
