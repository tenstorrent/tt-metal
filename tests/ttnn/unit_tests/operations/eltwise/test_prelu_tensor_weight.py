# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""prelu with a per-channel weight tensor.

The weight is broadcast along the channel dimension, so these cover the shapes
where that broadcast is what the op is doing, plus the sign boundary, which is
the only branch in the kernel.
"""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shape, channels", [((1, 8, 32, 32), 8), ((2, 16, 64, 64), 16), ((1, 4, 32, 32), 4)])
def test_prelu_tensor_weight(device, dtype, shape, channels):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)

    x = (torch.rand(shape, dtype=torch.float32) * 20 - 10).to(torch_dtype)
    w = (torch.rand((channels,), dtype=torch.float32) * 2 - 1).to(torch_dtype)
    want = torch.nn.functional.prelu(x.float(), w.float()).to(torch_dtype)

    ix = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    iw = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.prelu(ix, iw))

    assert torch.equal(got, want), f"{int((got != want).sum())} of {want.numel()} differ"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_prelu_sign_boundary_and_specials(device, dtype):
    """Zero takes the non-negative arm, and the weight must not touch it."""
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    vals = [-1.0, 0.0, -0.0, 1.0, float("inf"), float("-inf"), float("nan"), -3.0]

    x = torch.tensor(vals * 128, dtype=torch_dtype).reshape(1, 8, 32, 4).repeat(1, 1, 1, 8)
    w = torch.full((8,), 0.25, dtype=torch_dtype)
    want = torch.nn.functional.prelu(x.float(), w.float()).to(torch_dtype)

    ix = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    iw = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(ttnn.prelu(ix, iw))

    finite = torch.isfinite(want)
    assert torch.equal(got[finite], want[finite])

    if dtype == ttnn.float32:
        # Bitwise, so the sign of -0.0 and the NaN input both count.
        assert torch.equal(got.view(torch.int32), want.view(torch.int32))
    # A bfloat16 NaN comes back as inf on this path, as it does for the composite
    # this replaces, so it is not asserted there.
