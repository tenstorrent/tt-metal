# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""A preallocated output_tensor of a different dtype gets its own cached program.

The unary program hash was built from the input dtype and attributes.output_dtype,
which carries the input's dtype when a preallocated output is given. Two calls that
differed only in the output tensor's dtype collided, and the second was served the
first one's packer: it wrote every other element, or read the wrong stride.
"""

import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)
TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

OPS = [(ttnn.neg, 1.5, -1.5), (ttnn.abs, -2.0, 2.0), (ttnn.relu, 1.5, 1.5)]


def _t(v, dtype, device):
    return ttnn.from_torch(
        torch.full(SHAPE, v, dtype=TORCH_DTYPE[dtype]), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )


@pytest.mark.parametrize("op, value, expected", OPS)
@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.float32])
def test_preallocated_output_dtype_survives_a_warmed_cache(device, op, value, expected, in_dtype, out_dtype):
    # Warm the cache with an output of the input's own dtype, which is what the hash saw.
    warm = _t(0.0, in_dtype, device)
    op(_t(value, in_dtype, device), output_tensor=warm)

    out = _t(0.0, out_dtype, device)
    op(_t(value, in_dtype, device), output_tensor=out)
    got = ttnn.to_torch(out).float()
    assert (got == expected).all(), f"{(got != expected).sum().item()} of {got.numel()} elements are not {expected}"


@pytest.mark.parametrize("op, value, expected", OPS)
def test_both_output_dtypes_in_either_order(device, op, value, expected):
    for first, second in ((ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16)):
        a, b = _t(0.0, first, device), _t(0.0, second, device)
        op(_t(value, ttnn.float32, device), output_tensor=a)
        op(_t(value, ttnn.float32, device), output_tensor=b)
        assert (ttnn.to_torch(a).float() == expected).all(), f"{first} first"
        assert (ttnn.to_torch(b).float() == expected).all(), f"{second} second"
