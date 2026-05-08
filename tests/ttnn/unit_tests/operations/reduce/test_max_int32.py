# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Phase 1 of the SFPU reduce plan (issue #43736) for the int32 reduction work
# tracked in issue #26726 enables MAX and MIN over Int32 along REDUCE_W and
# REDUCE_H.  MIN is lowered to MAX via the negate trick (-MAX(-x)) at the
# SFPU compute kernel level, so both ops share the same reduce_sfpu.cpp path.
# INT32 HW (REDUCE_SCALAR) is not supported directly, so we use the same
# two-step W-then-H strategy for both single-core and multi-core cases.
# The test includes the verbatim repro from issue #21071
# (shape=(1,1,32,32), dim=-1, op=max) and extends it across additional shapes,
# the H axis, multi-axis (HW) reduction, and the MIN op.
# https://github.com/tenstorrent/tt-metal/issues/21071

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device


INT32_INFO = torch.iinfo(torch.int32)


@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 1, 32, 32),  # issue #21071 repro: single tile
        (1, 1, 64, 60),  # 2x2 tiles with partial tile
        (1, 1, 100, 120),  # 4x4 tiles with partial tile
        (1, 1, 30, 96),  # Ht=1, Wt=3
        (1, 1, 90, 32),  # Ht=3, Wt=1
        (2, 3, 64, 64),  # multi-batch, NC>1
    ],
)
@pytest.mark.parametrize("dim", [-1, -2, (-1, -2), None])
@pytest.mark.parametrize("op", ["max", "min"])
def test_max_min_int32(device, input_shape, dim, op):
    torch.manual_seed(0)

    # Keep the issue #21071 repro verbatim for (1,1,32,32) W-reduce + MAX. All other
    # cases use a deterministic signed-int32 input.  Exclude INT32_MIN from random
    # inputs because the on-chip int32 format is sign-magnitude: 0x80000000 is "-0"
    # there but maps to torch's -2^31, so a -2^31 sample has no representable
    # negation -- the MIN-via-negate-MAX kernel path would disagree with torch.amin
    # on that single value.
    if input_shape == (1, 1, 32, 32) and dim == -1 and op == "max":
        torch_input_tensor = torch.arange(32 * 32, dtype=torch.int32).reshape(input_shape)
    else:
        torch_input_tensor = torch.randint(INT32_INFO.min + 1, INT32_INFO.max, input_shape, dtype=torch.int32)

    torch_op = torch.amax if op == "max" else torch.amin
    ttnn_op = ttnn.max if op == "max" else ttnn.min

    torch_output_tensor = torch_op(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    output_tensor = ttnn_op(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.dtype == torch.int32, f"Expected int32 output, got {output_tensor.dtype}"
    assert torch.equal(
        output_tensor.reshape(torch_output_tensor.shape), torch_output_tensor
    ), f"\nop={op} shape={input_shape} dim={dim}\nexpected:\n{torch_output_tensor}\nactual:\n{output_tensor}"
