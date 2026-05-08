# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Phase 1 of the SFPU reduce plan (issue #43736) for the int32 reduction work
# tracked in issue #26726 enables MAX over Int32 along REDUCE_W and REDUCE_H.
#
# This module is the gate for Phase 1: it includes the verbatim repro from
# issue #21071 (shape=(1,1,32,32), dim=-1) and extends it across additional
# tile-aligned shapes and the H axis. Multi-axis (HW) reduction is intentionally
# out of scope for Phase 1.
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
@pytest.mark.parametrize("dim", [-1, -2])
def test_max_int32(device, input_shape, dim):
    torch.manual_seed(0)

    # Keep the issue #21071 repro verbatim for (1,1,32,32) W-reduce. All other
    # cases use a deterministic signed-int32 input.
    if input_shape == (1, 1, 32, 32) and dim == -1:
        torch_input_tensor = torch.arange(32 * 32, dtype=torch.int32).reshape(input_shape)
    else:
        # torch.randint upper bound is exclusive; use INT32 max as the high bound.
        torch_input_tensor = torch.randint(INT32_INFO.min, INT32_INFO.max, input_shape, dtype=torch.int32)

    torch_output_tensor = torch.amax(torch_input_tensor, dim=dim)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.int32)
    output_tensor = ttnn.max(input_tensor, dim=dim)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.dtype == torch.int32, f"Expected int32 output, got {output_tensor.dtype}"
    assert torch.equal(
        output_tensor.reshape(torch_output_tensor.shape), torch_output_tensor
    ), f"\nshape={input_shape} dim={dim}\nexpected:\n{torch_output_tensor}\nactual:\n{output_tensor}"
