# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

# split_size is the divisor of the last dim, and the resulting tensor count divides the width in
# turn, so test that split_size of 0, or wider than the input, or a value that does not divide the width is rejected,
# otherwise it could lead to division by 0 or silently produce incorrect results.
_INVALID_SPLIT_SIZES = pytest.mark.parametrize(
    "split_size", [0, 512, 96], ids=["zero", "wider_than_input", "not_a_divisor"]
)
_MESSAGE = "must be greater than 0 and divide the last dim"


@_INVALID_SPLIT_SIZES
def test_deepseek_moe_fast_reduce_nc_invalid_split_size(device, expect_error, split_size):
    x = ttnn.from_torch(torch.randn(2, 1, 32, 256).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, _MESSAGE):
        ttnn.experimental.deepseek_moe_fast_reduce_nc(x, 0, split_size=split_size)


@_INVALID_SPLIT_SIZES
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_deepseek_moe_fast_reduce_nc_fused_invalid_split_size(mesh_device, expect_error, split_size):
    experts_k, tokens, hidden = 8, 32, 256
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def to_device(tensor, layout, dtype):
        return ttnn.from_torch(tensor, device=mesh_device, layout=layout, dtype=dtype, mesh_mapper=replicate)

    activation = to_device(torch.rand(experts_k, 1, tokens, hidden), ttnn.TILE_LAYOUT, ttnn.bfloat16)
    scores = to_device(torch.rand(tokens, 1, 1, experts_k), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16)
    indices = to_device(
        torch.arange(experts_k).repeat(tokens, 1, 1, 1).to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16
    )
    mapping = to_device(torch.zeros(1, 2 * experts_k, dtype=torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint16)
    with expect_error(RuntimeError, _MESSAGE):
        ttnn.experimental.deepseek_moe_fast_reduce_nc_fused(
            activation,
            indices,
            mapping,
            reduce_dim=0,
            split_size=split_size,
            cluster_axis=0,
            scores_tensor=scores,
        )
