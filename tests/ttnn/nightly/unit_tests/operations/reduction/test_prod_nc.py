# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose_and_pcc, skip_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_prod

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, device, npu_dtype=ttnn.bfloat16):
    torch.manual_seed(2023)
    cpu_dtype = torch.float32 if npu_dtype == ttnn.float32 else torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(-100, 100, input_shape, dtype=cpu_dtype)
    torch_output = torch.randint(-100, 100, output_shape, dtype=cpu_dtype)

    tt_input = ttnn.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "input_shape",
    (
        ([2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1]),
        ([9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1]),
        ([4, 3, TILE_HEIGHT * 3 - 1, TILE_WIDTH * 11 - 1]),
        ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1]),
        ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1]),
        ([8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1",
        "9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1",
        "4, 3, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 11 - 1",
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        "4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1",
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1",
        "8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [
            0,
        ],
        [
            1,
        ],
    ),
    ids=["0", "1"],
)
@pytest.mark.parametrize("npu_dtype", (ttnn.bfloat16, ttnn.float32), ids=["bfloat16", "float32"])
# Support for dim 2,3 in composite_ops
def test_prod_dims(input_shape, dims, npu_dtype, device):
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device, npu_dtype)

    torch_output = torch.prod(torch_input, dims[0], True)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu = ttnn_prod(tt_input, tt_output, dims=dims).cpu().to(cpu_layout).to_torch()

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing


DIM_PATH_BLOCK_FLOAT_SUPPORT = {
    ttnn.bfloat8_b: True,
    ttnn.bfloat4_b: False,
}
_dim_supported = [d for d, ok in DIM_PATH_BLOCK_FLOAT_SUPPORT.items() if ok]
_dim_unsupported = [d for d, ok in DIM_PATH_BLOCK_FLOAT_SUPPORT.items() if not ok]


@pytest.mark.parametrize("npu_dtype", _dim_supported, ids=lambda d: d.name.lower())
@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_prod_dims_block_float(dim, npu_dtype, device):
    # Mostly-ones input with a few 2.0s: exact in block-float, product stays a small power of two.
    torch.manual_seed(2023)
    torch_input = torch.ones([2, 3, TILE_HEIGHT, TILE_WIDTH * 2], dtype=torch.float32)
    torch_input.view(-1)[::13][:8] = 2.0

    tt_input = ttnn.from_torch(torch_input, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.prod(tt_input, dim=dim)
    assert tt_output.dtype == npu_dtype, f"expected {npu_dtype} result, got {tt_output.dtype}"

    out = ttnn.to_torch(tt_output)
    ref = torch.prod(torch_input, dim=dim)
    ref = ttnn.to_torch(ttnn.from_torch(ref, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device))
    assert torch.allclose(out, ref, atol=1e-2), f"dim={dim} {npu_dtype}: expected {ref}, got {out}"
