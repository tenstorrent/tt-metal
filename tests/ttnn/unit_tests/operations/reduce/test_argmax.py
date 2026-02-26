# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn

from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize(
    argnames="tensor_shape, tensor_layout, dim, keepdim, use_multicore, dtype",
    argvalues=[
        ([], ttnn.ROW_MAJOR_LAYOUT, None, True, True, torch.bfloat16),
        ([32], ttnn.ROW_MAJOR_LAYOUT, -1, False, False, torch.float32),
        ([32, 0], ttnn.ROW_MAJOR_LAYOUT, 1, True, True, torch.bfloat16),
        ([64], ttnn.ROW_MAJOR_LAYOUT, -1, True, False, torch.bfloat16),
        ([1, 512], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.float32),
        ([1, 1024], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.int32),
        ([1, 65], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.uint8),
        ([8, 10, 129], ttnn.ROW_MAJOR_LAYOUT, 2, True, False, torch.bfloat16),
        ([1, 8, 160], ttnn.ROW_MAJOR_LAYOUT, -1, False, True, torch.bfloat16),
        ([1, 256, 1024 * 8], ttnn.ROW_MAJOR_LAYOUT, -1, False, True, torch.float32),
        ([32, 32, 32, 1], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.float32),
        ([128], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.float32),
        ([256], ttnn.ROW_MAJOR_LAYOUT, -1, False, False, torch.bfloat16),
        ([128], ttnn.TILE_LAYOUT, -1, True, False, torch.float32),
        ([256], ttnn.TILE_LAYOUT, -1, False, False, torch.bfloat16),
        ([64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.float32),
        ([64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, False, True, torch.int32),
        ([64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, False, torch.float32),
        ([32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.float32),
        ([32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, False, torch.bfloat16),
        ([32, 64, 128], ttnn.TILE_LAYOUT, -1, False, False, torch.bfloat16),
        ([16, 32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.bfloat16),
        ([16, 32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, False, torch.float32),
        ([16, 32, 64, 128], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.int32),
        ([16, 32, 64, 128], ttnn.TILE_LAYOUT, -1, True, False, torch.bfloat16),
        ([16, 32, 64, 128], ttnn.TILE_LAYOUT, -1, False, False, torch.float32),
        ([16, 32, 70, 130], ttnn.TILE_LAYOUT, -1, True, False, torch.bfloat16),
        ([16, 32, 70, 130], ttnn.TILE_LAYOUT, -1, False, False, torch.bfloat16),
        ([8, 16, 32, 64], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.float32),
        ([8, 16, 32, 64], ttnn.ROW_MAJOR_LAYOUT, -1, False, True, torch.bfloat16),
        ([4, 8, 16, 32], ttnn.ROW_MAJOR_LAYOUT, -1, False, False, torch.float32),
        ([100, 200], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.bfloat16),
        ([100, 200], ttnn.ROW_MAJOR_LAYOUT, -1, False, False, torch.float32),
        ([50, 100, 200], ttnn.ROW_MAJOR_LAYOUT, -1, True, True, torch.int32),
        ([25, 50, 100], ttnn.ROW_MAJOR_LAYOUT, -1, False, True, torch.uint8),
        ([12, 24, 48, 96], ttnn.ROW_MAJOR_LAYOUT, -1, True, False, torch.bfloat16),
    ],
)
def test_argmax(device, tensor_shape, tensor_layout, dim, keepdim, use_multicore, dtype):
    """
    Test the compatibility of the torch and ttnn output for argmax of different
    tensor shapes, dim values, and data types.
    Checks for the exactness of shape, values, and dtype of the output tensors.
    Some operations raise exceptions in torch, we check if the same behavior is observed in ttnn.
    Note: We do not enforce the same exception type or message.
    """
    rank = len(tensor_shape)

    # Create tensor based on data type
    if dtype in [torch.int32, torch.uint8]:
        # Use randint for integer types
        torch_tensor = (
            torch.randint(0, 100, tensor_shape, dtype=dtype) if rank > 0 else torch.randint(0, 100, (), dtype=dtype)
        )
    else:
        # Use randn for floating point types
        torch_tensor = torch.randn(*tensor_shape, dtype=dtype) if rank > 0 else torch.randn((), dtype=dtype)

    # Convert torch uint8 to appropriate ttnn type
    if dtype == torch.uint8:  # PyTorch does not have uint32/uint16, so we use uint8
        ttnn_dtype = ttnn.uint32
        ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, dtype=ttnn_dtype, layout=tensor_layout)
    else:
        ttnn_tensor = ttnn.from_torch(torch_tensor, device=device, layout=tensor_layout)

    torch_op, ttnn_op = getattr(torch, "argmax"), getattr(ttnn, "argmax")

    # Run on both and flag exceptions
    torch_errored = False
    torch_error_msg = ""
    try:
        torch_result = torch_op(torch_tensor, dim=dim, keepdim=keepdim) if dim is not None else torch_op(torch_tensor)
    except (IndexError, RuntimeError) as e:
        torch_errored = True
        torch_error_msg = str(e)

    ttnn_errored = False
    ttnn_error_msg = ""
    try:
        if dim is not None:
            ttnn_result = ttnn_op(ttnn_tensor, dim=dim, keepdim=keepdim, use_multicore=use_multicore)
        else:
            ttnn_result = ttnn_op(ttnn_tensor, use_multicore=use_multicore)
    except RuntimeError as e:
        ttnn_errored = True
        ttnn_error_msg = str(e)

    assert (
        torch_errored == ttnn_errored
    ), f"mismatch in errors raised: torch: {torch_errored} ({torch_error_msg}), ttnn: {ttnn_errored} ({ttnn_error_msg})"

    # Skip the rest of the test if an exception was raised in both
    if torch_errored:
        logger.warning(f"both torch and ttnn raised errors: torch: {torch_error_msg}, ttnn: {ttnn_error_msg}")
        return

    ttnn_result = ttnn.to_torch(ttnn.from_device(ttnn_result)).to(torch.int32)

    pcc_result, msg = check_with_pcc(torch_result, ttnn_result, 0.99)

    assert pcc_result, msg + f"mismatch in pcc: torch: {torch_result}, ttnn: {ttnn_result}"

    # Convert torch dtype from uint64 to int32
    # Note: torch does not have uint32
    torch_result = torch_result.to(torch.int32)

    atol = rtol = 0.1
    assert torch.allclose(
        torch_result, ttnn_result, atol=atol, rtol=rtol, equal_nan=True
    ), f"mismatch in allclose: torch: {torch_result}, ttnn: {ttnn_result}"
