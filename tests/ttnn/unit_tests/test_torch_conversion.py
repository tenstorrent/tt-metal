# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import pytest
import ttnn


@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
    ],
)
@pytest.mark.parametrize(
    "ttnn_dtype",
    [
        ttnn.uint16,
        ttnn.uint32,
    ],
)
@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_torch_conversion_unsigned_edge_cases_random(device, shape, ttnn_dtype, ttnn_layout, seed):
    torch.manual_seed(seed)

    if ttnn_dtype == ttnn.uint16:
        low = np.iinfo(np.uint16).min
        high = np.iinfo(np.uint16).max

    elif ttnn_dtype == ttnn.uint32:
        low = np.iinfo(np.uint32).min
        high = np.iinfo(np.uint32).max

    ttnn_input_tensor = ttnn.rand(
        shape,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn_layout,
        low=low,
        high=high,
    )

    torch_result_tensor: torch.Tensor = ttnn.to_torch(ttnn_input_tensor)

    torch.testing.assert_close(torch.tensor(torch_result_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))


@pytest.mark.parametrize(
    "tensor_data,ttnn_dtype,torch_input_type",
    [
        ([np.iinfo(np.uint16).max], ttnn.uint16, torch.uint16),
        ([np.iinfo(np.uint16).min], ttnn.uint16, torch.uint16),
        ([np.iinfo(np.uint32).max], ttnn.uint32, torch.uint32),
        ([np.iinfo(np.uint32).min], ttnn.uint32, torch.uint32),
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_device", [True, False])
def test_torch_conversion_unsigned_edge_cases_fixed(
    device, tensor_data, ttnn_dtype, torch_input_type, ttnn_layout, with_device
):
    torch_input_tensor = torch.tensor(tensor_data, dtype=torch_input_type)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn_dtype,
        layout=ttnn_layout,
        device=device if with_device else None,
    )

    torch_result_tensor: torch.Tensor = ttnn.to_torch(ttnn_input_tensor)

    torch.testing.assert_close(torch.tensor(torch_input_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))
    torch.testing.assert_close(torch.tensor(torch_result_tensor.tolist()), torch.tensor(ttnn_input_tensor.to_list()))
    torch.testing.assert_close(torch_input_tensor, torch_result_tensor)


@pytest.mark.parametrize(
    "tensor_data,ttnn_dtype,numpy_input_type",
    [
        ([np.iinfo(np.uint16).max], ttnn.uint16, np.uint16),
        ([np.iinfo(np.uint16).min], ttnn.uint16, np.uint16),
        ([np.iinfo(np.uint32).max], ttnn.uint32, np.uint32),
        ([np.iinfo(np.uint32).min], ttnn.uint32, np.uint32),
    ],
)
@pytest.mark.parametrize("ttnn_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("with_device", [True, False])
def test_numpy_conversion_unsigned_edge_cases_fixed(
    device, tensor_data, ttnn_dtype, numpy_input_type, ttnn_layout, with_device
):
    numpy_input_tensor = np.array(tensor_data, dtype=numpy_input_type)
    ttnn_input_tensor = ttnn.from_torch(
        numpy_input_tensor,
        dtype=ttnn_dtype,
        layout=ttnn_layout,
        device=device if with_device else None,
    )
    numpy_result_tensor = ttnn_input_tensor.cpu().to_numpy()
    np.testing.assert_allclose(np.array(numpy_input_tensor.tolist()), np.array(ttnn_input_tensor.to_list()))
    np.testing.assert_allclose(np.array(numpy_result_tensor.tolist()), np.array(ttnn_input_tensor.to_list()))
    np.testing.assert_allclose(numpy_input_tensor, numpy_result_tensor)
