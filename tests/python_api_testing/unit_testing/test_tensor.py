import pytest

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib
from python_api_testing.models.utility_functions import print_diff_argmax


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize("tt_dtype", [tt_lib.tensor.DataType.FLOAT32, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B])
def test_tensor(shape, tt_dtype):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)
    host = tt_lib.device.GetHost()

    dtype = {
        tt_lib.tensor.DataType.FLOAT32:   torch.float,
        tt_lib.tensor.DataType.BFLOAT16:  torch.bfloat16,
        tt_lib.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_dtype]

    atol = 1e-3
    if tt_dtype == tt_lib.tensor.DataType.BFLOAT8_B:
        atol = 1e-2

    torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = (
        tt_lib.tensor.Tensor(
            torch_tensor.reshape(-1).tolist(),
            torch_tensor.shape,
            tt_dtype,
            tt_lib.tensor.Layout.TILE,
        )
    )
    if tt_dtype in {tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B}:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.to(host)

    tt_tensor_data = tt_tensor.data()
    torch_tensor_after_round_trip = torch.frombuffer(tt_tensor_data, dtype=dtype).reshape(shape)

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    torch_tensor_after_round_trip.flatten()[0] = 2
    assert tt_tensor_data[0] == torch_tensor_after_round_trip.flatten()[0]

    # Data mismatches because tt_tensor_data was modified
    assert not torch.allclose(torch_tensor , torch_tensor_after_round_trip, atol=atol, rtol=1e-2)

    # Data matches because torch_tensor  was modified the same way
    torch_tensor.flatten()[0] = 2
    assert torch.allclose(torch_tensor , torch_tensor_after_round_trip, atol=atol, rtol=1e-2)

    tt_lib.device.CloseDevice(device)
