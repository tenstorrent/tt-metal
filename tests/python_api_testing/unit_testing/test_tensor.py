import pytest

from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")

import torch

import tt_lib


@pytest.mark.parametrize("shape", [(2, 3, 64, 96)])
@pytest.mark.parametrize("tt_dtype", [tt_lib.tensor.DataType.UINT32, tt_lib.tensor.DataType.FLOAT32, tt_lib.tensor.DataType.BFLOAT16])
def test_tensor_conversion_between_torch_and_tt(shape, tt_dtype):
    # Initialize the device
    device = tt_lib.device.CreateDevice(tt_lib.device.Arch.GRAYSKULL, 0)
    tt_lib.device.InitializeDevice(device)

    dtype = {
        tt_lib.tensor.DataType.UINT32:    torch.int32,
        tt_lib.tensor.DataType.FLOAT32:   torch.float,
        tt_lib.tensor.DataType.BFLOAT16:  torch.bfloat16,
        tt_lib.tensor.DataType.BFLOAT8_B: torch.float,
    }[tt_dtype]

    if dtype == torch.int32:
        torch_tensor = torch.randint(0, 1024, shape, dtype=dtype)
    else:
        torch_tensor = torch.rand(shape, dtype=dtype)

    tt_tensor = tt_lib.tensor.Tensor(torch_tensor, tt_dtype)
    if tt_dtype in {tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B}:
        tt_tensor = tt_tensor.to(device)
        tt_tensor = tt_tensor.cpu()

    tt_tensor_data = tt_tensor.to_torch()
    torch_tensor_after_round_trip = tt_tensor_data.reshape(shape)

    assert torch_tensor.dtype == torch_tensor_after_round_trip.dtype
    assert torch_tensor.shape == torch_tensor_after_round_trip.shape

    assert torch.allclose(torch_tensor , torch_tensor_after_round_trip)

    tt_lib.device.CloseDevice(device)
