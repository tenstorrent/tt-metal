import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
@skip_for_wormhole_b0()
def test_head_concat_ttnn(device, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor0 = torch.randn(
        (1, 1, 6400, 4), dtype=torch.bfloat16
    )  # Batch size of 1, 1024 input channels, 10x10 height and width
    torch_input_tensor1 = torch.randn((1, 1, 6400, 80), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn((1, 1, 6400, 1), dtype=torch.bfloat16)

    torch_output_tensor = torch.concat([torch_input_tensor0, torch_input_tensor1, torch_input_tensor2], dim=3)

    input_tensor0 = ttnn.from_torch(torch_input_tensor0, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)

    print(input_tensor0.shape, input_tensor1.shape, input_tensor2.shape)

    output_tensor = ttnn.concat([input_tensor0, input_tensor1, input_tensor2], dim=3)

    print(output_tensor.shape)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
@skip_for_wormhole_b0()
def test_head_concat_row_major(device, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor0 = torch.randn(
        (1, 1, 6400, 4), dtype=torch.bfloat16
    )  # Batch size of 1, 1024 input channels, 10x10 height and width
    torch_input_tensor1 = torch.randn((1, 1, 6400, 80), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn((1, 1, 6400, 1), dtype=torch.bfloat16)

    torch_output_tensor = torch.concat([torch_input_tensor0, torch_input_tensor1, torch_input_tensor2], dim=3)

    input_tensor0 = ttnn.from_torch(torch_input_tensor0, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    print(input_tensor0.shape, input_tensor1.shape, input_tensor2.shape)

    output_tensor = ttnn.concat([input_tensor0, input_tensor1, input_tensor2], dim=3)

    print(output_tensor.shape)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.99)
