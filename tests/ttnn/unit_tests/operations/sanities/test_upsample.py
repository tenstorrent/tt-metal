import pytest
import torch
import ttnn
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 128, 15, 15],
        [1, 256, 15, 15],
        [1, 384, 15, 15],
        [1, 512, 15, 15],
        [1, 640, 15, 15],
        [1, 2048, 7, 7],
        [1, 64, 7, 7],
        [1, 72, 7, 7],
        [1, 120, 7, 7],
        [1, 128, 7, 7],
        [1, 160, 7, 7],
        [1, 176, 7, 7],
        [1, 192, 7, 7],
        [1, 256, 7, 7],
    ],
)
@pytest.mark.parametrize("scale_h", [2])
@pytest.mark.parametrize("scale_w", [2])
def test_upsample_single_core_odd(device, input_shapes, scale_h, scale_w):
    torch.manual_seed(0)
    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    print("shape of input", input.shape)
    tt_input = input.permute(0, 3, 1, 2)

    scale_factor = (scale_h, scale_w)
    m = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = m(tt_input)
    torch_result = torch_result.permute(0, 2, 3, 1)
    print("torch output shape", torch_result.shape)

    scale_factor = (scale_h, scale_w)
    input_tensor = ttnn.from_torch(input, device=device)
    output_tensor = ttnn.upsample(input_tensor, scale_factor)
    output_tensor = ttnn.to_torch(output_tensor)
    print("ttnn output shape", output_tensor.shape)

    assert_with_pcc(torch_result, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 256, 16, 16],
        [1, 16, 14, 14],
        [1, 32, 7, 7],
        [1, 18, 14, 14],
        [1, 36, 7, 7],
        [1, 30, 14, 14],
        [1, 60, 7, 7],
        [1, 32, 14, 14],
        [1, 64, 7, 7],
        [1, 40, 14, 14],
        [1, 80, 7, 7],
        [1, 44, 14, 14],
        [1, 88, 7, 7],
        [1, 48, 14, 14],
        [1, 96, 7, 7],
        [1, 64, 14, 14],
        [1, 128, 7, 7],
    ],
)
@pytest.mark.parametrize("scale_h", [4])
@pytest.mark.parametrize("scale_w", [4])
def test_upsample_single_core_sf4(device, input_shapes, scale_h, scale_w):
    torch.manual_seed(0)
    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    print("shape of input", input.shape)
    tt_input = input.permute(0, 3, 1, 2)

    scale_factor = (scale_h, scale_w)
    m = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = m(tt_input)
    torch_result = torch_result.permute(0, 2, 3, 1)
    print("torch output shape", torch_result.shape)

    scale_factor = (scale_h, scale_w)
    input_tensor = ttnn.from_torch(input, device=device)
    output_tensor = ttnn.upsample(input_tensor, scale_factor)
    output_tensor = ttnn.to_torch(output_tensor)
    print("ttnn output shape", output_tensor.shape)

    assert_with_pcc(torch_result, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 16, 7, 7],
        [1, 18, 7, 7],
        [1, 30, 7, 7],
        [1, 32, 7, 7],
        [1, 40, 7, 7],
        [1, 44, 7, 7],
        [1, 48, 7, 7],
        [1, 64, 7, 7],
    ],
)
@pytest.mark.parametrize("scale_h", [8])
@pytest.mark.parametrize("scale_w", [8])
def test_upsample_single_core_sf8(device, input_shapes, scale_h, scale_w):
    torch.manual_seed(0)
    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    print("shape of input", input.shape)
    tt_input = input.permute(0, 3, 1, 2)

    scale_factor = (scale_h, scale_w)
    m = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = m(tt_input)
    torch_result = torch_result.permute(0, 2, 3, 1)
    print("torch output shape", torch_result.shape)

    scale_factor = (scale_h, scale_w)
    input_tensor = ttnn.from_torch(input, device=device)
    output_tensor = ttnn.upsample(input_tensor, scale_factor)
    output_tensor = ttnn.to_torch(output_tensor)
    print("ttnn output shape", output_tensor.shape)

    assert_with_pcc(torch_result, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 64, 112, 112],  # height and width not divisble by 32
        [1, 128, 64, 128],  # height and width divisble by 32 but not equal
        [1, 256, 64, 64],  # height and width divisble by 32 & equal
    ],
)
@pytest.mark.parametrize("scale_h", [2])
@pytest.mark.parametrize("scale_w", [2])
def test_upsample_single_core_bilinear(device, input_shapes, scale_h, scale_w):
    batch_size, height, width, num_channels = input_shapes

    torch.manual_seed(0)
    input = torch.rand(input_shapes, dtype=torch.bfloat16)
    print("shape of input", input.shape)
    tt_input = input.permute(0, 3, 1, 2)

    scale_factor = (scale_h, scale_w)
    m = nn.Upsample(scale_factor=scale_factor, mode="bilinear")
    torch_result = m(tt_input)
    torch_result = torch_result.permute(0, 2, 3, 1)
    print("torch output shape", torch_result.shape)

    scale_factor = (scale_h, scale_w)
    input_tensor = ttnn.from_torch(input, device=device)
    output_tensor = ttnn.upsample(input_tensor, scale_factor)
    output_tensor = ttnn.to_torch(output_tensor)
    print("ttnn output shape", output_tensor.shape)

    assert_with_pcc(torch_result, output_tensor)
