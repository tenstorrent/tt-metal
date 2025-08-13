import torch
import ttnn
import pytest
from models.experimental.oft.reference.resnet import BasicBlock
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from models.experimental.oft.tt.custom_preprocessor import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride",
    [
        (1, 128, 128, 80, 64, 1),
        (1, 128, 128, 48, 160, 1),
        (1, 128, 256, 64, 80, 2),
        (1, 128, 256, 48, 160, 2),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
def test_tt_basicblock(device, n, in_ch, out_ch, h, w, stride):
    torch.manual_seed(42)
    input_tensor = torch.randn(n, in_ch, h, w)
    torch_model = BasicBlock(inplanes=in_ch, planes=out_ch, stride=stride)
    out = torch_model.forward(input_tensor)

    params = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )
    print("-----------------------------------------")
    # print(f"{params=}")
    print("-----------------------------------------")

    additional_params = {}
    additional_params["conv1"] = {
        "batch_size": input_tensor.shape[0],
        "input_height": input_tensor.shape[2],  # // stride,
        "input_width": input_tensor.shape[3],  # // stride,
        "in_channels": params.conv1.weight.shape[1],
        "out_channels": params.conv1.weight.shape[0],
    }
    additional_params["conv2"] = {
        "batch_size": input_tensor.shape[0],
        "input_height": input_tensor.shape[2] // stride,
        "input_width": input_tensor.shape[3] // stride,
        "in_channels": params.conv2.weight.shape[1],
        "out_channels": params.conv2.weight.shape[0],
    }
    if "downsample" in params:
        additional_params["conv_downsample"] = {}
        additional_params["conv_downsample"]["batch_size"] = input_tensor.shape[0]
        additional_params["conv_downsample"]["input_height"] = input_tensor.shape[2]  # // stride
        additional_params["conv_downsample"]["input_width"] = input_tensor.shape[3]  # // stride
        additional_params["conv_downsample"]["in_channels"] = params.downsample[0].weight.shape[1]
        additional_params["conv_downsample"]["out_channels"] = params.downsample[0].weight.shape[0]

    block = TTBasicBlock(
        device,
        params,
        additional_params,
        inplanes=in_ch,
        planes=out_ch,
        channels=out_ch,
        cell_size=h,
        grid_height=w,
        stride=stride,
    )

    x_for_ttnn = input_tensor.permute(0, 2, 3, 1)
    ttnn_x = ttnn.from_torch(x_for_ttnn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_out = block.forward(device, ttnn_x)

    print(f"Output shape: {ttnn_out.shape}, torch out {out.shape}")
    B, C, H, W = out.shape
    ttnn_out = ttnn.to_torch(ttnn_out)

    out = out.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
    pcc, message = assert_with_pcc(ttnn_out, out, 0.99)
    print(f"PCC: {pcc}, Message: {message}")


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride",
    [
        (1, 128, 128, 80, 64, 1),
        (1, 128, 128, 48, 160, 1),
        # (1, 128, 256, 48, 160, 2),
    ],
)
def test_basic_block_torch(n, in_ch, out_ch, h, w, stride):
    torch.manual_seed(42)
    input_tensor = torch.randn(n, in_ch, h, w)
    block = BasicBlock(inplanes=in_ch, planes=out_ch, stride=stride)
    out = block.forward(input_tensor)
    assert out.shape == (n, out_ch, h // stride, w // stride)
    assert torch.isfinite(out).all()
    print("BasicBlock forward test passed.")


@pytest.mark.parametrize(
    "inplanes, planes, stride, input_shape",
    [
        # (64, 64, 1, (1, 64, 56, 56)),   # identity path
        # (64, 128, 2, (1, 64, 56, 56)),  # downsample path
        (128, 256, 2, (1, 128, 64, 80)),
    ],
)
def test_basicblock_forward(inplanes, planes, stride, input_shape):
    torch.manual_seed(0)
    block = BasicBlock(inplanes, planes, stride)
    x = torch.randn(*input_shape)
    out = block.forward(x)
    # Output shape should match expected ResNet block output
    expected_h = input_shape[2] // stride
    expected_w = input_shape[3] // stride
    assert out.shape == (input_shape[0], planes, expected_h, expected_w)
    assert not torch.isnan(out).any()
