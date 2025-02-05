import ttnn
import torch
import pytest
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc, check_with_pcc_without_tensor_printout


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("B, H, W", [(1, 60, 106)])
@pytest.mark.parametrize(
    "I, O, KH, KW, padding, stride, padding_mode",
    [
        (12, 768, 1, 1, (0, 0), (1, 1), None),
        (768, 768, 3, 3, (1, 1), (1, 1), "replicate"),
    ],
)
def test_interleaved_conv2d(
    device, use_program_cache, reset_seeds, B, H, W, I, O, KH, KW, padding, stride, padding_mode
):
    device.enable_async(True)

    """
    Conv3d(12, 768, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    ContextParallelConv3d(768, 768, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), padding_mode=replicate)
    """

    kernel_shape = (O, I, KH, KW)
    img_h, img_w = H, W
    input_shape = (B, I, img_h, img_w)

    dilation = (1, 1)

    torch_kernel = torch.randn(kernel_shape, dtype=torch.bfloat16)
    torch_bias = torch.randn(O, dtype=torch.bfloat16)
    tt_kernel = ttnn.from_torch(torch_kernel)
    tt_bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, O))

    torch_input = torch.randn(input_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(torch_input, device=device)
    tt_input = ttnn.permute(tt_input, (0, 2, 3, 1))
    tt_input = ttnn.reshape(tt_input, (1, 1, B * img_h * img_w, I))
    tt_input = ttnn.to_layout(tt_input, ttnn.TILE_LAYOUT)
    print(f"{tt_input.shape=}")
    [tt_out, [out_height, out_width], [_, _]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_kernel,
        in_channels=I,
        out_channels=O,
        device=device,
        bias_tensor=tt_bias,
        kernel_size=(KH, KW),
        stride=stride,
        padding=padding,
        dilation=dilation,
        batch_size=B,
        input_height=H,
        input_width=W,
        groups=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    print(f"{tt_out.shape=}")
    print(f"{tt_out.memory_config()=}")
    print(f"{tt_out.layout=}")

    tt_output_tensor = ttnn.from_device(tt_out)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    # torch_output_tensor is in row major layout and NHWC shape
    # NHWC to NCHW
    torch_output_tensor = torch_output_tensor.reshape(B, out_height, out_width, torch_output_tensor.shape[-1])
    torch_output_tensor = torch_output_tensor[:, :, :, :O]

    torch_output_tensor = torch.permute(torch_output_tensor, (0, 3, 1, 2))

    torch_out_golden_tensor = torch.nn.functional.conv2d(
        torch_input, torch_kernel, bias=torch_bias, stride=stride, padding=padding, dilation=dilation, groups=1
    )

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_output_tensor, torch_out_golden_tensor, pcc=0.99)
    logger.info(f"PCC = {pcc_msg}. Threshold = 0.99")
    assert passing
