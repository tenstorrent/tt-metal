from loguru import logger
import torch
import pytest
import ttnn
import torch.nn as nn
from models.experimental.mochi.common import compute_metrics

from models.experimental.mochi.tests.vae.test_vol2col import _out_size, torch_vol2col


def run_vol2col_test(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    torch.manual_seed(42)

    # Define input dimensions.
    N, C, D, H, W = input_shape
    D_out = _out_size(D, padding[0], kernel_size[0])
    H_out = _out_size(H, padding[1], kernel_size[1])
    W_out = _out_size(W, padding[2], kernel_size[2])
    num_patches = N * D_out * H_out * W_out
    patch_size = kernel_size[0] * kernel_size[1] * kernel_size[2] * C
    # D = math.ceil(D / T_parallel_factor)

    # Create a random input tensor.
    # input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)
    input_tensor = torch.full((N, C, D, H, W), 0.1, dtype=torch.float32)
    # input_tensor = torch.arange(N * C * D * H * W, dtype=torch.int).reshape(N, C, D, H, W)
    # input_tensor = torch.ones(N, C, D, H, W, dtype=torch.float32)
    print(f"input_tensor.shape NCTHW = {input_tensor.shape}")

    # Create a Conv3d module with chosen parameters.
    in_channels = C
    dilation = (1, 1, 1)
    conv3d_module = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=True,
        padding_mode=padding_mode,
    )

    gt_output = conv3d_module(input_tensor)

    gt_vol2col = torch_vol2col(
        input_tensor,
        conv3d_module,
        depth_block=2,
        hw_block=2,
        out_chan_block=2,
    )  # (num_patches, C_in * kD * kH * kW)
    torch_weight = conv3d_module.weight.data  # out_chan, C, kD, kH, kW
    torch_weight = torch_weight.permute(1, 2, 3, 4, 0)  # C, kD, kH, kW, out_chan
    torch_weight = torch_weight.reshape(-1, out_channels)
    gt_matmul_out = torch.matmul(gt_vol2col, torch_weight)

    # Shape input for TTNN
    tt_input = input_tensor.permute(0, 2, 3, 4, 1)
    ALIGNMENT = 16
    ALIGN_PAD = ALIGNMENT - C % ALIGNMENT
    if C % ALIGNMENT != 0:
        tt_input = torch.nn.functional.pad(tt_input, (0, ALIGN_PAD))
    tt_input = ttnn.from_torch(tt_input, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)

    w = conv3d_module.weight.data  # out_chan, C, kD, kH, kW
    w = w.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    if C % ALIGNMENT != 0:
        w = torch.nn.functional.pad(w, (0, 0, 0, ALIGN_PAD))
    w = w.reshape(-1, out_channels)

    tt_weight = ttnn.from_torch(w, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT)
    tt_bias = ttnn.from_torch(
        conv3d_module.bias.data, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT
    )

    config = ttnn.Conv3dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1,
        W_out_block=1,
        H_out_block=1,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=1,
    )

    tt_vol2col_output = ttnn.conv3d(
        input_tensor=tt_input,
        config=config,
    )

    tt_vol2col_output = ttnn.to_layout(tt_vol2col_output, ttnn.TILE_LAYOUT)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    tt_output = ttnn.linear(tt_vol2col_output, tt_weight, bias=tt_bias, compute_kernel_config=cfg)

    tt_output = ttnn.to_torch(tt_output, device=device, dtype=torch.float32)

    tt_output = tt_output.reshape(N, D_out, H_out, W_out, out_channels)
    tt_output = tt_output.permute(0, 4, 1, 2, 3)

    print(f"gt output shape = {gt_output.shape}")
    print(f"tt output shape = {tt_output.shape}")
    assert tt_output.shape == gt_output.shape
    pcc, mse, mae = compute_metrics(gt_output, tt_output)
    logger.info(f"Compare conv3d torch vs ttnn: PCC = {pcc}, MSE = {mse}, MAE = {mae}")
    if not pcc > 0.99:
        import pdb

        pdb.set_trace()
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        # [(1, 32, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 1, 1), "zeros"],
        # [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 5, 5, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        # [(1, 12, 5, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 768, 5, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 128, 5, 120, 212), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 128, 5, 240, 424), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 128, 5, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # no padding
        # [(1, 5, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 1, 4, 4, 4), 1, (3, 3, 3), (1, 1, 1), (0, 1, 1), "zeros"],
        # smaller
        # [(1, 1, 3, 3, 3), 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 5, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 64, 28, 60, 106), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 64, 82, 120, 212), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 64, 163, 240, 424), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # [(1, 64, 163, 480, 848), 64, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        # Same as above but with no padding
        # [(1, 1, 3, 4, 3), 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 16, 4, 5, 6), 32, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 16, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 5, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 8, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 12, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 0, 0), "replicate"],
        # [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), "replicate"],
        # [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), "replicate"],
        # [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "replicate"],
    ],
    # ids=["test0", "variant0", "variant1", "variant2", "variant3", "variant4"],
)
def test_vol2col_torch_mochi_shapes(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    # def test_vol2col_torch(device, N, C, D, H, W, out_channels, kernel_size, stride, padding, padding_mode):
    # Set a manual seed for reproducibility.
    run_vol2col_test(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode)


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("C_in", [12, 32, 64])
@pytest.mark.parametrize("C_out", [12, 32, 64])
@pytest.mark.parametrize("T", [5, 8, 11])
@pytest.mark.parametrize("H", [7, 10, 13])
@pytest.mark.parametrize("W", [10, 13, 16])
@pytest.mark.parametrize("kernel_size", [(3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize("stride", [(1, 1, 1)])
@pytest.mark.parametrize("padding", [(0, 0, 0), (0, 1, 1)])
@pytest.mark.parametrize("padding_mode", ["zeros", "replicate"])
def test_vol2col_sweep(device, B, C_in, C_out, T, H, W, kernel_size, stride, padding, padding_mode):
    if padding == (0, 0, 0) and padding_mode == "replicate":
        pytest.skip("Skipping padding (0, 0, 0) and padding_mode replicate because it's duplicate")
    input_shape = (B, C_in, T, H, W)
    out_channels = C_out
    kernel_size = kernel_size
    stride = stride
    padding = padding
    padding_mode = padding_mode
    run_vol2col_test(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode)
