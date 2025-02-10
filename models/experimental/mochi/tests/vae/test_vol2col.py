import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics
import math


def torch_vol2col(
    input: torch.Tensor,
    conv3d_module: torch.nn.Conv3d,
    depth_block: int = 1,
    hw_block: int = 1,
    out_chan_block: int = 1,
):
    """
    Emulates a 3D convolution by partitioning the *output volume* in (D, H/W, out_channels),
    then using a vol2col + matrix-multiply (GEMM) on each sub-volume.

    - Stride is forced to (1,1,1)
    - Dilation is forced to (1,1,1)
    - Groups are forced to 1
    - We gather patches via Python slicing (no advanced index tensors).
    - This is a teaching example, not optimized for speed.

    Args:
        input: [N, C_in, D_in, H_in, W_in]
        conv3d_module: an nn.Conv3d with:
            - stride=(1,1,1)
            - dilation=(1,1,1)
            - groups=1
        depth_parallel: how many depth slices of the output to process at once
        hw_parallel: how many height/width slices of the output to process at once
        out_chan_parallel: how many output channels to process at once

    Returns:
        output: [N, out_channels, D_out, H_out, W_out]
    """

    # -----------------
    # 1) Extract Params
    # -----------------
    assert conv3d_module.stride == (1, 1, 1), "This example only supports stride=1"
    assert conv3d_module.dilation == (1, 1, 1), "This example only supports dilation=1"
    assert conv3d_module.groups == 1, "This example assumes groups=1"

    weight = conv3d_module.weight  # [out_channels, C_in, kD, kH, kW]
    bias = conv3d_module.bias
    pad_d, pad_h, pad_w = conv3d_module.padding
    kD, kH, kW = conv3d_module.kernel_size
    out_channels = conv3d_module.out_channels

    N, C_in, D_in, H_in, W_in = input.shape

    # ----------------------
    # 2) Compute Output Size
    # ----------------------
    # For stride=1, dilation=1, groups=1:
    # out_dim = in_dim + 2*pad - (kernel - 1)
    def _out_size(in_size, pad, k):
        return in_size + 2 * pad - (k - 1)

    D_out = _out_size(D_in, pad_d, kD)
    H_out = _out_size(H_in, pad_h, kH)
    W_out = _out_size(W_in, pad_w, kW)

    # -------------------------------------------
    # 3) Pad input along D, H, W (if needed)
    # -------------------------------------------
    if conv3d_module.padding_mode == "zeros":
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode="constant", value=0)
    elif conv3d_module.padding_mode == "replicate":
        input_padded = F.pad(input, (pad_w, pad_w, pad_h, pad_h, pad_d, pad_d), mode="replicate")
    else:
        raise ValueError(f"Unsupported padding_mode {conv3d_module.padding_mode}")

    # input_padded now has shape [N, C_in, D_in + 2*pad_d, H_in + 2*pad_h, W_in + 2*pad_w]
    D_pad, H_pad, W_pad = input_padded.shape[2:]

    num_patches = N * D_out * H_out * W_out
    patch_size = C_in * kD * kH * kW  # (groups=1)

    patches = []
    for d_start in range(0, D_out):
        for h_start in range(0, H_out):
            for w_start in range(0, W_out):
                slice_5d = input_padded[:, :, d_start : d_start + kD, h_start : h_start + kH, w_start : w_start + kW]
                assert slice_5d.shape == (N, C_in, kD, kH, kW)
                # slice_5d = slice_5d.permute(0, 2, 3, 4, 1).contiguous()
                # (N, C_in * kD * kH * kW)
                slice_2d = slice_5d.reshape(N, -1)
                patches.append(slice_2d)

    vol2col_output = torch.cat(patches, dim=0)
    assert vol2col_output.shape == (num_patches, patch_size)

    return vol2col_output


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        # [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 1, 1), "zeros"],
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
        # smaller
        # [(1, 1, 3, 3, 3), 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 5, 24, 10, 15), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
        # [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 0, 0), "zeros"],
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
# @pytest.mark.parametrize("N", [1])
# @pytest.mark.parametrize("C", [5, 12, 16, 20, 128])
# @pytest.mark.parametrize("D", [5, 8, 10, 13])
# @pytest.mark.parametrize("H", [5, 8, 10, 13])
# @pytest.mark.parametrize("W", [5, 8, 10, 13])
# @pytest.mark.parametrize("out_channels", [5, 12, 16, 20, 128])
# @pytest.mark.parametrize("kernel_size", [(3, 3, 3)])
# @pytest.mark.parametrize("stride", [(1, 1, 1)])
# @pytest.mark.parametrize("padding", [(0, 0, 0)])
# @pytest.mark.parametrize("padding_mode", ["zeros"])
# @pytest.mark.parametrize("T_parallel_factor", [8])
def test_vol2col_torch(device, input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    # def test_vol2col_torch(device, N, C, D, H, W, out_channels, kernel_size, stride, padding, padding_mode):
    # Set a manual seed for reproducibility.
    torch.manual_seed(42)

    # Define input dimensions.
    N, C, D, H, W = input_shape
    # D = math.ceil(D / T_parallel_factor)

    # Create a random input tensor.
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)
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

    # Compute the output using the decomposed conv3d (based on conv2d).
    gt_output = torch_vol2col(
        input_tensor,
        conv3d_module,
        depth_block=2,
        hw_block=2,
        out_chan_block=2,
    )  # (num_patches, C_in * kD * kH * kW)

    num_patches, patch_size = gt_output.shape

    # gt patches with channels last
    gt_out_chan_last = gt_output.reshape(num_patches, C, *kernel_size).permute(0, 2, 3, 4, 1).reshape(num_patches, -1)
    assert gt_out_chan_last.shape == (num_patches, kernel_size[0] * kernel_size[1] * kernel_size[2] * C)

    # Shape input for TTNN
    tt_input = input_tensor.permute(0, 2, 3, 4, 1)
    ALIGNMENT = 16
    if C % ALIGNMENT != 0:
        tt_input = torch.nn.functional.pad(tt_input, (0, ALIGNMENT - C % ALIGNMENT))
    tt_input = ttnn.from_torch(tt_input, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_output = ttnn.conv3d(
        input_tensor=tt_input,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=1,
    )
    tt_output = ttnn.to_torch(tt_output, device=device, dtype=torch.float32)
    if C % ALIGNMENT != 0:
        tt_output = tt_output.reshape(num_patches, kernel_size[0], kernel_size[1], kernel_size[2], -1)[..., :C].reshape(
            num_patches, -1
        )

    print(f"gt output shape = {gt_output.shape}")
    print(f"tt output shape = {tt_output.shape}")
    assert tt_output.shape == gt_output.shape

    pcc, mse, mae = compute_metrics(gt_out_chan_last, tt_output)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")
    if not pcc > 0.99:
        import pdb

        pdb.set_trace()
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"
