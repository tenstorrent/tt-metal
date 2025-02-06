import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import pytest
from loguru import logger
from models.experimental.mochi.common import compute_metrics


def decomposed_conv3d_torch(input, conv3d_module):
    """
    A decomposed conv3d that computes the 3D convolution by iterating
    over output depth indices and summing over kernel depth slices via 2D conv.

    Parameters:
      input         : Tensor of shape [N, C, D, H, W]
      conv3d_module : An nn.Conv3d module (with weight, bias, stride, padding, dilation, etc.)

    Returns:
      output        : Tensor of shape [N, out_channels, D_out, H_out, W_out]
    """
    # Extract conv3d parameters.
    weight = conv3d_module.weight  # [out_channels, in_channels, kD, kH, kW]
    bias = conv3d_module.bias
    stride_d, stride_h, stride_w = conv3d_module.stride
    pad_d, pad_h, pad_w = conv3d_module.padding
    dilation_d, dilation_h, dilation_w = conv3d_module.dilation
    kD, kH, kW = conv3d_module.kernel_size
    N, C, D, H, W = input.shape
    padding_mode = conv3d_module.padding_mode

    # Pad only along the depth dimension.
    # (Note: For H and W we rely on F.conv2d’s internal padding.)
    # F.pad takes padding in the order: (pad_left_W, pad_right_W, pad_top_H, pad_bottom_H, pad_front_D, pad_back_D)
    if padding_mode == "zeros":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d))
    elif padding_mode == "replicate":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d), mode="replicate")
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

    # Compute effective kernel sizes and output dimensions.
    eff_kD = dilation_d * (kD - 1) + 1
    D_out = (D + 2 * pad_d - eff_kD) // stride_d + 1
    eff_kH = dilation_h * (kH - 1) + 1
    H_out = (H + 2 * pad_h - eff_kH) // stride_h + 1
    eff_kW = dilation_w * (kW - 1) + 1
    W_out = (W + 2 * pad_w - eff_kW) // stride_w + 1

    # Allocate the output tensor.
    output = torch.zeros((N, conv3d_module.out_channels, D_out, H_out, W_out), dtype=input.dtype, device=input.device)

    # Loop over each output depth index.
    for d in range(D_out):
        # For each output depth position d, conv3d computes:
        #   input_depth_index = d * stride_d + t * dilation_d   for each kernel depth slice t.
        # We accumulate the contributions from each t.
        out_slice = 0
        for t in range(kD):
            depth_index = d * stride_d + t * dilation_d
            # Extract the 2D slice from the padded input at this depth.
            # Shape: [N, C, H, W]
            slice_2d = input_padded[:, :, depth_index, :, :]

            # Use conv2d to process the H/W dimensions with the corresponding kernel slice.
            # Note: We pass padding=(pad_h, pad_w) so that conv2d’s padding matches nn.Conv3d’s H/W padding.
            out2d = F.conv2d(
                slice_2d,
                weight[:, :, t, :, :],
                bias=None,
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
                dilation=(dilation_h, dilation_w),
            )
            # out2d has shape: [N, out_channels, H_out, W_out]
            out_slice = out_slice + out2d
        # Save the accumulated result into the output tensor at depth index d.
        output[:, :, d, :, :] = out_slice

    # Add bias (if any) once at the end.
    if bias is not None:
        output += bias.view(1, -1, 1, 1, 1)
    return output


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
    ids=["variant0", "variant1", "variant2", "variant3", "variant4"],
)
def test_decomposed_conv3d_torch(input_shape, out_channels, kernel_size, stride, padding, padding_mode):
    # Set a manual seed for reproducibility.
    torch.manual_seed(42)

    # Define input dimensions.
    N, C, D, H, W = input_shape

    # Create a random input tensor.
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

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

    # Compute the output using PyTorch's built-in conv3d.
    output_builtin = conv3d_module(input_tensor)

    # Compute the output using the decomposed conv3d (based on conv2d).
    output_decomposed = decomposed_conv3d_torch(input_tensor, conv3d_module)

    pcc, mse, mae = compute_metrics(output_builtin, output_decomposed)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")
    assert pcc > 0.99, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"
    # Compare the two outputs.
    # assert torch.allclose(output_builtin, output_decomposed, atol=1e-5), f"Outputs do not match!\nBuilt-in:\n{output_builtin}\n\nDecomposed:\n{output_decomposed}"


def decomposed_conv3d_tt(device, input, conv3d_module):
    """
    A decomposed conv3d that computes the 3D convolution by iterating
    over output depth indices and summing over kernel depth slices via 2D conv.

    Parameters:
      input         : Tensor of shape [N, C, D, H, W]
      conv3d_module : An nn.Conv3d module (with weight, bias, stride, padding, dilation, etc.)

    Returns:
      output        : Tensor of shape [N, out_channels, D_out, H_out, W_out]
    """
    # Extract conv3d parameters.
    weight = conv3d_module.weight  # [out_channels, in_channels, kD, kH, kW]
    bias = conv3d_module.bias
    stride_d, stride_h, stride_w = conv3d_module.stride
    pad_d, pad_h, pad_w = conv3d_module.padding
    dilation_d, dilation_h, dilation_w = conv3d_module.dilation
    kD, kH, kW = conv3d_module.kernel_size
    N, C, D, H, W = input.shape
    out_channels = conv3d_module.out_channels
    padding_mode = conv3d_module.padding_mode
    # Pad only along the depth dimension.
    # (Note: For H and W we rely on F.conv2d’s internal padding.)
    # F.pad takes padding in the order: (pad_left_W, pad_right_W, pad_top_H, pad_bottom_H, pad_front_D, pad_back_D)
    # TODO: Pad and permute on device
    # Check padding_mode
    if padding_mode == "zeros":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d))
    elif padding_mode == "replicate":
        input_padded = F.pad(input, (0, 0, 0, 0, pad_d, pad_d), mode="replicate")
    else:
        raise ValueError(f"Unsupported padding mode: {padding_mode}")

    # Reshape to get depth in upper dim, and collapse NHW to rows
    input_padded = input_padded.permute(2, 0, 3, 4, 1).reshape(1, D + pad_d * 2, N * H * W, C)
    tt_input = ttnn.from_torch(input_padded, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    # Slice weight on kD
    tt_weights = [ttnn.from_torch(weight[:, :, t, :, :], dtype=ttnn.bfloat16) for t in range(kD)]
    if bias is not None:
        tt_bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )  # Applied at the end

    # Compute effective kernel sizes and output dimensions.
    eff_kD = dilation_d * (kD - 1) + 1
    D_out = (D + 2 * pad_d - eff_kD) // stride_d + 1
    eff_kH = dilation_h * (kH - 1) + 1
    H_out = (H + 2 * pad_h - eff_kH) // stride_h + 1
    eff_kW = dilation_w * (kW - 1) + 1
    W_out = (W + 2 * pad_w - eff_kW) // stride_w + 1

    # Loop over each output depth index.
    print(f"Iterating over {D_out} depth indices and {kD} kernel depth slices")
    out_tensors = []
    for d in range(D_out):
        # For each output depth position d, conv3d computes:
        #   input_depth_index = d * stride_d + t * dilation_d   for each kernel depth slice t.
        # We accumulate the contributions from each t.

        for t in range(kD):
            depth_index = d * stride_d + t * dilation_d
            # Extract the 2D slice from the padded input at this depth.
            # Shape: [N, C, H, W]
            slice_2d = tt_input[:, depth_index : depth_index + 1]

            # Use conv2d to process the H/W dimensions with the corresponding kernel slice.
            [out2d, [out_height, out_width], [_, _]] = ttnn.conv2d(
                input_tensor=slice_2d,
                weight_tensor=tt_weights[t],
                in_channels=C,
                out_channels=out_channels,
                device=device,
                bias_tensor=None,
                kernel_size=(kH, kW),
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
                dilation=(dilation_h, dilation_w),
                batch_size=N,
                input_height=H,
                input_width=W,
                groups=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            # out2d has shape: [N, out_channels, H_out, W_out]
            if t == 0:
                out_slice = out2d
            else:
                out_slice = out_slice + out2d

        # Save the accumulated result into the output tensor at depth index d.
        out_tensors.append(out_slice)

    output = ttnn.concat(out_tensors, dim=1)
    if bias is not None:
        output = output + tt_bias

    tt_output_tensor = ttnn.from_device(output)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    torch_output_tensor = torch_output_tensor.reshape(D_out, N, H_out, W_out, out_channels).permute(1, 4, 0, 2, 3)

    return torch_output_tensor


@pytest.mark.parametrize(
    "input_shape, out_channels, kernel_size, stride, padding, padding_mode",
    [
        [(1, 12, 28, 60, 106), 768, (1, 1, 1), (1, 1, 1), (0, 0, 0), "zeros"],
        [(1, 768, 28, 60, 106), 768, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 512, 82, 120, 212), 512, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 256, 163, 240, 424), 256, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
        [(1, 128, 163, 480, 848), 128, (3, 3, 3), (1, 1, 1), (0, 1, 1), "replicate"],
    ],
    ids=["variant0", "variant1", "variant2", "variant3", "variant4"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_decomposed_conv3d_tt(
    device, input_shape, out_channels, kernel_size, stride, padding, padding_mode, use_program_cache
):
    device.enable_async(True)
    # Set a manual seed for reproducibility.
    torch.manual_seed(42)
    required_pcc = 0.98  # TODO: tighten up

    # Define input dimensions.
    N, C, D, H, W = input_shape

    # Create a random input tensor.
    input_tensor = torch.randn(N, C, D, H, W, dtype=torch.float32)

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

    # Compute the output using PyTorch's built-in conv3d.
    import time

    start = time.perf_counter()
    output_builtin = conv3d_module(input_tensor)
    end = time.perf_counter()
    logger.info(f"Built-in latency: {end - start} seconds")

    # Compute the output using the decomposed conv3d (based on conv2d).
    output_decomposed = decomposed_conv3d_tt(device, input_tensor, conv3d_module)

    pcc, mse, mae = compute_metrics(output_builtin, output_decomposed)
    logger.info(f"PCC = {pcc}, MSE = {mse}, MAE = {mae}")
    assert pcc > required_pcc, f"PCC = {pcc}, MSE = {mse}, MAE = {mae}"

    # start = time.perf_counter()
    # output_decomposed = decomposed_conv3d_tt(device, input_tensor, conv3d_module)
    # end = time.perf_counter()
    # logger.info(f"Compiled decomposed latency: {end - start} seconds")
