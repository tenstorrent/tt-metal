# Minimal conv2d test for fast-tilize debugging.
# Constant inputs make accumulation errors visible as clean ratios.
# Random inputs catch face-swap / addressing bugs that constants mask.
import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
from loguru import logger


def run_conv2d_check(device, input_width, filter_h, const_input):
    batch_size = 1
    input_channels = 32
    output_channels = 32
    input_height = 8
    filter_w = 3
    stride_h = 1
    stride_w = 1
    pad_h = 0
    pad_w = 0

    torch.manual_seed(42)
    if const_input:
        torch_input_nchw = torch.ones(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16) * 7
        torch_weight = torch.ones(output_channels, input_channels, filter_h, filter_w, dtype=torch.bfloat16) * 3
    else:
        torch_input_nchw = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)
        torch_weight = torch.randn(output_channels, input_channels, filter_h, filter_w, dtype=torch.bfloat16)

    # Golden: standard PyTorch conv2d (NCHW)
    torch_out_nchw = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=None, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
    )

    # ttnn.conv2d expects NHWC input
    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1)

    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    [tt_out, [out_height, out_width], [_w, _b]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        device=device,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(filter_h, filter_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_out_torch = ttnn.to_torch(tt_out)

    # Slice ttnn output to actual (non-padded) size, reshape to match golden
    golden_nhwc = torch_out_nchw.permute(0, 2, 3, 1)
    tt_result = tt_out_torch.reshape(-1, output_channels)[: out_height * out_width, :].reshape(
        1, 1, -1, output_channels
    )
    golden = golden_nhwc.reshape(1, 1, -1, output_channels)

    # PCC check (catches correlation-level errors)
    passing_pcc, pcc_msg = check_with_pcc_without_tensor_printout(tt_result, golden, pcc=0.99)
    logger.info(f"PCC = {pcc_msg}")

    # Allclose check (catches absolute value errors that PCC misses with constant data)
    result_f = tt_result.float()
    golden_f = golden.float()
    atol = golden_f.abs().max().item() * 0.05  # 5% of max value
    allclose = torch.allclose(result_f, golden_f, rtol=0.05, atol=max(atol, 1.0))
    if not allclose:
        diff = (result_f - golden_f).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        logger.error(f"allclose FAILED: max_diff={max_diff:.2f}, mean_diff={mean_diff:.2f}, atol={atol:.2f}")

    assert passing_pcc, f"PCC check failed: {pcc_msg}"
    assert allclose, f"allclose failed: max_diff={max_diff:.2f}, mean_diff={mean_diff:.2f}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_width,filter_h",
    [
        (4, 1),
        (4, 2),
        (4, 3),
        (3, 3),
    ],
)
@pytest.mark.parametrize("const_input", [True, False], ids=["const", "random"])
def test_conv2d_ones(device, input_width, filter_h, const_input):
    run_conv2d_check(device, input_width, filter_h, const_input)


# Repro for test_conv_features failures: isolate which parameter triggers the bug.
# Failing config: 256x256, 16ch, filter=3, stride=2, asymmetric padding, batch=2
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size,input_channels,output_channels,input_height,input_width,filter_h,filter_w,stride_h,stride_w,pad_h,pad_w",
    [
        # Simplest failing case: 80x80, 32ch, filter=3, stride=1, no padding
        (1, 32, 32, 8, 4, 3, 3, 1, 1, 0, 0),
        (1, 32, 32, 80, 80, 3, 3, 1, 1, 0, 0),
    ],
    ids=["8x4_pass", "80x80_fail"],
)
def test_conv2d_features_repro(
    device,
    batch_size,
    input_channels,
    output_channels,
    input_height,
    input_width,
    filter_h,
    filter_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
):
    torch.manual_seed(42)
    torch_input_nchw = torch.randn(batch_size, input_channels, input_height, input_width, dtype=torch.bfloat16)
    torch_weight = torch.randn(output_channels, input_channels, filter_h, filter_w, dtype=torch.bfloat16)

    # Handle asymmetric padding
    if isinstance(pad_h, tuple):
        pad_top, pad_bottom = pad_h
        pad_left, pad_right = pad_w
        torch_padded = torch.nn.functional.pad(torch_input_nchw, (pad_left, pad_right, pad_top, pad_bottom))
        torch_out_nchw = torch.nn.functional.conv2d(
            torch_padded, torch_weight, bias=None, stride=(stride_h, stride_w), padding=(0, 0)
        )
        padding_arg = (pad_top, pad_bottom, pad_left, pad_right)
    else:
        torch_out_nchw = torch.nn.functional.conv2d(
            torch_input_nchw, torch_weight, bias=None, stride=(stride_h, stride_w), padding=(pad_h, pad_w)
        )
        padding_arg = (pad_h, pad_w)

    torch_input_nhwc = torch_input_nchw.permute(0, 2, 3, 1)
    tt_input = ttnn.from_torch(torch_input_nhwc, dtype=ttnn.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )

    [tt_out, [out_height, out_width], [_w, _b]] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        device=device,
        in_channels=input_channels,
        out_channels=output_channels,
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        kernel_size=(filter_h, filter_w),
        stride=(stride_h, stride_w),
        padding=padding_arg,
        conv_config=conv_config,
        dtype=ttnn.bfloat16,
        return_output_dim=True,
        return_weights_and_bias=True,
    )

    tt_out_torch = ttnn.to_torch(tt_out)
    golden_nhwc = torch_out_nchw.permute(0, 2, 3, 1)
    tt_result = tt_out_torch.reshape(-1, output_channels)[: out_height * out_width * batch_size, :].reshape(
        1, 1, -1, output_channels
    )
    golden = golden_nhwc.reshape(1, 1, -1, output_channels)

    passing_pcc, pcc_msg = check_with_pcc_without_tensor_printout(tt_result, golden, pcc=0.985)
    logger.info(f"PCC = {pcc_msg}")

    # Find where corruption starts: check PCC per output tile row (32 pixels)
    result_f = tt_result.reshape(-1, output_channels).float()
    golden_f = golden.reshape(-1, output_channels).float()
    n_pixels = result_f.shape[0]
    tile_row_size = 32  # pixels per tile row
    n_tile_rows = (n_pixels + tile_row_size - 1) // tile_row_size
    first_bad = None
    for tr in range(n_tile_rows):
        s = tr * tile_row_size
        e = min(s + tile_row_size, n_pixels)
        r_slice = result_f[s:e].flatten()
        g_slice = golden_f[s:e].flatten()
        if r_slice.std() > 0 and g_slice.std() > 0:
            pcc_val = torch.corrcoef(torch.stack([r_slice, g_slice]))[0, 1].item()
        else:
            pcc_val = 1.0 if torch.allclose(r_slice, g_slice, atol=1.0) else 0.0
        if pcc_val < 0.98 and first_bad is None:
            first_bad = tr
            logger.error(f"First bad tile_row={tr} (pixels {s}-{e-1}), PCC={pcc_val:.4f}")
            logger.error(f"  result[{s}:4]={result_f[s,:4].tolist()}, golden[{s}:4]={golden_f[s,:4].tolist()}")
    if first_bad is not None:
        logger.error(f"Corruption starts at tile_row {first_bad}/{n_tile_rows} (pixel {first_bad*32}/{n_pixels})")
    else:
        logger.info(f"All {n_tile_rows} tile rows OK")

    assert passing_pcc, f"PCC check failed: {pcc_msg}"
