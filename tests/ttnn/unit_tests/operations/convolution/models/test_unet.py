from ..conv_test_utils import *


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        # unet convs with batch size 2
        # unique convs in unet (complete list)
        (2, 16, 3, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}, True),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}, True),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, True),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, True),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, True),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, True),
        # fails. mismatch. It passes when input_channels=64. Probably an issue with padding when input_channels % 32 != 0.
        (
            2,
            16,
            48,
            528,
            80,
            3,
            3,
            1,
            1,
            1,
            1,
            HS,
            {"act_block_h": 32},
            False,
        ),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, False),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, False),
        (2, 1, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, BS], ids=["auto_shard", "no_auto_shard"])
def test_unet_conv(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    auto_shard,
):
    if is_blackhole():
        pytest.skip("This test is for Grayskull only")

    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        output_layout=output_layout,
        auto_shard=auto_shard,
    )


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        # unet convs with batch size 2
        # unique convs in unet (complete list)
        (2, 16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (2, 16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (2, 16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        # (2, 1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 5 * 32}, False) # Enable when issue #11490 resolved
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b, ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_unet_conv_wh(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    auto_shard,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        output_layout=output_layout,
        auto_shard=auto_shard,
    )


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [2],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, True),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, True),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 8 * 32}, True),
        (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 16 * 32}, True),
        (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 5 * 32}, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
def test_unet_conv_groups_2_wh(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    auto_shard,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        output_layout=output_layout,
        auto_shard=auto_shard,
        groups=groups,
    )


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [4, 6],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        # (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        # (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, HS, {"act_block_h": 2 * 32}, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_unet_conv_groups_4_6_wh(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        output_layout=output_layout,
        groups=groups,
    )


@skip_for_grayskull()
@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "groups",
    [8],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (16, 4, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        # (16, 16, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        (32, 16, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 32, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (64, 64, 66, 10, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 96, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        (32, 32, 132, 20, 3, 3, 1, 1, 1, 1, HS, None, False),
        # (32, 64, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False), # OOM - need inplace convolution
        (32, 32, 264, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        # (16, 48, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        (16, 16, 528, 80, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True),
        # (16, 32, 1056, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
        # (1, 16, 1056, 160, 1, 1, 1, 1, 0, 0, True, {"act_block_h": 2 * 32}, True), # OOM - need inplace convolution
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [False], ids=["no_auto_shard"])
def test_unet_conv_groups_8_wh(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
    auto_shard,
    groups,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and activations_dtype == ttnn.bfloat8_b:
        pytest.skip("Row major layout not compatible with bfloat8_b")
    if output_layout == ttnn.ROW_MAJOR_LAYOUT and input_height >= 1056:
        pytest.skip("OOM")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        groups * output_channels,
        groups * input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        output_layout=output_layout,
        auto_shard=auto_shard,
        groups=groups,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels,input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override, use_shallow_conv_variant",
    (
        (1, 32, 3, 480, 640, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}, True),
        (1, 32, 32, 480, 640, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 64, 32, 240, 320, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 64, 64, 240, 320, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 64}, False),
        (1, 128, 64, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 128, 128, 120, 160, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 256, 128, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 256, 256, 60, 80, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 512, 256, 30, 40, 3, 3, 1, 1, 1, 1, HS, None, False),
        (1, 512, 512, 30, 40, 3, 3, 1, 1, 1, 1, BS, None, False),
        (1, 256, 512, 60, 80, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}, False),
        (1, 128, 256, 120, 160, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 64, 128, 240, 320, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 32, 64, 256, 256, 3, 3, 1, 1, 1, 1, HS, {"act_block_h": 32}, False),
        (1, 1, 32, 480, 640, 1, 1, 1, 1, 0, 0, HS, None, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@skip_for_grayskull()
def test_conv_for_vanilla_unet(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    shard_layout,
    config_override,
    use_shallow_conv_variant,
    output_layout,
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")
    run_conv(
        device,
        torch_tensor_map,
        math_fidelity,
        activations_dtype,
        weights_dtype,
        batch_size,
        output_channels,
        input_channels,
        input_height,
        input_width,
        filter_height,
        filter_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        config_override,
        shard_layout=shard_layout,
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=1,
        output_layout=output_layout,
        has_bias=False,
    )
