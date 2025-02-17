from ..conv_test_utils import *


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, shard_layout, config_override",
    (
        # sd convs with HxW=32x32
        (1, 320, 320, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        (1, 320, 320, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),
        (1, 640, 640, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),
        (1, 640, 640, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 activations doesnt fit
        (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),  # slighlty low pcc with 0.99689. bfloat16 weights doesnt fit
        (1, 1280, 1280, 8, 8, 3, 3, 2, 2, 1, 1, BS, None),  # fails to parallelize with sharding
        (1, 1280, 1280, 4, 4, 3, 3, 1, 1, 1, 1, BS, None),  # fails to parallelize with sharding
        (
            1,
            1280,
            1280,
            16,
            16,
            3,
            3,
            1,
            1,
            1,
            1,
            BS,
            None,
        ),  # slightly low pcc with 0.99698. bfloat16 weights doesnt fit
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),  # doesnt fit at all.. for all data types
        # sd convs with HxW=64x64 with batch size = 1
        (1, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        (1, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (1, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  #
        (1, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        (1, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit.
        (1, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 weights doesnt fit
        (1, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, None),
        (1, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (1, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),
        (1, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        # # sd convs with HxW=64x64 with batch size=2
        (2, 320, 16, 64, 64, 3, 3, 1, 1, 1, 1, HS, None),
        (2, 320, 320, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 64}),
        (2, 320, 320, 64, 64, 3, 3, 2, 2, 1, 1, BS, None),  # fits with bfloat8_b
        (2, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 640, 32, 32, 3, 3, 2, 2, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),  # bfloat16 doesnt fit
        (2, 1280, 1280, 16, 16, 3, 3, 2, 2, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        (2, 1280, 1280, 8, 8, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 1280, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),  # bfloat16 doesnt fit
        # (2, 640, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}), L1 Allocation Error
        (2, 1280, 2560, 8, 8, 3, 3, 1, 1, 1, 1, BS, None),
        (2, 1280, 2560, 16, 16, 3, 3, 1, 1, 1, 1, BS, None),
        (2, 1280, 1920, 16, 16, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 1920, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 1280, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 640, 960, 32, 32, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 320, 960, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        (2, 320, 640, 64, 64, 3, 3, 1, 1, 1, 1, BS, {"act_block_h": 32}),
        # 1x1 conv
        (2, 320, 960, 64, 64, 1, 1, 1, 1, 0, 0, HS, None),
        # Small conv
        (1, 32, 32, 16, 16, 3, 3, 2, 2, 1, 1, HS, None),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "fp32_accum",
    [
        False,
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
def test_stable_diffusion(
    device,
    torch_tensor_map,
    use_program_cache,
    math_fidelity,
    activations_dtype,
    weights_dtype,
    fp32_accum,
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
):
    if device.core_grid.y == 7:
        pytest.skip("This test is not supported for N300")

    # Skip test cases raising OOM, but do not affect the SD e2e test
    if (
        (input_channels == 320 and config_override == None and activations_dtype == ttnn.bfloat16)
        or (input_channels == 960 and config_override == None and fp32_accum == True)
        or (
            output_channels == 1280
            and input_height == 32
            and activations_dtype == ttnn.bfloat16
            and weights_dtype == ttnn.bfloat16
        )
    ):
        pytest.skip("Skip the test cases raising OOM but not affecting e2e test")

    if filter_height > 1 and (input_channels > 1280 or (input_channels > 640 and input_height > 16)):
        run_conv_with_split(
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
            split_factor=3 if input_channels == 1920 else 2,
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
        )
    else:
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
            use_shallow_conv_variant=(input_channels == 16),
            fp32_accum=fp32_accum,
            packer_l1_acc=True,
        )
