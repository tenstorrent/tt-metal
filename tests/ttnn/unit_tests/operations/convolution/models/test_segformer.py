from ..conv_test_utils import *


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups, shard_layout, config_override, use_shallow_conv_variant",
    (
        # mlp sub_module
        (1, 3, 32, 512, 512, 7, 7, 4, 4, 3, 3, 1, HS, {"act_block_h": 64}, False),  # ncrisc build failed
        # efficient selfattention sub_module
        (1, 32, 32, 128, 128, 8, 8, 8, 8, 0, 0, 1, HS, None, False),  # ncrisc build failed, Two times called in model
        (1, 64, 64, 64, 64, 4, 4, 4, 4, 0, 0, 1, HS, None, False),  # ncrisc build failed, Two times called in model
        (1, 160, 160, 32, 32, 2, 2, 2, 2, 0, 0, 1, HS, None, False),  # pass , Two times called in model
        # dwconv sub_module
        (1, 128, 128, 128, 128, 3, 3, 1, 1, 1, 1, 128, HS, {"act_block_h": 64}, False),
        (1, 256, 256, 64, 64, 3, 3, 1, 1, 1, 1, 256, HS, None, False),  # pass , Two times called in model
        (1, 640, 640, 32, 32, 3, 3, 1, 1, 1, 1, 640, ttnn.TensorMemoryLayout.BLOCK_SHARDED, {"act_block_h": 32}, False),
        # (1,1024, 1024, 16, 16, 3, 3, 1, 1, 1, 1, 1024, BS, None, False), #Switch to Width Sharding
        # decode_head sub_module
        # (1,1024, 256, 128, 128, 1, 1, 1, 1, 0, 0, 1, BS, {"act_block_h": 32}, False), #pass for activation_dtype=bf8 but fails for bf16
        (1, 256, 150, 128, 128, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 32, 16, 64, 64, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 96, 24, 32, 32, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 576, 576, 8, 8, 3, 3, 1, 1, 0, 0, 576, WS, None, False),
        (1, 576, 576, 8, 8, 3, 3, 2, 2, 0, 0, 576, WS, None, False),
        (1, 960, 960, 4, 4, 3, 3, 1, 1, 0, 0, 960, WS, None, False),
        (1, 144, 24, 32, 32, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
        (1, 144, 32, 16, 16, 1, 1, 1, 1, 0, 0, 1, HS, None, False),
    ),
)
@pytest.mark.parametrize(
    "weights_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "activations_dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("auto_shard", [True, False], ids=["auto_shard", "no_auto_shard"])
@skip_for_grayskull()
def test_conv_for_segformer_512x512(
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
    groups,
    output_layout,
    auto_shard,
):
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
        use_shallow_conv_variant=use_shallow_conv_variant,
        groups=groups,
        output_layout=output_layout,
        has_bias=False,
        auto_shard=auto_shard,
        shard_layout=shard_layout,
    )
