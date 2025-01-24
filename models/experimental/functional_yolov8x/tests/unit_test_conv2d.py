import pytest
import ttnn
from tests.ttnn.unit_tests.operations.test_new_conv2d import run_conv


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, pad_h, pad_w, groups",
    (
        # (1, 80, 160, 320, 320, 3, 3, 2, 2, 1, 1, 1), # model.0.conv
        (1, 80, 160, 320, 320, 3, 3, 2, 2, 1, 1, 1),
    ),
)
@pytest.mark.parametrize(
    "deallocate_activation",
    [True],
)
@pytest.mark.parametrize(
    "use_1d_systolic_array",
    [False],
)
@pytest.mark.parametrize(
    "auto_shard",
    [True],
)
@pytest.mark.parametrize(
    "weights_dtype",
    # [ttnn.bfloat16, ttnn.bfloat8_b],
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "activations_dtype",
    # [ttnn.bfloat8_b, ttnn.bfloat16],
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "shard_layout, config_override",
    [
        (None, None),
        # (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, None),
        # (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, {"act_block_h": 32}),
        #  (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, {"act_block_h": 64}),
        #  (ttnn.TensorMemoryLayout.WIDTH_SHARDED, None),
        #  (ttnn.TensorMemoryLayout.BLOCK_SHARDED, None)
    ],
)
@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.LoFi])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
def test_conv_yolov8x(
    device,
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
    use_1d_systolic_array,
    groups,
    output_layout,
    auto_shard,
    shard_layout,
    config_override,
    deallocate_activation,
):
    run_conv(
        device,
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
        use_1d_systolic_array,
        config_override,
        groups=groups,
        output_layout=output_layout,
        auto_shard=auto_shard,
        shard_layout=shard_layout,
        deallocate_activation=deallocate_activation,
    )
