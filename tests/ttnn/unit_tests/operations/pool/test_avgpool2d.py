# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.nightly.unit_tests.operations.pool.test_avgpool2d import run_avg_pool2d

pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}

    return tensor_map


@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        # Normal reduction cases are when channels <= 8 * 32 and kernel_hw <= 16
        # Wide reduction cases channels > 8 * 32
        # Large reduction cases (channels < 32 and kernel_hw > 16) or (channels > 32 and kernel_hw > 32)
        [2, 32, 16, 16],
        [1, 512, 112, 32],
        [1, 320, 48, 48],
        [1, 290, 47, 47],
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        # Wide and normal reductions go to normal kernels
        # Large reductions go to large kernels
        # Reductions which are large and wide at the same time
        # go to large kernels
        (3, 3),
        (9, 9),
        (36, 36),
    ),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize(
    "padding",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "ceil_mode",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "count_include_pad",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "divisor_override",
    [
        None,
        5,
    ],
)
@pytest.mark.parametrize(
    "shard_scheme",
    [
        None,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
def test_avg_pool2d_post_commit(
    device,
    tensor_map,
    input_shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    divisor_override,
    count_include_pad,
    shard_scheme,
    in_dtype,
):
    # we only want to test the largest kernel size with a specific input shape
    # to test otherwise untouched paths in the large kernel, other shapes run OOM
    # or will just slow the test down doing redundant work
    if kernel_size == (36, 36) and input_shape != [1, 320, 48, 48] and input_shape != [1, 290, 47, 47]:
        pytest.skip("Skipping, only run shapes [1, 320, 48, 48] and [1, 290, 47, 47] with kernel size (36, 36)")
    if in_dtype == ttnn.bfloat8_b and input_shape != [1, 320, 48, 48] and input_shape != [1, 512, 112, 32]:
        pytest.skip("Skipping, only run shape [1, 320, 48, 48] with bfloat8_b input dtype")
    run_avg_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        shard_scheme=shard_scheme,
        in_dtype=in_dtype,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize(
    "input_shape, num_slices",  # NCHW
    (
        # Normal reduction cases are when channels <= 8 * 32 and kernel_hw <= 16
        # Wide reduction cases channels > 8 * 32
        # Large reduction cases (channels < 32 and kernel_hw > 16) or (channels > 32 and kernel_hw > 32)
        ([2, 32, 1024, 1024], 8),
        ([1, 320, 384, 384], 6),
        ([1, 64, 96, 96], 4),
        ([1, 64, 96, 96], 0),  # auto num_slices: exercises L1 estimation -> slice determination path
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    ((3, 3),),
)
@pytest.mark.parametrize(
    "stride",
    ((2, 2),),
)
@pytest.mark.parametrize(
    "padding",
    ((1, 1),),
)
@pytest.mark.parametrize(
    "ceil_mode",
    [True],
)
@pytest.mark.parametrize(
    "divisor_override",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "count_include_pad",
    [True, False],  # False exercises per-element scalar CB path in DRAM slicing L1 estimation
)
@pytest.mark.parametrize(
    "shard_scheme",
    [
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ],
)
@pytest.mark.parametrize(
    "in_dtype",
    [ttnn.bfloat16],
)
def test_avg_pool2d_dram_post_commit(
    device,
    tensor_map,
    input_shape,
    num_slices,
    kernel_size,
    stride,
    padding,
    divisor_override,
    ceil_mode,
    count_include_pad,
    shard_scheme,
    in_dtype,
):
    dram_slice_config = ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceWidth)

    run_avg_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        divisor_override=divisor_override,
        count_include_pad=count_include_pad,
        shard_scheme=shard_scheme,
        in_dtype=in_dtype,
        nightly_skips=False,
        dram_slice_config=dram_slice_config,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        [1, 2048, 7, 7],  # ResNet-50 final layer
        [1, 144, 7, 7],  # EfficientNet (non-tile-aligned channels)
        [2, 64, 8, 8],  # Multi-batch
        [4, 144, 7, 7],  # Multi-batch with non-tile-aligned channels
        [1, 512, 1, 1],  # Already 1x1 spatial
        [1, 16, 56, 56],  # MobileNetV3 SqueezeExcitation entry shape
    ),
)
@pytest.mark.parametrize(
    "divisor_override",
    [None, 5],
)
@pytest.mark.parametrize(
    "in_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
def test_avg_pool2d_global_pool_post_commit(
    device,
    tensor_map,
    input_shape,
    divisor_override,
    in_dtype,
):
    """Global average pooling via avg_pool2d (kernel == input spatial dims, no padding).
    Exercises the reduction-based fast path that replaces the standalone global_avg_pool op.
    bfloat8_b coverage exercises the block-float dtype conversion branch."""
    in_n, in_c, in_h, in_w = input_shape
    run_avg_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        kernel_size=(in_h, in_w),
        stride=(1, 1),
        padding=(0, 0),
        ceil_mode=False,
        divisor_override=divisor_override,
        count_include_pad=True,
        shard_scheme=None,
        in_dtype=in_dtype,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize(
    "input_shape",  # NCHW
    (
        [1, 2048, 7, 7],
        [1, 16, 56, 56],
    ),
)
@pytest.mark.parametrize(
    "memory_config",
    [
        ttnn.DRAM_MEMORY_CONFIG,
        ttnn.L1_MEMORY_CONFIG,
    ],
    ids=["DRAM", "L1"],
)
def test_global_avg_pool2d_memory_config(device, input_shape, memory_config):
    """Verify ttnn.global_avg_pool2d honors the requested output memory_config.
    The reduction fast path is separate from the regular pool pipeline; this guards
    against accidentally ignoring the caller's memory_config (PR #41063 review)."""
    import torch
    from tests.ttnn.utils_for_testing import assert_with_pcc

    torch.manual_seed(0)
    torch_input = torch.randn(input_shape)
    torch_expected = torch.nn.functional.adaptive_avg_pool2d(torch_input, (1, 1))

    nhwc = torch.permute(torch_input, (0, 2, 3, 1))
    tt_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.global_avg_pool2d(tt_input, memory_config=memory_config)

    assert (
        tt_output.memory_config().buffer_type == memory_config.buffer_type
    ), f"output buffer type {tt_output.memory_config().buffer_type} != requested {memory_config.buffer_type}"

    tt_torch = ttnn.to_torch(tt_output)
    tt_torch = torch.permute(tt_torch, (0, 3, 1, 2))
    assert_with_pcc(torch_expected, tt_torch, 0.999)


def test_avg_pool2d_global_rejects_zero_batch(device, expect_error):
    in_c, in_h, in_w = 32, 8, 8
    ttnn_input = ttnn.from_torch(
        torch.randn(1, 1, in_h * in_w, in_c, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error(RuntimeError, "Batch size must be greater than 0"):
        ttnn.avg_pool2d(
            input_tensor=ttnn_input,
            batch_size=0,
            input_h=in_h,
            input_w=in_w,
            channels=in_c,
            kernel_size=[in_h, in_w],
            stride=[1, 1],
            padding=[0, 0],
        )


@pytest.mark.parametrize(
    "out_dtype, in_c",
    [
        # Two or three channel cores whose row-major input shards were 24 wide against 32-wide output shards
        (ttnn.bfloat16, 33),
        (ttnn.bfloat16, 48),
        (ttnn.bfloat16, 65),
        (ttnn.bfloat16, 72),
        (ttnn.bfloat8_b, 48),
        # Input and output shards were already both 32 wide
        (ttnn.bfloat16, 64),
        (ttnn.bfloat16, 96),
        (ttnn.bfloat8_b, 64),
    ],
)
def test_avg_pool2d_tiled_out_auto_shard_channels(device, out_dtype, in_c):
    in_n, in_h, in_w = 2, 16, 16
    kernel_size, stride = [2, 2], [2, 2]
    torch.manual_seed(0)
    torch_input = torch.randn(in_n, in_c, in_h, in_w, dtype=torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, in_n * in_h * in_w, in_c),
        ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    ttnn_output = ttnn.avg_pool2d(
        input_tensor=ttnn_input,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=[0, 0],
        dtype=out_dtype,
        output_layout=ttnn.TILE_LAYOUT,
    )
    torch_output = torch.nn.functional.avg_pool2d(torch_input.float(), kernel_size, stride)
    out_h, out_w = torch_output.shape[-2:]
    ttnn_output = ttnn.to_torch(ttnn_output).float().reshape(in_n, out_h, out_w, in_c).permute(0, 3, 1, 2)
    if out_dtype == ttnn.bfloat16:
        assert torch.allclose(ttnn_output, torch_output, rtol=0.02, atol=0.05)
    else:
        assert torch.allclose(ttnn_output, torch_output, rtol=0.04, atol=0.15)
