# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
import pytest
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d


# Cache map used for torch tensor reuse - the tensor will not be generated if a tensor of the same dimensions has already been generated
@pytest.fixture(scope="module")
def tensor_map(request):
    tensor_map = {}

    return tensor_map


HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
SliceWidth = ttnn.Op2DDRAMSliceWidth
SliceHeight = ttnn.Op2DDRAMSliceHeight

parameters = {
    "dram_slice_tests": {
        "in_specs": [[ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT], [ttnn.bfloat8_b, ttnn.TILE_LAYOUT]],
        "input_specs": [
            # Contains following parameters
            # [in_n, in_c, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode, num_slices, shard_layout, slice_type]
            [1, 128, 1024, 1024, 2, 2, 2, 2, 0, 0, 1, 1, False, 8, HS, SliceWidth],
            [1, 480, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, False, 8, BS, SliceWidth],
            [1, 32768, 32, 32, 2, 2, 1, 1, 0, 0, 1, 1, False, 4, WS, SliceHeight],
            [1, 128, 1024, 1024, 2, 2, 2, 2, 0, 0, 1, 1, True, 8, HS, SliceWidth],
            [1, 480, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, True, 8, BS, SliceWidth],
            [1, 256, 81, 81, 2, 2, 2, 2, 0, 0, 1, 1, True, 2, HS, SliceHeight],
            # Pooling dimension has been changed from width to height. Otherwise, with tile layout, the output width of 1 gets rounded up to 32.
            [1, 256, 64, 1024, 64, 1, 1, 1, 0, 0, 1, 1, False, 8, BS, SliceWidth],
            [1, 256, 32, 1024, 32, 1, 1, 1, 0, 0, 1, 1, False, 8, BS, SliceWidth],
            [1, 256, 64, 2048, 64, 1, 1, 1, 0, 0, 1, 1, False, 8, BS, SliceWidth],
        ],
    },
    "height_shard_tests": {
        "in_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            # Contains following parameters
            # [in_n, in_c, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode]
            [1, 128, 150, 150, 2, 2, 2, 2, 0, 0, 1, 1, False],
            [1, 16, 25, 23, 2, 2, 2, 2, 0, 0, 1, 1, False],  # C=16
            [1, 480, 28, 28, 3, 3, 2, 2, 1, 1, 1, 1, True],
            [1, 7, 24, 24, 3, 3, 1, 1, 0, 0, 2, 2, False],  # dilation, C = 7
            [1, 1, 59, 59, 3, 5, 4, 2, 1, 1, 5, 4, True],  # dilation with ceil mode, C = 1
            [1, 64, 400, 544, 3, 3, 2, 2, 1, 1, 1, 1, False],  # massive NHW
            [1, 832, 14, 14, 4, 4, 2, 2, 0, 0, 1, 1, True],  # > 800 channels, 16 kernel
            [1, 160, 30, 30, 15, 15, 1, 1, 7, 5, 1, 1, False],  # 15x15 kernel, uneven padding
            [1, 224, 20, 20, 8, 8, 6, 6, 2, 4, 1, 1, False],  # 8x8 kernel, uneven padding
            [1, 320, 48, 48, 36, 36, 1, 1, 0, 0, 1, 1, False],  # massive kernel, wide
            [1, 290, 47, 47, 36, 36, 1, 1, 0, 0, 1, 1, False],  # non-tile multiple NHW
            [1, 320, 48, 48, 36, 36, 2, 2, 0, 0, 1, 1, True],  # massive kernel, wide, ceil mode
            [1, 290, 47, 47, 36, 36, 2, 2, 0, 0, 1, 1, True],  # non-tile multiple NHW, ceil mode
            [1, 32, 6, 6, 3, 3, 1, 1, 1, 1, 1, 1, False],  # partial grid on WH to use noop cores
            [1, 32, 13, 8, 4, 3, 6, 5, 2, 1, 1, 1, True],  # ceil mode output shape adjustment edge case
            # requires reversed local reads on some cores, and forward reads on others
            [8, 64, 112, 112, 3, 3, 2, 2, 1, 1, 1, 1, True],
            # requires reversed local reads on some cores, and forward reads on others, large kernel
            [32, 32, 264, 40, 5, 5, 2, 2, 2, 2, 1, 1, True],
        ],
    },
    "width_shard_tests": {
        "in_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            [1, 32768, 6, 6, 2, 2, 1, 1, 0, 0, 1, 1, False],  # wide in place untilize
            [1, 16384, 8, 8, 2, 2, 1, 1, 0, 0, 1, 1, False],  # normal in place untilize
            [1, 6144, 20, 20, 11, 11, 1, 1, 5, 5, 1, 1, False],  # 11x11 kernel
        ],
    },
    "block_shard_tests": {
        "in_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            [1, 4096, 16, 16, 2, 2, 1, 1, 0, 0, 1, 1, False],  # wide in place untilize
            [1, 2048, 16, 16, 2, 2, 1, 1, 0, 0, 1, 1, False],  # normal in place untilize
            # requires reversed local reads on some cores, and forward reads on others, wide in place untilize, large kernel
            [1, 4096, 16, 16, 5, 5, 2, 2, 2, 2, 1, 1, True],
            # requires reversed local reads on some cores, and forward reads on others, normal in place untilize, large kernel
            [1, 2048, 16, 16, 5, 5, 2, 2, 2, 2, 1, 1, True],
            [1, 512, 25, 25, 12, 12, 1, 1, 6, 6, 1, 1, False],  # 12x12 kernel
        ],
    },
    "in_mem_config_tests": {
        "cores": [[8, 7]],  # N300 core grid
        "input_specs": [
            # trace from openpdn_mnist test, no work on core 47 for 8x7 N300 grid
            [2, 32, 78, 78, 3, 3, 3, 3, 0, 0, 1, 1, False],
        ],
    },
    "out_mem_config_tests": {
        "in_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "input_specs": [
            [1, 32, 224, 224, 3, 3, 2, 2, 1, 1, 1, 1, False],
        ],
    },
    "tiled_out_tests": {
        "in_dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "out_dtype": [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
        "input_specs": [
            [1, 320, 224, 224, 3, 3, 1, 1, 1, 1, 1, 1, False],
        ],
    },
}


@pytest.mark.parametrize("input_spec", parameters["dram_slice_tests"]["input_specs"])
@pytest.mark.parametrize("in_specs", parameters["dram_slice_tests"]["in_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_dram_slice(device, in_specs, input_spec):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
        num_slices,
        shard_scheme,
        slice_type,
    ) = input_spec
    [in_dtype, output_layout] = in_specs
    dram_slice_config = ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=slice_type)
    torch_tensor_map = {}
    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        [kernel_h, kernel_w],
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        device,
        torch_tensor_map,
        in_dtype,
        shard_scheme=shard_scheme,
        ceil_mode=ceil_mode,
        nightly_skips=False,
        dram_slice_config=dram_slice_config,
        output_layout=output_layout,
        config_tensor_in_dram=False,
    )


@pytest.mark.parametrize("input_spec", parameters["height_shard_tests"]["input_specs"])
@pytest.mark.parametrize("in_dtype", parameters["height_shard_tests"]["in_dtype"])
def test_max_pool2d_height_shard(device, in_dtype, input_spec, tensor_map):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        in_dtype,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize("input_spec", parameters["width_shard_tests"]["input_specs"])
@pytest.mark.parametrize("in_dtype", parameters["width_shard_tests"]["in_dtype"])
def test_max_pool2d_width_shard(device, in_dtype, input_spec, tensor_map):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        in_dtype,
        shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ceil_mode=ceil_mode,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize("input_spec", parameters["block_shard_tests"]["input_specs"])
@pytest.mark.parametrize("in_dtype", parameters["block_shard_tests"]["in_dtype"])
def test_max_pool2d_block_shard(device, in_dtype, input_spec, tensor_map):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        in_dtype,
        shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ceil_mode=ceil_mode,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize("input_spec", parameters["out_mem_config_tests"]["input_specs"])
@pytest.mark.parametrize("in_dtype", parameters["out_mem_config_tests"]["in_dtype"])
@pytest.mark.parametrize("out_memory_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_max_pool2d_mem_config(device, in_dtype, input_spec, out_memory_config, tensor_map):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        in_dtype,
        out_memory_config=out_memory_config,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize("input_spec", parameters["tiled_out_tests"]["input_specs"])
@pytest.mark.parametrize("in_dtype", parameters["tiled_out_tests"]["in_dtype"])
@pytest.mark.parametrize("out_dtype", parameters["tiled_out_tests"]["out_dtype"])
def test_max_pool2d_tiled_out(device, in_dtype, input_spec, out_dtype, tensor_map):
    output_layout = ttnn.TILE_LAYOUT

    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        in_dtype,
        shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=ceil_mode,
        output_layout=output_layout,
        out_dtype=out_dtype,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )


@pytest.mark.parametrize("input_spec", parameters["in_mem_config_tests"]["input_specs"])
@pytest.mark.parametrize("cores", parameters["in_mem_config_tests"]["cores"])
def test_max_pool2d_in_mem_config(device, input_spec, cores, tensor_map):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        ceil_mode,
    ) = input_spec

    (cores_x, cores_y) = cores

    # shard shape calculations are only accurate for height sharded tensors
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    in_nhw = in_n * in_h * in_w
    num_cores = cores_x * cores_y
    # we use tile shape to mimic tile based tensors passed from other ops like conv2d
    shard_height = math.ceil(math.ceil(in_nhw / num_cores) / 32) * 32
    shard_width = math.ceil(in_c / 32) * 32

    in_memory_config = ttnn.MemoryConfig(
        memory_layout=shard_scheme,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_x - 1, cores_y - 1))}),
            [shard_height, shard_width],
            shard_orientation,
        ),
    )

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        (kernel_h, kernel_w),
        (pad_h, pad_w),
        (stride_h, stride_w),
        (dilation_h, dilation_w),
        device,
        tensor_map,
        ttnn.bfloat16,
        in_memory_config=in_memory_config,
        ceil_mode=ceil_mode,
        nightly_skips=False,
        config_tensor_in_dram=True,
    )
