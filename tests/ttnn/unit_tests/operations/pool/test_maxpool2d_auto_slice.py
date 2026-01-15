# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.ttnn.nightly.unit_tests.operations.pool.test_maxpool2d import run_max_pool2d


# Test automatic DRAM slicing (no manual num_slices)
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED
SliceWidth = ttnn.Op2DDRAMSliceWidth
SliceHeight = ttnn.Op2DDRAMSliceHeight

parameters = {
    "auto_dram_slice_tests": {
        "in_specs": [[ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT], [ttnn.bfloat8_b, ttnn.TILE_LAYOUT]],
        "input_specs": [
            # Testing automatic num_slices determination for DRAM slicing
            # [in_n, in_c, in_h, in_w, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, ceil_mode, shard_layout, slice_type]
            # NOTE: slice_type=None means fully automatic (framework chooses direction AND num_slices)
            #       slice_type=WIDTH/HEIGHT means semi-automatic (framework only determines num_slices)
            #       shard_layout specifies how L1 slices are sharded (HS/BS/WS)
            [1, 128, 1024, 1024, 2, 2, 2, 2, 0, 0, 1, 1, False, HS, None],  # Fully automatic
            [1, 480, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, False, BS, SliceWidth],  # Auto num_slices, width slice
            [1, 32768, 32, 32, 2, 2, 1, 1, 0, 0, 1, 1, False, WS, SliceHeight],  # Auto num_slices, height slice
            [1, 128, 1024, 1024, 2, 2, 2, 2, 0, 0, 1, 1, True, HS, None],  # Fully automatic, ceil_mode
            [1, 480, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, True, BS, SliceWidth],  # Auto num_slices, width slice, ceil_mode
            [1, 256, 81, 81, 2, 2, 2, 2, 0, 0, 1, 1, True, HS, SliceHeight],  # Auto num_slices, height slice, ceil_mode
            [1, 256, 64, 1024, 64, 1, 1, 1, 0, 0, 1, 1, False, BS, None],  # Fully automatic
        ],
    },
}


@pytest.mark.parametrize("input_spec", parameters["auto_dram_slice_tests"]["input_specs"])
@pytest.mark.parametrize("in_specs", parameters["auto_dram_slice_tests"]["in_specs"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_max_pool2d_auto_dram_slice(device, in_specs, input_spec):
    """Test Pool2D with automatic DRAM slicing - framework determines num_slices"""
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
        shard_scheme,
        slice_type,
    ) = input_spec
    [in_dtype, output_layout] = in_specs

    # Now Pool2D matches Conv2D behavior:
    # - dram_slice_config=None → Automatic slicing (chooses direction AND num_slices)
    # - dram_slice_config=Op2DSliceConfig(num_slices=0, slice_type=...) → Semi-automatic (only num_slices)
    if slice_type is None:
        dram_slice_config = None  # Fully automatic
        slice_desc = "Fully automatic (direction + num_slices)"
    else:
        dram_slice_config = ttnn.Op2DSliceConfig(num_slices=0, slice_type=slice_type)
        slice_desc = f"Semi-automatic (direction={slice_type}, num_slices=auto)"

    print(f"\n{'='*80}")
    print(f"Testing automatic DRAM slicing for Pool2D:")
    print(f"  Input: {in_n}x{in_c}x{in_h}x{in_w}")
    print(f"  Kernel: {kernel_h}x{kernel_w}, Stride: {stride_h}x{stride_w}")
    print(f"  Mode: {slice_desc}")
    print(f"{'='*80}\n")

    torch_tensor_map = {}

    # For DRAM slicing tests:
    # - Input must be in DRAM (since L1 is constrained)
    # - shard_scheme specifies how L1 slices are sharded during DRAM slicing
    in_memory_config = ttnn.DRAM_MEMORY_CONFIG

    run_max_pool2d(
        [in_n, in_c, in_h, in_w],
        [kernel_h, kernel_w],
        [pad_h, pad_w],
        [stride_h, stride_w],
        [dilation_h, dilation_w],
        device,
        torch_tensor_map,
        in_dtype,
        in_memory_config=in_memory_config,
        shard_scheme=shard_scheme,
        ceil_mode=ceil_mode,
        nightly_skips=False,
        dram_slice_config=dram_slice_config,
        output_layout=output_layout,
    )
