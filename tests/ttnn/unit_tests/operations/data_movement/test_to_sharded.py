# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc


def check_mem_config(tensor, expected_memory_config, is_nd_sharded):
    out_mc = tensor.memory_config()
    assert out_mc.is_sharded(), "Output tensor is not sharded"
    if is_nd_sharded:
        expected_shard_spec = expected_memory_config.nd_shard_spec
        actual_shard_spec = out_mc.nd_shard_spec
        assert (
            actual_shard_spec.shard_shape == expected_shard_spec.shard_shape
        ), f"ND Shard shape mismatch: {actual_shard_spec.shard_shape} != {expected_shard_spec.shard_shape}"
        assert actual_shard_spec.shard_distribution_strategy == expected_shard_spec.shard_distribution_strategy, (
            f"Distribution strategy mismatch: "
            f"{actual_shard_spec.shard_distribution_strategy} != {expected_shard_spec.shard_distribution_strategy}"
        )
    else:
        expected_shard_spec = expected_memory_config.shard_spec
        actual_shard_spec = out_mc.shard_spec
        assert (
            actual_shard_spec.shape == expected_shard_spec.shape
        ), f"Shard shape mismatch: {actual_shard_spec.shape} != {expected_shard_spec.shape}"

    assert (
        actual_shard_spec.grid == expected_shard_spec.grid
    ), f"Shard grid mismatch: {actual_shard_spec.grid} != {expected_shard_spec.grid}"
    assert (
        actual_shard_spec.orientation == expected_shard_spec.orientation
    ), f"Shard orientation mismatch: {actual_shard_spec.orientation} != {expected_shard_spec.orientation}"


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single shard (1 core)
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards, 1×3 grid
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across first two dims → 6 shards, 1x3 grid
        ([6, 8, 64], [2, 4, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards, 4×1 grid
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D tensor sharded across batch+channel → 6 shards, disjoint grid (2 + 4 cores)
        (
            [4, 3, 16, 32],
            [2, 1, 16, 32],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(3, 1)),
                }
            ),
        ),
        # 3-D tensor with uneven shards → 3 shards, grid with more cores than shards (6 cores)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})),
        # 3-D tensor sharded across all dims → 8 shards, 1x3 grid
        ([6, 8, 64], [3, 4, 32], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across all dims → 24 shards, 1x3 grid, uneven sharded, unaligned shard width
        ([6, 8, 64], [4, 3, 20], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        (
            [1, 1, 1024, 1],
            [1, 1, 256, 1],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1])
def test_to_sharded_rm_interleaved_to_nd_sharded(
    device, tensor_shape, shard_shape, grid, shard_orientation, buffer_type
):
    torch.manual_seed(0)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(buffer_type, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_sharded(input_tensor, sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


@pytest.mark.parametrize(
    "tensor_shape, shard_shape, grid",
    [
        # 3-D tensor, single DRAM bank
        ([1, 32, 64], [1, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        # 3-D tensor sharded across first dim → 3 shards on 3 DRAM banks
        ([6, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))})),
        # 3-D tensor sharded across first two dims → 6 shards on 6 DRAM banks
        ([6, 8, 64], [2, 4, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))})),
        # 4-D tensor sharded across batch dim → 4 shards on 4 DRAM banks
        (
            [4, 1, 32, 64],
            [1, 1, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # 4-D tensor → 6 shards on disjoint DRAM banks (banks 0-1 and 4-7)
        (
            [4, 3, 16, 32],
            [2, 1, 16, 32],
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 0)),
                }
            ),
        ),
        # 3-D tensor with uneven shards → 3 shards, more DRAM banks than shards (8 banks)
        ([5, 32, 64], [2, 32, 64], ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))})),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_sharded_rm_interleaved_to_nd_sharded_dram(device, tensor_shape, shard_shape, grid, shard_orientation):
    torch.manual_seed(0)

    num_dram_banks = device.dram_grid_size().x
    num_cores_in_grid = grid.num_cores()
    if num_cores_in_grid > num_dram_banks:
        pytest.skip(f"Test requires {num_cores_in_grid} DRAM banks but this device only has {num_dram_banks}")

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=shard_orientation)
    sharded_memory_config = ttnn.MemoryConfig(ttnn.BufferType.DRAM, nd_shard_spec)
    assert sharded_memory_config.is_sharded()

    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.to_sharded(input_tensor, sharded_memory_config)

    check_mem_config(output_tensor, sharded_memory_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)


@pytest.mark.parametrize(
    "hw, kernel, stride, pad, shard_shape, grid",
    [
        # avg_pool2d output: [1, 1, 1024, 1] with padded width 16 → 4 ND shards of [1, 1, 256, 1]
        (
            (64, 64),
            (2, 2),
            (2, 2),
            0,
            [1, 1, 256, 1],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # avg_pool2d output: [1, 1, 256, 1] with padded width > 1 → 4 ND shards of [1, 1, 64, 1]
        (
            (32, 32),
            (3, 3),
            (2, 2),
            1,
            [1, 1, 64, 1],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
    ],
)
def test_to_sharded_rm_interleaved_avg_pool2d_output_to_nd_sharded_with_input_row_major_interleaved_larger_padded_width(
    device, hw, kernel, stride, pad, shard_shape, grid
):
    """
    Runs avg_pool2d to produce a row-major interleaved tensor whose padded_shape width
    is larger than its logical_shape width, then converts to ND sharded via to_sharded.
    Validates that the sharded output data matches the avg_pool2d result.
    """
    h, w = hw
    kh, kw = kernel
    sh, sw = stride

    torch_input = torch.randn(1, 1, h, w, dtype=torch.bfloat16)
    ref = torch.nn.functional.avg_pool2d(
        torch_input, kernel_size=(kh, kw), stride=(sh, sw), padding=pad, count_include_pad=True
    )

    mem_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    x_tt = ttnn.from_torch(
        torch_input.reshape(h, w),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_cfg,
    )
    x_flat = ttnn.reshape(x_tt, [1, 1, h * w, 1], memory_config=mem_cfg)
    x_rm = ttnn.to_layout(x_flat, ttnn.ROW_MAJOR_LAYOUT, None, memory_config=None)

    y = ttnn.avg_pool2d(
        x_rm,
        1,
        h,
        w,
        1,
        [kh, kw],
        [sh, sw],
        [pad, pad],
        False,
        True,
        None,
        memory_config=mem_cfg,
        applied_shard_scheme=None,
        compute_kernel_config=None,
        reallocate_halo_output=False,
        config_tensor_in_dram=True,
    )

    assert (
        y.padded_shape[-1] > y.shape[-1]
    ), "Precondition failed: avg_pool2d output should have padded width > logical width"

    y_torch = ttnn.to_torch(y)

    print("y_torch.shape:", y.shape)
    print("y_torch.padded_shape:", y.padded_shape)

    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=ttnn.ShardOrientation.ROW_MAJOR)
    sharded_memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    y_sharded = ttnn.to_sharded(y, sharded_memory_config)

    check_mem_config(y_sharded, sharded_memory_config, is_nd_sharded=True)

    y_sharded_torch = ttnn.to_torch(y_sharded)

    print("y_sharded_torch.shape:", y_sharded.shape)
    print("y_sharded_torch.padded_shape:", y_sharded.padded_shape)

    assert_equal(y_torch, y_sharded_torch)

    ref_flat = ref.reshape(-1)
    result_flat = y_sharded_torch.reshape(-1)[: ref_flat.numel()]
    assert_with_pcc(ref_flat, result_flat, pcc=0.999)


@pytest.mark.parametrize(
    "tensor_shape, input_shard_layout, input_shard_shape, input_grid, " "output_nd_shard_shape, output_grid",
    [
        # HEIGHT_SHARDED even → ND sharded (reshard height into 3-D ND shard)
        (
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # HEIGHT_SHARDED uneven input (100 rows / 32 → 4 shards, last has 4 rows) → ND sharded
        (
            [1, 1, 100, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 50, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED even → ND sharded
        (
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED uneven input (100 cols / 32 → 4 shards, last has 4 cols) → ND sharded
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
            [1, 1, 64, 50],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # WIDTH_SHARDED uneven input with unaligned shard width (100 cols / 21 → 5 shards, last has 5 cols) → ND sharded
        (
            [1, 1, 64, 100],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 21),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 0))}),
            [1, 1, 64, 50],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK_SHARDED even (2×2 grid) → ND sharded
        (
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [1, 1, 64, 128],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # BLOCK_SHARDED uneven input (100×100 on 2×2, last row/col shards smaller) → ND sharded
        (
            [1, 1, 100, 100],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))}),
            [1, 1, 50, 100],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_sharded_legacy_sharded_to_nd_sharded(
    device,
    tensor_shape,
    input_shard_layout,
    input_shard_shape,
    input_grid,
    output_nd_shard_shape,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_shard_spec = ttnn.ShardSpec(input_grid, input_shard_shape, shard_orientation)
    input_sharded_mem_config = ttnn.MemoryConfig(input_shard_layout, ttnn.BufferType.L1, input_shard_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_sharded_mem_config,
    )

    nd_shard_spec = ttnn.NdShardSpec(output_nd_shard_shape, output_grid, orientation=shard_orientation)
    output_sharded_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, nd_shard_spec)

    output_tensor = ttnn.to_sharded(input_tensor, output_sharded_mem_config)

    check_mem_config(output_tensor, output_sharded_mem_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)

    assert_equal(torch_input, output_torch)


@pytest.mark.parametrize(
    "tensor_shape, input_nd_shard_shape, input_grid, output_nd_shard_shape, output_grid",
    [
        # ---- Even input & output, sharded across dim 0 (batch) ----
        # 3-D: [6,32,64] sharded [2,32,64]→3 shards, reshard to [3,32,64]→2 shards
        (
            [6, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [3, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Even input & output, sharded across dims 0 and 1 ----
        # 3-D: [6,8,64] sharded [2,4,64]→6 shards → reshard to [3,8,64]→2 shards
        (
            [6, 8, 64],
            [2, 4, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 8, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- 4-D sharded across batch+channel (dims 0,1), even ----
        # [4,6,16,32] sharded [1,2,16,32]→12 shards → reshard to [2,3,16,32]→4 shards
        (
            [4, 6, 16, 32],
            [1, 2, 16, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            [2, 3, 16, 32],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # ---- Uneven input shards (dim 0 doesn't divide evenly) ----
        # [5,32,64] sharded [2,32,64]→3 shards (last shard has 1 along dim0), output even [5,32,64]→1 shard
        (
            [5, 32, 64],
            [2, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [5, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        ),
        # ---- Uneven output shards (dim 0 doesn't divide evenly) ----
        # [6,32,64] sharded [6,32,64]→1 shard, reshard to [4,32,64]→2 shards (last shard has 2 along dim0)
        (
            [6, 32, 64],
            [6, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [4, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Both input and output uneven ----
        # [7,32,64] sharded [3,32,64]→3 shards (last has 1), reshard to [4,32,64]→2 shards (last has 3)
        (
            [7, 32, 64],
            [3, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))}),
            [4, 32, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Unaligned shard width on input ----
        # [6,8,50] sharded [2,4,17]→6 shards (shard width 17, not 16/32 aligned), output aligned
        (
            [6, 8, 50],
            [2, 4, 17],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 8, 50],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        ),
        # ---- Unaligned shard width on output ----
        # [6,8,64] sharded [2,4,64]→6 shards (aligned), output [3,4,21]→unaligned width 21
        (
            [6, 8, 64],
            [2, 4, 64],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 4, 21],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
        # ---- Both input and output unaligned shard widths ----
        # [6,8,50] sharded [2,4,17]→18 shards, reshard to [3,4,13]→16 shards
        (
            [6, 8, 50],
            [2, 4, 17],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 0))}),
            [3, 4, 13],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
        ),
        # ---- Uneven + unaligned combined ----
        # [5,7,50] sharded [2,3,17]→12 shards (uneven dim0, dim1; unaligned width), reshard to [3,4,25]→4 shards (uneven; width 25)
        (
            [5, 7, 50],
            [2, 3, 17],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(5, 1))}),
            [3, 4, 25],
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
        ),
    ],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_sharded_nd_sharded_to_nd_sharded(
    device,
    tensor_shape,
    input_nd_shard_shape,
    input_grid,
    output_nd_shard_shape,
    output_grid,
    shard_orientation,
):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_nd_spec = ttnn.NdShardSpec(input_nd_shard_shape, input_grid, orientation=shard_orientation)
    input_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, input_nd_spec)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
    )

    output_nd_spec = ttnn.NdShardSpec(output_nd_shard_shape, output_grid, orientation=shard_orientation)
    output_mem_config = ttnn.MemoryConfig(ttnn.BufferType.L1, output_nd_spec)

    output_tensor = ttnn.to_sharded(input_tensor, output_mem_config)

    check_mem_config(output_tensor, output_mem_config, is_nd_sharded=True)

    output_torch = ttnn.to_torch(output_tensor)
    assert_equal(torch_input, output_torch)
