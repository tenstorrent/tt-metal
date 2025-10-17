# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest

from tests.ttnn.utils_for_testing import assert_with_pcc

try:
    pass

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def calculate_per_core_dims(n, h, w, in_ch, out_ch, core_grid, sharding_strategy):
    nhw = n * h * w
    if sharding_strategy == "height":
        per_core_M = (nhw // (core_grid.y * core_grid.x) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        per_core_N = (out_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    else:
        per_core_M = (nhw // core_grid.y + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
        per_core_N = (out_ch // core_grid.x + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    return per_core_M, per_core_N


def calculate_shard_dims(per_core_M, per_core_N, in_ch, out_ch, core_grid, sharding_strategy):
    if sharding_strategy == "height":
        shard_height = per_core_M * ttnn.TILE_SIZE
        in_shard_width = (in_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        out_shard_width = (out_ch + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    else:
        shard_height = per_core_M * ttnn.TILE_SIZE
        in_shard_width = (in_ch // core_grid.x + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
        out_shard_width = per_core_N * ttnn.TILE_SIZE
    return shard_height, in_shard_width, out_shard_width


def get_matmul_config(core_grid, in0_block_w, out_subblock, per_core_M, per_core_N, out_block, sharding_strategy):
    if sharding_strategy == "height":
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock[0],
            out_subblock_w=out_subblock[1],
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            out_block_h=out_block[0],
            out_block_w=out_block[1],
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            mcast_in0=False,
        )
    else:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(core_grid.x, core_grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock[0],
            out_subblock_w=out_subblock[1],
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            out_block_h=out_block[0],
            out_block_w=out_block[1],
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            transpose_mcast=False,
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 12 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, in0_block_w, out_subblock, sharding_strategy, num_slices",
    [
        (1, 1792, 256, 1, 25344, 4, (3, 2), "block", 11),
    ],
)
@pytest.mark.parametrize(
    "core_grid",
    [
        ttnn.CoreGrid(y=4, x=5),
    ],
)
def test_sliced_ortho_feats(
    device, n, in_ch, out_ch, h, w, in0_block_w, out_subblock, sharding_strategy, num_slices, core_grid
):
    input_dtype = ttnn.bfloat16
    weights_bias_dtype = ttnn.bfloat8_b
    output_dtype = ttnn.bfloat8_b

    torch_top_left = torch.randn([n, h, w, in_ch], dtype=torch.float32)
    torch_top_right = torch.randn([n, h, w, in_ch], dtype=torch.float32)
    torch_bottom_left = torch.randn([n, h, w, in_ch], dtype=torch.float32)
    torch_bottom_right = torch.randn([n, h, w, in_ch], dtype=torch.float32)

    torch_area = torch.randn([n, h, w, in_ch], dtype=torch.float32)
    torch_input_tensor = (torch_top_left - torch_top_right + torch_bottom_right - torch_bottom_left) * torch_area

    torch_weight_tensor = torch.randn([n, 1, out_ch, in_ch], dtype=torch.float32)
    torch_bias_tensor = torch.randn([n, 1, 1, out_ch], dtype=torch.float32)

    torch_out_tensor = torch.nn.functional.linear(
        torch_input_tensor[0, 0, :, :],
        torch_weight_tensor[0, 0, :, :],
        bias=torch_bias_tensor[0, 0, 0, :],
    )
    torch_out_tensor = torch.nn.functional.relu(torch_out_tensor)

    top_left = ttnn.from_torch(
        torch_top_left, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    top_right = ttnn.from_torch(
        torch_top_right, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bottom_left = ttnn.from_torch(
        torch_bottom_left, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bottom_right = ttnn.from_torch(
        torch_bottom_right, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    area = ttnn.from_torch(
        torch_area, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    tt_weight_tensor = ttnn.from_torch(
        torch.permute(torch_weight_tensor, (0, 1, 3, 2)),
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_bias_tensor = ttnn.from_torch(
        torch_bias_tensor,
        weights_bias_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    w_sliced = w // num_slices

    per_core_M, per_core_N = calculate_per_core_dims(n, h, w_sliced, in_ch, out_ch, core_grid, sharding_strategy)
    shard_height, in_shard_width, out_shard_width = calculate_shard_dims(
        per_core_M, per_core_N, in_ch, out_ch, core_grid, sharding_strategy
    )

    out_block = (per_core_M, per_core_N)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )

    matmul_config = get_matmul_config(
        core_grid, in0_block_w, out_subblock, per_core_M, per_core_N, out_block, sharding_strategy
    )
    shard_strategy = ttnn.ShardStrategy.HEIGHT if sharding_strategy == "height" else ttnn.ShardStrategy.BLOCK

    output_mem_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid,
        shard_strategy,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    print(f"per_core_M: {per_core_M}, per_core_N: {per_core_N}")
    print(f"shard_height: {shard_height}, in_shard_width: {in_shard_width}, out_shard_width: {out_shard_width}")

    out_initial = torch.randn([n, h, w, out_ch], dtype=torch.float32)
    out_tt_tensor = ttnn.from_torch(
        out_initial, input_dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    for i in range(num_slices):
        vox_feats_slice = ttnn.interleaved_to_sharded_partial(
            top_left,
            (core_grid.x, core_grid.y),
            [shard_height, in_shard_width],
            num_slices,
            i,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        top_right_slice = ttnn.interleaved_to_sharded_partial(
            top_right,
            (core_grid.x, core_grid.y),
            [shard_height, in_shard_width],
            num_slices,
            i,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        vox_feats_slice = vox_feats_slice - top_right_slice
        ttnn.deallocate(top_right_slice)

        bottom_right_slice = ttnn.interleaved_to_sharded_partial(
            bottom_right,
            (core_grid.x, core_grid.y),
            [shard_height, in_shard_width],
            num_slices,
            i,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        vox_feats_slice = vox_feats_slice + bottom_right_slice
        ttnn.deallocate(bottom_right_slice)

        bottom_left_slice = ttnn.interleaved_to_sharded_partial(
            bottom_left,
            (core_grid.x, core_grid.y),
            [shard_height, in_shard_width],
            num_slices,
            i,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        vox_feats_slice = vox_feats_slice - bottom_left_slice
        ttnn.deallocate(bottom_left_slice)

        area_slice = ttnn.interleaved_to_sharded_partial(
            area,
            (core_grid.x, core_grid.y),
            [shard_height, in_shard_width],
            num_slices,
            i,
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )

        vox_feats_slice = vox_feats_slice * area_slice
        ttnn.deallocate(area_slice)

        vox_feats_slice = ttnn.linear(
            vox_feats_slice,
            tt_weight_tensor,
            bias=tt_bias_tensor,
            program_config=matmul_config,
            memory_config=output_mem_config,
            dtype=output_dtype,
            compute_kernel_config=compute_config,
        )

        ttnn.sharded_to_interleaved_partial(
            vox_feats_slice, out_tt_tensor, num_slices, i, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tt_output_tensor = ttnn.from_device(out_tt_tensor)
    torch_output_tensor = ttnn.to_torch(tt_output_tensor)
    assert_with_pcc(torch_out_tensor, torch_output_tensor[0, 0, :, :], pcc=0.99)
