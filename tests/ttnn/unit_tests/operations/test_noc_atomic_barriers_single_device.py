# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_no_unflushed_noc_atomics


pytestmark = pytest.mark.skipif(
    os.environ.get("TT_METAL_NOC_DEBUG_DUMP") != "1",
    reason="Set TT_METAL_NOC_DEBUG_DUMP=1 before the test process starts",
)


def _require_grid(device, x, y):
    grid = device.compute_with_storage_grid_size()
    if grid.x < x or grid.y < y:
        pytest.skip(f"This test needs a {x}x{y} compute grid")


def test_block_sharded_2d_matmul_flushes_receiver_atomics(device):
    _require_grid(device, 2, 2)
    torch.manual_seed(0)

    in0 = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    in1 = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    in0_memory_config = ttnn.create_sharded_memory_config(
        in0.shape,
        core_grid=ttnn.CoreGrid(x=2, y=2),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_device = ttnn.from_torch(
        in0,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    in1_device = ttnn.from_torch(
        in1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(2, 2),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fused_activation=None,
    )
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
    )

    with assert_no_unflushed_noc_atomics(device, min_atomic_events=2):
        ttnn.matmul(
            in0_device,
            in1_device,
            program_config=program_config,
            memory_config=output_memory_config,
            dtype=ttnn.bfloat16,
        )


@pytest.mark.parametrize("operation_name", ["layer_norm", "rms_norm"])
def test_sharded_norm_flushes_receiver_atomics(device, operation_name):
    _require_grid(device, 1, 2)
    torch.manual_seed(1)

    input_tensor = ttnn.from_torch(
        torch.randn((1, 1, 32, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid_size = [1, 2]
    sharded_input = ttnn.interleaved_to_sharded(
        input_tensor,
        grid_size,
        [32, 32],
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.COL_MAJOR,
    )
    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=grid_size,
        subblock_w=1,
        block_h=1,
        block_w=1,
        inplace=False,
        use_welford=False,
    )
    output_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        sharded_input.memory_config().shard_spec,
    )
    operation = getattr(ttnn, operation_name)

    with assert_no_unflushed_noc_atomics(device, min_atomic_events=1):
        operation(
            sharded_input,
            epsilon=1e-5,
            memory_config=output_memory_config,
            program_config=program_config,
        )


def test_noncausal_sdpa_flushes_receiver_atomics(device):
    _require_grid(device, 2, 1)
    torch.manual_seed(2)

    def make_input():
        return ttnn.from_torch(
            torch.randn((1, 1, 64, 32), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    query = make_input()
    key = make_input()
    value = make_input()
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 1),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=True,
    )

    with assert_no_unflushed_noc_atomics(device, min_atomic_events=2):
        ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            program_config=program_config,
        )


@pytest.mark.parametrize(
    "shape, layout",
    [
        ((1, 1, 160, 32), ttnn.TILE_LAYOUT),
        ((1, 1, 64, 24), ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["tile_layout", "row_major_layout"],
)
def test_overlapping_move_flushes_worker_atomics(device, shape, layout):
    _require_grid(device, 2, 2)
    torch.manual_seed(3)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    input_data = torch.randn(shape)

    dummy = ttnn.Tensor(torch.randn((1, 1, 1, 8)), ttnn.bfloat16).to(ttnn.ROW_MAJOR_LAYOUT).to(
        device, memory_config
    )
    input_tensor = ttnn.Tensor(input_data, ttnn.bfloat16).to(layout).to(device, memory_config)
    dummy.deallocate()

    with assert_no_unflushed_noc_atomics(device, min_atomic_events=1):
        output = ttnn.move(input_tensor, memory_config=memory_config)

    actual = output.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    torch.testing.assert_close(actual, input_data.to(torch.bfloat16), rtol=0, atol=0)


CONV_CASES = [
    pytest.param(
        2, 16, 16, 256, 256, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 32, (2, 2), (1, 2, 2, 3), id="height_sharded"
    ),
    pytest.param(
        2, 384, 353, 8, 8, ttnn.TensorMemoryLayout.WIDTH_SHARDED, None, (2, 2), (1, 2, 2, 3), id="width_sharded"
    ),
    pytest.param(1, 8, 64, 8, 8, ttnn.TensorMemoryLayout.BLOCK_SHARDED, None, (1, 1), (1, 1), id="block_sharded"),
]


@pytest.mark.parametrize(
    "batch_size, input_channels, output_channels, input_height, input_width, shard_layout, act_block_h, stride, padding",
    CONV_CASES,
)
def test_conv2d_flushes_receiver_atomics(
    device,
    batch_size,
    input_channels,
    output_channels,
    input_height,
    input_width,
    shard_layout,
    act_block_h,
    stride,
    padding,
):
    _require_grid(device, 4, 4)
    torch.manual_seed(4)

    input_tensor = ttnn.from_torch(
        torch.randn((batch_size, input_height, input_width, input_channels), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    weight_tensor = ttnn.from_torch(
        torch.randn((output_channels, input_channels, 3, 3), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
    )
    bias_tensor = ttnn.from_torch(
        torch.randn((1, 1, 1, output_channels), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
    )
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        config_tensors_in_dram=True,
        shard_layout=shard_layout,
        output_layout=ttnn.TILE_LAYOUT,
    )
    if act_block_h is not None:
        conv_config.act_block_h_override = act_block_h
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    with assert_no_unflushed_noc_atomics(device, min_atomic_events=1):
        ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=weight_tensor,
            in_channels=input_channels,
            out_channels=output_channels,
            device=device,
            bias_tensor=bias_tensor,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )


def _make_interleaved_group_norm_inputs(device):
    input_tensor = torch.randn((1, 128, 1, 512), dtype=torch.bfloat16)
    weight = torch.randn((128,), dtype=torch.bfloat16)
    bias = torch.randn((128,), dtype=torch.bfloat16)
    input_tensor = input_tensor.permute(0, 2, 3, 1).reshape(1, 1, 512, 128)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor = ttnn.tilize_with_zero_padding(input_tensor, use_multicore=True)
    grid = ttnn.CoreGrid(x=4, y=4)
    (gamma, beta), input_mask = ttnn.dram_group_norm_params_from_torch(
        [weight, bias],
        128,
        32,
        device,
        core_grid=grid,
        return_mask=True,
    )
    return input_tensor, gamma, beta, input_mask, grid


@pytest.mark.parametrize("use_welford", [False, True], ids=["tile_reduction", "two_pass"])
def test_interleaved_group_norm_flushes_receiver_atomics(device, use_welford):
    _require_grid(device, 4, 4)
    torch.manual_seed(5)
    input_tensor, gamma, beta, input_mask, grid = _make_interleaved_group_norm_inputs(device)

    with assert_no_unflushed_noc_atomics(device, min_atomic_events=1):
        ttnn.group_norm(
            input_tensor,
            num_groups=32,
            input_mask=input_mask,
            weight=gamma,
            bias=beta,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=grid,
            inplace=False,
            num_out_blocks=2,
            use_welford=use_welford,
        )


def _make_sharded_group_norm_inputs(device):
    input_tensor = torch.randn((1, 128, 1, 512), dtype=torch.bfloat16)
    weight = torch.randn((128,), dtype=torch.bfloat16)
    bias = torch.randn((128,), dtype=torch.bfloat16)
    input_tensor = input_tensor.permute(0, 2, 3, 1).reshape(1, 1, 512, 128)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    grid = ttnn.CoreGrid(x=4, y=1)
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
    )
    shard_spec = ttnn.ShardSpec(shard_grid, (128, 128), ttnn.ShardOrientation.COL_MAJOR)
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config)

    input_mask = ttnn.create_group_norm_input_mask(128, 16, grid.y, ttnn.bfloat8_b)
    input_mask = ttnn.to_device(input_mask, device)
    gamma = ttnn.create_group_norm_weight_bias_rm(weight, 128, grid.y)
    beta = ttnn.create_group_norm_weight_bias_rm(bias, 128, grid.y)
    gamma = ttnn.from_torch(
        gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta = ttnn.from_torch(
        beta,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return input_tensor, gamma, beta, input_mask, grid, memory_config


@pytest.mark.parametrize("use_welford", [False, True], ids=["tile_reduction", "two_pass"])
def test_sharded_group_norm_flushes_receiver_atomics(device, use_welford):
    _require_grid(device, 4, 1)
    torch.manual_seed(6)
    input_tensor, gamma, beta, input_mask, grid, memory_config = _make_sharded_group_norm_inputs(device)

    with assert_no_unflushed_noc_atomics(device, min_atomic_events=1):
        ttnn.group_norm(
            input_tensor,
            num_groups=16,
            input_mask=input_mask,
            weight=gamma,
            bias=beta,
            memory_config=memory_config,
            core_grid=grid,
            inplace=False,
            use_welford=use_welford,
        )
