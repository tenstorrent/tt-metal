# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

import ttnn
from models.utility_functions import comp_pcc
from models.utility_functions import is_grayskull, skip_for_blackhole
import ttnn


def generate_input_shapes():
    batch_size = 64
    kv_heads = 1
    q_len = 1
    q_heads = 10
    seq_len = 32
    K = 96
    yield [q_len, q_heads, batch_size, K], [batch_size, kv_heads, K, seq_len]

    batch_size = 32
    kv_heads = 1
    q_len = 1
    q_heads = 71
    seq_len = 128
    K = 64
    yield [q_len, q_heads, batch_size, K], [batch_size, kv_heads, K, seq_len]


@skip_for_blackhole("Hanging on BH, see #12349")
@pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 20), (False, 1)),
)
def test_attn_matmul(num_loops, enable_async, in0_dtype, in1_dtype, out_dtype, device):
    torch.manual_seed(0)
    device.enable_async(enable_async)

    for input_shape_a, input_shape_b in generate_input_shapes():
        for _ in range(num_loops):
            input_tensor_a = torch.randn(input_shape_a).bfloat16()
            input_tensor_b = torch.randn(input_shape_b).bfloat16()
            tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT)
            tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT)
            # Test python syntax in async mode -> tensor handle for inputs should get properly updated when sending to device
            tt_input_tensor_a = tt_input_tensor_a.to(device)
            tt_input_tensor_b = tt_input_tensor_b.to(device)
            compute_grid_size = device.compute_with_storage_grid_size()
            tt_output_tensor_on_device = ttnn.experimental.attn_matmul(
                tt_input_tensor_a,
                tt_input_tensor_b,
                compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=out_dtype,
            )
            tt_input_tensor_a.deallocate()
            tt_input_tensor_b.deallocate()
            tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            tt_output_tensor_on_device.deallocate()
            golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

            allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
            assert allclose, f"FAILED: {output}"

    device.enable_async(False)


@skip_for_blackhole("Hanging on BH, see #12349")
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 20), (False, 1)),
)
def test_attn_matmul_fp32(num_loops, enable_async, in_dtype, device):
    torch.manual_seed(0)
    device.enable_async(enable_async)

    for input_shape_a, input_shape_b in generate_input_shapes():
        for _ in range(num_loops):
            input_tensor_a = torch.randn(input_shape_a).bfloat16()
            input_tensor_b = torch.randn(input_shape_b).bfloat16()

            tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in_dtype).to(ttnn.TILE_LAYOUT).to(device)
            tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in_dtype).to(ttnn.TILE_LAYOUT).to(device)

            compute_grid_size = device.compute_with_storage_grid_size()

            compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            tt_output_tensor_on_device = ttnn.experimental.attn_matmul(
                tt_input_tensor_a,
                tt_input_tensor_b,
                compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=in_dtype,
                compute_kernel_config=compute_kernel_config,
            )
            tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

            golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

            allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
            assert allclose, f"FAILED: {output}"

    device.enable_async(False)


@skip_for_blackhole("Hanging on BH, see #12349")
@pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 20), (False, 1)),
)
def test_attn_matmul_with_program_cache(
    num_loops, enable_async, in0_dtype, in1_dtype, out_dtype, device, use_program_cache
):
    torch.manual_seed(0)
    device.enable_async(enable_async)
    for input_shape_a, input_shape_b in generate_input_shapes():
        for _ in range(num_loops):
            input_tensor_a = torch.randn(input_shape_a).bfloat16()
            input_tensor_b = torch.randn(input_shape_b).bfloat16()

            tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(device)
            tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(device)

            compute_grid_size = device.compute_with_storage_grid_size()

            tt_output_tensor_on_device = ttnn.experimental.attn_matmul(
                tt_input_tensor_a,
                tt_input_tensor_b,
                compute_with_storage_grid_size=ttnn.CoreCoord(compute_grid_size.x, compute_grid_size.y),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=out_dtype,
            )
            tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

            golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

            allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
            assert allclose, f"FAILED: {output}"
    device.enable_async(False)


@skip_for_blackhole("Hanging on BH, see #12349")
@pytest.mark.parametrize(
    "shard_orientation",
    (ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR),
)
@pytest.mark.parametrize(
    "output_sharded",
    (False, True),
)
@pytest.mark.parametrize(
    "in1_sharded",
    (False, True),
)
@pytest.mark.parametrize(
    "in0_sharded",
    (False, True),
)
@pytest.mark.parametrize(
    "batch, K, seq_len, q_heads, kv_heads",
    (
        (32, 64, 512 + 96, 32, 2),
        (32, 1024 + 32, 64, 32, 2),
        (32, 64, 128, 16, 1),
    ),
)
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 5), (False, 1)),
)
def test_group_attn_matmul(
    num_loops,
    enable_async,
    batch,
    K,
    seq_len,
    q_heads,
    kv_heads,
    in0_sharded,
    in1_sharded,
    output_sharded,
    shard_orientation,
    device,
):
    torch.manual_seed(0)

    device.enable_async(enable_async)

    compute_grid_size = device.compute_with_storage_grid_size()

    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG

    # NOTE: Mixed precision is supported as well; but might not have enough space for larger seq_len with BFLOAT16
    in0_dtype = ttnn.bfloat8_b
    in1_dtype = ttnn.bfloat8_b
    output_dtype = ttnn.bfloat8_b

    q_len = 1
    input_shape_a = [q_len, q_heads, batch, K]
    input_shape_b = [batch, kv_heads, K, seq_len]
    for _ in range(num_loops):
        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = (
            ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
        )
        tt_input_tensor_b = (
            ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
        )

        if in0_sharded:
            tt_input_tensor_a = ttnn.interleaved_to_sharded(
                tt_input_tensor_a,
                compute_grid_size,
                [q_len * batch, K],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_orientation,
            )

        if in1_sharded:
            tt_input_tensor_b = ttnn.interleaved_to_sharded(
                tt_input_tensor_b,
                compute_grid_size,
                [kv_heads * K, seq_len],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_orientation,
            )

        if output_sharded:
            output_mem_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
            )
        else:
            output_mem_config = interleaved_mem_config

        tt_output_tensor_on_device = ttnn.experimental.group_attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=compute_grid_size,
            memory_config=output_mem_config,
            dtype=output_dtype,
        )

        tt_input_tensor_a.deallocate()
        tt_input_tensor_b.deallocate()

        if output_sharded:
            tt_output_tensor_on_device = ttnn.sharded_to_interleaved(tt_output_tensor_on_device, interleaved_mem_config)

        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        tt_output_tensor_on_device.deallocate()
        input_tensor_a = input_tensor_a.to(torch.float)
        input_tensor_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
        golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"

    device.enable_async(False)


@skip_for_blackhole("Hanging on BH, see #12349")
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 5), (False, 1)),
)
def test_group_attn_matmul_with_program_cache(
    num_loops, enable_async, in0_dtype, in1_dtype, output_dtype, sharded, device, use_program_cache
):
    torch.manual_seed(0)

    device.enable_async(enable_async)

    compute_grid_size = device.compute_with_storage_grid_size()

    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG

    shard_orientation = ttnn.ShardOrientation.COL_MAJOR  # Only used if sharded

    q_len = 1
    batch = 32
    num_cache_entries = 0  # Only track cache entries of group_attn_matmul
    # NOTE: Program is cached on out_subblock_w as well, so only seq_len >= 256 (out_subblock_w = 8) will share cache
    # For seq_len < = 256, recompile at worst 8 times.
    for K, seq_len, q_heads, kv_heads in ((96, 512 + 64, 10, 2), (64, 256, 50, 5)):
        for _ in range(num_loops):
            input_shape_a = [q_len, q_heads, batch, K]
            input_shape_b = [batch, kv_heads, K, seq_len]

            input_tensor_a = torch.randn(input_shape_a).bfloat16()
            input_tensor_b = torch.randn(input_shape_b).bfloat16()

            tt_input_tensor_a = (
                ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
            )
            tt_input_tensor_b = (
                ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
            )

            if sharded:
                tt_input_tensor_a = ttnn.interleaved_to_sharded(
                    tt_input_tensor_a,
                    compute_grid_size,
                    [q_len * batch, K],
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    shard_orientation,
                )

                tt_input_tensor_b = ttnn.interleaved_to_sharded(
                    tt_input_tensor_b,
                    compute_grid_size,
                    [kv_heads * K, seq_len],
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    shard_orientation,
                )

                output_mem_config = ttnn.MemoryConfig(
                    memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    buffer_type=ttnn.BufferType.L1,
                )
            else:
                output_mem_config = interleaved_mem_config

            num_cache_entries_start = device.num_program_cache_entries()
            tt_output_tensor_on_device = ttnn.experimental.group_attn_matmul(
                tt_input_tensor_a,
                tt_input_tensor_b,
                compute_with_storage_grid_size=compute_grid_size,
                memory_config=output_mem_config,
                dtype=output_dtype,
            )
            num_cache_entries += device.num_program_cache_entries() - num_cache_entries_start

            if sharded:
                tt_output_tensor_on_device = ttnn.sharded_to_interleaved(
                    tt_output_tensor_on_device, interleaved_mem_config
                )

            tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

            input_tensor_a = input_tensor_a.to(torch.float)
            input_tensor_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
            golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

            allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
            assert allclose, f"FAILED: {output}"

    assert num_cache_entries == 1

    device.enable_async(False)


@skip_for_blackhole("Hanging on BH, see #12349")
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize(
    "shard_orientation",
    (ttnn.ShardOrientation.ROW_MAJOR,),
)
@pytest.mark.parametrize(
    "output_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "in1_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "in0_sharded",
    (True,),
)
@pytest.mark.parametrize(
    "batch, K, seq_len, q_heads, kv_heads",
    (
        (32, 64, 512 + 96, 32, 2),
        (32, 64 + 32, 64, 32, 2),
        (32, 32, 32, 2, 1),
    ),
)
@pytest.mark.parametrize(
    "enable_async, num_loops",
    ((True, 5), (False, 1)),
)
def test_group_attn_matmul_fp32(
    num_loops,
    enable_async,
    batch,
    K,
    seq_len,
    q_heads,
    kv_heads,
    in0_sharded,
    in1_sharded,
    output_sharded,
    shard_orientation,
    in_dtype,
    device,
):
    torch.manual_seed(0)

    device.enable_async(enable_async)

    compute_grid_size = device.compute_with_storage_grid_size()

    interleaved_mem_config = ttnn.DRAM_MEMORY_CONFIG

    # NOTE: Mixed precision is supported as well; but might not have enough space for larger seq_len with BFLOAT16
    in0_dtype = in_dtype
    in1_dtype = in_dtype
    output_dtype = in_dtype

    q_len = 1
    input_shape_a = [q_len, q_heads, batch, K]
    input_shape_b = [batch, kv_heads, K, seq_len]
    for _ in range(num_loops):
        input_tensor_a = torch.randn(input_shape_a).bfloat16()
        input_tensor_b = torch.randn(input_shape_b).bfloat16()

        tt_input_tensor_a = (
            ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
        )
        tt_input_tensor_b = (
            ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_mem_config)
        )

        if in0_sharded:
            tt_input_tensor_a = ttnn.interleaved_to_sharded(
                tt_input_tensor_a,
                compute_grid_size,
                [q_len * batch, K],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_orientation,
            )

        if in1_sharded:
            tt_input_tensor_b = ttnn.interleaved_to_sharded(
                tt_input_tensor_b,
                compute_grid_size,
                [kv_heads * K, seq_len],
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_orientation,
            )

        if output_sharded:
            output_mem_config = ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
            )
        else:
            output_mem_config = interleaved_mem_config

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        tt_output_tensor_on_device = ttnn.experimental.group_attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            compute_with_storage_grid_size=compute_grid_size,
            memory_config=output_mem_config,
            dtype=output_dtype,
            compute_kernel_config=compute_kernel_config,
        )
        if output_sharded:
            tt_output_tensor_on_device = ttnn.sharded_to_interleaved(tt_output_tensor_on_device, interleaved_mem_config)

        tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

        input_tensor_a = input_tensor_a.to(torch.float)
        input_tensor_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
        golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

        allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
        assert allclose, f"FAILED: {output}"
    device.enable_async(False)
