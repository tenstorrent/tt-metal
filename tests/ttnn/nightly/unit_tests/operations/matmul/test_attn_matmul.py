# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
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


@pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_loops", [20])
def test_attn_matmul(num_loops, in0_dtype, in1_dtype, out_dtype, device):
    torch.manual_seed(0)

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


@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_loops", [20])
def test_attn_matmul_fp32(num_loops, in_dtype, device):
    torch.manual_seed(0)

    for input_shape_a, input_shape_b in generate_input_shapes():
        for _ in range(num_loops):
            input_tensor_a = torch.randn(input_shape_a).bfloat16()
            input_tensor_b = torch.randn(input_shape_b).bfloat16()

            tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in_dtype).to(ttnn.TILE_LAYOUT).to(device)
            tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in_dtype).to(ttnn.TILE_LAYOUT).to(device)

            compute_grid_size = device.compute_with_storage_grid_size()

            compute_kernel_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
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


@pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("out_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_loops", [20])
def test_attn_matmul_with_program_cache(num_loops, in0_dtype, in1_dtype, out_dtype, device):
    torch.manual_seed(0)
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
@pytest.mark.parametrize("num_loops", [5])
def test_group_attn_matmul(
    num_loops,
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


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("in0_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("num_loops", [5])
def test_group_attn_matmul_with_program_cache(num_loops, in0_dtype, in1_dtype, output_dtype, sharded, device):
    torch.manual_seed(0)

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
@pytest.mark.parametrize("num_loops", [5])
def test_group_attn_matmul_fp32(
    num_loops,
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

        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
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


@pytest.mark.use_module_device
@pytest.mark.parametrize(
    "batch, K,seq_len, q_heads, kv_heads",
    [
        pytest.param(32, 128, 96, 32, 4, id="b32-K128-s96-q32-kv4"),
        pytest.param(32, 64, 128, 16, 2, id="b32-K64-s128-q16-kv2"),
        pytest.param(32, 160, 128, 64, 4, id="b32-K160-s128-q64-kv4"),
        pytest.param(32, 64, 256, 32, 2, id="b32-K64-s256-q32-kv2"),
        pytest.param(32, 64, 384, 48, 4, id="b32-K64-s384-q48-kv4"),
        pytest.param(32, 96, 512, 24, 2, id="b32-K96-s512-q24-kv2"),
    ],
)
@pytest.mark.parametrize(
    "use_optional_output_tensor", [pytest.param(False, id="optout-none"), pytest.param(True, id="optout-prealloc")]
)
@pytest.mark.parametrize(
    "shard_orientation",
    [
        pytest.param(ttnn.ShardOrientation.ROW_MAJOR, id="shardorient-row"),
        pytest.param(ttnn.ShardOrientation.COL_MAJOR, id="shardorient-col"),
    ],
)
@pytest.mark.parametrize(
    "in0_sharded",
    [
        pytest.param(False, id="in0I"),
        pytest.param(True, id="in0S"),
    ],
)
@pytest.mark.parametrize(
    "in1_sharded",
    [
        pytest.param(False, id="in1I"),
        pytest.param(True, id="in1S"),
    ],
)
@pytest.mark.parametrize(
    "output_sharded",
    [
        pytest.param(False, id="outI"),
        pytest.param(True, id="outS"),
    ],
)
@pytest.mark.parametrize(
    "input_buffer_type",
    [
        pytest.param("dram", id="inbuf-dram"),
        pytest.param("l1", id="inbuf-l1"),
    ],
)
def test_group_attn_matmul_with_program_cache_exhaustive(
    batch,
    K,
    seq_len,
    q_heads,
    kv_heads,
    input_buffer_type,
    in0_sharded,
    in1_sharded,
    output_sharded,
    shard_orientation,
    use_optional_output_tensor,
    device,
):
    torch.manual_seed(42)
    full_grid = device.compute_with_storage_grid_size()
    compute_grid = ttnn.CoreCoord(full_grid.x, full_grid.y)

    in0_dtype = ttnn.bfloat16
    in1_dtype = ttnn.bfloat16
    output_dtype = ttnn.bfloat16

    num_cores_required = max(q_heads, 32)
    if compute_grid.x * compute_grid.y < num_cores_required:
        pytest.skip("compute grid too small for q_heads / minimum 32 cores")

    interleaved_input_mem = ttnn.DRAM_MEMORY_CONFIG if input_buffer_type == "dram" else ttnn.L1_MEMORY_CONFIG
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    if use_optional_output_tensor and output_sharded:
        pytest.skip("optional_output_tensor path only covered for interleaved (DRAM) output in this suite")

    if (not in0_sharded) and (not in1_sharded) and shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
        pytest.skip("orientation is unused, dupilcate test")

    q_len = 1
    TILE_SIZE = 32

    if in0_sharded:
        if (q_len * batch) % TILE_SIZE != 0 or K % TILE_SIZE != 0:
            pytest.skip("Input 0 shard not supported")
    if in1_sharded:
        if (kv_heads * K) % TILE_SIZE != 0 or seq_len % TILE_SIZE != 0:
            pytest.skip("Input 1 shard not supported")

    input_shape_a = [q_len, q_heads, batch, K]
    input_shape_b = [batch, kv_heads, K, seq_len]

    input_tensor_a = torch.randn(input_shape_a).bfloat16()
    input_tensor_b = torch.randn(input_shape_b).bfloat16()

    tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_input_mem)
    tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_input_mem)

    if in0_sharded:
        tt_padded_shape_a = tt_input_tensor_a.padded_shape
        tt_input_tensor_a = ttnn.interleaved_to_sharded(
            tt_input_tensor_a,
            compute_grid,
            [tt_padded_shape_a[0] * tt_padded_shape_a[2], tt_padded_shape_a[3]],  # [q_len * batch, K]
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if in1_sharded:
        tt_padded_shape_b = tt_input_tensor_b.padded_shape
        tt_input_tensor_b = ttnn.interleaved_to_sharded(
            tt_input_tensor_b,
            compute_grid,
            [tt_padded_shape_b[1] * tt_padded_shape_b[2], tt_padded_shape_b[3]],  # [kv_heads * K, seq_len]
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if output_sharded:
        output_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        output_mem_config = dram_interleaved

    optional_output_tensor = None
    if use_optional_output_tensor:
        out_shape = [q_len, q_heads, batch, seq_len]
        torch_out_buf = torch.zeros(out_shape, dtype=torch.bfloat16)
        optional_output_tensor = (
            ttnn.Tensor(torch_out_buf, output_dtype).to(ttnn.TILE_LAYOUT).to(device, dram_interleaved)
        )

    kwargs = dict(
        compute_with_storage_grid_size=compute_grid,
        memory_config=output_mem_config,
        dtype=output_dtype,
    )
    if optional_output_tensor is not None:
        kwargs["optional_output_tensor"] = optional_output_tensor

    tt_output_tensor_on_device = ttnn.experimental.group_attn_matmul(
        tt_input_tensor_a,
        tt_input_tensor_b,
        **kwargs,
    )

    tt_input_tensor_a.deallocate()
    tt_input_tensor_b.deallocate()

    if output_sharded:
        tt_output_tensor_on_device = ttnn.sharded_to_interleaved(tt_output_tensor_on_device, dram_interleaved)

    tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_output_tensor_on_device.deallocate()

    input_tensor_a = input_tensor_a.to(torch.float)
    input_tensor_b = torch.repeat_interleave(input_tensor_b.to(torch.float), q_heads // kv_heads, dim=1)
    golden_output_tensor = (input_tensor_a.transpose(0, 2) @ input_tensor_b).transpose(0, 2)

    allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
    assert allclose, f"FAILED: {output}"


def _attn_round_up_tokens(n: int) -> int:
    return ((max(1, int(n)) - 1) // 32 + 1) * 32


def _attn_align_up(x: int, tile: int = 32) -> int:
    return ((int(x) + tile - 1) // tile) * tile


# Covers: standard attn_matmul vs attn_matmul_from_cache (pre/post)
@pytest.mark.use_module_device
@pytest.mark.parametrize(
    "batch, K, seq_len, q_heads",  # kv_heads is always 1
    [
        pytest.param(32, 32, 64, 8, id="b32-K32-s64-q8"),
        pytest.param(32, 64, 128, 16, id="b32-K64-s128-q16"),
        pytest.param(32, 64, 256, 32, id="b32-K64-s256-q32"),
        pytest.param(32, 96, 320, 10, id="b32-K96-s320-q10"),
        pytest.param(32, 128, 96, 32, id="b32-K128-s96-q32"),
        pytest.param(32, 48, 192, 24, id="b32-K48-s192-q24"),
        pytest.param(32, 160, 128, 64, id="b32-K160-s128-q64"),
    ],
)
@pytest.mark.parametrize(
    "cache_mode, from_cache_num_tokens",
    [
        pytest.param("standard", None, id="cache-std"),
        pytest.param("from_cache_pre", 32, id="cache-pre-32"),
        pytest.param("from_cache_pre", 64, id="cache-pre-64"),
        pytest.param("from_cache_post", 32, id="cache-post-32"),
        pytest.param("from_cache_post", 64, id="cache-post-64"),
    ],
)
@pytest.mark.parametrize(
    "use_optional_output_tensor", [pytest.param(False, id="optout-none"), pytest.param(True, id="optout-prealloc")]
)
@pytest.mark.parametrize(
    "in0_sharded",
    [
        pytest.param(False, id="in0I"),
        pytest.param(True, id="in0S"),
    ],
)
@pytest.mark.parametrize(
    "in1_sharded",
    [
        pytest.param(False, id="in1I"),
        pytest.param(True, id="in1S"),
    ],
)
@pytest.mark.parametrize(
    "output_sharded",
    [
        pytest.param(False, id="outI"),
    ],
)
def test_attn_matmul_with_program_cache_exhaustive(
    batch,
    K,
    seq_len,
    q_heads,
    in0_sharded,
    in1_sharded,
    output_sharded,
    use_optional_output_tensor,
    cache_mode,
    from_cache_num_tokens,
    device,
):
    torch.manual_seed(42)

    in0_dtype = ttnn.bfloat16
    in1_dtype = ttnn.bfloat16
    output_dtype = ttnn.bfloat16
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    interleaved_input_mem = ttnn.L1_MEMORY_CONFIG
    dram_interleaved = ttnn.DRAM_MEMORY_CONFIG

    full_grid = device.compute_with_storage_grid_size()
    compute_grid = ttnn.CoreCoord(full_grid.x, full_grid.y)

    num_cores_required = max(q_heads, 32)
    if compute_grid.x * compute_grid.y < num_cores_required:
        pytest.skip("compute grid too small for q_heads / minimum 32 cores")

    if use_optional_output_tensor and output_sharded:
        pytest.skip("optional output_tensor path only covered for interleaved (DRAM) output")

    q_len = 1
    kv_heads = 1
    TILE_SIZE = 32

    if cache_mode == "standard":
        assert from_cache_num_tokens is None
    else:
        assert from_cache_num_tokens is not None

    n_round = _attn_round_up_tokens(from_cache_num_tokens) if from_cache_num_tokens is not None else None

    if cache_mode == "from_cache_pre" and seq_len < from_cache_num_tokens:
        pytest.skip("max_seq (seq_len) must be >= num_tokens for from_cache_pre")

    if in0_sharded:
        a_k = n_round if cache_mode == "from_cache_post" else K
        if (q_len * batch) % TILE_SIZE != 0 or a_k % TILE_SIZE != 0:
            pytest.skip("input 0 shard not supported for this shape")

    if in1_sharded:
        if cache_mode == "standard":
            if (kv_heads * K) % TILE_SIZE != 0 or seq_len % TILE_SIZE != 0:
                pytest.skip("input 1 shard not supported for standard layout")
        elif cache_mode == "from_cache_pre":
            if (kv_heads * seq_len) % TILE_SIZE != 0 or K % TILE_SIZE != 0:
                pytest.skip("input 1 shard not supported for from_cache_pre layout")
        elif cache_mode == "from_cache_post":
            cache_dim = _attn_align_up(max(n_round, K, seq_len))
            if cache_dim % TILE_SIZE != 0 or seq_len % TILE_SIZE != 0:
                pytest.skip("input 1 shard not supported for from_cache_post layout")

    if cache_mode == "from_cache_post":
        cache_dim_pre = _attn_align_up(max(n_round, K, seq_len))
        if from_cache_num_tokens > cache_dim_pre:
            pytest.skip("num_tokens must be <= B padded dim 2")

    if cache_mode == "standard":
        input_shape_a = [q_len, q_heads, batch, K]
        input_shape_b = [batch, kv_heads, K, seq_len]
    elif cache_mode == "from_cache_pre":
        input_shape_a = [q_len, q_heads, batch, K]
        input_shape_b = [batch, kv_heads, seq_len, K]
    else:
        cache_dim = _attn_align_up(max(n_round, K, seq_len))
        input_shape_a = [q_len, q_heads, batch, n_round]
        input_shape_b = [batch, kv_heads, cache_dim, seq_len]

    input_tensor_a = torch.randn(input_shape_a).bfloat16()
    input_tensor_b = torch.randn(input_shape_b).bfloat16()

    tt_input_tensor_a = ttnn.Tensor(input_tensor_a, in0_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_input_mem)
    tt_input_tensor_b = ttnn.Tensor(input_tensor_b, in1_dtype).to(ttnn.TILE_LAYOUT).to(device, interleaved_input_mem)

    if in0_sharded:
        tt_padded_shape_a = tt_input_tensor_a.padded_shape
        tt_input_tensor_a = ttnn.interleaved_to_sharded(
            tt_input_tensor_a,
            compute_grid,
            [tt_padded_shape_a[0] * tt_padded_shape_a[2], tt_padded_shape_a[3]],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if in1_sharded:
        tt_padded_shape_b = tt_input_tensor_b.padded_shape
        tt_input_tensor_b = ttnn.interleaved_to_sharded(
            tt_input_tensor_b,
            compute_grid,
            [tt_padded_shape_b[1] * tt_padded_shape_b[2], tt_padded_shape_b[3]],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_orientation,
        )

    if output_sharded:
        output_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        output_mem_config = dram_interleaved

    optional_output_tensor = None
    if use_optional_output_tensor:
        if cache_mode == "from_cache_pre":
            out_seq = n_round
        else:
            out_seq = seq_len
        out_shape = [q_len, q_heads, batch, out_seq]
        torch_out_buf = torch.zeros(out_shape, dtype=torch.bfloat16)
        optional_output_tensor = (
            ttnn.Tensor(torch_out_buf, output_dtype).to(ttnn.TILE_LAYOUT).to(device, output_mem_config)
        )

    kwargs = dict(
        compute_with_storage_grid_size=compute_grid,
        dtype=output_dtype,
    )
    if optional_output_tensor is not None:
        kwargs["output_tensor"] = optional_output_tensor
    else:
        kwargs["memory_config"] = output_mem_config

    if cache_mode == "standard":
        tt_output_tensor_on_device = ttnn.experimental.attn_matmul(
            tt_input_tensor_a,
            tt_input_tensor_b,
            **kwargs,
        )
    else:
        transpose_hw = cache_mode == "from_cache_pre"
        tt_output_tensor_on_device = ttnn.experimental.attn_matmul_from_cache(
            tt_input_tensor_a,
            tt_input_tensor_b,
            num_tokens=from_cache_num_tokens,
            transpose_hw=transpose_hw,
            **kwargs,
        )

    tt_input_tensor_a.deallocate()
    tt_input_tensor_b.deallocate()

    if output_sharded:
        tt_output_tensor_on_device = ttnn.sharded_to_interleaved(tt_output_tensor_on_device, dram_interleaved)

    tt_output_tensor = tt_output_tensor_on_device.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    tt_output_tensor_on_device.deallocate()

    input_tensor_a_f = input_tensor_a.to(torch.float32)
    input_tensor_b_f = input_tensor_b.to(torch.float32)

    if cache_mode == "standard":
        golden_output_tensor = (input_tensor_a_f.transpose(0, 2) @ input_tensor_b_f).transpose(0, 2)
    elif cache_mode == "from_cache_pre":
        b_part = input_tensor_b_f[:, :, :n_round, :].transpose(-1, -2)
        golden_output_tensor = torch.matmul(input_tensor_a_f.transpose(0, 2), b_part).transpose(0, 2)
    else:
        b_eff = input_tensor_b_f[:, :, :n_round, :]
        golden_output_tensor = (input_tensor_a_f.transpose(0, 2) @ b_eff).transpose(0, 2)

    allclose, output = comp_pcc(tt_output_tensor, golden_output_tensor)
    assert allclose, f"FAILED: {output}"
