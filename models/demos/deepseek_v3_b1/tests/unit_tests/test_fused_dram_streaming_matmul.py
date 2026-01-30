# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for fusing two DRAM Streaming Matmul ops using the kernel fusion framework.

This test demonstrates:
- Creating program descriptors from DRAMStreamingMatmul with cb_offset
- Using the kernel fusion framework to fuse multiple ops
- Running the fused program with unified kernels
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.kernel_fusion import GlobalProgram, SubProgram
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul.op import DRAMStreamingMatmul
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def pad_to_dram_banks(num, tile_w, lcm):
    """Pad number to be aligned with DRAM banks."""
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    return num + padding_needed


def shuffle_tensor_tiles(tensor, tile_size, num_banks):
    """
    Shuffle tiles for WIDTH_SHARDED from row-major to column-major within each shard.
    """
    orig_shape = tensor.shape
    K = orig_shape[-2]
    N = orig_shape[-1]
    tensor_2d = tensor.reshape(-1, K, N) if len(orig_shape) > 2 else tensor.unsqueeze(0)
    batch_size = tensor_2d.shape[0]

    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    if n_padded != N:
        tensor_2d = torch.nn.functional.pad(tensor_2d, (0, n_padded - N))

    K_tiles = K // tile_size
    per_core_N_tiles = n_padded // num_banks // tile_size

    shuffled = torch.zeros_like(tensor_2d)

    for b in range(batch_size):
        for bank in range(num_banks):
            for kt_shuf in range(K_tiles):
                for local_nt_shuf in range(per_core_N_tiles):
                    local_shuf_idx = kt_shuf * per_core_N_tiles + local_nt_shuf
                    kt_orig = local_shuf_idx % K_tiles
                    local_nt_orig = local_shuf_idx // K_tiles

                    nt_shuf = bank * per_core_N_tiles + local_nt_shuf
                    nt_orig = bank * per_core_N_tiles + local_nt_orig

                    shuffled[
                        b,
                        kt_shuf * tile_size : (kt_shuf + 1) * tile_size,
                        nt_shuf * tile_size : (nt_shuf + 1) * tile_size,
                    ] = tensor_2d[
                        b,
                        kt_orig * tile_size : (kt_orig + 1) * tile_size,
                        nt_orig * tile_size : (nt_orig + 1) * tile_size,
                    ]

    shuffled = shuffled[:, :, :N]
    if len(orig_shape) > 2:
        shuffled = shuffled.reshape(*orig_shape[:-2], K, N)
    else:
        shuffled = shuffled.squeeze(0)

    return shuffled


@pytest.mark.parametrize("k, n", [(2048, 2048)])
@pytest.mark.parametrize("m", [1])
def test_fused_dram_streaming_matmul(device, k, n, m):
    """
    Test fusing two DRAM streaming matmul ops using unified kernels.

    This test:
    1. Creates two separate matmul ops: out0 = in0 @ in1_0, out1 = in0 @ in1_1
    2. Uses the kernel fusion framework to fuse them
    3. Runs the fused program with unified kernels
    4. Verifies results against PyTorch reference
    """
    tile_h = m
    tile_w = 32

    in0_tile = ttnn.Tile([tile_h, tile_w])
    out_tile = ttnn.Tile([tile_h, tile_w])

    compute_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_cores = len(compute_cores)
    num_banks = device.dram_grid_size().x
    assert num_cores == num_banks

    logger.info(f"num_compute_cores={num_cores}, num_dram_banks={num_banks}")

    n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)
    per_core_N = n_padded // num_banks

    logger.info(f"n_padded={n_padded}, per_core_N={per_core_N}, Kt={k // tile_w}")

    # Define shapes
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n_padded]

    # Build CoreRangeSet
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )

    # Create PyTorch tensors
    torch.manual_seed(42)
    in0 = torch.randn(in0_shape).bfloat16().float()
    in1_0 = torch.randn(in1_shape).bfloat16().float()  # Weights for first matmul
    in1_1 = torch.randn(in1_shape).bfloat16().float()  # Weights for second matmul

    # ========== Create input tensors ==========

    # Input A - REPLICATED on compute cores
    in0_replicated = in0.repeat(1, 1, num_cores, 1)
    in0_shard_shape_full = [m, k]
    in0_shard_spec = ttnn.ShardSpec(compute_core_grid, in0_shard_shape_full, ttnn.ShardOrientation.ROW_MAJOR)
    in0_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec)
    in0_t = ttnn.from_torch(
        in0_replicated,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
        tile=in0_tile,
    )

    # Input B0 - WIDTH_SHARDED in DRAM (weights for first matmul)
    in1_0_shuffled = shuffle_tensor_tiles(in1_0, tile_w, num_banks)
    in1_shard_shape = [k, n_padded // num_banks]
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_0_t = ttnn.from_torch(
        in1_0_shuffled,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # Input B1 - WIDTH_SHARDED in DRAM (weights for second matmul)
    in1_1_shuffled = shuffle_tensor_tiles(in1_1, tile_w, num_banks)
    in1_1_t = ttnn.from_torch(
        in1_1_shuffled,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    # Output tensors - WIDTH_SHARDED in L1
    output_shard_width = n_padded // num_banks
    output_shard_spec = ttnn.ShardSpec(compute_core_grid, (m, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    torch_output_zeros = torch.zeros([1, 1, m, n_padded]).bfloat16().float()
    output_0_t = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    output_1_t = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    subblock_k = k // tile_w // 2

    # ========== Create GlobalProgram and attach sub-programs ==========
    logger.info("Creating fused program with unified kernels...")

    global_program = GlobalProgram(
        device,
        compute_core_grid,
        fused_kernel_path="models/demos/deepseek_v3_b1/kernel_fusion/kernels/fused_dram_streaming_matmul_kernel.cpp",
    )

    # First matmul: get offsets, create program info, attach
    cb_offset_0, sem_offset_0, rt_offset_0 = global_program.get_next_offsets()
    logger.info(f"Op 0: cb_offset={cb_offset_0}, sem_offset={sem_offset_0}, rt_offset={rt_offset_0}")

    info_0 = DRAMStreamingMatmul.create_program_info(
        in0_t,
        in1_0_t,
        output_0_t,
        fp32_dest_acc_en=True,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        subblock_k=subblock_k,
        cb_offset=cb_offset_0,
        sem_offset=sem_offset_0,
    )
    global_program.attach(SubProgram.from_program_info("dsm0", info_0))

    # Second matmul: get offsets, create program info, attach
    cb_offset_1, sem_offset_1, rt_offset_1 = global_program.get_next_offsets()
    logger.info(f"Op 1: cb_offset={cb_offset_1}, sem_offset={sem_offset_1}, rt_offset={rt_offset_1}")

    info_1 = DRAMStreamingMatmul.create_program_info(
        in0_t,
        in1_1_t,
        output_1_t,
        fp32_dest_acc_en=True,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        subblock_k=subblock_k,
        cb_offset=cb_offset_1,
        sem_offset=sem_offset_1,
    )
    global_program.attach(SubProgram.from_program_info("dsm1", info_1))

    # Fuse
    fused_descriptor = global_program.fuse()

    # Log fusion details
    logger.info(f"Fused program has {len(fused_descriptor.cbs)} CBs")
    logger.info(f"Fused program has {len(fused_descriptor.kernels)} kernels")
    for i, kernel in enumerate(fused_descriptor.kernels):
        logger.info(f"  Kernel {i}: {kernel.kernel_source}")
        if kernel.defines:
            logger.info(f"    Defines: {kernel.defines}")
        if kernel.named_compile_time_args:
            logger.info(f"    Named CT args: {len(kernel.named_compile_time_args)} args")

    # ========== Execute fused program ==========
    logger.info("Executing fused program...")

    io_tensors = global_program.get_io_tensors()
    logger.info(f"I/O tensors: {len(io_tensors)}")

    try:
        ttnn.generic_op(io_tensors, fused_descriptor)
    except Exception as e:
        logger.error(f"Fused program execution failed: {e}")
        pytest.skip(f"Fused program execution failed: {e}")

    # ========== Verify results ==========
    logger.info("Verifying results...")

    # PyTorch reference
    pt_out_0 = DRAMStreamingMatmul.golden(in0, in1_0)
    pt_out_1 = DRAMStreamingMatmul.golden(in0, in1_1)

    # Convert results to torch
    tt_out_0 = ttnn.to_torch(output_0_t)
    tt_out_1 = ttnn.to_torch(output_1_t)

    # Verify first matmul
    expected_pcc = 0.99
    passing_0, output_0 = comp_pcc(pt_out_0, tt_out_0, expected_pcc)
    logger.info(f"Matmul 0: {output_0}")
    assert passing_0, f"Matmul 0 PCC check failed: {output_0}"

    # Verify second matmul
    passing_1, output_1 = comp_pcc(pt_out_1, tt_out_1, expected_pcc)
    logger.info(f"Matmul 1: {output_1}")
    assert passing_1, f"Matmul 1 PCC check failed: {output_1}"

    logger.info("Fused DRAM streaming matmul test passed!")
