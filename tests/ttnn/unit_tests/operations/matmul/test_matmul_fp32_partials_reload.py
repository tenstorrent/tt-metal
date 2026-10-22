# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""fp32 partials must be reloaded without loss in every matmul factory that spills them.

With fp32_dest_acc_en the partials of each K block are packed to the fp32 intermediate buffer and reloaded for the
next block. The reuse, DRAM-sharded and batched DRAM-sharded factories reloaded them through SrcA, which truncates
to TF32, so the error grew with the number of K blocks (about 300x the mcast factories' error at 128 blocks).
in0_block_w=1 maximizes the number of reloads. Inputs are bfloat16-representable, so with HiFi4 the products are
exact and the remaining error is fp32 accumulation, the same as in the mcast_1d/2d factories (rel. error ~8e-5).
"""

import math

import pytest
import torch

import ttnn

# Relative max error vs. an fp64 reference. The lossless factories reach ~8e-5 here; the TF32 reload gives
# ~8e-4 with packer_l1_acc and 1e-2 to 2.5e-2 without it.
REL_TOL = 2e-4


def _compute_config(device, packer_l1_acc):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def _rel_max_err(output, reference):
    return ((output - reference).abs().max() / reference.abs().max()).item()


def _bf16_rand(*shape):
    return torch.rand(*shape).to(torch.bfloat16).to(torch.float32)


@pytest.mark.parametrize("packer_l1_acc", [False, True])
def test_matmul_reuse_fp32_partials_reload(device, packer_l1_acc):
    torch.manual_seed(0)
    batch, m, k, n = 8, 256, 4096, 256
    a, b = _bf16_rand(batch, 1, m, k), _bf16_rand(batch, 1, k, n)
    grid = device.compute_with_storage_grid_size()
    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=4,
        per_core_M=m // 32,
        per_core_N=n // 32,
    )
    a_tt = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(
        a_tt,
        b_tt,
        program_config=program_config,
        dtype=ttnn.float32,
        compute_kernel_config=_compute_config(device, packer_l1_acc),
    )
    err = _rel_max_err(ttnn.to_torch(output).double(), torch.matmul(a.double(), b.double()))
    assert err < REL_TOL, f"relative max error {err:.3e}"


@pytest.mark.parametrize("packer_l1_acc", [False, True])
def test_matmul_dram_sharded_fp32_partials_reload(device, packer_l1_acc):
    torch.manual_seed(0)
    num_banks = device.dram_grid_size().x
    num_cores = 8  # in0 width-shard grid (8, 1); N must split evenly across both the cores and the DRAM banks
    m, k, n = 32, 32 * num_cores * 16, 32 * math.lcm(num_banks, num_cores) * 5
    a, b = _bf16_rand(1, 1, m, k), _bf16_rand(1, 1, k, n)
    in0_memory_config = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=1, x=num_cores),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    dram_grid_size = device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
    )
    in1_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [k, n // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=1, per_core_M=m // 32, per_core_N=n // num_cores // 32, fused_activation=None
    )
    a_tt = ttnn.from_torch(
        a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_memory_config
    )
    b_tt = ttnn.from_torch(
        b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_memory_config
    )
    output = ttnn.matmul(
        a_tt,
        b_tt,
        program_config=program_config,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
        dtype=ttnn.float32,
        compute_kernel_config=_compute_config(device, packer_l1_acc),
    )
    output = ttnn.to_torch(ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)).double()
    err = _rel_max_err(output, torch.matmul(a.double(), b.double()))
    assert err < REL_TOL, f"relative max error {err:.3e}"


@pytest.mark.parametrize("packer_l1_acc", [False, True])
def test_matmul_batched_dram_sharded_fp32_partials_reload(device, packer_l1_acc):
    torch.manual_seed(0)
    workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(workers)
    batch, m, k, n = num_banks, 32, 2048, 64
    a, b = _bf16_rand(1, batch, m, k), _bf16_rand(1, batch, k, n)
    worker_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in workers]
    )
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})

    def height_sharded(grid, buffer_type, shard_shape):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type,
            ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=1, per_core_M=m // 32, per_core_N=n // 32, fused_activation=None
    )
    a_tt = ttnn.from_torch(
        a,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=height_sharded(worker_grid, ttnn.BufferType.L1, (m, k)),
    )
    b_tt = ttnn.from_torch(
        b,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=height_sharded(dram_grid, ttnn.BufferType.DRAM, (k, n)),
    )
    output = ttnn.matmul(
        a_tt,
        b_tt,
        program_config=program_config,
        memory_config=height_sharded(worker_grid, ttnn.BufferType.L1, (m, n)),
        dtype=ttnn.float32,
        compute_kernel_config=_compute_config(device, packer_l1_acc),
    )
    reference = torch.matmul(a.double(), b.double())
    output = (
        ttnn.to_torch(ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)).double().reshape(reference.shape)
    )
    err = _rel_max_err(output, reference)
    assert err < REL_TOL, f"relative max error {err:.3e}"


# With a fused bias the partials buffer is also read by the bias add (through SrcA), so the reload uses a separate
# UnpackToDest view of it. The final bias add itself still goes through SrcA in every factory (mcast included),
# which bounds the error at ~1.3e-3 to 1.8e-3 here; with packer_l1_acc=True there is no reload (control case).
BIAS_REL_TOL = 3e-3
# A bfloat16 output adds up to 2^-9 relative rounding on top (measured 4.1e-3 to 4.7e-3, the same as without reload).
BIAS_REL_TOL_BF16 = 6e-3


def _fused_bias_matmul(device, a_tt, b_tt, bias, program_config, packer_l1_acc, output_dtype, output_mem_config=None):
    bias_tt = ttnn.from_torch(
        bias, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    params = ttnn.MatmulParams()
    params.program_config = program_config
    params.output_dtype = output_dtype
    params.compute_kernel_config = _compute_config(device, packer_l1_acc)
    if output_mem_config is not None:
        params.output_mem_config = output_mem_config
    attributes = ttnn.create_matmul_attributes(a_tt, b_tt, params, [])
    output = ttnn.MatmulDeviceOperation.invoke(a_tt, b_tt, bias_tt, attributes)[0]
    if output.is_sharded():
        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_torch(output).double()


# A bfloat16 output keeps the Float32 partials in their own buffer instead of sharing it with the output.
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("packer_l1_acc", [False, True])
@pytest.mark.parametrize("factory", ["reuse", "dram_sharded", "batched_dram_sharded"])
def test_matmul_fp32_partials_reload_fused_bias(device, factory, packer_l1_acc, output_dtype):
    torch.manual_seed(0)
    if factory == "reuse":
        batch, m, k, n = 8, 256, 4096, 256
        a, b, bias = _bf16_rand(batch, 1, m, k), _bf16_rand(batch, 1, k, n), _bf16_rand(1, 1, m, n)
        grid = device.compute_with_storage_grid_size()
        program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=m // 32,
            per_core_N=n // 32,
        )
        a_tt = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        output = _fused_bias_matmul(device, a_tt, b_tt, bias, program_config, packer_l1_acc, output_dtype)
    elif factory == "dram_sharded":
        num_banks = device.dram_grid_size().x
        num_cores = 8
        m, k, n = 32, 32 * num_cores * 16, 32 * math.lcm(num_banks, num_cores) * 5
        a, b, bias = _bf16_rand(1, 1, m, k), _bf16_rand(1, 1, k, n), _bf16_rand(1, 1, 1, n)
        in0_memory_config = ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=ttnn.CoreGrid(y=1, x=num_cores),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        dram_grid_size = device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        )
        in1_memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, [k, n // num_banks], ttnn.ShardOrientation.ROW_MAJOR),
        )
        program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1, per_core_M=m // 32, per_core_N=n // num_cores // 32, fused_activation=None
        )
        a_tt = ttnn.from_torch(
            a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in0_memory_config
        )
        b_tt = ttnn.from_torch(
            b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in1_memory_config
        )
        output = _fused_bias_matmul(
            device,
            a_tt,
            b_tt,
            bias,
            program_config,
            packer_l1_acc,
            output_dtype,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1),
        )
    else:
        workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        num_banks = len(workers)
        batch, m, k, n = num_banks, 32, 2048, 64
        a, b, bias = _bf16_rand(1, batch, m, k), _bf16_rand(1, batch, k, n), _bf16_rand(1, 1, 1, n)
        worker_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in workers]
        )
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})

        def height_sharded(grid, buffer_type, shard_shape):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type,
                ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
            )

        program_config = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
            in0_block_w=1, per_core_M=m // 32, per_core_N=n // 32, fused_activation=None
        )
        a_tt = ttnn.from_torch(
            a,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=height_sharded(worker_grid, ttnn.BufferType.L1, (m, k)),
        )
        b_tt = ttnn.from_torch(
            b,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=height_sharded(dram_grid, ttnn.BufferType.DRAM, (k, n)),
        )
        output = _fused_bias_matmul(
            device,
            a_tt,
            b_tt,
            bias,
            program_config,
            packer_l1_acc,
            output_dtype,
            height_sharded(worker_grid, ttnn.BufferType.L1, (m, n)),
        )
    reference = torch.matmul(a.double(), b.double()) + bias.double()
    err = _rel_max_err(output.reshape(reference.shape), reference)
    tol = BIAS_REL_TOL if output_dtype == ttnn.float32 else BIAS_REL_TOL_BF16
    assert err < tol, f"relative max error {err:.3e}"
