# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused real-weight projection matrix for Phi optimized decode."""

import statistics
import time

import pytest
import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _real_state,
)


def _pcc(reference, actual):
    x, y = reference.float().flatten(), actual.float().flatten()
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def _cores(mesh_device, count):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.num_cores_to_corerangeset(count, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)


def _l1_width(mesh_device, width, count):
    return ttnn.create_sharded_memory_config_(
        shape=(32, width // count),
        core_grid=_cores(mesh_device, count),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_width(mesh_device, k, n):
    dram = mesh_device.dram_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, 0))})
    shard_width = ((n + dram.x * 32 - 1) // (dram.x * 32)) * 32
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, shard_width), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _kernel(fidelity):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _trace_ms(mesh_device, op):
    op()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    samples = []
    try:
        for _ in range(100):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter_ns() - start) / 1.0e6)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return statistics.mean(samples)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_projection_matrix(mesh_device, batch):
    config = _config()
    state = _real_state()
    prefix = f"model.layers.{LAYER_IDX}."
    roles = {
        "qkv": ("self_attn.qkv_proj.weight", config.hidden_size, 3 * config.hidden_size, ttnn.bfloat4_b),
        "output": ("self_attn.o_proj.weight", config.hidden_size, config.hidden_size, ttnn.bfloat4_b),
        "gate_up": ("mlp.gate_up_proj.weight", config.hidden_size, 2 * config.intermediate_size, ttnn.bfloat4_b),
        "down": ("mlp.down_proj.weight", config.intermediate_size, config.hidden_size, ttnn.bfloat4_b),
    }
    torch.manual_seed(9700 + batch)
    for role, (suffix, k, n, weight_dtype) in roles.items():
        logical = (torch.randn(1, 1, batch, k) * 0.2).to(torch.bfloat16)
        padded = torch.zeros(1, 1, 32, k, dtype=torch.bfloat16)
        padded[:, :, :batch, :] = logical
        weight = state[prefix + suffix].transpose(-2, -1).contiguous()
        reference = logical.float() @ weight.float()
        activation = ttnn.from_torch(
            padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        baseline_weight = ttnn.from_torch(
            weight,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def baseline():
            return ttnn.linear(activation, baseline_weight, dtype=ttnn.bfloat16)

        output = baseline()
        print(
            f"PROJ_RESULT batch={batch} role={role} candidate=baseline "
            f"pcc={_pcc(reference, ttnn.to_torch(output)[:, :, :batch, :]):.8f} "
            f"mean_ms={_trace_ms(mesh_device, baseline):.6f}"
        )
        dram_weight = ttnn.from_torch(
            weight,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=_dram_width(mesh_device, k, n),
        )
        block_matrix = (
            ((16, (16, 8, 4, 2, 1)), (32, (8, 4, 2, 1))) if role == "down" else ((16, (6, 3, 2, 1)), (32, (3, 1)))
        )
        for cores, block_ws in block_matrix:
            sharded = ttnn.to_memory_config(activation, _l1_width(mesh_device, k, cores))
            output_config = _l1_width(mesh_device, n, cores)
            for fidelity_name, fidelity in (("lofi", ttnn.MathFidelity.LoFi),):
                for block_w in block_ws:
                    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=block_w,
                        per_core_M=1,
                        per_core_N=(n // 32) // cores,
                        fused_activation=None,
                    )

                    def candidate():
                        return ttnn.linear(
                            sharded,
                            dram_weight,
                            dtype=ttnn.bfloat16,
                            memory_config=output_config,
                            program_config=pc,
                            compute_kernel_config=_kernel(fidelity),
                        )

                    try:
                        output = candidate()
                        output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
                        pcc = _pcc(reference, ttnn.to_torch(output)[:, :, :batch, :])
                        mean_ms = _trace_ms(mesh_device, candidate)
                        print(
                            f"PROJ_RESULT batch={batch} role={role} candidate=dram cores={cores} "
                            f"in0_block_w={block_w} fidelity={fidelity_name} pcc={pcc:.8f} mean_ms={mean_ms:.6f}"
                        )
                    except RuntimeError as error:
                        print(
                            f"PROJ_REJECT batch={batch} role={role} cores={cores} "
                            f"in0_block_w={block_w} fidelity={fidelity_name} error={error}"
                        )
