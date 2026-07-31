# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused AutoFix H1 experiment for Phi's decode down projection."""

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
    x = reference.float().flatten()
    y = actual.float().flatten()
    return float(torch.corrcoef(torch.stack((x, y)))[0, 1])


def _core_grid(num_cores, mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.num_cores_to_corerangeset(num_cores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)


def _l1_width_config(width, num_cores, mesh_device):
    assert width % num_cores == 0
    return ttnn.create_sharded_memory_config_(
        shape=(32, width // num_cores),
        core_grid=_core_grid(num_cores, mesh_device),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _dram_weight_config(k, n, mesh_device):
    dram = mesh_device.dram_grid_size()
    assert dram.y == 1
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, 0))})
    padded_n_per_bank = ((n + dram.x * 32 - 1) // (dram.x * 32)) * 32
    shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n_per_bank), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


def _kernel(fidelity):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _trace_mean_ms(mesh_device, op, iterations=100):
    op()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    op()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    samples = []
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        for _ in range(iterations):
            start = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            samples.append((time.perf_counter_ns() - start) / 1.0e6)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    return statistics.mean(samples)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_h1_dram_sharded_down(mesh_device, batch):
    config = _config()
    state = _real_state()
    weight = state[f"model.layers.{LAYER_IDX}.mlp.down_proj.weight"].transpose(-2, -1).contiguous()
    torch.manual_seed(9510 + batch)
    logical_activation = (torch.randn(1, 1, batch, config.intermediate_size) * 0.2).to(torch.bfloat16)
    padded_activation = torch.zeros(1, 1, 32, config.intermediate_size, dtype=torch.bfloat16)
    padded_activation[:, :, :batch, :] = logical_activation
    reference = logical_activation.float() @ weight.float()

    interleaved = ttnn.from_torch(
        padded_activation,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    baseline_weight = ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def baseline():
        return ttnn.linear(interleaved, baseline_weight, dtype=ttnn.bfloat16)

    baseline_out = baseline()
    baseline_pcc = _pcc(reference, ttnn.to_torch(baseline_out)[:, :, :batch, :])
    baseline_ms = _trace_mean_ms(mesh_device, baseline)
    print(f"H1_RESULT batch={batch} candidate=baseline pcc={baseline_pcc:.8f} mean_ms={baseline_ms:.6f}")

    dram_weight = ttnn.from_torch(
        weight,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=_dram_weight_config(config.intermediate_size, config.hidden_size, mesh_device),
    )
    for cores, block_ws in ((16, (16, 8, 4, 2)), (32, (8, 4, 2))):
        input_config = _l1_width_config(config.intermediate_size, cores, mesh_device)
        output_config = _l1_width_config(config.hidden_size, cores, mesh_device)
        sharded_input = ttnn.to_memory_config(interleaved, input_config)
        for fidelity_name, fidelity in (("hifi2", ttnn.MathFidelity.HiFi2), ("lofi", ttnn.MathFidelity.LoFi)):
            for block_w in block_ws:
                program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block_w,
                    per_core_M=1,
                    per_core_N=(config.hidden_size // 32) // cores,
                    fused_activation=None,
                )

                def candidate():
                    return ttnn.linear(
                        sharded_input,
                        dram_weight,
                        dtype=ttnn.bfloat16,
                        memory_config=output_config,
                        program_config=program_config,
                        compute_kernel_config=_kernel(fidelity),
                    )

                try:
                    output = candidate()
                    output = ttnn.sharded_to_interleaved(output, ttnn.DRAM_MEMORY_CONFIG)
                    pcc = _pcc(reference, ttnn.to_torch(output)[:, :, :batch, :])
                    mean_ms = _trace_mean_ms(mesh_device, candidate)
                    print(
                        f"H1_RESULT batch={batch} candidate=dram cores={cores} "
                        f"in0_block_w={block_w} fidelity={fidelity_name} pcc={pcc:.8f} mean_ms={mean_ms:.6f}"
                    )
                except RuntimeError as error:
                    print(
                        f"H1_REJECT batch={batch} cores={cores} in0_block_w={block_w} "
                        f"fidelity={fidelity_name} error={error}"
                    )
