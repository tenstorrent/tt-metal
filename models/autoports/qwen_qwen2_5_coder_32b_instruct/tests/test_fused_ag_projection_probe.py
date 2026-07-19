# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused exact-shape probes for TP4 fused all-gather + decode projection.

These probes intentionally avoid the complete decoder so a launch, static-CB
failure, or device stall can be attributed to one fused projection shape.
They are manual-only because a failing experimental fused kernel may require a
device reset before another case is attempted.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.modules.tt_ccl import (
    CCL_CHUNKS_PER_SYNC,
    CCL_NUM_BUFFERS_PER_CHANNEL,
    CCL_NUM_WORKERS_PER_LINK,
    get_tt_ccl,
)

PROBE_ENV = "QWEN2_5_CODER_32B_MULTICHIP_FUSED_AG_PROBE"


def _width_sharded(rows: int, width: int, grid: ttnn.CoreGrid) -> ttnn.MemoryConfig:
    assert width % grid.num_cores == 0
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // grid.num_cores),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _largest_divisor(value: int, limit: int = 8) -> int:
    return max(candidate for candidate in range(1, limit + 1) if value % candidate == 0)


@pytest.mark.skipif(
    os.getenv(PROBE_ENV) not in {"qkv_8x1", "gate_up_padded_10x1", "gate_then_up_8x1"}, reason="manual probe"
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(300)
def test_exact_shape_fused_all_gather_projection(mesh_device):
    role = os.environ[PROBE_ENV]
    hidden_size = 5120
    local_hidden_size = hidden_size // 4
    batch = 32
    if role == "qkv_8x1":
        grid = ttnn.CoreGrid(x=8, y=1)
        output_width = 2048
        weight_dtype = ttnn.bfloat8_b
        fidelity = ttnn.MathFidelity.HiFi2
    elif role == "gate_up_padded_10x1":
        # The logical packed gate/up width is 14,336. Two internal zero
        # columns tiles reduce per-core N from 56 on 8 cores to 45 on 10.
        grid = ttnn.CoreGrid(x=10, y=1)
        output_width = 14400
        weight_dtype = ttnn.bfloat4_b
        fidelity = ttnn.MathFidelity.LoFi
    else:
        # Decompose the material packed projection while preserving one AG:
        # the fused gate returns the full gathered activation, and up consumes
        # that tensor directly rather than launching a second collective.
        grid = ttnn.CoreGrid(x=8, y=1)
        output_width = 7168
        weight_dtype = ttnn.bfloat4_b
        fidelity = ttnn.MathFidelity.LoFi

    torch.manual_seed(190726)
    host_input = torch.empty((1, 1, batch, hidden_size), dtype=torch.bfloat16)
    for rank, chunk in enumerate(host_input.chunk(4, dim=-1)):
        chunk.fill_(rank + 1)
    tt_input = ttnn.from_torch(
        host_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    input_memory = _width_sharded(batch, hidden_size, grid)
    tt_input = ttnn.to_memory_config(tt_input, input_memory)

    # Replication is enough for launch/CB isolation: every device receives a
    # legal local column-parallel weight with the exact model K and local N.
    host_weight = torch.zeros((1, 1, hidden_size, output_width), dtype=torch.bfloat16)
    host_weight[..., ::32, ::32] = 1
    tt_weight = ttnn.from_torch(
        host_weight,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_up_weight = None
    if role == "gate_then_up_8x1":
        tt_up_weight = ttnn.from_torch(
            host_weight,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    per_core_n = output_width // 32 // grid.num_cores
    program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=hidden_size // 32 // grid.num_cores,
        out_subblock_h=1,
        out_subblock_w=_largest_divisor(per_core_n),
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    ccl = get_tt_ccl(mesh_device)
    output_memory = _width_sharded(batch, output_width, grid)
    gathered, projected = ttnn.experimental.all_gather_matmul_async(
        tt_input,
        tt_weight,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
        all_gather_core_grid_offset=(0, 4),
        barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
        # The fused primitive's validated model integrations use one link even
        # where the standalone TP collectives use all available links.
        num_links=1,
        memory_config_ag=input_memory,
        topology=ttnn.Topology.Ring,
        memory_config_mm=output_memory,
        dtype=ttnn.bfloat16,
        program_config=program,
        compute_kernel_config=compute,
        chunks_per_sync=CCL_CHUNKS_PER_SYNC,
        num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
        num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
    )
    up = None
    if role == "gate_then_up_8x1":
        up = ttnn.matmul(
            gathered,
            tt_up_weight,
            dtype=ttnn.bfloat16,
            program_config=program,
            compute_kernel_config=compute,
            memory_config=output_memory,
        )
    ttnn.synchronize_device(mesh_device)

    gather_orders = []
    for device_tensor in ttnn.get_device_tensors(gathered):
        host = ttnn.to_torch(device_tensor).float()
        gather_orders.append([round(float(chunk.mean())) for chunk in host.chunk(4, dim=-1)])
    projected_hosts = [ttnn.to_torch(tensor).float() for tensor in ttnn.get_device_tensors(projected)]
    assert all(torch.isfinite(tensor).all() for tensor in projected_hosts)
    up_hosts = None
    if up is not None:
        up_hosts = [ttnn.to_torch(tensor).float() for tensor in ttnn.get_device_tensors(up)]
        assert all(torch.isfinite(tensor).all() for tensor in up_hosts)
    print(
        {
            "role": role,
            "grid": [grid.x, grid.y],
            "in0_block_w": hidden_size // 32 // grid.num_cores,
            "per_core_n": per_core_n,
            "gather_orders": gather_orders,
            "projected_shape_per_device": [list(tensor.shape) for tensor in projected_hosts],
            "up_shape_per_device": None if up_hosts is None else [list(tensor.shape) for tensor in up_hosts],
        }
    )


@pytest.mark.skipif(os.getenv(PROBE_ENV) != "persistent_trace_4shard_ag_8x1_mm", reason="manual probe")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.timeout(300)
def test_persistent_four_shard_ag_eight_core_matmul_trace(mesh_device):
    """Prove a trace-stable TP4 AG buffer can feed an 8-core fused matmul."""

    hidden_size = 5120
    batch = 32
    ag_grid = ttnn.CoreGrid(x=4, y=1)
    mm_grid = ttnn.CoreGrid(x=8, y=1)
    ag_memory = _width_sharded(batch, hidden_size, ag_grid)

    host_input = torch.empty((1, 1, batch, hidden_size), dtype=torch.bfloat16)
    for rank, chunk in enumerate(host_input.chunk(4, dim=-1)):
        chunk.fill_(rank + 1)
    tt_input = ttnn.from_torch(
        host_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    tt_input = ttnn.to_memory_config(tt_input, ag_memory)

    persistent = []
    for _ in range(2):
        persistent.append(
            ttnn.from_torch(
                torch.zeros((1, 1, batch, hidden_size), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ag_memory,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        )

    role_configs = []
    for output_width, weight_dtype, fidelity in (
        (2048, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),
        (7168, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi),
    ):
        host_weight = torch.zeros((1, 1, hidden_size, output_width), dtype=torch.bfloat16)
        host_weight[..., ::32, ::32] = 1
        weight = ttnn.from_torch(
            host_weight,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        per_core_n = output_width // 32 // mm_grid.num_cores
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(mm_grid.x, mm_grid.y),
            in0_block_w=hidden_size // 32 // mm_grid.num_cores,
            out_subblock_h=1,
            out_subblock_w=_largest_divisor(per_core_n),
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        role_configs.append((weight, program, compute, _width_sharded(batch, output_width, mm_grid), output_width))

    ccl = get_tt_ccl(mesh_device)

    def run_sequence():
        outputs = []
        for slot, (weight, program, compute, output_memory, _) in enumerate(role_configs):
            _, projected = ttnn.experimental.all_gather_matmul_async(
                tt_input,
                weight,
                persistent_output_buffer=persistent[slot],
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
                all_gather_core_grid_offset=(0, 4),
                barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=1,
                memory_config_ag=ag_memory,
                topology=ttnn.Topology.Ring,
                memory_config_mm=output_memory,
                dtype=ttnn.bfloat16,
                program_config=program,
                compute_kernel_config=compute,
                chunks_per_sync=CCL_CHUNKS_PER_SYNC,
                num_workers_per_link=CCL_NUM_WORKERS_PER_LINK,
                num_buffers_per_channel=CCL_NUM_BUFFERS_PER_CHANNEL,
            )
            outputs.append(projected)
        return outputs

    warm_outputs = run_sequence()
    ttnn.synchronize_device(mesh_device)
    assert all(
        torch.isfinite(ttnn.to_torch(device_tensor).float()).all()
        for output in warm_outputs
        for device_tensor in ttnn.get_device_tensors(output)
    )

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_outputs = run_sequence()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        first = [
            [ttnn.to_torch(device_tensor).clone() for device_tensor in ttnn.get_device_tensors(output)]
            for output in trace_outputs
        ]
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        second = [
            [ttnn.to_torch(device_tensor).clone() for device_tensor in ttnn.get_device_tensors(output)]
            for output in trace_outputs
        ]
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    assert all(torch.equal(a, b) for role_a, role_b in zip(first, second) for a, b in zip(role_a, role_b))
    gather_orders = []
    for buffer in persistent:
        gather_orders.append(
            [
                [round(float(chunk.mean())) for chunk in ttnn.to_torch(device_tensor).float().chunk(4, dim=-1)]
                for device_tensor in ttnn.get_device_tensors(buffer)
            ]
        )
    assert all(order == [1, 2, 3, 4] for role in gather_orders for order in role)
    print(
        {
            "role": "persistent_trace_4shard_ag_8x1_mm",
            "gather_orders": gather_orders,
            "projected_shape_per_device": [[list(tensor.shape) for tensor in role_outputs] for role_outputs in first],
            "trace_bitwise_deterministic": True,
        }
    )
