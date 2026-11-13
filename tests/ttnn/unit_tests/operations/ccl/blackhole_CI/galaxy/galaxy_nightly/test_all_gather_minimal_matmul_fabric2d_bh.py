# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather_minimal_matmul_async on a Blackhole Galaxy (8 x 4) over a 2D fabric (TorusXY).

The activation's K is gathered along one ring (the TP ring of 4 or the SP ring of 8) and multiplied by that axis'
weight shard. Each case runs the op back to back on one gathered buffer and one set of semaphores, as a model's layers
do, and checks every call on every device.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

NUM_CALLS = 3
NUM_LINKS = 2


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _run(mesh_device, *, gather_axis, rows_per_device, K, n_per_device, M_block, K_block, N_block, subblock_h, subblock_w):
    other_axis = 1 - gather_axis
    ring, others = mesh_device.shape[gather_axis], mesh_device.shape[other_axis]
    full_grid = mesh_device.compute_with_storage_grid_size()
    grid = ttnn.CoreCoord(full_grid.x - 1, full_grid.y)
    assert grid.y % NUM_LINKS == 0

    rows = others * rows_per_device
    torch.manual_seed(0)
    torch_inputs = [torch.randn(rows, K) for _ in range(NUM_CALLS)]
    torch_weight = torch.randn(K, ring * n_per_device)

    def dims(row_dim, gather_dim):
        # (tensor dim split over mesh axis 0, tensor dim split over mesh axis 1)
        d = [None, None]
        d[other_axis], d[gather_axis] = row_dim, gather_dim
        return tuple(d)

    shape = tuple(mesh_device.shape)
    tt_inputs = [
        ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=dims(0, 1)),
        )
        for x in torch_inputs
    ]
    tt_weight = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=dims(None, 1)),
    )

    matmul_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    gathered = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows_per_device, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    counters = [ttnn.create_global_semaphore(mesh_device, matmul_cores, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh_device, matmul_cores, 0)

    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    config = ttnn.MinimalMatmulConfig(
        M_block_size=M_block,
        K_block_size=K_block,
        N_block_size=N_block,
        subblock_h=subblock_h,
        subblock_w=subblock_w,
        compute_with_storage_grid_size=grid,
    )

    # Back to back, no synchronisation in between.
    outputs = [
        ttnn.experimental.all_gather_minimal_matmul_async(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            config=config,
            compute_kernel_config=compute_config,
            multi_device_global_semaphore=counters,
            barrier_semaphore=barrier,
            persistent_output_buffer=gathered,
            topology=ttnn.Topology.Ring,
            cluster_axis=gather_axis,
            num_links=NUM_LINKS,
            num_workers_per_link=grid.y // NUM_LINKS,
            num_buffers_per_channel=24,
            force_transpose=False,
        )[0]
        for tt_input in tt_inputs
    ]
    ttnn.synchronize_device(mesh_device)

    # bfloat8_b weights: compare against the weight as the device holds it.
    device_weight = ttnn.to_torch(
        ttnn.from_torch(torch_weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT), dtype=torch.float32
    )
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=shape, dims=dims(0, 1))
    for call, (torch_input, tt_output) in enumerate(zip(torch_inputs, outputs)):
        expected = torch_input.to(torch.bfloat16).float() @ device_weight
        actual = ttnn.to_torch(tt_output, mesh_composer=composer).float()
        assert actual.shape == expected.shape, (call, actual.shape, expected.shape)
        for i in range(others):
            for j in range(ring):
                m = slice(i * rows_per_device, (i + 1) * rows_per_device)
                n = slice(j * n_per_device, (j + 1) * n_per_device)
                passed, pcc = comp_pcc(expected[m, n], actual[m, n], 0.9999)
                assert passed, f"call {call}, rows {i}, ring position {j}: {pcc}"


# Kimi-K3 KDA's input projection geometry on the TP ring (640 rows per device, 11 x 10 matmul grid, 2 x 36 tiles per
# core, 8-tile K blocks, 1 x 4 subblocks) with a shorter K. bfloat8_b weights, as in KDA.
_KDA = dict(
    gather_axis=1,
    rows_per_device=640,
    K=4096,
    n_per_device=11 * 36 * 32,
    M_block=2,
    K_block=8,
    N_block=36,
    subblock_h=1,
    subblock_w=4,
)

_CASES = [
    pytest.param(_KDA, id="kda_input_projection"),
    # 19 tiles over 10 rows of cores: the last rows of the output are past the tensor.
    pytest.param(dict(_KDA, rows_per_device=608), id="partial_m_block"),
    # 4 rows per core in 2-row blocks: two output blocks per core.
    pytest.param(dict(_KDA, rows_per_device=1280), id="two_m_blocks"),
    # 3 rows per core: a 2-row block, then a 1-row block.
    pytest.param(dict(_KDA, rows_per_device=960), id="short_last_m_block"),
    # One K block per device: each fabric core's first forward is also its last.
    pytest.param(dict(_KDA, K=1024), id="one_k_block_per_device"),
    pytest.param(dict(_KDA, subblock_h=2, subblock_w=2), id="subblock_2x2"),
    # The SP ring of 8: K gathered over 8 devices, 2 K blocks per device.
    pytest.param(dict(_KDA, gather_axis=0, rows_per_device=320, n_per_device=11 * 20 * 32, N_block=20), id="sp_ring_8"),
]


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
                "fabric_router_config": create_fabric_router_config(7168),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            },
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("case", _CASES)
def test_all_gather_minimal_matmul_fabric2d(mesh_device, case):
    _run(mesh_device, **case)
