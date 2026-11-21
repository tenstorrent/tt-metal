# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather_minimal_matmul_async with gather_rows on a Blackhole Galaxy (8 x 4), Fabric2D torus.

Each rank of the TP ring (4) holds its own rows of the whole K; the op gathers the ring's rows in rank order and
multiplies them by the rank's weight shard: [rows * 4, K] @ [K, N]. The sparse MLA q stem's shape (q_b_proj after
the q row gather). Each case runs the op back to back on one gathered buffer and one set of semaphores and checks
every call on every device.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

NUM_CALLS = 3
NUM_LINKS = 2
TP_AXIS = 1


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _run(mesh_device, *, rows, K, n_per_device, M_block, K_block, N_block, subblock_h, subblock_w, use_mcast=True):
    ring, others = mesh_device.shape[TP_AXIS], mesh_device.shape[1 - TP_AXIS]
    full_grid = mesh_device.compute_with_storage_grid_size()
    grid = ttnn.CoreCoord(full_grid.x - 1, full_grid.y)  # the in0 muxes take the column right of the grid
    shape = tuple(mesh_device.shape)

    torch.manual_seed(0)
    # [SP, TP, rows, K]: device (i, j) holds x[i, j]; distinguishable per-rank rows catch row-order errors.
    torch_inputs = [
        torch.randn(others, ring, rows, K) + torch.arange(ring).view(1, ring, 1, 1) for _ in range(NUM_CALLS)
    ]
    torch_weight = torch.randn(K, ring * n_per_device)
    tt_inputs = [
        ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(0, 1)),
        )
        for x in torch_inputs
    ]
    tt_weight = ttnn.from_torch(
        torch_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(None, 1)),
    )

    gathered = ttnn.allocate_tensor_on_device(
        ttnn.Shape([ring * rows, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    # On every worker core: the row movers sit beside the matmul grid.
    all_cores = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))]
    )
    counters = [ttnn.create_global_semaphore(mesh_device, all_cores, 0) for _ in range(2)]
    barrier = ttnn.create_global_semaphore(mesh_device, all_cores, 0)
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
            cluster_axis=TP_AXIS,
            num_links=NUM_LINKS,
            num_workers_per_link=grid.y // NUM_LINKS,
            num_buffers_per_channel=8,
            force_transpose=False,
            use_mcast=use_mcast,
            gather_rows=True,
        )[0]
        for tt_input in tt_inputs
    ]
    ttnn.synchronize_device(mesh_device)

    weight = torch_weight.to(torch.bfloat16).float()
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=shape, dims=(0, 1))
    for call, (torch_input, tt_output) in enumerate(zip(torch_inputs, outputs)):
        actual = ttnn.to_torch(tt_output, mesh_composer=composer).float()  # [SP, TP, ring * rows, n]
        assert tuple(actual.shape[-2:]) == (ring * rows, n_per_device), (call, actual.shape)
        for i in range(others):
            gathered_rows = torch_input[i].to(torch.bfloat16).float().reshape(ring * rows, K)
            for j in range(ring):
                expected = gathered_rows @ weight[:, j * n_per_device : (j + 1) * n_per_device]
                passed, pcc = comp_pcc(expected, actual[i, j], 0.9999)
                assert passed, f"call {call}, SP row {i}, TP rank {j}: {pcc}"
                for r in range(ring):  # each rank's rows in their place
                    band = slice(r * rows, (r + 1) * rows)
                    passed, pcc = comp_pcc(expected[band], actual[i, j][band], 0.9999)
                    assert passed, f"call {call}, SP row {i}, TP rank {j}, rows of rank {r}: {pcc}"


# Kimi K3.1 sparse MLA q_b_proj: 160 rows per rank (640 gathered, 2 tiles per core row; bands of 5 tiles split core
# rows), [1536, 4608] per rank.
_KIMI = dict(rows=160, K=1536, n_per_device=4608, M_block=2, K_block=8, N_block=14, subblock_h=2, subblock_w=2)
_CASES = [
    pytest.param(_KIMI, id="kimi_q_b_proj"),
    # GLM-5.3: [2048, 4096] per rank.
    pytest.param(dict(_KIMI, K=2048, n_per_device=4096, N_block=12), id="glm_q_b_proj"),
    # 128 rows per rank: 16 tiles over 10 core rows, the last rows of the M split are past the gather.
    pytest.param(dict(_KIMI, rows=128), id="padded_m_split"),
    # 320 rows per rank: two M blocks per core row.
    pytest.param(dict(_KIMI, rows=320), id="two_m_blocks"),
    # 12-tile K blocks: fewer, larger steps (the (tile row, K block) count stays a multiple of the DRAM banks).
    pytest.param(dict(_KIMI, K_block=12), id="k_block_12"),
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
@pytest.mark.parametrize("use_mcast", [True, False], ids=["mcast", "relay"])
def test_all_gather_minimal_matmul_gather_rows(mesh_device, case, use_mcast):
    _run(mesh_device, **case, use_mcast=use_mcast)
