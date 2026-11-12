# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather_minimal_matmul_async on a Blackhole Galaxy over a 2D fabric (TorusXY), TP ring of 4.

Covers the gather-sender mode (dedicated senders on the column right of the matmul grid, in1 column chunks, direct
output) and its relay fallback. Each case runs the op back to back on one gathered buffer and one set of semaphores,
as a model's layers do, and checks every call on every device.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

SP_AXIS, TP_AXIS = 0, 1
NUM_CALLS = 3
NUM_LINKS = 2


def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _run(
    mesh_device,
    *,
    rows_per_device,
    K,
    n_per_device,
    M_block,
    K_block,
    N_block,
    subblock_h,
    subblock_w,
    weight_dtype,
    barrier_on_sender_column=True,
):
    sp, tp = mesh_device.shape[SP_AXIS], mesh_device.shape[TP_AXIS]
    full_grid = mesh_device.compute_with_storage_grid_size()
    grid = ttnn.CoreCoord(full_grid.x - 1, full_grid.y)  # the free column right of it hosts the gather senders
    num_workers_per_link = grid.y // NUM_LINKS
    assert grid.y % NUM_LINKS == 0

    rows = sp * rows_per_device
    torch.manual_seed(0)
    torch_inputs = [torch.randn(rows, K) for _ in range(NUM_CALLS)]
    torch_weight = torch.randn(K, tp * n_per_device)

    # The activation's rows are split across SP and its K across TP; the weight's N across TP.
    tt_inputs = [
        ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
        )
        for x in torch_inputs
    ]
    tt_weight = ttnn.from_torch(
        torch_weight,
        dtype=weight_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 1)),
    )

    # Persistent state shared by every call: the gathered buffer, one arrival counter per source peer (backward,
    # forward, opposite) and the entry barrier.
    workers = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full_grid.x - 1, full_grid.y - 1))]
    )
    matmul_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    gathered = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows_per_device, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
    )
    counters = [ttnn.create_global_semaphore(mesh_device, workers, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(mesh_device, workers if barrier_on_sender_column else matmul_cores, 0)

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
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=24,
            force_transpose=False,
        )[0]
        for tt_input in tt_inputs
    ]
    ttnn.synchronize_device(mesh_device)

    # bfloat8_b weights: compare against the weight as the device holds it.
    device_weight = ttnn.to_torch(
        ttnn.from_torch(torch_weight, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT), dtype=torch.float32
    )
    for call, (torch_input, tt_output) in enumerate(zip(torch_inputs, outputs)):
        expected = torch_input.to(torch.bfloat16).float() @ device_weight
        actual = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1)),
        ).float()
        assert actual.shape == expected.shape, (call, actual.shape, expected.shape)
        for i in range(sp):
            for j in range(tp):
                m = slice(i * rows_per_device, (i + 1) * rows_per_device)
                n = slice(j * n_per_device, (j + 1) * n_per_device)
                passed, pcc = comp_pcc(expected[m, n], actual[m, n], 0.9999)
                assert passed, f"call {call}, device ({i}, {j}): {pcc}"


# Kimi-K3 KDA's input projection geometry (640 rows per device, 11 x 10 matmul grid, 2 x 36 tiles per core, 8-tile K
# blocks, 1 x 4 subblocks) with a shorter K. in1 goes in 3 column chunks of 12. The weights are bfloat8_b, as in KDA:
# double-buffered 8 x 36-tile bfloat16 in1 blocks alone would not fit in L1.
_KDA = dict(
    rows_per_device=640,
    K=4096,
    n_per_device=11 * 36 * 32,
    M_block=2,
    K_block=8,
    N_block=36,
    subblock_h=1,
    subblock_w=4,
    weight_dtype=ttnn.bfloat8_b,
)

_CASES = [
    pytest.param(_KDA, id="kda_input_projection"),
    # 19 tiles over 10 rows of cores: some cores have one row of a 2-row block (empty output pieces).
    pytest.param(dict(_KDA, rows_per_device=608), id="partial_m_block"),
    # 4 rows per core in 2-row blocks: two output blocks per core, so the output buffer keeps the whole block.
    pytest.param(dict(_KDA, rows_per_device=1280), id="two_m_blocks"),
    # Two-high subblocks: no direct output; in1 still in column chunks (3 of 12).
    pytest.param(dict(_KDA, subblock_h=2, subblock_w=2), id="subblock_2x2"),
    # The barrier semaphore stops at the matmul grid: the op relays the gather through the in0 cores instead.
    pytest.param(dict(_KDA, barrier_on_sender_column=False), id="relay_fallback"),
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
