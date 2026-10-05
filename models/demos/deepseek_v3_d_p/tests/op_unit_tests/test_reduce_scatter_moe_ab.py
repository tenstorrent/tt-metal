# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A/B the two reduce-scatters the GLM-5.3 prefill block uses on the same [1, 1, rows, 6144] tensor over the
4-chip column axis: ``ttnn.reduce_scatter`` (MoE TtReduceModule, dense TtFfn) vs
``ttnn.experimental.reduce_scatter_minimal_async`` (shared expert). rows = 640 (single-user TP), 1024 / 2560
(batch-4 at 2k / 5k chunks). Each variant is preceded by a signpost; profile with tracy."""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology

ITERS = 4


@pytest.mark.parametrize("rows", [640, 1024, 2560])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=GLM53Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
def test_reduce_scatter_moe_ab(mesh_device, device_params, num_links, rows):
    _, col_topology = per_axis_topology(device_params["fabric_config"])
    hidden = GLM53Config.EMB_SIZE
    cols = mesh_device.shape[1]
    torch.manual_seed(0)
    # Distinct partial per column: host [cols, 1, rows, hidden], column c gets slice c (rows replicated on SP).
    host = torch.randn(cols, 1, rows, hidden, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 0)),
    )
    ref = host.float().sum(dim=0)  # [1, rows, hidden]
    tt_ccl = get_tt_ccl(mesh_device)

    def check(out, name):
        got = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
        )[:1, 0]
        ok, pcc = comp_pcc(ref, got.float(), 0.999)
        logger.info(f"[rs-ab rows={rows}] {name} PCC {pcc}")
        assert ok, f"{name} PCC {pcc}"

    for topo in (col_topology, ttnn.Topology.Linear):
        tname = "ring" if topo == ttnn.Topology.Ring else "linear"
        signpost(f"rs_ab_reduce_scatter_{tname}_rows{rows}")
        for _ in range(ITERS):
            out = ttnn.reduce_scatter(x, dim=-1, cluster_axis=1, num_links=num_links, topology=topo)
            ttnn.synchronize_device(mesh_device)
        check(out, f"reduce_scatter[{tname}]")

        inter = tt_ccl.get_shared_rs_intermediate(x, topo) if topo == col_topology else None
        if inter is None or tuple(inter.shape) != (tuple(x.shape) if topo == ttnn.Topology.Ring else (2, *x.shape)):
            inter_shape = list(x.shape) if topo == ttnn.Topology.Ring else [2] + list(x.shape)
            inter = ttnn.from_torch(
                torch.zeros(inter_shape, dtype=torch.bfloat16),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        signpost(f"rs_ab_minimal_async_{tname}_rows{rows}")
        for _ in range(ITERS):
            out = ttnn.experimental.reduce_scatter_minimal_async(
                x,
                persistent_output_buffers=[inter],
                dim=-1,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
                num_links=num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topo,
                cluster_axis=1,
            )
            ttnn.synchronize_device(mesh_device)
        check(out, f"reduce_scatter_minimal_async[{tname}]")
    ttnn.ReadDeviceProfiler(mesh_device)
