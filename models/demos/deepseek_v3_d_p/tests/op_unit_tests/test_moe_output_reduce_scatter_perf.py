# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
The MoE output reduce-scatter (TtReduceModule: [rows, emb] bf16 per chip -> [rows, emb / tp] across TP) at
one chunk's rows and at a packed forward's, with each candidate op. Record only: logs the median of
timed eager calls (each followed by a device sync).

At 640 -> 2560 rows per chip the plain ttnn.reduce_scatter on one link took 0.72 -> 4.38 ms in the packed
GLM forward's profile, 6x for 4x the data, while the shared expert's reduce_scatter_minimal_async scaled
linearly.
"""

import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, moe_fabric_payload
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

_ITERS = 10


@pytest.mark.skipif(not is_blackhole(), reason="Blackhole")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("op", ["rs", "rs_minimal_async"])
@pytest.mark.parametrize("num_links", [1, 2], ids=["link1", "link2"])
@pytest.mark.parametrize("rows", [640, 2560], ids=["rows640", "rows2560"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(fabric_payload_size=moe_fabric_payload(GLM53Config)),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        ),
    ],
    indirect=True,
)
def test_moe_output_reduce_scatter_perf(mesh_device, device_params, rows, num_links, op):
    emb = GLM53Config.EMB_SIZE
    topology = ttnn.Topology.Linear
    x = ttnn.from_torch(
        torch.randn(1, 1, rows, emb, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_ccl = get_tt_ccl(mesh_device)
    intermediate = None
    if op == "rs_minimal_async":
        # Linear keeps a forward/backward pair of accumulators (see TT_CCL.get_shared_rs_intermediate).
        intermediate = ttnn.from_torch(
            torch.zeros(2, 1, rows, emb),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def call():
        if op == "rs":
            return ttnn.reduce_scatter(x, dim=-1, cluster_axis=1, num_links=num_links, topology=topology)
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=[intermediate],
            dim=-1,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            cluster_axis=1,
        )

    # Correctness: every chip's slice is the sum over the TP row of identical inputs, i.e. tp * x.
    out = call()
    ttnn.synchronize_device(mesh_device)
    tp = mesh_device.shape[1]
    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
    )
    want = (
        tp
        * ttnn.to_torch(
            x, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape)
        )[:1, :1].float()
    )
    assert torch.allclose(got[:1].float(), want, rtol=0.05, atol=0.5), "reduce-scatter result mismatch"
    ttnn.deallocate(out)

    times = []
    for i in range(_ITERS + 2):
        t0 = time.perf_counter()
        out = call()
        ttnn.synchronize_device(mesh_device)
        if i >= 2:
            times.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(out)
    logger.warning(
        f"MOE_RS op={op} links={num_links} rows={rows} median_ms={statistics.median(times):.3f} "
        f"min_ms={min(times):.3f}"
    )
