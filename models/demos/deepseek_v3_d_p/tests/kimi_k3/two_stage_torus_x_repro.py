# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal reproducer for the two-stage TORUS_X hang, with no model.

Two-stage K3 prefill on one galaxy (one 4x4 half per rank) hangs intermittently under
FABRIC_2D_TORUS_X and not under TORUS_Y / FABRIC_2D. Triage of the hangs: on rank 0, the rows of the
mesh that carry the inter-mesh links to rank 1 sit in a TP-axis (ring) `reduce_scatter_minimal_async`
while the other rows moved on; rank 1 sits in `inbound_socket_service_sync` waiting for the chunk.

This reproduces exactly that state with the runner's own D2D service (same spec, mapper, FIFO, worker
grid, lease/reclaim order) and the same collective (K3's KDA output projection reduce-scatter, Ring on
the TP axis), without the model:

    rank 0, per iteration: reclaim links -> N ring reduce-scatters on the TP axis -> D2D send -> release
    rank 1, per iteration: reclaim links -> release -> receive (blocks in inbound_socket_service_sync)

Launch with the 2-rank binding from `gen_pipeline_binding.py --stage-shape 4x4`:

    PREFILL_FABRIC_MODE=2d_torus_x tt-run --rank-binding <binding.yaml> \
        python3 models/demos/deepseek_v3_d_p/tests/kimi_k3/two_stage_torus_x_repro.py

Each rank prints `TORUS_X_REPRO rank=<r> PASS` when all iterations complete; a hang leaves the ranks in
the state above for tt-triage.
"""

import os
import sys

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.runner_utils import activation_global_spec, open_mesh_device
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology

MESH_SHAPE = (4, 4)
TP_AXIS = 1
ROWS = 5120  # one chunk
PLANES = 2  # K3 sends [live | one sealed snapshot] at the layer-12 boundary
FIFO_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", "32768"))
METADATA_BYTES = 12
ITERATIONS = int(os.environ.get("REPRO_ITERATIONS", "20"))
RS_PER_ITERATION = int(os.environ.get("REPRO_RS_PER_ITERATION", "12"))  # one per KDA/attention layer of a stage
# Narrowing switches: no D2D service at all (rank 1 idles), and the TP link count.
NO_D2D = os.environ.get("REPRO_NO_D2D", "0") == "1"
NUM_LINKS = os.environ.get("REPRO_NUM_LINKS")
# Candidate fix: one D2D round before any collective, so the service's start-up handshake has drained
# before the first ring reduce-scatter crosses the boundary chips.
WARMUP_XFER = os.environ.get("REPRO_WARMUP_XFER", "0") == "1"


def _worker_cores(mesh):
    grid = mesh.compute_with_storage_grid_size()
    return ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(min(8, grid.x) - 1, min(8, grid.y) - 1))


def _d2d_kwargs(mesh):
    mapper = ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
    )
    workers = _worker_cores(mesh)
    return dict(
        global_spec=activation_global_spec(ROWS, KimiK3Config.EMB_SIZE, PLANES),
        mapper=mapper,
        fifo_size_bytes=FIFO_BYTES,
        sender_worker_cores=workers,
        receiver_worker_cores=workers,
        metadata_size_bytes=METADATA_BYTES,
        share_fabric_links=True,
        socket_buffer_type=ttnn.BufferType.L1,
    )


def _tp_reduce_scatter(mesh, tt_ccl, x, topology):
    return ttnn.experimental.reduce_scatter_minimal_async(
        x,
        dim=-1,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(TP_AXIS),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(TP_AXIS),
        num_links=int(NUM_LINKS) if NUM_LINKS else tt_ccl.get_num_links(TP_AXIS),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=TP_AXIS,
    )


def main():
    mesh = open_mesh_device(MESH_SHAPE, KimiK3Config, l1_small_size=KimiK3Config.L1_SMALL_SIZE)
    rank = int(ttnn.distributed_context_get_rank())
    assert int(ttnn.distributed_context_get_size()) == 2, "run with exactly 2 ranks"
    topology = per_axis_topology()
    logger.info(f"rank {rank}: fabric {ttnn.get_fabric_config()}, per-axis topology {topology}")

    if NO_D2D:
        d2d = None
    elif rank == 0:
        d2d = ttnn.D2DStreamService.create_sender(sender_mesh=mesh, sender_rank=0, receiver_rank=1, **_d2d_kwargs(mesh))
    else:
        d2d = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh, sender_rank=0, receiver_rank=1, **_d2d_kwargs(mesh)
        )
    ttnn.distributed_context_barrier()
    logger.info(f"rank {rank}: D2D endpoint up")

    tt_ccl = get_tt_ccl(mesh) if rank == 0 else None
    # KDA output projection result: rows / SP per chip, full hidden before the TP reduce-scatter.
    x = None
    if rank == 0:
        x = ttnn.from_torch(
            torch.randn(1, 1, ROWS, KimiK3Config.EMB_SIZE * MESH_SHAPE[1]),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=MESH_SHAPE),
        )
    metadata = ttnn.from_torch(
        torch.tensor([0, 0, ROWS], dtype=torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
        ),
    )

    if NO_D2D:
        for it in range(ITERATIONS):
            if rank == 0:
                for _ in range(RS_PER_ITERATION):
                    out = _tp_reduce_scatter(mesh, tt_ccl, x, topology[TP_AXIS])
                    ttnn.deallocate(out)
                ttnn.synchronize_device(mesh)
            logger.info(f"rank {rank}: iteration {it} OK")
        ttnn.distributed_context_barrier()
        print(
            f"TORUS_X_REPRO rank={rank} fabric={ttnn.get_fabric_config()} no_d2d iterations={ITERATIONS} PASS",
            flush=True,
        )
        ttnn.close_mesh_device(mesh)
        return 0

    if WARMUP_XFER:
        d2d.wait_for_fabric_links()
        if rank == 1:
            d2d.release_fabric_links()
            ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(d2d, metadata_size_bytes=METADATA_BYTES)
        else:
            warm = ttnn.allocate_tensor_on_device(d2d.get_backing_tensor().spec, mesh)
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2d, warm, metadata=metadata)
            d2d.release_fabric_links()
        ttnn.synchronize_device(mesh)
        ttnn.distributed_context_barrier()
        logger.info(f"rank {rank}: warm-up transfer done")

    for it in range(ITERATIONS):
        # The runner's lease order (`prefill_runner._lease_reclaim` / `_compute_and_send`): take the links
        # back from the service before using the fabric for compute; a receiver hands them straight back so
        # its service can deliver; a sender hands them back after its send.
        d2d.wait_for_fabric_links()
        if rank == 1:
            d2d.release_fabric_links()
        if rank == 0:
            out = None
            for _ in range(RS_PER_ITERATION):
                out = _tp_reduce_scatter(mesh, tt_ccl, x, topology[TP_AXIS])
            ttnn.synchronize_device(mesh)
            payload = ttnn.allocate_tensor_on_device(d2d.get_backing_tensor().spec, mesh)
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2d, payload, metadata=metadata)
            ttnn.deallocate(out)
            ttnn.deallocate(payload)
        else:
            ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(d2d, metadata_size_bytes=METADATA_BYTES)
            ttnn.synchronize_device(mesh)
        if rank == 0:
            d2d.release_fabric_links()
        logger.info(f"rank {rank}: iteration {it} OK")

    ttnn.distributed_context_barrier()
    print(f"TORUS_X_REPRO rank={rank} fabric={ttnn.get_fabric_config()} iterations={ITERATIONS} PASS", flush=True)
    del d2d  # shut the service down while its command queue still exists
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
