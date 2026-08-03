# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-replay device-time perf for high_bw_all_gather.

Two cases, both run the collective inside a captured metal trace and replay it so the
tracy device profiler records per-op device zones under the traced dispatch path (the path
the chunked-prefill runner actually uses):

    Case A (test_..._single_glx, 1 rank):  all_gather loop on one galaxy.
    Case B (test_..._d2d, 2 ranks):        all_gather loop on galaxy 0 -> D2D hop ->
                                           all_gather loop on galaxy 1.

The op's own bandwidth suite measures via the realtime device-profiler callback, which fires
per host program enqueue; trace replay is a single EnqueueTrace, so that path and its GB/s
assertion cannot run here. We do not measure bandwidth in-test: the traced loop is exercised
and ReadDeviceProfiler drains the device zones, and per-op device time is read afterwards from
the ops_perf_results CSV (filter to the all_gather op code).

Setup (fabric, sharding, D2D socket, semaphore) mirrors test_allgather_d2d_probe.py.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

_NUM_LINKS = 2
_CLUSTER_AXIS = 0  # all_gather along the 8-long mesh dim; line schedule under FABRIC_2D
_ROWS_PER_DEVICE = int(os.environ.get("PERF_ROWS_PER_DEVICE", 32))
_WIDTH = int(os.environ.get("PERF_WIDTH", 512))

# Ops captured per trace and trace replays; product stays under tracy's ~1000-op device buffer.
_CAPTURE_COUNT = int(os.environ.get("PERF_CAPTURE_COUNT", 10))
_REPLAY_COUNT = int(os.environ.get("PERF_REPLAY_COUNT", 20))

_TRACE_REGION_SIZE = int(os.environ.get("PERF_TRACE_REGION_SIZE", 90 * 1024 * 1024))

# D2D socket knobs, same shape as the prefill runner's endpoints.
_SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
_METADATA_SIZE_BYTES = 12
_FIFO_SIZE_BYTES = int(os.environ.get("PERF_D2D_FIFO_BYTES", 64 * 1024))
_REPLICATE_2D = ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])


def _fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024
    return config


_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": _fabric_router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "l1_small_size": 2048,
    "trace_region_size": _TRACE_REGION_SIZE,
}


def _data_valid_semaphore(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return ttnn.create_global_semaphore(mesh_device, cores, 0, buffer_type=ttnn.BufferType.L1_SMALL)


def _gathered_spec(gathered_rows):
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, gathered_rows, _WIDTH]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _persistent_replica(mesh_device, rows):
    return ttnn.from_torch(
        torch.zeros((1, 1, rows, _WIDTH), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )


def _sharded_input(mesh_device, gathered_rows):
    host_input = torch.rand((1, 1, gathered_rows, _WIDTH), dtype=torch.bfloat16)
    return ttnn.from_torch(
        host_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=tuple(mesh_device.shape)),
        device=mesh_device,
    )


def _one_all_gather(mesh_device, device_input, output, semaphore):
    ttnn.experimental.deepseek_prefill.high_bw_all_gather(
        device_input,
        dim=2,
        output_tensor=output,
        cluster_axis=_CLUSTER_AXIS,
        data_valid_semaphore=semaphore,
        num_links=_NUM_LINKS,
    )


def _trace_all_gather_loop(mesh_device, device_input, semaphore, label):
    """Capture _CAPTURE_COUNT all_gathers into a trace, replay _REPLAY_COUNT times, drain the profiler.

    The persistent output is allocated once and reused every iteration so the captured trace has a
    stable address to replay against. Device zones for the replayed ops land in the profiler CSV.
    """
    local_padded = ttnn.get_device_tensors(device_input)[0].padded_shape
    gathered_rows = local_padded[2] * mesh_device.shape[_CLUSTER_AXIS]
    output = _persistent_replica(mesh_device, gathered_rows)

    # Eager warmup compiles kernels and fills the program cache; trace capture forbids device writes
    # (e.g. binary loads) on the fast-dispatch path, so everything must be resident before capture.
    _one_all_gather(mesh_device, device_input, output, semaphore)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{label}] eager warmup all_gather done")

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(_CAPTURE_COUNT):
        _one_all_gather(mesh_device, device_input, output, semaphore)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    logger.info(f"[{label}] trace capture done ({_CAPTURE_COUNT} ops)")

    # One unmeasured replay, then drain, so the warmup's device zones are not in the measured set.
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{label}] warmup replay done")

    for _ in range(_REPLAY_COUNT):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[{label}] measured replay done ({_REPLAY_COUNT} replays)")

    ttnn.release_trace(mesh_device, tid)
    logger.info(
        f"[{label}] captured {_CAPTURE_COUNT}/replayed {_REPLAY_COUNT} all_gathers "
        f"gathered_rows={gathered_rows} width={_WIDTH}"
    )
    return output, gathered_rows


def _socket_common(mesh_device, gathered_rows):
    return dict(
        global_spec=_gathered_spec(gathered_rows),
        mapper=ttnn.create_mesh_mapper(mesh_device, _REPLICATE_2D),
        fifo_size_bytes=_FIFO_SIZE_BYTES,
        sender_worker_cores=_SYNC_WORKER_CORES,
        receiver_worker_cores=_SYNC_WORKER_CORES,
        metadata_size_bytes=_METADATA_SIZE_BYTES,
        share_fabric_links=True,
        socket_buffer_type=ttnn.BufferType.L1,
    )


def _send(outbound, tensor):
    backing = outbound.get_backing_tensor()
    md = ttnn.from_torch(
        torch.zeros((1, 1, 1, 3), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=backing.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(backing.device(), _REPLICATE_2D),
    )
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(outbound, tensor, metadata=md)


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_allgather_trace_perf_single_glx(mesh_device):
    """Case A: traced all_gather loop on a single galaxy."""
    axis = mesh_device.shape[_CLUSTER_AXIS]
    gathered_rows = _ROWS_PER_DEVICE * axis
    logger.info(f"[case A] mesh={tuple(mesh_device.shape)} axis={axis} gathered_rows={gathered_rows}")

    semaphore = _data_valid_semaphore(mesh_device)
    device_input = _sharded_input(mesh_device, gathered_rows)
    _trace_all_gather_loop(mesh_device, device_input, semaphore, "case A")


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_allgather_trace_perf_d2d(mesh_device):
    """Case B: traced all_gather loop on galaxy 0 -> D2D hop -> traced all_gather loop on galaxy 1."""
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())
    assert num_ranks == 2, f"case B needs exactly 2 ranks (2 galaxies), got {num_ranks}"

    axis = mesh_device.shape[_CLUSTER_AXIS]
    gathered_rows = _ROWS_PER_DEVICE * axis
    logger.info(f"[case B rank {rank}] mesh={tuple(mesh_device.shape)} axis={axis} gathered_rows={gathered_rows}")

    # Endpoints rendezvous point-to-point and each ctor blocks on its peer: the sync point before compute.
    common = _socket_common(mesh_device, gathered_rows)
    if rank == 0:
        outbound = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=0, receiver_rank=1, **common
        )
    else:
        inbound = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=0, receiver_rank=1, **common
        )

    semaphore = _data_valid_semaphore(mesh_device)

    if rank == 0:
        device_input = _sharded_input(mesh_device, gathered_rows)
        # Lease mode: sender holds no fabric link until granted, so the traced all_gather owns the links.
        outbound.wait_for_fabric_links()
        gathered, _ = _trace_all_gather_loop(mesh_device, device_input, semaphore, "case B rank 0")

        _send(outbound, gathered)
        outbound.release_fabric_links()
        outbound.wait_for_fabric_links()
        logger.info("[case B rank 0] D2D send complete")
    else:
        # Runner receiver-lease order (prefill_runner._lease_reclaim): reclaim (drain any prior transfer),
        # then grant the receiver so the recv drains. Do NOT reclaim again before the traced collective —
        # the runner leaves inbound granted across the traced forward; reclaiming here parks the receiver's
        # resident fabric connection and the captured all_gather deadlocks on replay.
        inbound.wait_for_fabric_links()
        inbound.release_fabric_links()
        received, _ = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            inbound, metadata_size_bytes=_METADATA_SIZE_BYTES
        )
        logger.info(f"[case B rank 1] D2D recv done -> {tuple(received.shape)}")

        # Same row count as rank 0: gather a freshly-sharded gathered_rows input (32/device -> 256),
        # not the received replicated tensor. The receiver endpoint stays resident and granted (D2D
        # always-on, as in the runner) while the traced all_gather runs on the shared links.
        device_input = _sharded_input(mesh_device, gathered_rows)
        _trace_all_gather_loop(mesh_device, device_input, semaphore, "case B rank 1")

    ttnn.distributed_context_barrier()
    logger.info(f"[case B rank {rank}] done")
