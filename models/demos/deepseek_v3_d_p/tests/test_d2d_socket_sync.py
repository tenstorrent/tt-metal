# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op test for ``ttnn.experimental.deepseek_prefill.d2d_socket_sync`` (the D2D sender op).

Mirrors the ``test_d2d_stream_service`` gtest's single-host ``create_pair`` flow, but
drives the native sender op instead of the placeholder worker kernel:

  1. Carve a sender and a receiver ``1x2`` submesh from one mesh (adjacent rows, so
     tt-fabric routes between them).
  2. Stand up a ``D2DStreamService`` pair in OWN mode (``share_fabric_links=False``) so
     the sender forwards as soon as it has ``num_workers`` acks -- no host lease driving.
  3. Push an activation with ``d2d_socket_sync`` (the op under test). It is NON-BLOCKING:
     it copies the input into the sender backing, inc's ``data_ready_counter``, and
     returns; the persistent sender service does the fabric forward.
  4. Drain it on the receiver with the service-agnostic ``h2d_socket_sync`` (it blocks on
     the receiver's ``data_ready_sem``, so it returns only once the forward has landed).
  5. Assert the round trip is bit-exact, and (optionally) that inline metadata survives.

A second iteration with fresh data exercises the program cache: the op builds its
program once, and iteration 2 must reuse it with the per-dispatch input address patched
in as a ``Buffer*`` BufferBinding (if the address were baked in, iter 2 would read
iter 1's freed buffer and the data check would fail).

Requires Blackhole service cores + a FABRIC_2D mesh; skips cleanly elsewhere.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.utility_functions import is_blackhole

# Single worker core per side (num_workers == 1), matching the gtest's kWorkerCores.
_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
# "Fixed work per core": 32 ROW_MAJOR pages of 512 u32 (2048 B page) -> 4096 B FIFO.
_GLOBAL_SHAPE = [1, 1, 32, 512]
_FIFO_SIZE_BYTES = 4096


def _replicate_mapper(mesh):
    # 2D submesh -> one placement per mesh dim; replicate the global tensor on every chip.
    return ttnn.create_mesh_mapper(
        mesh,
        ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]),
    )


def _global_spec():
    return ttnn.TensorSpec(
        shape=ttnn.Shape(_GLOBAL_SHAPE),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _to_device(torch_u32, mesh):
    return ttnn.from_torch(
        torch_u32,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(mesh),
    )


@pytest.mark.parametrize("metadata_words", [None, [7, 128, 256, 1]], ids=["no_metadata", "metadata"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            {"fabric_config": ttnn.FabricConfig.FABRIC_2D},
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_d2d_socket_sync(mesh_device, metadata_words):
    if not is_blackhole():
        pytest.skip("D2DStreamService needs Blackhole (or UBB Galaxy) service cores")
    if mesh_device.shape[0] < 2:
        pytest.skip(f"need >= 2 mesh rows to carve sender+receiver submeshes, got {mesh_device.shape}")

    # Two adjacent 1x2 submeshes (gtest 'row pair' layout). create_pair wires them 1:1 by
    # coord and requires identical shapes.
    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(0, 0))
    receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 2), offset=ttnn.MeshCoordinate(1, 0))

    # Program cache lives on the dispatching device: the sender op caches on sender_mesh.
    sender_mesh.enable_program_cache()
    receiver_mesh.enable_program_cache()

    metadata_size_bytes = 0 if metadata_words is None else len(metadata_words) * 4

    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=_global_spec(),
        mapper=_replicate_mapper(sender_mesh),
        fifo_size_bytes=_FIFO_SIZE_BYTES,
        sender_worker_cores=_WORKER_CORES,
        receiver_worker_cores=_WORKER_CORES,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=metadata_size_bytes,
        share_fabric_links=False,  # OWN mode: forward without host wait/release_fabric_links
    )

    numel = _GLOBAL_SHAPE[-1] * _GLOBAL_SHAPE[-2]
    cache_after_first = None

    for it in range(2):
        # Distinct per-iteration iota so a stuck / mis-addressed / stale-cache transfer is
        # caught (a different base every iteration).
        torch_in = (torch.arange(numel, dtype=torch.int64) + it * 100_000).reshape(_GLOBAL_SHAPE).to(torch.int32)
        tt_in = _to_device(torch_in, sender_mesh)

        md_tensor = None
        if metadata_words is not None:
            md_torch = torch.tensor(metadata_words, dtype=torch.int64).to(torch.int32).reshape(1, 1, 1, -1)
            md_tensor = _to_device(md_torch, sender_mesh)

        # ---- op under test: non-blocking push into the sender backing + data_ready ----
        ttnn.experimental.deepseek_prefill.d2d_socket_sync(sender, tt_in, metadata=md_tensor)

        # ---- drain on the receiver (blocks until the forward lands) ----
        if metadata_words is None:
            recv = h2d_socket_sync(receiver, _WORKER_CORES)
            recv_md = None
        else:
            recv, recv_md = h2d_socket_sync(receiver, _WORKER_CORES, metadata_size_bytes=metadata_size_bytes)

        # ---- verify: replicate => every receiver device holds the original input ----
        for dev_t in ttnn.get_device_tensors(recv):
            got = ttnn.to_torch(dev_t).reshape(_GLOBAL_SHAPE).to(torch.int32)
            assert torch.equal(got, torch_in), f"iter {it}: data round-trip mismatch"

        if metadata_words is not None:
            for dev_md in ttnn.get_device_tensors(recv_md):
                got_md = ttnn.to_torch(dev_md).flatten().to(torch.int64).tolist()
                assert got_md == metadata_words, f"iter {it}: metadata mismatch {got_md} != {metadata_words}"

        ttnn.deallocate(tt_in)
        if md_tensor is not None:
            ttnn.deallocate(md_tensor)

        # Program cache: built once on iter 0, reused on iter 1 (input addr is a BufferBinding).
        n_cached = getattr(sender_mesh, "num_program_cache_entries", lambda: None)()
        if it == 0:
            cache_after_first = n_cached
        elif n_cached is not None:
            assert n_cached == cache_after_first, "d2d_socket_sync should hit the program cache on iter 2 (no rebuild)"
