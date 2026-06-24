# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op test for ``ttnn.experimental.deepseek_prefill.d2d_socket_sync`` (the D2D sender op).

Realistic pipeline-parallel shape: a sender stage streams one chunk of activations to a
receiver stage over tt-fabric. Parametrized by stage size (``submesh_rows``) — two stacked
submeshes carved from the mesh:

  * ``stage-1row``: sender = row 0 (1xC), receiver = row 1 (1xC). Small footprint — only
    2 mesh rows are used for the transfer (the rest of the 8x4 stays idle). 640 tokens.
  * ``stage-half``: sender = top half, receiver = bottom half (rows 0-3 / 4-7 on 8x4).
    Full-mesh case. 2560 tokens (4 rows * 640).

Each mesh row carries ``_TOKENS_PER_ROW`` (640) tokens; the hidden dim is sharded across
the columns, so the per-chip shard is always [640 tokens, 1792 hidden] uint32 ROW_MAJOR DRAM.
NOTE: the 8x4 mesh is always OPENED (conftest's requires_mesh_topology mandates all 32
Blackhole devices); ``stage-1row`` just confines the D2D transfer to 2 rows.

Flow (mirrors the gtest create_pair flow, but drives the real op):
  1. Carve sender + receiver submeshes from the 8x4 mesh.
  2. ``D2DStreamService.create_pair`` in OWN mode (``share_fabric_links=False``).
  3. ``d2d_socket_sync`` (op under test) pushes the sharded activation: copy -> sender
     backing, inc data_ready, return (NON-BLOCKING). Pages split across a 2x2 worker grid.
  4. The native ``h2d_socket_sync`` (D2DStreamServiceReceiver overload) drains it on the
     receiver (blocks until the forward lands).
  5. Assert each receiver chip's shard equals the sender chip's shard it is wired to
     (create_pair wires coord (r,c) -> (r,c)), bit-exact; (optionally) metadata survives.

A second iteration with fresh data exercises the program cache (built once; the input
address is patched per dispatch as a ``Buffer*`` BufferBinding).

NOTE: the D2D service V0 wire format is UINT32 / ROW_MAJOR / DRAM, so the activation is
uint32 here — a stand-in for the real bf16 hidden state; the SHAPE is the realistic part.
Requires Blackhole service cores + a FABRIC_2D mesh; skips cleanly elsewhere.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

_TOKENS_PER_ROW = 640  # tokens of the chunk carried per mesh row (the realistic knob)
_HIDDEN = 7168  # DeepSeek-V3 / Kimi-K2 hidden size, sharded across the mesh columns
# 2x2 producer/consumer worker grid -> the 640 backing pages split 160/core (multi-worker path).
_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))
# Socket FIFO must hold >= one per-shard page. Per-shard page = (7168/4 cols) * 4 B = 7168 B.
_FIFO_SIZE_BYTES = 16384


def _shard_mapper(mesh):
    # SP x TP layout: shard tokens (tensor dim 2) across mesh rows and hidden (dim 3)
    # across mesh columns. create_pair wires shard (r,c) -> (r,c).
    return ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
    )


def _replicate_mapper(mesh):
    # The tiny metadata blob is replicated to every chip.
    return ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
    )


def _to_device(torch_u32, mesh, mapper):
    return ttnn.from_torch(
        torch_u32,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


@pytest.mark.parametrize("submesh_rows", [1, None], ids=["stage-1row", "stage-half"])
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
def test_d2d_socket_sync(mesh_device, metadata_words, submesh_rows):
    if not is_blackhole():
        pytest.skip("D2DStreamService needs Blackhole (or UBB Galaxy) service cores")
    rows_total, cols = mesh_device.shape[0], mesh_device.shape[1]
    if _HIDDEN % cols != 0:
        pytest.skip(f"hidden {_HIDDEN} must shard evenly across {cols} columns")

    # Rows per stage. None => half the mesh (full-mesh case); 1 => a 1xC sender (row 0) +
    # 1xC receiver (row 1) -- small footprint. The two stages are stacked: sender rows
    # [0, R), receiver rows [R, 2R). create_pair requires equal shapes and wires chip
    # (r,c) on the sender 1:1 to (r,c) on the receiver.
    stage_rows = submesh_rows or (rows_total // 2)
    if 2 * stage_rows > rows_total:
        pytest.skip(f"need >= {2 * stage_rows} rows for two {stage_rows}-row stages, got {rows_total}")

    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(stage_rows, cols), offset=ttnn.MeshCoordinate(0, 0))
    receiver_mesh = mesh_device.create_submesh(
        ttnn.MeshShape(stage_rows, cols), offset=ttnn.MeshCoordinate(stage_rows, 0)
    )

    sender_mesh.enable_program_cache()
    receiver_mesh.enable_program_cache()

    g_tokens = stage_rows * _TOKENS_PER_ROW  # per-stage chunk length (640/row): 1 row=>640, 4 rows=>2560
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, g_tokens, _HIDDEN]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )

    metadata_size_bytes = 0 if metadata_words is None else len(metadata_words) * 4

    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=global_spec,
        mapper=_shard_mapper(sender_mesh),
        fifo_size_bytes=_FIFO_SIZE_BYTES,
        sender_worker_cores=_WORKER_CORES,
        receiver_worker_cores=_WORKER_CORES,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=metadata_size_bytes,
        share_fabric_links=False,  # OWN mode: forward without host wait/release_fabric_links
    )

    numel = g_tokens * _HIDDEN
    cache_after_first = None

    try:
        for it in range(2):
            # Distinct per-iteration iota (different base each iter) so a stuck / mis-addressed
            # / stale-cache transfer reads back wrong values. Stays well within int32.
            torch_in = (torch.arange(numel, dtype=torch.int32) + it * 100_000).reshape(1, 1, g_tokens, _HIDDEN)
            tt_in = _to_device(torch_in, sender_mesh, _shard_mapper(sender_mesh))

            md_tensor = None
            if metadata_words is not None:
                md_torch = torch.tensor(metadata_words, dtype=torch.int32).reshape(1, 1, 1, -1)
                md_tensor = _to_device(md_torch, sender_mesh, _replicate_mapper(sender_mesh))

            # ---- op under test: non-blocking push of the sharded activation ----
            ttnn.experimental.deepseek_prefill.d2d_socket_sync(sender, tt_in, metadata=md_tensor)

            # ---- drain on the receiver: the native h2d_socket_sync, D2DStreamServiceReceiver
            #      overload (blocks until the forward lands). It derives worker_cores from the
            #      service and returns a list -> [tokens] or [tokens, metadata]. ----
            drained = ttnn.experimental.deepseek_prefill.h2d_socket_sync(
                receiver, metadata_size_bytes=metadata_size_bytes
            )
            recv = drained[0]
            recv_md = drained[1] if metadata_words is not None else None

            # ---- verify per chip: receiver shard (r,c) == sender shard (r,c) it was wired to.
            #      Comparing device input vs device output is layout-agnostic and tests the
            #      coord-to-coord fabric wiring directly. ----
            in_shards = ttnn.get_device_tensors(tt_in)
            out_shards = ttnn.get_device_tensors(recv)
            assert len(in_shards) == len(out_shards), f"shard count {len(in_shards)} != {len(out_shards)}"
            for k, (a, b) in enumerate(zip(in_shards, out_shards)):
                ta = ttnn.to_torch(a).to(torch.int32)
                tb = ttnn.to_torch(b).to(torch.int32)
                assert torch.equal(ta, tb), f"iter {it}: shard {k} round-trip mismatch"

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
                assert (
                    n_cached == cache_after_first
                ), "d2d_socket_sync should hit the program cache on iter 2 (no rebuild)"
    finally:
        # Drop the service handles BEFORE the submeshes/mesh tear down: each destructor
        # finishes the command queue, terminates its persistent kernel, releases the
        # claimed service core, and resets the MeshSocket -- all of which need the submesh
        # + its command queue still alive (else GC frees the queues first and the
        # destructors fatal with "cq_id 0 is out of range"). See the `del endpoint` before
        # close_device in tests/ttnn/unit_tests/base_functionality/test_d2d_stream_service_multiprocess.py.
        del receiver
        del sender
