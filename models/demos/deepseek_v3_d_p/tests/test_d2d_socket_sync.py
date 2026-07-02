# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Op test for ``ttnn.experimental.deepseek_prefill.outbound_socket_service_sync`` (the D2D sender op).

Splits the 8x4 Galaxy into a 4x4 sender stage (rows 0-3) and a 4x4 receiver stage (rows
4-7) and streams one sharded activation chunk between them over tt-fabric: per-chip shard
[640 tokens, 1792 hidden] uint32 ROW_MAJOR DRAM ([2560, 7168] total). ``create_pair`` runs
in OWN mode; ``outbound_socket_service_sync`` pushes it (non-blocking, 2x2 worker grid) and the native
``inbound_socket_service_sync`` (D2DStreamServiceReceiver overload) drains it. Asserts each receiver
shard equals its wired sender shard (coord (r,c)->(r,c)) bit-exact, plus metadata; a 2nd
iteration checks the program cache (input address is a per-dispatch BufferBinding).

uint32 is a V0 wire-format stand-in for the real bf16 activation.
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
    rows_total, cols = mesh_device.shape[0], mesh_device.shape[1]
    if rows_total < 2 or rows_total % 2 != 0:
        pytest.skip(f"need an even row count to split into sender/receiver halves, got {mesh_device.shape}")
    if _HIDDEN % cols != 0:
        pytest.skip(f"hidden {_HIDDEN} must shard evenly across {cols} columns")

    # Top half = sender stage, bottom half = receiver stage. create_pair requires equal
    # shapes and wires chip (r,c) on the sender 1:1 to chip (r,c) on the receiver.
    half = rows_total // 2
    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(0, 0))
    receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(half, 0))

    sender_mesh.enable_program_cache()
    receiver_mesh.enable_program_cache()

    g_tokens = half * _TOKENS_PER_ROW  # full per-stage chunk length (e.g. 4 * 640 = 2560)
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
            torch_in = (torch.arange(numel, dtype=torch.int32) + it * 100_000).reshape(1, 1, g_tokens, _HIDDEN)
            tt_in = _to_device(torch_in, sender_mesh, _shard_mapper(sender_mesh))

            md_tensor = None
            if metadata_words is not None:
                md_torch = torch.tensor(metadata_words, dtype=torch.int32).reshape(1, 1, 1, -1)
                md_tensor = _to_device(md_torch, sender_mesh, _replicate_mapper(sender_mesh))

            # non-blocking push of the sharded activation
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(sender, tt_in, metadata=md_tensor)

            # drain on the receiver
            drained = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                receiver, metadata_size_bytes=metadata_size_bytes
            )
            recv = drained[0]
            recv_md = drained[1] if metadata_words is not None else None

            # compare
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

            # Program cache: built once on iter 0, reused on iter 1 (input addr is a BufferBinding)
            n_cached = getattr(sender_mesh, "num_program_cache_entries", lambda: None)()
            if it == 0:
                cache_after_first = n_cached
            elif n_cached is not None:
                assert (
                    n_cached == cache_after_first
                ), "outbound_socket_service_sync should hit the program cache on iter 2 (no rebuild)"
    finally:
        del receiver
        del sender
