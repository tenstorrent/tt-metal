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

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params

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
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
        # 4-chip QuietBox variant: 2 rows -> 1 sender row + 1 receiver row, 2 cols. Per-shard
        # page = (7168/2)*4 = 14336 B (< the 16384 B FIFO), so the same op path runs on a
        # small multi-card box (no 32-chip Galaxy needed).
        pytest.param(
            (2, 2),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric2d-2x2",
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
            print(f"[d2d cache] iter {it}: sender program-cache entries = {n_cached}")
            if it == 0:
                cache_after_first = n_cached
            elif n_cached is not None:
                assert (
                    n_cached == cache_after_first
                ), "outbound_socket_service_sync should hit the program cache on iter 2 (no rebuild)"
    finally:
        del receiver
        del sender


@pytest.mark.parametrize("relay", [False, True], ids=["direct", "relay"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_d2d_socket_sync_shared_links(mesh_device, relay):
    """Share mode (the prefill runner's mode), with and without relay mode.

    Same transfer as test_d2d_socket_sync, but the host hands out the fabric links per
    transfer (release_fabric_links / wait_for_fabric_links). With relay=True, only the
    row-3 chips (the cable chips) send across rows 3 -> 4; the other senders go through them.
    """
    if not is_blackhole():
        pytest.skip("D2DStreamService needs Blackhole (or UBB Galaxy) service cores")
    rows_total, cols = mesh_device.shape[0], mesh_device.shape[1]
    half = rows_total // 2
    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(0, 0))
    receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(half, cols), offset=ttnn.MeshCoordinate(half, 0))

    g_tokens = half * _TOKENS_PER_ROW
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, g_tokens, _HIDDEN]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    metadata_words = [7, 128, 256, 1]
    metadata_size_bytes = len(metadata_words) * 4

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
        share_fabric_links=True,
        relay=relay,
    )

    numel = g_tokens * _HIDDEN
    try:
        # 3 transfers: the relay slots and credits must be reused correctly after the first.
        for it in range(3):
            torch_in = (torch.arange(numel, dtype=torch.int32) + it * 100_000).reshape(1, 1, g_tokens, _HIDDEN)
            tt_in = _to_device(torch_in, sender_mesh, _shard_mapper(sender_mesh))
            md_torch = torch.tensor(metadata_words, dtype=torch.int32).reshape(1, 1, 1, -1) + it
            md_tensor = _to_device(md_torch, sender_mesh, _replicate_mapper(sender_mesh))

            # Same order as prefill_runner.py: push, then grant the links on both sides.
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(sender, tt_in, metadata=md_tensor)
            sender.release_fabric_links()
            receiver.release_fabric_links()
            recv, recv_md = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                receiver, metadata_size_bytes=metadata_size_bytes
            )
            sender.wait_for_fabric_links()
            receiver.wait_for_fabric_links()

            in_shards = ttnn.get_device_tensors(tt_in)
            out_shards = ttnn.get_device_tensors(recv)
            assert len(in_shards) == len(out_shards)
            for k, (a, b) in enumerate(zip(in_shards, out_shards)):
                ta = ttnn.to_torch(a).to(torch.int32)
                tb = ttnn.to_torch(b).to(torch.int32)
                assert torch.equal(ta, tb), f"relay={relay} iter {it}: shard {k} mismatch"
            for dev_md in ttnn.get_device_tensors(recv_md):
                got_md = ttnn.to_torch(dev_md).flatten().to(torch.int64).tolist()
                want = [w + it for w in metadata_words]
                assert got_md == want, f"relay={relay} iter {it}: metadata {got_md} != {want}"
            print(f"[d2d shared-links] relay={relay} iter {it}: {len(out_shards)} shards bit-exact")

            ttnn.deallocate(tt_in)
            ttnn.deallocate(md_tensor)
    finally:
        del receiver
        del sender


@pytest.mark.parametrize("relay", [False, True], ids=["direct", "relay"])
# (first sender row, first receiver row, rows per side). Every column sends straight down,
# so all flows of a column use the same Y links. 3->4 and 0->7 have one flow per link (1 and
# 7 hops); 2-3->4-5 and 0-3->4-7 put 2 and 4 flows on the link between rows 3 and 4.
@pytest.mark.parametrize(
    "rows",
    [(0, 4, 4), (3, 4, 1), (0, 7, 1), (2, 4, 2)],
    ids=["rows0-3_to_4-7", "row3_to_4", "row0_to_7", "rows2-3_to_4-5"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_d2d_socket_sync_shared_links_timing(mesh_device, relay, rows):
    """Host-measured time of one share-mode transfer, with and without relay mode.

    Sizes follow prefill_runner.py: 8x8 worker grid, 256 B socket FIFO, about 2 MB per chip
    (640 rows x 768 uint32 words per chip). One timed transfer = push, grant links on both
    sides, drain, and wait until both meshes are idle. Two inputs with different values are
    sent in turn and every timed transfer is checked, so a transfer that did not arrive fails.
    """
    import time

    if not is_blackhole():
        pytest.skip("D2DStreamService needs Blackhole (or UBB Galaxy) service cores")
    send_row0, recv_row0, n_rows = rows
    cols = mesh_device.shape[1]
    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(n_rows, cols), offset=ttnn.MeshCoordinate(send_row0, 0))
    receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(n_rows, cols), offset=ttnn.MeshCoordinate(recv_row0, 0))

    hidden = 768 * cols
    g_tokens = n_rows * _TOKENS_PER_ROW
    workers = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))
    global_spec = ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, g_tokens, hidden]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )
    metadata_size_bytes = 12
    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=global_spec,
        mapper=_shard_mapper(sender_mesh),
        fifo_size_bytes=256,
        sender_worker_cores=workers,
        receiver_worker_cores=workers,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=metadata_size_bytes,
        share_fabric_links=True,
        relay=relay,
    )

    torch_a = torch.arange(g_tokens * hidden, dtype=torch.int32).reshape(1, 1, g_tokens, hidden)
    torch_ins = [torch_a, -torch_a - 1]  # every word differs between the two inputs
    tt_ins = [_to_device(t, sender_mesh, _shard_mapper(sender_mesh)) for t in torch_ins]
    want_shards = [[ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)] for t in tt_ins]
    # 30 samples were too few: one relay case gave p50 1.215 ms in one run and 0.788 ms in the next.
    warmup = int(os.environ.get("D2D_TIMING_WARMUP", "3"))
    timed = int(os.environ.get("D2D_TIMING_ITERS", "30"))
    times_ms, recv_ms = [], []  # until both meshes are idle / until the receiver mesh is idle
    try:
        for it in range(warmup + timed):
            md_words = [it, it + 1, it + 2]
            md_tensor = _to_device(
                torch.tensor(md_words, dtype=torch.int32).reshape(1, 1, 1, -1),
                sender_mesh,
                _replicate_mapper(sender_mesh),
            )
            ttnn.synchronize_device(sender_mesh)
            t0 = time.perf_counter()
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(sender, tt_ins[it % 2], metadata=md_tensor)
            sender.release_fabric_links()
            receiver.release_fabric_links()
            recv, recv_md = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
                receiver, metadata_size_bytes=metadata_size_bytes
            )
            sender.wait_for_fabric_links()
            receiver.wait_for_fabric_links()
            # wait_for_fabric_links only enqueues; wait for both meshes so the sender tail is timed too
            ttnn.synchronize_device(receiver_mesh)
            t_recv = time.perf_counter()
            ttnn.synchronize_device(sender_mesh)
            if it >= warmup:
                times_ms.append((time.perf_counter() - t0) * 1000.0)
                recv_ms.append((t_recv - t0) * 1000.0)
                for k, (want, got) in enumerate(zip(want_shards[it % 2], ttnn.get_device_tensors(recv))):
                    assert torch.equal(want, ttnn.to_torch(got)), f"relay={relay} iter {it}: shard {k} mismatch"
                for k, dev_md in enumerate(ttnn.get_device_tensors(recv_md)):
                    got_md = ttnn.to_torch(dev_md).flatten().to(torch.int64).tolist()
                    assert got_md == md_words, f"relay={relay} iter {it}: shard {k} metadata {got_md} != {md_words}"
            ttnn.deallocate(md_tensor)
        times_ms.sort()
        recv_ms.sort()
        mb = g_tokens * hidden * 4 / 1e6
        p50 = times_ms[timed // 2]
        print(
            f"[d2d timing] relay={relay} receiver mesh idle: min={recv_ms[0]:.3f} p50={recv_ms[timed // 2]:.3f} "
            f"p90={recv_ms[int(timed * 0.9)]:.3f} ms"
        )
        print(
            f"[d2d timing] relay={relay} rows={send_row0}..{send_row0 + n_rows - 1}->{recv_row0}..{recv_row0 + n_rows - 1} "
            f"total={mb:.1f} MB n={timed} min={times_ms[0]:.3f} p50={p50:.3f} p90={times_ms[int(timed * 0.9)]:.3f} "
            f"max={times_ms[-1]:.3f} ms, per column {mb / cols / p50:.2f} GB/s at p50"
        )
    finally:
        for t in tt_ins:
            ttnn.deallocate(t)
        del receiver
        del sender
