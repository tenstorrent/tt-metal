# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time of one GLM-5.4 G1->G2 chunk over a D2D socket, measured between two halves of one 8x4 Galaxy.

An intra-Galaxy proxy for the SC4 inter-Galaxy seam: the 8x4 mesh opens whole (sub-mesh fabric opens fail on
Galaxies) and is split with ``create_submesh`` into a sender and a receiver half: 4x4 (rows 0-3 -> rows 4-7) or 8x2
(cols 0-1 -> cols 2-3), which puts twice as many chips, and so relay cables, on the seam. Segments
and socket settings follow the prefill runner's stage boundary: lease mode (push, grant both sides, drain, wait
both sides), an 8x8 worker grid, the minimal L1 FIFO, and a 12 B chunk record as metadata.

Payload of one chunk of ``tokens`` rows (5120 = one GLM-5.3 chunk), all segments rows over SP; ``full`` sends
all three on one socket, the others one segment alone:

* activation: bf16 TILE ``[1, 1, tokens, 6144]``, embedding over TP (the runner's D2D mapper);
* KVPE rows: ``[1, 28, tokens, 176]`` uint32 ROW_MAJOR = 704 B pages for the 28 loop layers, layers over TP
  (GLM TP-dedups its KVPE cache, so each row crosses once);
* top-k: ``[1, 7, tokens, 2048]`` uint32 ROW_MAJOR, k over TP (the runner's index mapper).

Relay mode is the deployed GLM path (only cable chips cross; here chips directly linked to the other half).
Each case prints one ``[d2d-cost] {json}`` line with eager and traced per-chunk times and the bytes moved; the
relay cable plan (exit -> entry chips) is in the C++ log. Every segment is checked bit-exact, eager and traced,
on a transfer into zeroed destinations outside the timed loop.
"""

import json
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.prefill_chunk import CHUNK_BYTES
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    fabric2d_device_params,
    model_fabric_payload,
    torus_xy_device_params,
)
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import CapturedCall

_HIDDEN = 6144  # GLM-5.3 / GLM-5.4 hidden size
_LOOP_LAYERS = 28  # L27-54, the layers whose KVPE rows G2 copies
_KVPE_PAGE_WORDS = 704 // 4  # scaled-FP8 KVPE page: 576 B latent+rope + scales
_TOPK_SETS = 7  # one top-k set per indexer reuse group of 4 loop layers
_TOPK = 2048
_WORKERS = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))  # the runner's d2d_worker_cores on Galaxy
_METADATA_BYTES = CHUNK_BYTES
_FIFO_BYTES = 256  # the runner's minimum: the L1-aligned chunk record
_WARMUP = 2
_ITERS = 10
_TRACE_REGION_BYTES = 32 * 1024 * 1024


def _mapper(mesh, tp_dim):
    return ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(tp_dim)])
    )


def _segments(payload, tokens):
    """(name, global shape, dtype, layout, TP-sharded dim) per segment."""
    segs = [
        ("activation", [1, 1, tokens, _HIDDEN], ttnn.bfloat16, ttnn.TILE_LAYOUT, 3),
        ("kvpe", [1, _LOOP_LAYERS, tokens, _KVPE_PAGE_WORDS], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 1),
        ("topk", [1, _TOPK_SETS, tokens, _TOPK], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 3),
    ]
    return segs if payload == "full" else [s for s in segs if s[0] == payload]


def _nbytes(shape, dtype):
    n = 1
    for d in shape:
        n *= d
    return n * (2 if dtype == ttnn.bfloat16 else 4)


def _host_data(shape, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    if dtype == ttnn.bfloat16:
        return torch.randn(shape, generator=g, dtype=torch.float32).to(torch.bfloat16)
    return torch.randint(-(2**31), 2**31 - 1, shape, generator=g, dtype=torch.int32)


def _to_mesh(host, mesh, dtype, layout, tp_dim):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh, tp_dim),
    )


def _record(mesh, words):
    return ttnn.from_torch(
        torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
        ),
    )


def _zero_fill(t, zeros_host):
    ttnn.copy_host_to_device_tensor(zeros_host, t)


def _assert_equal(sent, got, what):
    a_shards, b_shards = ttnn.get_device_tensors(sent), ttnn.get_device_tensors(got)
    assert len(a_shards) == len(b_shards), f"{what}: {len(b_shards)} shards != {len(a_shards)}"
    for k, (a, b) in enumerate(zip(a_shards, b_shards)):
        assert a.spec == b.spec, f"{what}: shard {k} spec {b.spec} != {a.spec}"
        assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"{what}: shard {k} differs"


def _stats(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": [round(s, 3) for s in samples],
    }


@pytest.mark.parametrize("tokens", [5120, 2560], ids=["t5120", "t2560"])
@pytest.mark.parametrize("payload", ["activation", "kvpe", "topk", "full"])
@pytest.mark.parametrize("split", ["rows", "cols"])
@pytest.mark.parametrize("relay", [True, False], ids=["relay", "direct"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=model_fabric_payload("glm_5_3"), trace_region_size=_TRACE_REGION_BYTES
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
        pytest.param(
            (8, 4),
            fabric2d_device_params(
                fabric_payload_size=model_fabric_payload("glm_5_3"), trace_region_size=_TRACE_REGION_BYTES
            ),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_glm54_d2d_payload_cost(mesh_device, device_params, split, relay, payload, tokens):
    if not is_blackhole():
        pytest.skip("D2DStreamService needs Blackhole service cores")
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    stamp = lambda msg: logger.info(f"[d2d-cost] {time.strftime('%H:%M:%S')} {msg}")
    # rows: 4x4 halves (rows 0-3 -> 4-7); cols: 8x2 halves (cols 0-1 -> 2-3), twice the chips facing the seam.
    shape = (rows // 2, cols) if split == "rows" else (rows, cols // 2)
    peer = ttnn.MeshCoordinate(rows // 2, 0) if split == "rows" else ttnn.MeshCoordinate(0, cols // 2)
    sender_mesh = mesh_device.create_submesh(ttnn.MeshShape(*shape), offset=ttnn.MeshCoordinate(0, 0))
    receiver_mesh = mesh_device.create_submesh(ttnn.MeshShape(*shape), offset=peer)
    sender_mesh.enable_program_cache()
    receiver_mesh.enable_program_cache()

    segs = _segments(payload, tokens)
    spec = lambda shape, dtype, layout: ttnn.TensorSpec(
        shape=ttnn.Shape(shape), dtype=dtype, layout=layout, buffer_type=ttnn.BufferType.DRAM
    )
    stamp(f"create_pair split={split} relay={relay} payload={payload} tokens={tokens}")
    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=spec(*segs[0][1:4]),
        mapper=_mapper(sender_mesh, segs[0][4]),
        fifo_size_bytes=_FIFO_BYTES,
        sender_worker_cores=_WORKERS,
        receiver_worker_cores=_WORKERS,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=_METADATA_BYTES,
        share_fabric_links=True,
        relay=relay,
        extra_tensors=[(spec(*s[1:4]), _mapper(sender_mesh, s[4])) for s in segs[1:]],
    )

    stamp("host data")
    hosts = [_host_data(s[1], s[2], seed=i) for i, s in enumerate(segs)]
    zeros = [
        ttnn.from_torch(
            torch.zeros(s[1], dtype=torch.bfloat16 if s[2] == ttnn.bfloat16 else torch.int32),
            dtype=s[2],
            layout=s[3],
            mesh_mapper=_mapper(receiver_mesh, s[4]),
        )
        for s in segs
    ]
    inputs = [_to_mesh(h, sender_mesh, s[2], s[3], s[4]) for h, s in zip(hosts, segs)]
    outputs = [_to_mesh(torch.zeros_like(h), receiver_mesh, s[2], s[3], s[4]) for h, s in zip(hosts, segs)]
    del hosts
    send_md = _record(sender_mesh, [0, 0, tokens])
    recv_md = _record(receiver_mesh, [0, 0, 0])
    payload_arg = inputs if len(inputs) > 1 else inputs[0]

    def push():
        ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(sender, payload_arg, metadata=send_md)

    def drain():
        ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            receiver,
            metadata_size_bytes=_METADATA_BYTES,
            metadata_out=recv_md,
            tokens_out=outputs[0],
            extras_out=outputs[1:],
        )

    def transfer(send, recv):
        # The runner's order at a stage boundary: push, grant the links on both sides, drain, wait both sides.
        t0 = time.perf_counter()
        send()
        sender.release_fabric_links()
        receiver.release_fabric_links()
        recv()
        sender.wait_for_fabric_links()
        receiver.wait_for_fabric_links()
        ttnn.synchronize_device(sender_mesh)
        ttnn.synchronize_device(receiver_mesh)
        return (time.perf_counter() - t0) * 1e3

    def check(mode):
        for t, z in zip(outputs, zeros):
            _zero_fill(t, z)
        ttnn.synchronize_device(receiver_mesh)
        transfer(*mode)
        for (name, *_), a, b in zip(segs, inputs, outputs):
            _assert_equal(a, b, f"{name}")
        got_md = ttnn.to_torch(ttnn.get_device_tensors(recv_md)[0]).flatten().tolist()
        assert got_md == [0, 0, tokens], f"metadata {got_md}"

    send_trace = recv_trace = None
    try:
        stamp("eager warmup")
        for _ in range(_WARMUP):
            transfer(push, drain)
        stamp("eager timed")
        eager = [transfer(push, drain) for _ in range(_ITERS)]
        stamp(f"eager median {statistics.median(eager):.3f} ms; checking")
        check((push, drain))

        stamp("trace capture")
        send_trace, recv_trace = CapturedCall(sender_mesh), CapturedCall(receiver_mesh)
        send_trace.capture(push)
        recv_trace.capture(drain)
        traced_mode = (send_trace.replay, recv_trace.replay)
        for _ in range(_WARMUP):
            transfer(*traced_mode)
        stamp("traced timed")
        traced = [transfer(*traced_mode) for _ in range(_ITERS)]
        stamp(f"traced median {statistics.median(traced):.3f} ms; checking")
        check(traced_mode)

        seg_bytes = {name: _nbytes(shape, dtype) for name, shape, dtype, *_ in segs}
        total = sum(seg_bytes.values())
        result = {
            "fabric": str(device_params.get("fabric_config")),
            "relay": relay,
            "payload": payload,
            "tokens": tokens,
            "split": split,
            "half_shape": list(shape),
            "segment_bytes": seg_bytes,
            "total_bytes": total,
            "eager": _stats(eager),
            "traced": _stats(traced),
            "traced_GBps": total / (statistics.median(traced) * 1e-3) / 1e9,
        }
        print("[d2d-cost] " + json.dumps(result), flush=True)
    finally:
        for c in (send_trace, recv_trace):
            if c is not None:
                c.release()
        for t in inputs + outputs + [send_md, recv_md]:
            ttnn.deallocate(t)
        del receiver
        del sender
