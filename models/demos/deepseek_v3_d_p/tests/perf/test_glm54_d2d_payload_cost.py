# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time of one GLM-5.4 G1->G2 chunk over a D2D socket, measured between two halves of one 8x4 Galaxy.

An intra-Galaxy proxy for the SC4 inter-Galaxy seam: the 8x4 mesh opens whole (sub-mesh fabric opens fail on
Galaxies) and is split with ``create_submesh`` into a sender and a receiver half: 4x4 (rows 0-3 -> rows 4-7) or 8x2
(cols 0-1 -> cols 2-3), which puts twice as many chips, and so relay cables, on the seam. Segments
and socket settings follow the prefill runner's stage boundary: lease mode (push, grant both sides, drain, wait
both sides), an 8x8 worker grid, the minimal L1 FIFO, and a 12 B chunk record as metadata.

Payload of one chunk of ``tokens`` rows (5120 = one GLM-5.3 chunk), dim 2 over SP and dim 3 (or 1) over TP;
``full`` sends every segment on one socket, the others one kind alone:

* activation: bf16 TILE ``[1, 1, tokens, 6144]``, embedding over TP (the runner's D2D mapper);
* KVPE rows: ``[1, 28, tokens, 176]`` uint32 ROW_MAJOR = 704 B pages for the 28 loop layers, layers over TP
  (GLM TP-dedups its KVPE cache, so each row crosses once). The ``kvpe_pack`` cases (q171.3.20) send the handoff's
  own 656 B FP8 row (164 words) with ``m`` consecutive rows of a layer per page, ``[1, 28, tokens / m, 164 m]``: the
  same bytes in 656 m B pages;
* top-k: 7 segments, one per set, ``[1, 1, tokens // tp, tp * 2048]`` uint32 ROW_MAJOR: the runner's index
  layout (runner_utils.indexer_indices_global_spec), each chip holding its own rows with the full k = 2048 (8 KB
  pages).

Relay mode is the deployed GLM path (only cable chips cross; here chips directly linked to the other half).
Each case prints one ``[d2d-cost] {json}`` line with eager and traced per-chunk times and the bytes moved; the
relay cable plan (exit -> entry chips) is in the C++ log. Every segment is checked bit-exact, eager and traced,
on a transfer into zeroed destinations outside the timed loop.
"""

import json
import math
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
_PAYLOADS = ("activation", "kvpe", "topk", "full")
_HANDOFF_ROW_WORDS = 656 // 4  # the handoff's KVPE row: one stage-cache row (scaled FP8, 656 B wide)
_KVPE_ROWS_PER_PAGE = (1, 16, 32, 160)


def _mapper(mesh, tp_dim):
    return ttnn.create_mesh_mapper(
        mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(tp_dim)])
    )


def _segments(payload, tokens, tp, kvpe_pack=(_KVPE_PAGE_WORDS, 1)):
    """(kind, name, global shape, dtype, layout, TP-sharded dim) per segment. ``kvpe_pack`` = (row words, rows per
    page) of the KVPE segment."""
    row_words, rows_per_page = kvpe_pack
    assert tokens % rows_per_page == 0, f"{tokens} rows do not split into pages of {rows_per_page}"
    kvpe_shape = [1, _LOOP_LAYERS, tokens // rows_per_page, row_words * rows_per_page]
    segs = [
        ("activation", "activation", [1, 1, tokens, _HIDDEN], ttnn.bfloat16, ttnn.TILE_LAYOUT, 3),
        ("kvpe", "kvpe", kvpe_shape, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 1),
    ]
    segs += [
        ("topk", f"topk{i}", [1, 1, tokens // tp, tp * _TOPK], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, 3)
        for i in range(_TOPK_SETS)
    ]
    return segs if payload == "full" else [s for s in segs if s[0] == payload]


def _nbytes(shape, dtype):
    return math.prod(shape) * (2 if dtype == ttnn.bfloat16 else 4)


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


def _case(fabric, split, relay, payload, tokens=5120, kvpe_pack=None):
    profile = torus_xy_device_params if fabric == "torus-xy" else fabric2d_device_params
    return pytest.param(
        (8, 4),
        profile(fabric_payload_size=model_fabric_payload("glm_5_3"), trace_region_size=_TRACE_REGION_BYTES),
        split,
        relay,
        payload,
        tokens,
        kvpe_pack or (_KVPE_PAGE_WORDS, 1),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id=f"{fabric}-{split}-{'relay' if relay else 'direct'}-{payload}-t{tokens}"
        + (f"-row{kvpe_pack[0] * 4}B-x{kvpe_pack[1]}" if kvpe_pack else ""),
    )


# Relay (the deployed GLM path) per payload on 4 cables (rows) and 8 cables (cols); the 2560-token pair checks
# linearity in bytes; direct mode is context; one Fabric2D case checks the fabric profile does not matter.
_CASES = [
    *[_case("torus-xy", split, True, payload) for split in ("rows", "cols") for payload in _PAYLOADS],
    *[_case("torus-xy", "rows", True, payload, 2560) for payload in ("activation", "full")],
    *[_case("torus-xy", split, False, "full") for split in ("rows", "cols")],
    _case("fabric2d", "rows", True, "full"),
    # q171.3.20: the handoff's 656 B KVPE row, 1 (as sent today) or m rows per page, on the deployed 4-cable relay.
    *[
        _case("torus-xy", "rows", True, payload, kvpe_pack=(_HANDOFF_ROW_WORDS, m))
        for m in _KVPE_ROWS_PER_PAGE
        for payload in ("kvpe", "full")
    ],
]


@pytest.mark.parametrize(
    "mesh_device, device_params, split, relay, payload, tokens, kvpe_pack",
    _CASES,
    indirect=["mesh_device", "device_params"],
)
def test_glm54_d2d_payload_cost(mesh_device, device_params, split, relay, payload, tokens, kvpe_pack):
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

    segs = _segments(payload, tokens, tp=shape[1], kvpe_pack=kvpe_pack)
    spec = lambda shape, dtype, layout: ttnn.TensorSpec(
        shape=ttnn.Shape(shape), dtype=dtype, layout=layout, buffer_type=ttnn.BufferType.DRAM
    )
    stamp(f"create_pair split={split} relay={relay} payload={payload} tokens={tokens}")
    sender, receiver = ttnn.D2DStreamService.create_pair(
        sender_mesh=sender_mesh,
        receiver_mesh=receiver_mesh,
        global_spec=spec(*segs[0][2:5]),
        mapper=_mapper(sender_mesh, segs[0][5]),
        fifo_size_bytes=_FIFO_BYTES,
        sender_worker_cores=_WORKERS,
        receiver_worker_cores=_WORKERS,
        socket_buffer_type=ttnn.BufferType.L1,
        metadata_size_bytes=_METADATA_BYTES,
        share_fabric_links=True,
        relay=relay,
        extra_tensors=[(spec(*s[2:5]), _mapper(sender_mesh, s[5])) for s in segs[1:]],
    )

    stamp("host data")
    hosts = [_host_data(s[2], s[3], seed=i) for i, s in enumerate(segs)]
    zeros = [
        ttnn.from_torch(torch.zeros_like(h), dtype=s[3], layout=s[4], mesh_mapper=_mapper(receiver_mesh, s[5]))
        for h, s in zip(hosts, segs)
    ]
    inputs = [_to_mesh(h, sender_mesh, *s[3:6]) for h, s in zip(hosts, segs)]
    outputs = [_to_mesh(torch.zeros_like(h), receiver_mesh, *s[3:6]) for h, s in zip(hosts, segs)]
    del hosts
    send_md = _record(sender_mesh, [0, 0, tokens])
    recv_md = _record(receiver_mesh, [0, 0, 0])
    zero_md = ttnn.from_torch(
        torch.zeros(1, 1, 1, 3, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.create_mesh_mapper(
            receiver_mesh, ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
        ),
    )
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
        # Zero every destination, metadata included, so a transfer that delivers nothing fails the checks.
        for t, z in zip(outputs + [recv_md], zeros + [zero_md]):
            ttnn.copy_host_to_device_tensor(z, t)
        ttnn.synchronize_device(receiver_mesh)
        transfer(*mode)
        for (_, name, *_), a, b in zip(segs, inputs, outputs):
            _assert_equal(a, b, name)
        for dev_md in ttnn.get_device_tensors(recv_md):
            got_md = ttnn.to_torch(dev_md).flatten().tolist()
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

        seg_bytes = {name: _nbytes(shape, dtype) for _, name, shape, dtype, *_ in segs}
        total = sum(seg_bytes.values())
        result = {
            "fabric": str(device_params.get("fabric_config")),
            "relay": relay,
            "payload": payload,
            "tokens": tokens,
            "kvpe_pack": {"row_bytes": kvpe_pack[0] * 4, "rows_per_page": kvpe_pack[1]},
            "split": split,
            "half_shape": list(shape),
            "segment_shapes": {name: shape for _, name, shape, *_ in segs},
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
