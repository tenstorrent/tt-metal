# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trace capture / replay of DeepSeek-V4.1 prefill as the perf-measurement mode (dev-spec D-G (b), beads 8y7.9, 8y7.9.2).

Contract: the prefill forward writes nothing from the host (position tables are device slices of
``cache.V41ChunkTables``; checked by counting host writes over a steady-state forward, which must be zero), so it is
captured as is. For the same inputs and starting state a traced replay produces outputs and cache contents
bit-identical to the untraced forward, replays are deterministic, and the replay's wall time is the device time
without host dispatch (reported with the untraced time). The MoE's shared-expert / dispatch overlap loads a
sub-device manager mid-forward, which a trace cannot contain: ``TtV41Moe.set_trace_controller`` routes it through the
in-tree ``SubDeviceTraceController``, which splits the capture there (the replay is several trace segments).

Cases: one block (small dims and real production weights) and the transformer's chunk loop (small dims, several
chunks, a padded last chunk: embedding, every block, cache writes, state advance; the host-side token upload and
logits readback stay outside the capture).
"""

import json
import time
from dataclasses import asdict

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import expert_dtype_reference as R
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import _pack
from models.demos.deepseek_v3_d_p.tests.v41.test_transformer_v41 import WEIGHT_CACHE
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_expert_dtype import BLOCK_CONFIG, _weights
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.mhc import initial_pre_mix
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController

TRACE_REGION = 512 * 1024 * 1024
TIMED_ITERS = 5
SMALL_SEQ = 512
MESH = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(trace_region_size=TRACE_REGION),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]


class HostWriteGuard:
    """Counts the host writes (``ttnn.from_torch``; ``ttnn.full`` on a device, which fills on the host) and host reads
    (``ttnn.to_torch``) made while ``active``; a traced region may contain none."""

    def __init__(self, monkeypatch):
        self.active, self.calls = False, []
        real = {"from_torch": ttnn.from_torch, "full": ttnn.full, "to_torch": ttnn.to_torch}

        def wrap(name):
            def call(*args, **kwargs):
                if self.active and (name != "full" or kwargs.get("device") is not None):
                    self.calls.append(name)
                return real[name](*args, **kwargs)

            return call

        for name in real:
            monkeypatch.setattr(ttnn, name, wrap(name))

    def count(self, forward):
        """Run ``forward()`` untraced and return the host transfers it made."""
        self.active, self.calls = True, []
        try:
            forward()
        finally:
            self.active = False
        return list(self.calls)


def capture(mesh_device, forward, moes):
    """Capture ``forward()`` (the capture is always closed) with the MoEs' sub-device switches split out.
    Returns (controller, outputs); ``controller.replay()`` runs the whole forward."""
    controller = SubDeviceTraceController(mesh_device)
    for moe in moes:
        moe.set_trace_controller(controller)
    ttnn.synchronize_device(mesh_device)
    controller.begin_capture()
    try:
        outs = forward()
    finally:
        controller.end_capture()
        for moe in moes:
            moe.set_trace_controller(None)
    return controller, outs


def _timed_ms(mesh_device, fn, iters=TIMED_ITERS) -> float:
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) * 1e3 / iters


def _state_tensors(state) -> dict:
    """Every cache tensor of ``state`` (KV, index-K, window carries)."""
    tensors = {}
    for name in ("kv", "index_k", "window_carry"):
        for key, t in getattr(state, name, {}).items():
            if t is not None:
                tensors[f"{name}[{key}]"] = t
    return tensors


def _block_small(mesh_device, device_params, layer: int):
    spec = small_spec((layer,), SMALL_SEQ)
    reference = orc.build_reference(spec)
    result = orc.oracle(spec, orc.random_tokens(spec), model=reference)
    block = TtV41Block(
        mesh_device,
        SmallV41Config,
        layer,
        device_weights(reference, 0),
        SMALL_SEQ,
    )
    return block, SmallV41Config, SMALL_SEQ, result["blocks"][layer]


def _block_real(mesh_device, device_params, layer: int):
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    spec = R.block_spec(layer)
    result = orc.oracle(spec, orc.random_tokens(spec))
    root, w, marker = _weights(ckpt, layer, "bfp8")
    block = TtV41Block(
        mesh_device,
        BLOCK_CONFIG,
        layer,
        w,
        R.ORACLE_SEQ,
        weight_cache_path=root,
    )
    marker.touch()
    return block, BLOCK_CONFIG, R.ORACLE_SEQ, result["blocks"][layer]


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("layer", [2, 0, 20], ids=["L2", "L0", "L20"])
@pytest.mark.parametrize("weights", ["small", "real"])
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_block_trace(mesh_device, device_params, weights, layer, monkeypatch):
    start = time.perf_counter()
    build = _block_small if weights == "small" else _block_real
    block, cfg, seq, rec = build(mesh_device, device_params, layer)
    logger.info(f"block L{layer} ({weights}) built {time.perf_counter() - start:.1f}s")
    shape = tuple(mesh_device.shape)
    tp = shape[1]

    def to_device(t, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=dims),
        )

    x = to_device(_pack(rec["x_in"].float(), tp), (2, 3))
    pre = to_device(rec["pre_in"].float()[None, None], (2, None))
    fresh = lambda: V41PrefillState(mesh_device, cfg, seq, seq, [layer], kv_format=MlaKvCacheFormat.BF16_RM)
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))

    def snapshot(outs, state):
        """Outputs and every cache tensor of ``state`` on host (full per-chip bytes)."""
        tensors = {"x_out": outs[0], "pre_out": outs[1]} | _state_tensors(state)
        return {k: down(t) for k, t in tensors.items()}

    guard = HostWriteGuard(monkeypatch)
    block(x, pre, fresh(), seq)  # compiles every program and creates lazily built constants
    steady_state = fresh()
    host_calls = guard.count(lambda: block(x, pre, steady_state, seq))
    ttnn.synchronize_device(mesh_device)
    logger.info(f"host transfers per forward: {host_calls}")
    assert not host_calls, f"the forward transfers from/to the host, so it cannot be traced: {host_calls}"

    untraced_state = fresh()
    untraced = [snapshot(block(x, pre, untraced_state, seq), untraced_state) for _ in range(2)]

    traced_state = fresh()
    trace, traced_outs = capture(mesh_device, lambda: block(x, pre, traced_state, seq), [block.ffn])
    traced = []
    for _ in range(2):
        trace.replay()
        traced.append(snapshot(traced_outs, traced_state))

    mismatches = [
        f"replay {i}: {k}" for i in range(2) for k in untraced[i] if not torch.equal(untraced[i][k], traced[i][k])
    ]

    timing_state = fresh()
    untraced_ms = _timed_ms(mesh_device, lambda: block(x, pre, timing_state, seq))
    traced_ms = _timed_ms(mesh_device, trace.replay)
    segments = trace.num_segments
    trace.release()

    report = {
        "mesh": list(shape),
        "weights": weights,
        "layer": layer,
        "tokens": seq,
        "host_transfers_per_forward": len(host_calls),
        "trace_segments": segments,
        "compared": sorted(untraced[0]),
        "bit_identical": not mismatches,
        "mismatches": mismatches,
        "untraced_ms": round(untraced_ms, 3),
        "traced_ms": round(traced_ms, 3),
        "speedup": round(untraced_ms / traced_ms, 2),
    }
    logger.info(f"V41_TRACE_RESULT {json.dumps(report)}")
    assert not mismatches, report


LOOP_LAYERS = (0, 2, 3, 20, 21, 24)  # every block type: SWA, KV/index source, consumer, candidate source + users
LOOP_CHUNK, LOOP_TOTAL = SMALL_SEQ // 2, SMALL_SEQ - 12  # two chunks, the last one padded


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_chunk_loop_trace(mesh_device, device_params, monkeypatch):
    """The transformer's text chunk loop (``TtV41Transformer.prefill`` without the token upload and the logits
    readback) captured as one trace over all chunks of a prompt."""
    t0 = time.perf_counter()
    cfg, layers = SmallV41Config, LOOP_LAYERS
    spec = small_spec(layers, SMALL_SEQ)
    reference = orc.build_reference(spec)
    identity = orc._digest(asdict(spec.args), spec.seed, "synthetic", orc._reference_digest(synthetic=True))
    model = TtV41Transformer(
        mesh_device,
        cfg,
        list(layers),
        lambda layer, include_moe: device_weights(reference, layers.index(layer), include_moe),
        reference.embed.weight.detach(),
        reference.norm.weight.detach(),
        reference.head.weight.detach(),
        max_seq_len=SMALL_SEQ,
        chunk=LOOP_CHUNK,
        weight_cache_path=WEIGHT_CACHE / f"small-{identity}-mesh{mesh_device.shape[0]}x{mesh_device.shape[1]}",
    )
    logger.info(f"chunk loop: model built {time.perf_counter() - t0:.1f}s")
    tokens = orc.text_tokens(LOOP_TOTAL)[0]
    chunks = -(-LOOP_TOTAL // LOOP_CHUNK)
    ids = []
    for c in range(chunks):
        chunk_tokens = torch.zeros(LOOP_CHUNK, dtype=torch.int64)
        part = tokens[c * LOOP_CHUNK : (c + 1) * LOOP_CHUNK]
        chunk_tokens[: part.numel()] = part
        ids.append(model._token_ids(chunk_tokens))
    pre0 = initial_pre_mix(mesh_device, cfg, LOOP_CHUNK)
    fresh = lambda: V41PrefillState(mesh_device, cfg, SMALL_SEQ, LOOP_CHUNK, list(layers))

    def run(state):
        """The prefill chunk loop from ``state.start`` = 0 -> the last chunk's final collapsed stream (bf16)."""
        state.start, state.selection = 0, {}
        while state.start < LOOP_TOTAL:
            length = min(LOOP_CHUNK, LOOP_TOTAL - state.start)
            h = model.embedding(ids[state.start // LOOP_CHUNK])
            x, pre = mhc_expand(ttnn.typecast(h, ttnn.float32), cfg.HC_MULT), pre0
            for block in model.blocks:
                x, pre = block(x, pre, state, length)
            state.advance(length)
        return ttnn.typecast(model.blocks[-1].residual.final_collapse(x, pre), ttnn.bfloat16)

    shape = tuple(mesh_device.shape)
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))
    snapshot = lambda out, state: {k: down(t) for k, t in ({"final": out} | _state_tensors(state)).items()}

    guard = HostWriteGuard(monkeypatch)
    run(fresh())  # compile
    steady_state = fresh()
    host_calls = guard.count(lambda: run(steady_state))
    ttnn.synchronize_device(mesh_device)
    assert not host_calls, f"the chunk loop transfers from/to the host, so it cannot be traced: {host_calls}"

    untraced_state = fresh()
    untraced = [snapshot(run(untraced_state), untraced_state) for _ in range(2)]
    traced_state = fresh()
    trace, final = capture(mesh_device, lambda: run(traced_state), [b.ffn for b in model.blocks])
    traced = []
    for _ in range(2):
        trace.replay()
        traced.append(snapshot(final, traced_state))
    mismatches = [
        f"replay {i}: {k}" for i in range(2) for k in untraced[i] if not torch.equal(untraced[i][k], traced[i][k])
    ]
    timing_state = fresh()
    untraced_ms = _timed_ms(mesh_device, lambda: run(timing_state))
    traced_ms = _timed_ms(mesh_device, trace.replay)
    segments = trace.num_segments
    trace.release()
    report = {
        "mesh": list(shape),
        "layers": list(layers),
        "chunk": LOOP_CHUNK,
        "tokens": LOOP_TOTAL,
        "chunks": chunks,
        "trace_segments": segments,
        "compared": sorted(untraced[0]),
        "bit_identical": not mismatches,
        "mismatches": mismatches,
        "untraced_ms": round(untraced_ms, 3),
        "traced_ms": round(traced_ms, 3),
        "speedup": round(untraced_ms / traced_ms, 2),
    }
    logger.info(f"V41_CHUNK_LOOP_TRACE_RESULT {json.dumps(report)}")
    assert not mismatches, report
