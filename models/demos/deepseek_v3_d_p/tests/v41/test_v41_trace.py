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

Cases: one block (small dims and real production weights) and the transformer's perf mode ``V41PrefillTrace``
(``test_v41_prefill_trace``): small schedules (sharing, Engram, DSpark; two chunks, the last one padded) and real
production weights (one and two chunks at S=2048, a 5120-token chunk, and the real layers plus a synthetic Engram
layer 1 for the host Engram prepare pipelining).
"""

import json
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import expert_dtype_reference as R
from models.demos.deepseek_v3_d_p.tests.v41.reference_weights import device_weights
from models.demos.deepseek_v3_d_p.tests.v41.small_config import SmallV41Config, small_spec
from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import _pack
from models.demos.deepseek_v3_d_p.tests.v41.test_transformer_v41 import (
    PRODUCTION_CANDIDATE_BLOCKS,
    PRODUCTION_SEQ,
    SCHEDULES,
    SEQ,
)
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_expert_dtype import BLOCK_CONFIG, _weights
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import (
    WEIGHT_CACHE,
    conversion_digest,
    source_key,
    weight_cache_dir,
)
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.engram import TtV41Engram, V41EngramHash, V41EngramTable
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer, V41PrefillTrace
from models.demos.deepseek_v3_d_p.tt.v41.weights import (
    dequant_fp8_block,
    load_layer,
    load_layer_dense,
    resolve_checkpoint,
)
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
    root, w, marker = _weights(ckpt, layer, "bfp8", mesh_device.shape)
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


# --- the transformer's chunk loop: V41PrefillTrace vs the untraced prefill -----------------------------------------
# (weights, schedule, chunk, prompt tokens, scored positions); "engram" = the real sharing layers + a synthetic layer 1
# (real Engram tables are not downloaded), for the host Engram prepare pipelining (bead 8y7.17). Perf cases score the
# last position only (generation), so the 256-row logits readback (~130 MB) does not swamp the timings.
PREFILL_CASES = {
    "small-sharing": ("small", "sharing", SEQ // 2, SEQ - 12, 256),
    "small-engram": ("small", "engram", SEQ // 2, SEQ - 12, 256),
    "small-dspark": ("small", "dspark", SEQ // 2, SEQ - 12, 256),
    "real-one_chunk": ("real", "sharing", PRODUCTION_SEQ, PRODUCTION_SEQ, 256),
    "real-two_chunks": ("real", "sharing", PRODUCTION_SEQ // 2, PRODUCTION_SEQ, 256),
    "real-c2048": ("real", "sharing", 2048, 4096, 1),
    "real-c5120": ("real", "sharing", 5120, 5120, 1),
    "engram-c2048": ("engram", "sharing", 2048, 8192, 1),
    "engram-c5120": ("engram", "sharing", 5120, 10240, 1),
}


def _prompts(total: int) -> list[torch.Tensor]:
    """Two different real-text prompts [1, total]: the capture runs the first, replays run both."""
    text = orc.text_tokens(2 * total)
    return [text[:, :total], text[:, total:]]


def _engram_modules(mesh_device, cfg, reference, spec, layers, prompts) -> tuple[dict, V41EngramHash | None]:
    """TtV41Engram per Engram layer of ``reference`` (synthetic rows of both prompts) and the hasher."""
    if reference.engram_hash is None:
        return {}, None
    with torch.inference_mode():
        hashes = [reference.engram_hash(p, 0) for p in prompts]  # as orc.load_engram_rows, rows of both prompts
    for pos, block in enumerate(reference.layers):
        if block.engram is None:
            continue
        emb = block.engram.embed
        rows = torch.unique(torch.cat([h[:, :, block.engram.layer_hash_index].flatten() for h in hashes]))
        q, scale = orc.synthetic_engram_rows(spec.seed, spec.layer_ids[pos], rows, emb.dim)
        emb.weight = torch.nn.Parameter(q, requires_grad=False)
        emb.scale = torch.nn.Parameter(scale, requires_grad=False)
        emb.vocab_end_idx, emb.oracle_rows = len(rows), rows
    modules = {}
    for pos, layer in enumerate(layers):
        e = reference.layers[pos].engram
        if e is None:
            continue
        weights = {
            "wkv": dequant_fp8_block(e.wkv.weight.detach(), e.wkv.scale.detach()),
            "q_weight": e.q_weight.detach(),
            "k_weight": e.k_weight.detach(),
        }
        table = V41EngramTable(e.embed.weight.detach(), e.embed.scale.detach(), e.embed.oracle_rows)
        modules[layer] = TtV41Engram(mesh_device, cfg, layer, weights, table)
    return modules, V41EngramHash(cfg, reference.engram_hash.token_map)


def _small_model(mesh_device, schedule: str, chunk: int, prompts):
    """As ``test_transformer_v41.test_v41_transformer_small`` builds it."""
    layers = SCHEDULES[schedule]
    spec = small_spec(layers, SEQ, dspark=schedule == "dspark")
    reference = orc.build_reference(spec)
    engram, engram_hash = _engram_modules(mesh_device, SmallV41Config, reference, spec, layers, prompts)
    dspark = None
    if reference.mtp:
        fp8 = lambda linear: dequant_fp8_block(linear.weight.detach(), linear.scale.detach())
        dspark = {
            "main_proj": fp8(reference.mtp[0].main_proj),
            "main_norm": reference.mtp[0].main_norm.weight.detach(),
            "layers": [{"wkv": fp8(b.attn.wkv), "kv_norm": b.attn.kv_norm.weight.detach()} for b in reference.mtp],
        }
    return TtV41Transformer(
        mesh_device,
        SmallV41Config,
        list(layers),
        lambda layer, include_moe: device_weights(reference, layers.index(layer), include_moe),
        reference.embed.weight.detach(),
        reference.norm.weight.detach(),
        reference.head.weight.detach(),
        max_seq_len=SEQ,
        chunk=chunk,
        dspark_weights=dspark,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=weight_cache_dir(spec, mesh_device.shape),
    )


def _real_model(mesh_device, chunk: int, total: int, prompts, with_engram: bool):
    """The real sharing layers (as ``test_v41_transformer_production``); ``with_engram`` adds a synthetic layer 1
    (attention, MoE and Engram at real dims) for timing."""
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    real_layers = SCHEDULES["sharing"]
    spec = orc.real_spec(
        real_layers, PRODUCTION_SEQ, candidate_topk_blocks=PRODUCTION_CANDIDATE_BLOCKS, checkpoint=ckpt.root
    )
    cache = weight_cache_dir(spec, mesh_device.shape)
    cfg = type("V41TestConfig", (C,), {"CANDIDATE_TOPK_BLOCKS": PRODUCTION_CANDIDATE_BLOCKS})
    layers, engram, engram_hash, synthetic = list(real_layers), {}, None, None
    if with_engram:
        spec1 = orc.real_spec((1,), total)
        synthetic = orc.build_reference(spec1)
        engram, engram_hash = _engram_modules(mesh_device, cfg, synthetic, spec1, (1,), prompts)
        layers = sorted(layers + [1])
        sp, tp = tuple(mesh_device.shape)
        key = orc._digest(source_key(spec), source_key(spec1), ttnn.bfloat8_b.name, conversion_digest(), [sp, tp])
        cache = WEIGHT_CACHE / f"mixed-{key}-mesh{sp}x{tp}"  # never mixed into the real-weights directory

    def layer_weights(layer, include_moe):
        if layer == 1:
            return device_weights(synthetic, 0, include_moe)
        return (load_layer if include_moe else load_layer_dense)(ckpt, layer)

    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    return TtV41Transformer(
        mesh_device,
        cfg,
        layers,
        layer_weights,
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=total,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=cache,
    )


def _differences(name: str, expected: dict, actual: dict) -> list[str]:
    """``name: key`` of every unequal tensor, with the unequal element count and the row range (dim -2)."""
    out = []
    for k, want in expected.items():
        got = actual[k]
        if torch.equal(want, got):
            continue
        rows = (want != got).flatten(0, -3).any(-1).any(0).nonzero().flatten() if want.dim() >= 2 else torch.zeros(1)
        count = int((want != got).sum())
        out.append(f"{name}: {k} ({count} values, rows {int(rows.min())}..{int(rows.max())} of {want.shape[-2]})")
    return out


def _prefill_snapshot(mesh_device, logits, state) -> dict:
    """Logits and every tensor the request leaves in ``state`` (caches, carries, DSpark rings), on host."""
    shape = tuple(mesh_device.shape)
    down = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3)))
    tensors = _state_tensors(state)
    for source, carry in state.compressor_carry.items():
        for i, t in enumerate(carry or ()):
            tensors[f"compressor_carry[{source}][{i}]"] = t
    for k, ring in enumerate(state.dspark_rings or ()):
        tensors[f"dspark_ring[{k}]"] = ring
    return {"logits": logits} | {k: down(t) for k, t in tensors.items()}


def _replay_phases(mesh_device, trace, tokens, iters=TIMED_ITERS) -> dict:
    """Host wall per phase of ``trace.run(tokens)`` (mean over ``iters``): carry zeroing, chunk loop (input copies,
    non-blocking replays), waiting for the device, logits readback (also per iteration: it is host bound and
    varies); plus a device-only replay synchronized per iteration (no pipelining across iterations)."""
    model, phases, readback = trace.model, {}, []
    names = ("zero_carries", "chunk_loop", "device_wait", "readback", "device_only_sync")
    for _ in range(iters):
        ttnn.synchronize_device(mesh_device)
        t = [time.perf_counter()]
        request = model._request(tokens, trace.logit_positions)
        trace.state.start = 0
        for tensor in list(trace.state.window_carry.values()) + list(trace.state.dspark_rings or ()):
            ttnn.copy_host_to_device_tensor(trace._zeros(tensor), tensor)
        t.append(time.perf_counter())
        model._run(request, trace.state, trace.buffers, lambda c, _: trace.controllers[c].replay(blocking=False) or [])
        t.append(time.perf_counter())
        ttnn.synchronize_device(mesh_device)
        t.append(time.perf_counter())
        model._logits(trace.scored)
        t.append(time.perf_counter())
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for controller in trace.controllers:
            controller.replay(blocking=False)
        ttnn.synchronize_device(mesh_device)
        t.append(t[-1] + time.perf_counter() - start)
        readback.append(round((t[4] - t[3]) * 1e3, 2))
        for name, a, b in zip(names, t, t[1:]):
            phases[name] = phases.get(name, 0.0) + (b - a) * 1e3 / iters
    return {k: round(v, 2) for k, v in phases.items()} | {"readback_each": readback}


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("case", list(PREFILL_CASES))
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_prefill_trace(mesh_device, device_params, case, monkeypatch):
    """``V41PrefillTrace`` (per-chunk captures, per-chunk input copies into fixed buffers, the next chunk's host
    Engram prepare overlapping the device) == ``TtV41Transformer.prefill`` bit-identically (logits of the scored
    positions, caches, carries, DSpark rings) for the captured prompt and another prompt of the same
    length; the chunk forward makes no host transfers. Reports untraced / traced / serial / device-only times."""
    t0 = time.perf_counter()
    weights, schedule, chunk, total, scored = PREFILL_CASES[case]
    prompts = _prompts(total)
    if weights == "small":
        model = _small_model(mesh_device, schedule, chunk, prompts)
    else:
        model = _real_model(mesh_device, chunk, total, prompts, with_engram=weights == "engram")
    logger.info(f"prefill trace {case}: model built {time.perf_counter() - t0:.1f}s")
    tokens = [p[0] for p in prompts]

    t = time.perf_counter()
    model.prefill(tokens[0], scored)  # compile
    logger.info(f"prefill trace {case}: compile run {time.perf_counter() - t:.1f}s")
    guard, transfers, forward = HostWriteGuard(monkeypatch), [], model._chunk_forward

    def counted(*args, **kwargs):
        out = []
        transfers.extend(guard.count(lambda: out.append(forward(*args, **kwargs))))
        return out[0]

    model._chunk_forward = counted
    model.prefill(tokens[0], scored)
    del model._chunk_forward
    assert not transfers, f"the chunk forward transfers from/to the host, so it cannot be traced: {transfers}"

    t = time.perf_counter()
    untraced = [_prefill_snapshot(mesh_device, *model.prefill(p, scored)) for p in tokens]
    repeat = _prefill_snapshot(mesh_device, *model.prefill(tokens[0], scored))
    mismatches = _differences("untraced repeat", untraced[0], repeat)
    untraced_ms = _timed_ms(mesh_device, lambda: model.prefill(tokens[0], scored), iters=3)
    logger.info(f"prefill trace {case}: untraced runs {time.perf_counter() - t:.1f}s")

    # the host Engram prepare (hash + packed row gather) of every chunk, alone
    request = model._request(tokens[0], scored)
    t = time.perf_counter()
    host_chunks = list(model._host_chunks(request))
    host_prepare_ms = (time.perf_counter() - t) * 1e3 / len(host_chunks)
    del host_chunks

    t = time.perf_counter()
    trace = V41PrefillTrace(model, tokens[0], scored)
    logger.info(f"prefill trace {case}: warm-up + capture {time.perf_counter() - t:.1f}s")
    for i in (0, 1, 0):
        traced = _prefill_snapshot(mesh_device, *trace.run(tokens[i]))
        mismatches += _differences(f"prompt {i}", untraced[i], traced)
    traced_ms = _timed_ms(mesh_device, lambda: trace.run(tokens[0]))

    def serial():
        """The same loop with a blocking replay: the next chunk's host prepare waits for the device."""
        trace.state.start = 0
        model._run(request, trace.state, trace.buffers, lambda c, _: trace.controllers[c].replay() or [])
        model._logits(trace.scored)

    def device_only():
        for controller in trace.controllers:
            controller.replay(blocking=False)

    serial_ms = _timed_ms(mesh_device, serial)
    device_ms = _timed_ms(mesh_device, device_only)
    readback_ms = _timed_ms(mesh_device, lambda: model._logits(trace.scored))  # scored rows' logits to host
    phases = _replay_phases(mesh_device, trace, tokens[0])
    segments = trace.num_segments
    trace.release()
    report = {
        "case": case,
        "mesh": list(mesh_device.shape),
        "layers": model.layers,
        "chunk": chunk,
        "tokens": total,
        "chunks": -(-total // chunk),
        "scored": scored,
        "trace_segments": segments,
        "compared": sorted(untraced[0]),
        "bit_identical": not mismatches,
        "mismatches": mismatches,
        "untraced_ms": round(untraced_ms, 2),
        "traced_ms": round(traced_ms, 2),
        "traced_serial_ms": round(serial_ms, 2),
        "traced_device_only_ms": round(device_ms, 2),
        "host_prepare_ms_per_chunk": round(host_prepare_ms, 2),
        "logits_readback_ms": round(readback_ms, 2),
        # host time on the critical path besides the logits readback: input copies, the Engram prepare not hidden
        "host_critical_path_ms": round(traced_ms - device_ms - readback_ms, 2),
        "host_critical_path_serial_ms": round(serial_ms - device_ms - readback_ms, 2),
        "speedup": round(untraced_ms / traced_ms, 2),
        "traced_phases_ms": phases,
    }
    logger.info(f"V41_PREFILL_TRACE_RESULT {json.dumps(report)}")
    assert not mismatches, report
