# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trace capture / replay of one DeepSeek-V4.1 block as a perf-measurement mode (dev-spec D-G (b), bead 8y7.9).

Contract: for the same inputs and the same starting state, a traced replay produces outputs and cache contents
bit-identical to the untraced forward, replays are deterministic, and the replay's wall time is the block's device
time without host dispatch (reported with the untraced time and the speedup).

The prefill forward is not capturable as written: attention and the indexer upload per-chunk host tables
(RoPE cos/sin, window rows, selection masks) with ``ttnn.from_torch`` inside ``forward``, and a trace forbids host
writes. For a fixed chunk (fixed start and length) those tables are constant, so this test stages them: a recording
forward logs every ``from_torch`` of the forward, the tables are uploaded before capture, and the captured forward
receives the staged tensors in call order (checked equal to what it asks for) and may not deallocate them. The MoE's
shared-expert / dispatch overlap loads a sub-device manager mid-forward, which a trace cannot contain either; the
in-tree ``SubDeviceTraceController`` splits the capture at those points (the replay is several trace segments). This
proves the device program is replay-safe and measures it; production trace needs those tables hoisted out of the
forward (reported by the bead, not changed here).
"""

import json
import time

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
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_expert_dtype import BLOCK_CONFIG, _weights
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
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


class HostTableStager:
    """Records the host uploads of a forward (``ttnn.from_torch``, and ``ttnn.full`` on a device, which writes from
    the host too) and serves them pre-staged to a captured forward."""

    def __init__(self, monkeypatch):
        self.real = {"from_torch": ttnn.from_torch, "full": ttnn.full}
        self.real_deallocate = ttnn.deallocate
        self.mode, self.calls, self.staged, self.served, self.errors = "off", [], [], 0, []
        monkeypatch.setattr(ttnn, "from_torch", lambda *a, **k: self._upload("from_torch", *a, **k))
        monkeypatch.setattr(ttnn, "full", lambda *a, **k: self._upload("full", *a, **k))
        monkeypatch.setattr(ttnn, "deallocate", self._deallocate)

    @staticmethod
    def _key(fn, args, kwargs):
        """What identifies an upload: the host tensor for from_torch, the fill arguments for full."""
        if fn == "from_torch":
            return args[0] if args else kwargs["tensor"]
        return repr((args, sorted((k, v) for k, v in kwargs.items() if k != "device")))

    @staticmethod
    def _same(a, b) -> bool:
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
        return a == b

    def _upload(self, fn, *args, **kwargs):
        if fn == "full" and kwargs.get("device") is None:
            return self.real[fn](*args, **kwargs)  # a host tensor: no device write
        if self.mode == "record":
            key = self._key(fn, args, kwargs)
            self.calls.append((fn, key.clone() if isinstance(key, torch.Tensor) else key, args, kwargs))
        elif self.mode == "serve":
            # never raise inside a capture (an open capture hangs the device at teardown): record and check after
            i = self.served
            self.served += 1
            if i >= len(self.staged):
                self.errors.append(f"unrecorded {fn} upload {i}")
                return self.staged[-1]
            if self.calls[i][0] != fn or not self._same(self.calls[i][1], self._key(fn, args, kwargs)):
                self.errors.append(f"upload {i} ({fn}) differs from the recorded one")
            return self.staged[i]
        return self.real[fn](*args, **kwargs)

    def _deallocate(self, tensor, *args, **kwargs):
        if self.mode == "serve" and any(tensor is t for t in self.staged):
            return None  # staged tables outlive the trace
        return self.real_deallocate(tensor, *args, **kwargs)

    def stage(self):
        self.staged = [self.real[fn](*a, **k) for fn, _, a, k in self.calls]
        self.served, self.errors = 0, []

    def capture(self, mesh_device, forward, moes):
        """Capture ``forward()`` with the staged tables; the capture is always closed. The MoE's shared-expert /
        dispatch overlap loads a sub-device manager, which a trace cannot contain: ``SubDeviceTraceController``
        splits the capture there (``moes``: the ``TtMoe`` instances the forward runs). Returns
        (controller, outputs); ``controller.replay()`` runs the whole forward."""
        assert self.staged, "stage() the recorded tables first"
        controller = SubDeviceTraceController(mesh_device)
        for moe in moes:
            moe.set_trace_controller(controller)
        ttnn.synchronize_device(mesh_device)
        self.mode = "serve"
        controller.begin_capture()
        try:
            outs = forward()
        finally:
            controller.end_capture()
            self.mode = "off"
            for moe in moes:
                moe.set_trace_controller(None)
        if self.served != len(self.calls):
            self.errors.append(f"served {self.served} of {len(self.calls)} tables")
        if self.errors:
            controller.release()
            raise AssertionError(f"captured forward differs from the recorded one: {self.errors}")
        return controller, outs


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
        tensors = {"x_out": outs[0], "pre_out": outs[1]}
        for name in ("kv", "index_k", "window_carry"):
            for key, t in getattr(state, name, {}).items():
                if t is not None:
                    tensors[f"{name}[{key}]"] = t
        return {k: down(t) for k, t in tensors.items()}

    stager = HostTableStager(monkeypatch)
    block(x, pre, fresh(), seq)  # compiles every program and creates lazily built constants
    record_state = fresh()
    stager.mode = "record"
    block(x, pre, record_state, seq)  # logs a steady-state forward's host tables
    stager.mode = "off"
    ttnn.synchronize_device(mesh_device)
    logger.info(f"recorded {len(stager.calls)} host tables per forward")

    untraced_state = fresh()
    untraced = [snapshot(block(x, pre, untraced_state, seq), untraced_state) for _ in range(2)]

    traced_state = fresh()
    stager.stage()
    trace, traced_outs = stager.capture(mesh_device, lambda: block(x, pre, traced_state, seq), [block.ffn.moe])
    traced = []
    for _ in range(2):
        trace.replay()
        traced.append(snapshot(traced_outs, traced_state))

    mismatches = [
        f"replay {i}: {k}" for i in range(2) for k in untraced[i] if not torch.equal(untraced[i][k], traced[i][k])
    ]

    timing_state = fresh()
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        block(x, pre, timing_state, seq)
    ttnn.synchronize_device(mesh_device)
    untraced_ms = (time.perf_counter() - t0) * 1e3 / TIMED_ITERS
    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        trace.replay()
    ttnn.synchronize_device(mesh_device)
    traced_ms = (time.perf_counter() - t0) * 1e3 / TIMED_ITERS
    segments = trace.num_segments
    trace.release()

    report = {
        "mesh": list(shape),
        "weights": weights,
        "layer": layer,
        "tokens": seq,
        "host_tables_per_forward": len(stager.calls),
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
