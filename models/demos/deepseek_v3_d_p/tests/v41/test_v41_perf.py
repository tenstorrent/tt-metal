# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Measured block time vs the theoretical model (``utils/v41_perf_model.py``, G2), bead 8y7.9.

Production shapes: one 5120-token chunk at start 0 through the real-weight stack 2 -> 3 -> 20 -> 21 -> 24 (every
sharing role) and the sliding-window block 0, LoudBox 2x4, BF16 KV. The input of the first layer is its real
block-oracle input (2048 rows tiled to 5120); later layers take the device outputs. Measurements (warm, program
cache hot):

* ``layer_ms``: untraced forward with a device synchronize after each layer (includes host dispatch);
* ``parts_ms``: the same with synchronized timers around every sublayer and MoE / attention stage (serializes
  what might overlap; the stage times sum to more than ``layer_ms``);
* ``traced_ms``: per-layer trace replay (device time without host dispatch; host tables staged as in
  ``test_v41_trace``).

Each stage is set against the model's optimistic (``max(compute, DRAM, CCL)``) and conservative (serialized)
compositions of the graph nodes it executes; utilization = model optimistic / measured. Logged as
``V41_PERF_RESULT`` JSON lines; no pass/fail bar (G2 targets are reconciled in the bead report).
"""

import json
import time
from collections import defaultdict

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41 import expert_dtype_reference as R
from models.demos.deepseek_v3_d_p.tests.v41.test_block_v41 import _pack
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_expert_dtype import EXPERT_DTYPES, _weights
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_trace import TRACE_REGION, HostTableStager
from models.demos.deepseek_v3_d_p.tt.v41.block import TtV41Block
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.weights import resolve_checkpoint
from models.demos.deepseek_v3_d_p.utils import v41_perf_model as M
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

SCHEDULES = {"stack": (2, 3, 20, 21, 24), "swa": (0,)}
CHUNK = R.CHUNK
TIMED_ITERS = 3
LAYOUTS = {(2, 4): M.LOUDBOX_2X4, (4, 2): M.LOUDBOX_4X2}
# measured stage -> model graph nodes (op names where one node spans stages)
STAGE_NODES = {
    "mhc": {"B1", "B2", "B17", "B18", "B23"},
    "attn_norm": {"B3"},
    "attention": {"B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15", "B16"},
    "attention.compressor": {"B6"},
    "attention.index_keys": {"B7"},
    "attention.indexer": {"B9", "B10", "B11", "B12"},
    "ffn_norm": {"B19"},
    "moe": {"B20", "B21", "B22"},
    "moe.gate": {"B20"},
    "moe.dispatch_module": {"dispatch"},
    "moe.routed_expert": {"routed_experts"},
    "moe.combine_module": {"combine"},
    "moe.reduce_module": {"routed_reduce"},
    "moe.shared_expert": {"B22"},
}
MESH = [
    pytest.param(
        (2, 4),
        fabric2d_device_params(trace_region_size=TRACE_REGION),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-mesh-2x4",
    )
]


class _Timed:
    """Wraps a sublayer callable: device-synchronized wall time per call into ``sink[name]``."""

    def __init__(self, fn, name, sink, mesh_device):
        self._fn, self._name, self._sink, self._mesh = fn, name, sink, mesh_device

    def __call__(self, *args, **kwargs):
        if not self._sink.get("enabled"):
            return self._fn(*args, **kwargs)
        ttnn.synchronize_device(self._mesh)
        start = time.perf_counter()
        out = self._fn(*args, **kwargs)
        ttnn.synchronize_device(self._mesh)
        self._sink[self._name].append((time.perf_counter() - start) * 1e3)
        return out

    def __getattr__(self, name):
        return getattr(self._fn, name)


def _instrument(block, sink, mesh_device):
    def wrap(owner, attr, name):
        if getattr(owner, attr, None) is not None:
            setattr(owner, attr, _Timed(getattr(owner, attr), name, sink, mesh_device))

    attn, moe = block.attn, block.ffn.moe
    for attr in ("compressor", "index_keys", "indexer"):
        wrap(attn, attr, f"attention.{attr}")
    for attr in ("gate", "dispatch_module", "routed_expert", "combine_module", "reduce_module", "shared_expert"):
        wrap(moe, attr, f"moe.{attr}")
    wrap(block, "attn_norm", "attn_norm")
    wrap(block, "attn", "attention")
    wrap(block, "ffn_norm", "ffn_norm")
    wrap(block, "ffn", "moe")


def _model(layer: int, workload: M.Workload, layout: M.Layout) -> dict:
    ops = M.block_ops(layer, workload, layout)
    out = {}
    for stage, keys in STAGE_NODES.items():
        sel = [o for o in ops if o.node in keys or o.name in keys]
        if not sel:
            continue
        e = M.compose(sel)
        out[stage] = {
            "optimistic_ms": e.optimistic_ns / 1e6,
            "conservative_ms": e.conservative_ns / 1e6,
            "compute_ms": e.compute_ns / 1e6,
            "dram_ms": e.dram_ns / 1e6,
            "ccl_ms": e.ccl_ns / 1e6,
        }
    e = M.compose(ops)
    out["block"] = {
        "optimistic_ms": e.optimistic_ns / 1e6,
        "conservative_ms": e.conservative_ns / 1e6,
        "compute_ms": e.compute_ns / 1e6,
        "dram_ms": e.dram_ns / 1e6,
        "ccl_ms": e.ccl_ns / 1e6,
    }
    return out


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("expert_dtype", list(EXPERT_DTYPES))
@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_block_perf(mesh_device, device_params, schedule, expert_dtype, monkeypatch):
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    layers = SCHEDULES[schedule]
    spec = R.block_spec(layers[0])
    result = orc.oracle(spec, orc.random_tokens(spec))
    rec = result["blocks"][layers[0]]
    shape = tuple(mesh_device.shape)
    tp = shape[1]
    blocks = {}
    for layer in layers:
        start = time.perf_counter()
        root, w, marker = _weights(ckpt, layer, expert_dtype, spec)
        blocks[layer] = TtV41Block(
            mesh_device,
            C,
            layer,
            w,
            CHUNK,
            routed_expert_weights_dtype=EXPERT_DTYPES[expert_dtype],
            weight_cache_path=root,
        )
        marker.touch()
        del w
        logger.info(f"block {layer} built {time.perf_counter() - start:.1f}s")

    def to_device(t, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=dims),
        )

    x0 = to_device(_pack(R.tile_rows(rec["x_in"].float()), tp), (2, 3))
    pre0 = to_device(R.tile_rows(rec["pre_in"].float())[None, None], (2, None))
    fresh = lambda: V41PrefillState(mesh_device, C, CHUNK, CHUNK, list(layers), kv_format=MlaKvCacheFormat.BF16_RM)

    def stack(state, per_layer_ms=None, inputs=None):
        x, pre = x0, pre0
        for layer in layers:
            if inputs is not None:
                inputs[layer] = (x, pre)
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            x, pre = blocks[layer](x, pre, state, CHUNK)
            ttnn.synchronize_device(mesh_device)
            if per_layer_ms is not None:
                per_layer_ms[layer].append((time.perf_counter() - start) * 1e3)
        return x, pre

    start = time.perf_counter()
    stack(fresh())  # compile
    logger.info(f"warm-up stack {time.perf_counter() - start:.1f}s")
    layer_ms = defaultdict(list)
    state = fresh()
    for _ in range(TIMED_ITERS):
        stack(state, layer_ms)

    sink = defaultdict(list)
    for layer in layers:
        _instrument(blocks[layer], sink, mesh_device)
    parts = {layer: defaultdict(list) for layer in layers}
    state = fresh()
    for it in range(TIMED_ITERS + 1):  # the first instrumented pass is discarded
        x, pre = x0, pre0
        for layer in layers:
            sink.clear()
            sink["enabled"] = True
            ttnn.synchronize_device(mesh_device)
            start = time.perf_counter()
            x, pre = blocks[layer](x, pre, state, CHUNK)
            ttnn.synchronize_device(mesh_device)
            elapsed = (time.perf_counter() - start) * 1e3
            sink["enabled"] = False
            if it:
                for name, values in sink.items():
                    if name != "enabled":
                        parts[layer][name].append(sum(values))
                parts[layer]["block_instrumented"].append(elapsed)

    # traced replay per layer, each on the input the untraced stack gives it
    inputs = {}
    state = fresh()
    stack(state, inputs=inputs)
    stager = HostTableStager(monkeypatch)
    traced_ms = {}
    for layer in layers:
        x, pre = inputs[layer]
        stager.calls = []
        stager.mode = "record"
        blocks[layer](x, pre, state, CHUNK)
        stager.mode = "off"
        stager.stage()
        trace, _ = stager.capture(mesh_device, lambda: blocks[layer](x, pre, state, CHUNK), [blocks[layer].ffn.moe])
        trace.replay()
        start = time.perf_counter()
        for _ in range(TIMED_ITERS):
            trace.replay()
        ttnn.synchronize_device(mesh_device)
        traced_ms[layer] = (time.perf_counter() - start) * 1e3 / TIMED_ITERS
        trace.release()

    workload = M.Workload(chunk=CHUNK, expert_dtype=expert_dtype)
    layout = LAYOUTS[shape]
    for layer in layers:
        mean = lambda v: sum(v) / len(v)
        model = _model(layer, workload, layout)
        measured = {k: mean(v) for k, v in parts[layer].items()}
        measured["mhc"] = measured["block_instrumented"] - sum(
            measured[s] for s in ("attn_norm", "attention", "ffn_norm", "moe")
        )
        measured["block"] = mean(layer_ms[layer])
        stages = {}
        for stage, m in model.items():
            if stage not in measured:
                continue
            stages[stage] = {
                "measured_ms": round(measured[stage], 3),
                **{k: round(v, 3) for k, v in m.items()},
                "util_vs_optimistic": round(m["optimistic_ms"] / measured[stage], 3),
            }
        stages["block"]["traced_ms"] = round(traced_ms[layer], 3)
        stages["block"]["traced_util_vs_optimistic"] = round(model["block"]["optimistic_ms"] / traced_ms[layer], 3)
        report = {
            "mesh": list(shape),
            "layer": layer,
            "block_type": C.block_type(layer).value,
            "expert_dtype": expert_dtype,
            "chunk": CHUNK,
            "stages": stages,
            "block_instrumented_ms": round(measured["block_instrumented"], 3),
        }
        logger.info(f"V41_PERF_RESULT {json.dumps(report)}")
        assert torch.isfinite(torch.tensor(list(measured.values()))).all()
