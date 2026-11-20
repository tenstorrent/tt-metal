# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.4 (owner 2026-11-19, overseer 2026-11-20): is the device MoE over the 8192 packed rows lane-equivalent by
construction? The MoE block alone (one layer's TtMoEBlockV2Layer: head, dispatch, routed experts + combine, the
local tail), fed the same rows of a segment A with permuted / different companion rows, at different lanes, with
pad lanes routed (PACKED_MOE=off) or out of the dispatch (PACKED_MOE=on, tt/packed_device.MoEPadSpan): A's output
rows must be bit-identical to A alone in lane 0 with every other row a pad row routed as today (the reference run).
What can break it: per-expert counts (the planner's order and regions, capacity / overflow paths), the routed
experts' program by rows per expert, the combine's sum order (int32 snap), bfp8 / bfp4 shared exponents (per 16
values of one row), the pad span words.

Rows (bf16, the ffn_norm output's scale ~ N(0, 1)): A, B, C, D random (seeds 1-4), E one random row repeated 2048
times (every row on the same 6 experts: the busiest chip's bigger matrices), PAD one fixed row (the clamped pad id's
rows: the whole pad lane on one expert set, as the walls' 1 x 2k case).
Runs (each: the 8192 rows, lane i = rows [2048 i, 2048 i + 2048)):
    ref     [A, PAD, PAD, PAD]  off (every row routed: today's MoE)
    c1      [A, B, C, D]        off, on (all rows real: on passes no span, the same program)
    c2      [B, E, A, C]        off, on (A at lane 2 next to the hot lane)
    c3      [D, C, E, A]        off, on (A at lane 3)
    c4      [A, -, -, -]        off, on (3 empty lanes of one pad row: routed / out of the dispatch)
    c5      [E, A:777, B:1600, -] off, on (padded lanes, an empty lane; A's first 777 rows compared)
    c6      [-, -, A, -]        on
    c2'     c2 on again (determinism)
Metric (gated == 0): packed_moe_block_max_abs_diff = the max abs diff of A's rows over every run and layer vs ref
(0 when the bytes match; >= 1e-30 when bytes differ). Not gated: packed_moe_block_ms_<case>_<switch> (one call,
synchronized, layer-mean: the pad span's saving on the block alone).

Layers: PB4_MOE_LAYERS (comma list; default 1, 14, 24 - those the loaded model holds). The model is the gate's
(BRINGUP_CHUNK=8192, every layer: one load with the oracle, OV-61); only the listed layers' MoE runs.

    BRINGUP_CHUNK=8192 DS41_PACKED=on scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_moe_block.py
"""

from __future__ import annotations

import importlib
import os
import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("PB4_MOE_LAYERS",)  # (the layers whose MoE runs; the model is the gate's)
CHUNK, LANE = _pc.CHUNK, _pc.CHUNK // _pc.LANES
DEFAULT_LAYERS = (1, 14, 24)
PAD = "PAD"
RUNS = {  # name: (lanes: (row set, real rows) or None for an empty lane, switch values)
    "c1": ([("A", LANE), ("B", LANE), ("C", LANE), ("D", LANE)], ("off", "on")),
    "c2": ([("B", LANE), ("E", LANE), ("A", LANE), ("C", LANE)], ("off", "on")),
    "c3": ([("D", LANE), ("C", LANE), ("E", LANE), ("A", LANE)], ("off", "on")),
    "c4": ([("A", LANE), None, None, None], ("off", "on")),
    "c5": ([("E", LANE), ("A", 777), ("B", 1600), None], ("off", "on")),
    "c6": ([None, None, ("A", LANE), None], ("on",)),
}
TIMED = ("c1", "c4")  # (cases timed: no pad rows / 3 empty lanes)


def _layers(model) -> list:
    v = os.environ.get("PB4_MOE_LAYERS", "").strip()
    want = [int(x) for x in v.split(",") if x.strip()] if v else [i for i in DEFAULT_LAYERS if i in model.layers]
    assert want and all(i in model.layers for i in want), (want, model.layers)
    return want


def _rows(h: int) -> dict:
    g = lambda seed: torch.Generator().manual_seed(seed)  # noqa: E731
    out = {k: torch.randn(LANE, h, generator=g(s)).bfloat16() for k, s in (("A", 1), ("B", 2), ("C", 3), ("D", 4))}
    out["E"] = torch.randn(1, h, generator=g(5)).bfloat16().expand(LANE, h).contiguous()
    out[PAD] = torch.randn(1, h, generator=g(6)).bfloat16().expand(LANE, h).contiguous()
    return out


def _desc(lanes) -> ChunkDescriptor:
    return ChunkDescriptor.packed(CHUNK, [None if x is None else Segment(1 + i, 0, x[1]) for i, x in enumerate(lanes)])


def _x(model, rows: dict, lanes) -> ttnn.Tensor:
    """The chunk's rows as the MoE's OWN_ROWS input: chip q's rows [q R, (q + 1) R) (chip-linear = device order)."""
    parts = []
    for x in lanes:
        if x is None:
            parts.append(rows[PAD])
            continue
        name, n = x
        lane = rows[PAD].clone()
        lane[:n] = rows[name][:n]
        parts.append(lane)
    full = torch.cat(parts, dim=0).reshape(1, 1, CHUNK, -1)
    return ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=model.mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(model.mesh, dim=2),
    )


def _call(model, layer: int, x, desc, lanes, value: str, timed: bool = False):
    """One layer's MoE on x with PACKED_MOE=value -> (A's lane, A's rows as host fp32 [n, H], ms or None)."""
    from models.demos.deepseek_v41_flash_d_p.tt.ccl import dealloc
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import MoEPadSpan, moe_pad_on

    with _pc.switch("PACKED_MOE", value):
        span = None
        if moe_pad_on():
            valid = torch.zeros(CHUNK, dtype=torch.bool)
            for i, v in enumerate(lanes):
                if v is not None:
                    valid[i * LANE : i * LANE + v[1]] = True
            span = MoEPadSpan.of(model.mesh, CHUNK).span(desc, valid)
        moe = model.blocks[layer].moe_block
        kw = {} if span is None else {"span": span}
        if timed:
            ttnn.synchronize_device(model.mesh)
            t0 = time.perf_counter()
        y = moe(x, **kw)
        ms = None
        if timed:
            ttnn.synchronize_device(model.mesh)
            ms = (time.perf_counter() - t0) * 1e3
    lane = next(i for i, v in enumerate(lanes) if v is not None and v[0] == "A")
    n = lanes[lane][1]
    a = _pc.lane_rows(model, y, desc, lane, n)
    dealloc(y)  # (LEAN_PY's load-time output buffer is pinned: left alone)
    return a, ms


def _diff(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    if _pc.digest(a) == _pc.digest(b):
        return 0.0
    d = torch.nan_to_num((a - b).abs(), nan=float("inf"))
    return max(float(d.max()), 1e-30)


@mesh_parametrize
def test_packed_moe_block(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    assert model.chunk == CHUNK, model.chunk
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    assert settings.get("MOE_IMPL") == "block" and settings.get("MOE_BLOCK_SRC") == "block", "the shipped MoE block"
    layers = _layers(model)
    rows = _rows(int(model.cfg.dim))
    worst, bad, ms = 0.0, [], {}
    for i in layers:
        ref_lanes = [("A", LANE), None, None, None]
        ref_desc = _desc(ref_lanes)
        x = _x(model, rows, ref_lanes)
        ref, _ = _call(model, i, x, ref_desc, ref_lanes, "off")
        ttnn.deallocate(x)
        outs = {}
        for name, (lanes, values) in RUNS.items():
            desc = _desc(lanes)
            x = _x(model, rows, lanes)
            for v in values:
                for rep in range(2 if name in TIMED else 1):  # (timed: a warm second call)
                    a, t = _call(model, i, x, desc, lanes, v, timed=name in TIMED)
                    if t is not None and rep == 1:
                        ms.setdefault(f"{name}_{v}", []).append(t)
                outs[(name, v)] = a
                d = _diff(a, ref)
                worst = max(worst, d)
                if d:
                    bad.append(f"layer {i} {name} PACKED_MOE={v}: A's rows differ from ref (max abs {d:.3g})")
            if name == "c2":  # (determinism: c2 on again)
                a, _ = _call(model, i, x, desc, lanes, "on")
                d = _diff(a, outs[("c2", "on")])
                worst = max(worst, d)
                if d:
                    bad.append(f"layer {i} c2 PACKED_MOE=on repeated: differs (max abs {d:.3g})")
            ttnn.deallocate(x)
        print(f"PB.4 MoE block layer {i}: {len(outs) + 2} runs, worst so far {worst}")
    metrics.record("packed_moe_block_max_abs_diff", worst)
    metrics.record("packed_moe_block_layers", len(layers))
    for k, v in sorted(ms.items()):
        metrics.record(f"packed_moe_block_ms_{k}", sum(v) / len(v))
    print(
        f"PB.4 MoE block, layers {layers}: max abs diff {worst}; ms a call (layer mean) "
        + ", ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in sorted(ms.items()))
    )
    for b in bad:
        print("  " + b)
    assert worst == 0, "the MoE block is not lane-equivalent (see above)"
