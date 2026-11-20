# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.3 (owner 2026-11-19): the packed prefill's steady walls, serving-equivalent (bringup/PACKED_DESIGN.md s9 item 4).

DECODER_SKIP=decode (layers 0-19, layer 20's KV head, 21-39 never: the layers loaded and looped, BATCH_LAYERS
executed), PACKED=on, agentic text (BRINGUP_POS_TOKENS=agentic_cc; lane k reads the text shifted by k x 4096 tokens, so
the four lanes are four different stretches of the session). Each segment is a request at its real history position
with its own slot state (fresh caches sized to it, as test_skip_positions: the long-range history is the state's
zeros; its window carry and engram tail are the real previous chunk's).

Cases (metric suffix):
    4x2k_p139264    four 2048-token segments, each at 139264
    mixed           2048 tokens each at 61440 / 139264 / 307200 / 546816 (61k / 139k / 308k / 547k)
    1x2k_p139264    one 2048-token segment and three empty lanes (the per-segment overhead)
Every iteration (warm-up, timed, profiled) first runs the REAL previous packed chunk (each segment at p - 2048,
untimed, synced), then the timed chunk; the slots change lanes every chunk (rotated by one lane each iteration: no
lane keeps a slot's device state; every segment's history is gathered from its own slot's caches every chunk).
Timed: from the engine's device input (uint32 [SP, 1, W] lane-natural, already on the device: the engine's H2D) through
the ids readback (tt/packed_device.ids_from_input: the one readback a chunk) to the sync after the last layer. Median
of WALL_RUNS warm runs with program-cache misses refused (testing/profile.warm_up / misses_allowed). Then one run
under the real-time profiler: the slowest chip's device time and its phases.

Metrics: packed_wall_ms_<case> (gated > 0), packed_serving_tok_s_<case> (new tokens / wall), packed_device_ms_<case>,
packed_host_exposed_ms_<case> (wall - device), packed_device_ms_<phase>_<case>, packed_spread_ms_<case> and packed_runs_ms_<case> (the timed walls),
packed_dram_gib_<case> (the allocator's DRAM a chip with every segment's state and buffers held), packed_walls_table.

    BRINGUP_CHUNK=8192 DS41_PACKED=on BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_walls.py
"""

from __future__ import annotations

import importlib
import os
import re
import time
from collections import defaultdict

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import (
    WALL_RUNS,
    PositionGolden,
    cache_entries,
    misses_allowed,
    settled,
    warm_up,
)
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
CHUNK, packed = _pc.CHUNK, _pc.packed

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("PACKED_WALL_CASES",)
BATCH_LAYERS = "executed"  # F285: a performance measurement: the layers it executes (0-20 under decode)
LANE_ROWS = 2048
SHIFT = 4096  # lane k reads the text shifted by k x SHIFT
CASES = {
    "4x2k_p139264": [139264, 139264, 139264, 139264],
    "mixed": [61440, 139264, 307200, 546816],
    "1x2k_p139264": [139264, None, None, None],
}
GIB = float(1 << 30)


def _cases() -> list:
    v = os.environ.get("PACKED_WALL_CASES", "").strip()
    return [c for c in v.split(",") if c] if v else list(CASES)


def _device_split(prof: dict) -> dict:
    per_chip = defaultdict(float)
    for d in prof["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    phases = defaultdict(float)
    for sec, ns in prof["kernel_ns"].items():
        phases[profiler.phase_of(sec)] += ns
    return {
        "slowest": max(per_chip.values()) / 1e6 if per_chip else 0.0,
        "phases": {re.sub(r"[^A-Za-z0-9]+", "_", k): v / 1e6 for k, v in phases.items()},
    }


def _dram(mesh):
    try:
        from models.demos.deepseek_v41_flash_d_p.tests.bringup.test_dram import Dram

        return Dram(mesh)
    except Exception as e:  # noqa: BLE001
        print(f"DRAM view unavailable: {e}")
        return None


@mesh_parametrize(timeout_s=4 * 3600)
def test_packed_walls(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("DECODER_SKIP") == "decode", "the packed walls measure DECODER_SKIP=decode"
    # (no load-time switch set here, OV-61 / F283: the gate's one performance load serves every packed perf test)
    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    cases = _cases()
    top = max(p for c in cases for p in CASES[c] if p is not None)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    want = max(top + CHUNK, int(S.data["target"]["seq"]))
    assert top + CHUNK <= deploy + CHUNK, (top, deploy)
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    assert model.chunk == CHUNK and model.max_seq >= top + CHUNK, (model.chunk, model.max_seq, top)
    g = PositionGolden(S, top + 3 * SHIFT, CHUNK)
    text = g.tokens()
    dram = _dram(mesh_device)
    rows = []
    try:
        for case in cases:
            rows.append((case, _case(model, mesh_device, layers, case, text, dram)))
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    head = ["case", "positions", "new tokens", "wall ms (spread)", "device ms", "host exposed ms", "new tok/s"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for case, r in rows:
        lines.append(
            f"| {case} | {r['pos']} | {r['new']} | {r['wall']} ({r['spread']}) | {r['device']:.1f} | "
            f"{r['wall'] - r['device']:.1f} | {r['tok_s']:.0f} |"
        )
    table = "\n".join(lines)
    metrics.record("packed_walls_table", table)
    print("\n" + table)


def _case(model, mesh, layers, case: str, text, dram) -> dict:
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    pos = CASES[case]
    live = [k for k, p in enumerate(pos) if p is not None]
    states = {k: model.new_state(pos[k] + CHUNK) for k in live}
    toks = {k: text[k * SHIFT : k * SHIFT + pos[k] + LANE_ROWS].long() for k in live}
    it = [0]

    def chunk(prev: bool):
        """The packed chunk of this iteration: lane (k + it) % 4 holds segment k (slots move every chunk)."""
        lanes = [None] * 4
        for k in live:
            lanes[(k + it[0]) % 4] = k
        segs, ids = [], []
        for k in lanes:
            if k is None:
                segs.append(None)
                ids.append(None)
                continue
            s = pos[k] - (LANE_ROWS if prev else 0)
            segs.append(Segment(10 + k, s, s + LANE_ROWS))
            ids.append(toks[k][s : s + LANE_ROWS])
        desc = ChunkDescriptor.packed(CHUNK, segs)
        return desc, [None if k is None else states[k] for k in lanes], ids

    def run(prev: bool, timed: bool = False):
        desc, sts, ids = chunk(prev)
        inp = make_input(mesh, desc, ids)  # (the engine's H2D: not timed)
        model.sync()
        t0 = time.time()
        nat = ids_from_input(mesh, inp, desc)
        run_chunk(model, desc, nat, sts, layers, last=False)
        model.sync()
        wall = (time.time() - t0) * 1e3
        import ttnn

        ttnn.deallocate(inp)
        return wall

    def one_run():
        run(True)
        w = run(False)
        it[0] += 1
        return round(w, 1)

    try:
        with packed("on"):
            w = warm_up(mesh, one_run)
            assert w["warm"], f"{case}: not warm after {len(w['walls'])} runs"
            walls = []
            misses_allowed(mesh, False)
            try:
                for _ in range(WALL_RUNS):
                    walls.append(one_run())
            finally:
                misses_allowed(mesh, True)
            kept = sorted(walls)
            wall = kept[len(kept) // 2]
            run(True)
            n0 = cache_entries(mesh)
            profiler.enable(mesh, rt=True)
            try:
                profiler.set_layer(None)
                run(False)
                profiler.signpost("end")
                profiler.finish_rt()
                dev = _device_split(profiler.result())
            finally:
                profiler.disable()
            n1 = cache_entries(mesh)
            it[0] += 1
        new = LANE_ROWS * len(live)
        sfx = f"_{case}"
        metrics.record("packed_wall_ms" + sfx, wall)
        metrics.record("packed_runs_ms" + sfx, walls)
        metrics.record("packed_spread_ms" + sfx, round(kept[-1] - kept[0], 1))
        metrics.record("packed_wall_settled" + sfx, int(settled(w["walls"])))
        metrics.record("packed_serving_tok_s" + sfx, round(new / (wall / 1e3), 1))
        metrics.record("packed_device_ms" + sfx, round(dev["slowest"], 2))
        metrics.record("packed_host_exposed_ms" + sfx, round(wall - dev["slowest"], 1))
        metrics.record("packed_profiled_new_programs" + sfx, -1 if n0 is None or n1 is None else n1 - n0)
        for ph, ms in dev["phases"].items():
            metrics.record(f"packed_device_ms_{ph}" + sfx, round(ms, 2))
        if dram is not None:
            metrics.record("packed_dram_gib" + sfx, round(dram.max() / GIB, 3))
        print(
            f"CASE {case} positions {pos}: wall {wall} ms (runs {walls}), device {dev['slowest']:.2f} ms, "
            f"{new / (wall / 1e3):.0f} new tok/s; phases "
            + ", ".join(f"{k} {v:.2f}" for k, v in sorted(dev["phases"].items(), key=lambda x: -x[1])),
            flush=True,
        )
        return {
            "pos": "/".join(str(p) for p in pos if p is not None),
            "new": new,
            "wall": wall,
            "spread": round(kept[-1] - kept[0], 1),
            "device": dev["slowest"],
            "tok_s": new / (wall / 1e3),
        }
    finally:
        for st in states.values():
            st.free()
