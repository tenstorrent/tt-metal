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
    4xp50_p139264   (PB.5C) four 1600-token segments (the AgentX p50 turn), each at 139264: every lane padded 1600 ->
                    2048 (pad rows out of the MoE dispatch under PACKED_MOE=on)
    4x2k_p1m        (PB.5C) four 2048-token segments at 1,040,384 - k x 4096 (histories near 1M; lane k's text from
                    k x 8192, so the lanes' new tokens are four different stretches), every slot's history gathered
                    and resident at once: the DRAM budget at 1M
With 1x2k_p139264 also today's single-request chunk (PACKED=off, 8192 rows at 139264, as timed here): the per-segment
overhead's reference (packed_single8k_*_p139264, packed_segment_overhead_ms_p139264 = (4 x 2k - single) / 3).
Every iteration (warm-up, timed, profiled) first runs the REAL previous packed chunk (each segment at p - 2048,
untimed, synced), then the timed chunk; the slots change lanes every chunk (rotated by one lane each iteration: no
lane keeps a slot's device state; every segment's history is gathered from its own slot's caches every chunk).
Timed: from the engine's device input (uint32 [SP, 1, W] lane-natural, already on the device: the engine's H2D) through
the ids readback (tt/packed_device.ids_from_input: the one readback a chunk) to the sync after the last layer. Median
of WALL_RUNS warm runs with program-cache misses refused (testing/profile.warm_up / misses_allowed). Then one run
under the real-time profiler: the slowest chip's device time and its phases.

Slots (OV-62, serving's layout): every lane's slot state has ONE capacity, the deployment context rounded up to whole
chunks (PACKED_WALL_CAP overrides: a smaller one where four of them do not fit next to the model; printed with the
allocator's DRAM), so the one-ring indexer path is the one measured in every case, as tt-d-gen's equal slots take it.

History gathers (PB.3H): in the profiled run every history gather / prep op is traced (tt/history_gather.TRACE: its
device-operation ids, source layer and lane). The real-time profiler keeps one open record a command queue, so it
cannot show programs of two sub-devices at once (tt/gather_hide.py): a gather run beside the attention
(PACKED_GATHER_HIDE=on) is measured by device clock stamps instead (gather_hide.Stamps / LOG): its exposed time = the
compute's dispatch stalls on it + max(0, its end - the compute's end); a gather run serially is exposed whole (its
real-time record). The reference cost of every gather is the same chunk's serial gathers (one more profiled run with
the switch off, same lanes: the profiler's records of serial programs are exact). hidden = reference - exposed.
GATHER_RUNS profiled chunks a side, per segment the median (a host hiccup inside a region skews one run).
A full-mesh gather ends when every chip has reached it: on a chip that reached it early its time includes waiting for
the others (the lanes' skew, which the next collective absorbs in the serial design as well). So every gather (and
every region) is charged on the chip that waited least: its own cost (the reference's per-gather minimum over chips,
a region's exposure minimum over chips); packed_gather_exposed_worst_chip_ms_<case> keeps the worst chip's sum.
Metrics: packed_gather_ms_<case> (the reference),
packed_gather_hidden_ms_<case>, packed_gather_exposed_ms_<case>, packed_gather_exposed_frac_<case> (gated <= 0.25 for
4x2k_p139264 and mixed, recorded also without the text tag: the gate's names), the same per segment (_seg<k>: the case's k-th position; _segtails: the combined tails),
packed_gather_prep_ms_<case> (the packs / slices / slice_writes, local ops), packed_gather_regions_<case>.

Metrics: packed_wall_ms_<case> (gated > 0), packed_serving_tok_s_<case> (new tokens / wall), packed_device_ms_<case>,
packed_host_exposed_ms_<case> (wall - device), packed_device_ms_<phase>_<case>, packed_spread_ms_<case> and packed_runs_ms_<case> (the timed walls),
packed_dram_gib_<case> (PB.5C: the allocator's PEAK DRAM a chip while the chunk runs, every live slot's state and
history buffers held, sampled before every free (test_dram.Sampler), plus the layers the performance load does not hold
(weights and each live slot's state, charged per layer kind from a held layer: the serving model holds all 40);
packed_dram_measured_gib_<case> / packed_dram_unloaded_gib_<case> the two parts), packed_dram_headroom_gb_4x2k_p1m
(32 GiB minus packed_dram_gib_4x2k_p1m; _alloc_gib: the allocator's capacity minus it), packed_walls_table (also
<results>/<task>_packed_walls.md).

    BRINGUP_CHUNK=8192 DS41_PACKED=on BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_walls.py

PB.9F (OWNER 2026-11-20 19:00, "I absolutely need an honest measurement"): WALLS_FRESH=1 runs the FRESH walls instead of
the cases (tests/packed/fresh_common.py): case lanes4x2k, the fixed 4 x 2048 lanes (PACKED=on, PACKED_GLUED=off),
every timed chunk four NEW requests (each its own AgentX history draw, 64-aligned, >= 2048) in four NEW serving slots
(the contract's caches; warm-up and timed slots disjoint) in a random lane order, the real previous packed chunk (each
segment at p - 2048) untimed first as the cases do, program-cache misses allowed and counted; WALLS_FRESH_WARM warm-up
draws, WALLS_FRESH_RUNS timed (each run again right after: the warm-repeated number), WALLS_FRESH_PROF profiled,
WALLS_FRESH_PROBE more under the build probe (the ops that built).
PB.9C (OWNER 2026-11-21, the performance LADDER): LADDER_POS=<histories> runs the ladder instead of the cases
(tests/packed/ladder_common.py): PB.9G (chip-aligned lanes, PACKED_CHIPS): LADDER_CASES=4x2k (path lanes4x2k: four
2048-token requests, 8 chips each), 4x1600s (path lanes4x1600s: four 1600-token requests, 7 chips each, + a 1024-row
split piece, 4 chips), 4x1792s (PB.LAD, path lanes4x1792s: four 1792-token requests, 7 whole chips each, + the
5th request's first 1024 tokens, 4 chips: 8192 real tokens, no pad row; packed_fallback_chunks_<path> recorded, 0: no fallback exists), every
segment its own serving slot at history h, steady and resumed, the real previous chunk first, warm_up + WALL_RUNS median,
one profiled run (device ms), the serving wall through runtime.prefill_packed with the D2H ack a layer and segment.
Metrics ladder_<path>_{steady,resumed}_wall_ms_p<h>, ladder_<path>_tok_s_p<h>, ladder_<path>_device_ms_p<h>,
ladder_<path>_host_exposed_ms_p<h>, ladder_<path>_serving_wall_ms_p<h>, fits_<path>_p<h>; the table: the metric
ladder_<path>_table.

PB.9X (PACKED_ROUTE): the fresh lane cases (lanes4x2k; lanes4x1600: four ~1.6k requests, padded lanes) go through THE
SERVING ENTRY: the adapter's runtime (adapter.build_runtime through the model's one load path), the engine-owned KV
cache, runtime.compile (serving's warm-up, the glued layout kinds included), then runtime.prefill_packed with the
engine input made before the clock; under PACKED_ROUTE=glued (shipped) a lane header runs as its glued chunk, under
PACKED_ROUTE=lanes the fixed-lane path (PACKED_GLUED=off, PB.9F's row).
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
BATCH_ITEM_VARS = (
    "LADDER_POS",
    "LADDER_CASES",
    "LADDER_SERVING",
    "PACKED_WALL_CASES",
    "PACKED_WALL_CAP",
    "PB3H_STAMP_DUMP",
    "WALLS_FRESH",
    "WALLS_FRESH_SLOTS",
    "WALLS_FRESH_WARM",
    "WALLS_FRESH_RUNS",
    "WALLS_FRESH_PROBE",
    "WALLS_FRESH_PROF",
    "WALLS_FRESH_CASES",
)
BATCH_POS_VARS = ("LADDER_POS",)  # F288: the ladder's histories (the load's tables reach the furthest)
BATCH_LAYERS = "executed"  # F285: a performance measurement: the layers it executes (0-20 under decode)
LANE_ROWS = 2048
SHIFT = 4096  # lane k reads the text shifted by k x SHIFT
CASES = {
    "4x2k_p139264": [139264, 139264, 139264, 139264],
    "mixed": [61440, 139264, 307200, 546816],
    "1x2k_p139264": [139264, None, None, None],
    # PB.5C: the AgentX p50 turn (1600 new tokens) in every lane: each lane padded 1600 -> 2048 (pad rows out of the
    # MoE dispatch under PACKED_MOE=on)
    "4xp50_p139264": [139264, 139264, 139264, 139264],
    # PB.5C: four histories near 1M (1,040,384 - k x SHIFT), all four slots' histories gathered and resident at once
    "4x2k_p1m": [1040384 - k * SHIFT for k in range(4)],
}
LENS = {"4xp50_p139264": 1600}  # new tokens a segment (default LANE_ROWS)
# lane k reads the text from k x TEXT_SHIFT (default SHIFT). 4x2k_p1m: 2 x SHIFT, since its positions step down by
# SHIFT: with SHIFT the four lanes' new tokens would be the same text (offset + position equal); with 2 x SHIFT lane k's
# new tokens start at text position 1,040,384 + k x SHIFT, four different stretches of the session
TEXT_SHIFT = {"4x2k_p1m": 2 * SHIFT}
SINGLE_POS = 139264  # PB.5C: the single-request 8192-row chunk at the 1x2k case's history (the per-segment overhead)
GIB = float(1 << 30)
GATHER_RUNS = 3  # PB.3H: profiled chunks a side for the gather metrics (per segment the median)


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
    from models.demos.deepseek_v41_flash_d_p.tests.packed import fresh_common

    if fresh_common.on():  # PB.9F: the fresh walls (module docstring)
        return _fresh_walls(mesh_device, layers, monkeypatch)
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common

    if ladder_common.on():  # PB.9C: the ladder at real fixed histories (module docstring)
        return _ladder(mesh_device, layers, monkeypatch)
    cases = _cases()
    top = max(p for c in cases for p in CASES[c] if p is not None)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    assert top + CHUNK <= deploy + CHUNK, (top, deploy)
    cap = _cap(deploy)
    assert cap >= top + CHUNK, ("PACKED_WALL_CAP below the cases' positions", cap, top)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    assert model.chunk == CHUNK and model.max_seq >= top + CHUNK, (model.chunk, model.max_seq, top)
    need = max(_text_need(c) for c in cases)
    if "1x2k_p139264" in cases:
        need = max(need, SINGLE_POS + CHUNK)
    g = PositionGolden(S, need, CHUNK)
    text = g.tokens()
    dram = _dram(mesh_device)
    proj = _Unloaded(model, full) if dram is not None else None
    rows = []
    single = None
    try:
        for case in cases:
            rows.append((case, _case(model, mesh_device, layers, case, text, dram, cap, proj)))
        if "1x2k_p139264" in cases:  # PB.5C: the per-segment overhead's reference (today's single-request chunk)
            single = _single(model, mesh_device, layers, text, cap)
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    table = _table(rows, single, dram, proj, cap)
    metrics.record("packed_walls_table", table)
    out = metrics.results_dir() / f"{metrics.record_task()}_packed_walls.md"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n")
        print(f"walls table -> {out}")
    except OSError as e:
        print(f"walls table not written ({out}): {e}")
    print("\n" + table)


def _lens(case: str) -> int:
    return LENS.get(case, LANE_ROWS)


def _text_need(case: str) -> int:
    """Tokens of the text the case reads: lane k from k x TEXT_SHIFT to its segment's end."""
    sh, n = TEXT_SHIFT.get(case, SHIFT), _lens(case)
    return max(k * sh + p + n for k, p in enumerate(CASES[case]) if p is not None)


def _table(rows: list, single, dram, proj, cap: int) -> str:
    """The walls table (markdown): per case the chunk's wall, device, host exposed, serving new tok/s, the history
    gathers per segment (reference / exposed / hidden ms), DRAM; then the per-segment overhead."""
    head = [
        "case",
        "positions",
        "new tokens",
        "wall ms (spread)",
        "device ms",
        "host exposed ms",
        "serving new tok/s",
        "gather ms: reference / exposed / hidden",
        "per segment: exposed / hidden ms",
        "DRAM GiB a chip: loaded layers peak / 40-layer model",
    ]
    lines = [
        "# Packed prefill walls (test_packed_walls.py)",
        "",
        f"8192-row chunk, 4 lanes of {LANE_ROWS}; slots at one capacity {cap}; layers 0-20 run (DECODER_SKIP=decode). "
        "Wall = engine device input -> ids readback -> sync after the last layer (median of the warm runs). "
        "Gathers: each at its least chip (the reference = the same chunk's gathers serial). DRAM: the allocator's peak "
        "on the worst chip while the chunk runs, every live slot's state and history buffers held; the 40-layer column "
        "adds the layers the performance load does not hold (weights + each live slot's state, per layer kind, from a "
        "loaded layer of the same kind).",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for case, r in rows:
        g = r["gs"]
        per = "; ".join(f"seg{seg} {e:.2f} / {ms - e:.2f}" for seg, (ms, e) in r["segs"])
        dr = "-" if r["dram"] is None else f"{r['dram'][0]:.2f} / {r['dram'][1]:.2f}"
        lines.append(
            f"| {case} | {r['pos']} | {r['new']} | {r['wall']} ({r['spread']}) | {r['device']:.1f} | "
            f"{r['wall'] - r['device']:.1f} | {r['tok_s']:.0f} | {g[0]:.2f} / {g[1]:.2f} / {g[0] - g[1]:.2f} | {per} | {dr} |"
        )
    if dram is not None:
        capg = dram.capacity / GIB
        lines += ["", f"DRAM capacity a chip (allocator): {capg:.2f} GiB."]
        for case, r in rows:
            if r["dram"] is not None:
                lines.append(
                    f"- {case}: peak {r['dram'][0]:.2f} GiB measured ({r['dram'][2]}), 40-layer {r['dram'][1]:.2f} GiB, "
                    f"headroom {capg - r['dram'][1]:.2f} GiB (of the allocator's {capg:.2f}), "
                    f"{32.0 - r['dram'][1]:.2f} GiB of 32"
                )
        if proj is not None:
            lines.append(f"- not loaded here (projected per layer kind): {proj.describe()}")
    by = dict(rows)
    if single is not None:
        lines += [
            "",
            "## Per-segment overhead (history 139264)",
            "",
            "| chunk | new tokens | wall ms (spread) | device ms | host exposed ms | new tok/s |",
            "|---|---|---|---|---|---|",
            f"| single request, 8192 rows (PACKED=off) | {CHUNK} | {single['wall']} ({single['spread']}) | "
            f"{single['device']:.1f} | {single['wall'] - single['device']:.1f} | {single['tok_s']:.0f} |",
        ]
        for c in ("1x2k_p139264", "4x2k_p139264", "4xp50_p139264"):
            if c in by:
                r = by[c]
                lines.append(
                    f"| {c} | {r['new']} | {r['wall']} ({r['spread']}) | {r['device']:.1f} | "
                    f"{r['wall'] - r['device']:.1f} | {r['tok_s']:.0f} |"
                )
        lines.append("")
        if "1x2k_p139264" in by:
            r = by["1x2k_p139264"]
            lines.append(
                f"- 1 x 2k vs the single 8192 request: {r['wall']:.1f} vs {single['wall']:.1f} ms wall "
                f"({r['wall'] - single['wall']:+.1f}), device {r['device']:.1f} vs {single['device']:.1f} "
                f"({r['device'] - single['device']:+.1f}): a quarter of the tokens costs {r['wall'] / single['wall']:.0%} of the wall"
            )
        if "4x2k_p139264" in by:
            r = by["4x2k_p139264"]
            lines.append(
                f"- 4 x 2k vs the single 8192 request (the same 8192 new tokens): {r['wall'] - single['wall']:+.1f} ms "
                f"wall, {r['device'] - single['device']:+.1f} ms device; per extra segment "
                f"{(r['wall'] - single['wall']) / 3:+.1f} ms wall, {(r['device'] - single['device']) / 3:+.1f} ms device"
            )
            metrics.record("packed_segment_overhead_ms_p139264", round((r["wall"] - single["wall"]) / 3, 2))
    return "\n".join(lines)


class _Unloaded:
    """PB.5C (owner 2026-11-19: the DRAM budget honest at 1M): the performance load holds layers 0-20 only, but the
    serving model holds all of them (tt/runners/adapter.load_model: every layer; DECODER_SKIP=decode only skips running
    20-39). Each layer it does not hold is charged with a held layer of the same kind (ratio, KV / index / candidate
    source, engram, decoder), else of the same role (_role: its own source roles; the decoder layers 21-39 are plain
    layers, layer 20's KV / index / candidate head buffers are not theirs): its weights (every DRAM tensor its block
    reaches that no other block shares, its routed experts included) and, per live slot, its slot state (window cache,
    compressed caches). Neither: the largest held layer (an upper bound)."""

    def __init__(self, model, full: list):
        self.cfg, self.model = model.cfg, model
        self.held = list(model.layers)
        self.missing = [i for i in full if i not in set(self.held)]
        self.w, self.st, self.how = None, None, {}

    def _kind(self, i: int) -> tuple:
        c = self.cfg
        return (
            c.ratio(i),
            c.is_kv_source(i),
            c.is_index_source(i),
            c.is_cand_source(i),
            c.uses_candidates(i),
            c.is_engram(i),
            i in set(getattr(self.model, "decoder", ()) or ()),
        )

    def _role(self, i: int) -> tuple:
        """The kind without the ratio's value, the candidate use and the decoder flag: the layer's own sources (a KV /
        index / candidate source holds the history buffers and tables of its role; DECODER_SKIP=decode's decoder layers
        21-39 hold no source role: their weights and slot state are a plain layer's)."""
        c = self.cfg
        return (bool(c.ratio(i)), c.is_kv_source(i), c.is_index_source(i), c.is_cand_source(i), c.is_engram(i))

    def _weights(self) -> dict:
        from models.demos.deepseek_v41_flash_d_p.tests.bringup.test_dram import _dram_tensors

        m = self.model
        skip = {id(m), id(m.mesh), id(getattr(m, "tables", None)), id(getattr(m, "state", None))}
        found, owners = {}, defaultdict(set)
        for i in self.held:
            blk = m.blocks.get(i)
            if blk is None:
                continue
            f = _dram_tensors({f"L{i}": blk}, skip=set(skip))
            found[i] = f
            for a in f:
                owners[a].add(i)
        # (the routed experts are the MoE layer's own tensors, w_gu / w_d: tt/moe_block_v2_adapter, found by the walk)
        w = {i: sum(b for a, (b, _p) in f.items() if len(owners[a]) == 1) for i, f in found.items()}
        for i, f in found.items():  # (the evidence: each held layer's total and its largest own buffers)
            top = sorted(((b, p) for a, (b, p) in f.items() if len(owners[a]) == 1), reverse=True)[:6]
            print(
                f"[dram] L{i} kind {self._kind(i)} own {w[i] / GIB:.3f} GiB: "
                + "; ".join(f"{p} {b / 2**20:.1f} MiB" for b, p in top),
                flush=True,
            )
        return w

    def _state(self, st) -> dict:
        from models.demos.deepseek_v41_flash_d_p.tests.bringup.test_dram import _dram_tensors

        m = self.model
        skip = {id(m), id(m.mesh), id(getattr(m, "tables", None))}
        out = {}
        for i in self.held:
            roots = {"window": st.window.get(i)}
            if i in st.compress and i not in st.shared_compress:
                roots["compress"] = st.compress[i]
            out[i] = sum(b for b, _p in _dram_tensors(roots, skip=set(skip)).values())
        print(
            "[dram] slot state a layer (MiB): " + ", ".join(f"L{i} {v / 2**20:.1f}" for i, v in out.items()), flush=True
        )
        return out

    def bytes(self, st, slots: int) -> tuple:
        """(bytes a chip the missing layers add with ``slots`` live slots, how each was charged)."""
        if not self.missing:
            return 0, {}
        if self.w is None:
            self.w = self._weights()
        if self.st is None and st is not None:
            self.st = self._state(st)
        sts = self.st or {}
        by_kind, by_role = {}, {}
        for i in self.held:
            if i in self.w:
                by_kind.setdefault(self._kind(i), i)
                by_role.setdefault(self._role(i), i)
        big = max(self.w, key=lambda i: self.w[i] + slots * sts.get(i, 0))
        total, how = 0, {}
        for i in self.missing:
            j = by_kind.get(self._kind(i), by_role.get(self._role(i), big))
            how[i] = j
            total += self.w[j] + slots * sts.get(j, 0)
        self.how = how
        return total, how

    def describe(self) -> str:
        if not self.missing or self.w is None:
            return "none"
        sts, by = self.st or {}, defaultdict(list)
        for i, j in self.how.items():
            by[j].append(i)
        held = f"held layers' own weights {sum(self.w.values()) / GIB:.3f} GiB; "
        return held + "; ".join(
            f"layers {v[0]}-{v[-1]} ({len(v)}) as L{j}: weights {self.w.get(j, 0) / GIB:.3f} GiB, slot state "
            f"{sts.get(j, 0) / GIB:.3f} GiB a layer and slot"
            for j, v in sorted(by.items())
        )


def _peak_run(dram, fn) -> tuple:
    """(the allocator's peak DRAM a chip while fn(on_layer) runs, where): sampled before every DRAM free
    (ttnn.deallocate and tt/ccl's binding; allocation only grows between frees) and after every layer
    (tests/bringup/test_dram.Sampler, as P.27)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.bringup.test_dram import Sampler
    from models.demos.deepseek_v41_flash_d_p.tt import ccl

    sampler = Sampler(dram)
    sampler.mark("start")
    d0, td = ccl._DEALLOC[0], ttnn.deallocate
    ccl._DEALLOC[0] = sampler.wrap(d0)
    ttnn.deallocate = sampler.wrap(td)

    def on_layer(i, h):
        sampler.layer = i
        sampler.mark("layer_end")

    try:
        fn(on_layer)
    finally:
        ccl._DEALLOC[0], ttnn.deallocate = d0, td
    sampler.mark("done")
    peak = max(sampler.samples, key=lambda x: x[2])
    return peak[2], f"layer {peak[0]} {peak[1]}, {len(sampler.samples)} samples"


def _single(model, mesh, layers, text, cap: int) -> dict:
    """PB.5C: today's single-request chunk (PACKED=off, one 8192-row segment) at SINGLE_POS, as the walls measure
    a packed chunk (the real previous chunk first, untimed; wall from the engine's device input to the sync; median
    of WALL_RUNS warm runs; one real-time profiled run for the device time): the per-segment overhead's reference."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    st = model.new_state(cap)
    toks = text[: SINGLE_POS + CHUNK].long()

    def run(prev: bool, prof: bool = False):
        s = SINGLE_POS - (CHUNK if prev else 0)
        desc = ChunkDescriptor.single(CHUNK, 10, s, s + CHUNK)
        inp = make_input(mesh, desc, [toks[s : s + CHUNK]])
        model.sync()
        if prof:
            profiler.enable(mesh, rt=True)
            profiler.set_layer(None)
        t0 = time.time()
        try:
            nat = ids_from_input(mesh, inp, desc)
            run_chunk(model, desc, nat, [st], layers, last=False)
            model.sync()
            wall = (time.time() - t0) * 1e3
            res = None
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                res = profiler.result()
        finally:
            if prof:
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, res

    def one_run():
        run(True)
        return round(run(False)[0], 1)

    try:
        with packed("off"):
            w = warm_up(mesh, one_run)
            assert w["warm"], f"single 8192: not warm after {len(w['walls'])} runs"
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
            dev = _device_split(run(False, prof=True)[1])
    finally:
        st.free()
    sfx = f"_p{SINGLE_POS}"
    metrics.record("packed_single8k_wall_ms" + sfx, wall)
    metrics.record("packed_single8k_runs_ms" + sfx, walls)
    metrics.record("packed_single8k_device_ms" + sfx, round(dev["slowest"], 2))
    metrics.record("packed_single8k_tok_s" + sfx, round(CHUNK / (wall / 1e3), 1))
    print(
        f"SINGLE 8192 at {SINGLE_POS} (PACKED=off): wall {wall} ms (runs {walls}), device {dev['slowest']:.2f} ms, "
        f"{CHUNK / (wall / 1e3):.0f} new tok/s",
        flush=True,
    )
    return {
        "wall": wall,
        "spread": round(kept[-1] - kept[0], 1),
        "device": dev["slowest"],
        "tok_s": CHUNK / (wall / 1e3),
    }


def _cap(deploy: int) -> int:
    """OV-62: every slot's capacity (one for all lanes): PACKED_WALL_CAP, else the deployment context in whole chunks."""
    v = os.environ.get("PACKED_WALL_CAP", "").strip()
    return int(v) if v else -(-deploy // CHUNK) * CHUNK


def _serial_gathers(prof: dict, trace: list, skip: set) -> dict:
    """The traced history gathers (kinds gather / tail) the real-time profiler recorded, but those launched inside a
    gather_hide region (``skip``: their ids): {lane: ms} with each gather at its least chip time ("g"), {lane: ms} on
    the worst chip ("worst": the chip with the largest sum), and the prep ops' ms (per-op least chip time)."""
    recs, chip_of = prof.get("rt_records") or {}, prof.get("dev_to_chip") or {}
    g, prep, per = defaultdict(float), 0.0, defaultdict(lambda: defaultdict(float))
    for k, _src, lane, a, b in trace:
        for i in range(a, b):
            if i in skip or not recs.get(i):
                continue
            t = {chip_of.get(int(d), int(d)): (e[2] - e[1]) / 1e6 for d, e in recs[i].items()}
            if k in ("gather", "tail"):
                g[lane] += min(t.values())
                for c, v in t.items():
                    per[c][lane] += v
            else:
                prep += min(t.values())
    worst = max(per.values(), key=lambda v: sum(v.values())) if per else {}
    return {"g": dict(g), "worst": dict(worst), "prep": prep}


def _region_exposed(log: list, clocks: list, ghz: float) -> tuple:
    """From gather_hide.LOG and the stamps' clocks [chip][slot]: ({lane: exposed ms}: every region on the chip with
    its least exposure, {lane: ms} the worst chip's sum, the ids of the gathers launched in regions). A stall (the
    stamps "before" / "after" a gather j >= 1: the compute's launch waited for the gather before it) is charged to
    that gather's lane; the region's end (last gather end - compute end) to the last gather's lane."""
    least, total, ids = defaultdict(float), defaultdict(lambda: defaultdict(float)), set()
    for reg in log:
        out = defaultdict(lambda: defaultdict(float))
        prev, before, owed, endc = None, None, None, None
        for kind, lane, v in reg:
            if kind == "ids":
                ids.update(range(*v))
            elif kind == "before":
                before, owed = v, prev  # (the launch after it waits for the gather before: ``prev``)
            elif kind == "after":
                for c, t in enumerate(clocks):
                    out[c][owed] += max(0, int(t[v]) - int(t[before])) / ghz / 1e6
            elif kind == "end_compute":
                endc = v
            elif kind == "end_gather":
                for c, t in enumerate(clocks):
                    out[c][lane] += max(0, int(t[v]) - int(t[endc])) / ghz / 1e6
            if kind == "ids":
                prev = lane
        if out:
            c = min(out, key=lambda c: sum(out[c].values()))
            for lane, v in out[c].items():
                least[lane] += v
            for c, lanes in out.items():
                for lane, v in lanes.items():
                    total[c][lane] += v
    worst = max(total.values(), key=lambda v: sum(v.values())) if total else {}
    return dict(least), dict(worst), ids


def _gather_metrics(reg, ser, ref, regions: int, ghz: float) -> dict:
    """Per lane (reference ms, exposed ms), totals, the worst chip's exposed sum, prep ms (``reg``: _region_exposed)."""
    least, worst, _ = reg
    lanes = sorted(set(ref["g"]) | set(ser["g"]) | set(least), key=lambda k: (k is None, k))
    per = {k: (ref["g"].get(k, 0.0), ser["g"].get(k, 0.0) + least.get(k, 0.0)) for k in lanes}
    worst_exp = sum(worst.values()) + sum(ser["worst"].values())
    return {
        "ms": sum(v[0] for v in per.values()),
        "exposed": sum(v[1] for v in per.values()),
        "worst": worst_exp,
        "lanes": per,
        "prep": ser["prep"],
        "regions": regions,
        "ghz": ghz,
    }


def _gather_metrics_runs(runs_on: list, runs_ref: list, ghz: float) -> dict:
    """GATHER_RUNS profiled chunks a side: per segment the median of its runs' exposed and reference ms (a host
    hiccup inside one region of one run skews that run only)."""
    import statistics

    one = [_gather_metrics(reg, ser, ref, n, ghz) for (reg, ser, n), ref in zip(runs_on, runs_ref)]
    lanes = sorted({k for g in one for k in g["lanes"]}, key=lambda k: (k is None, k))
    per = {
        k: (
            statistics.median(g["lanes"].get(k, (0.0, 0.0))[0] for g in one),
            statistics.median(g["lanes"].get(k, (0.0, 0.0))[1] for g in one),
        )
        for k in lanes
    }
    print(
        f"history gathers, each run (reference / exposed ms): {[(round(g['ms'], 3), round(g['exposed'], 3)) for g in one]}"
    )
    return {
        "ms": sum(v[0] for v in per.values()),
        "exposed": sum(v[1] for v in per.values()),
        "worst": statistics.median(g["worst"] for g in one),
        "lanes": per,
        "prep": statistics.median(g["prep"] for g in one),
        "regions": one[0]["regions"],
        "ghz": ghz,
    }


def _ghz(mesh, st) -> float:
    """The stamps' clock rate (ticks a ns): two stamps 0.2 s apart on the host clock (chip 0)."""
    import ttnn

    t0 = time.perf_counter()
    a = st("compute")
    ttnn.synchronize_device(mesh)
    time.sleep(0.2)
    t1 = time.perf_counter()
    b = st("compute")
    ttnn.synchronize_device(mesh)
    c = st.read()[0]
    return float(int(c[b]) - int(c[a])) / ((t1 - t0) * 1e9)


def _record_gate_name(name: str, value) -> None:
    """The gate's threshold names carry no text tag: record the metric under both names."""
    metrics.record(name, value)
    tok = os.environ.pop("BRINGUP_POS_TOKENS", None)
    try:
        metrics.record(name, value)
    finally:
        if tok is not None:
            os.environ["BRINGUP_POS_TOKENS"] = tok


def _case(model, mesh, layers, case: str, text, dram, cap: int, proj=None) -> dict:
    from models.demos.deepseek_v41_flash_d_p.tt import gather_hide, history_gather
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    pos = CASES[case]
    live = [k for k, p in enumerate(pos) if p is not None]
    n_new, sh = _lens(case), TEXT_SHIFT.get(case, SHIFT)  # (PB.5C: a segment's new tokens; lane k's text offset)
    states = {k: model.new_state(cap) for k in live}  # (OV-62: one capacity, serving's slots)
    toks = {k: text[k * sh : k * sh + pos[k] + n_new].long() for k in live}
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
            s = pos[k] - (n_new if prev else 0)  # (the real previous turn chunk: the same length before it)
            segs.append(Segment(10 + k, s, s + n_new))
            ids.append(toks[k][s : s + n_new])
        desc = ChunkDescriptor.packed(CHUNK, segs)
        return desc, [None if k is None else states[k] for k in lanes], ids

    def run(prev: bool, timed: bool = False, on_layer=None):
        desc, sts, ids = chunk(prev)
        inp = make_input(mesh, desc, ids)  # (the engine's H2D: not timed)
        model.sync()
        t0 = time.time()
        nat = ids_from_input(mesh, inp, desc)
        run_chunk(model, desc, nat, sts, layers, last=False, on_layer=on_layer)
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

            def profiled():
                profiler.enable(mesh, rt=True)
                history_gather.TRACE = []
                gather_hide.LOG.clear()
                try:
                    profiler.set_layer(None)
                    run(False)
                    profiler.signpost("end")
                    profiler.finish_rt()
                    return profiler.result(), list(history_gather.TRACE), list(gather_hide.LOG)
                finally:
                    history_gather.TRACE = None
                    profiler.disable()

            gather_hide.warm_stamps(mesh)  # (the stamp programs' first use compiles: before the measured run)
            st = gather_hide.stamps(mesh)
            ghz = _ghz(mesh, st) if st is not None else 1.0
            runs_on, runs_ref = [], []
            for k in range(GATHER_RUNS):  # (each with the real previous chunk first; the lanes of this iteration)
                run(True)
                c0 = cache_entries(mesh)
                prof, trace, log = profiled()
                if k == 0:
                    dev = _device_split(prof)
                    n0, n1 = c0, cache_entries(mesh)
                clocks = st.read() if (st is not None and log) else []
                reg = _region_exposed(log, clocks, ghz)
                runs_on.append((reg, _serial_gathers(prof, trace, reg[2]), len(log)))
                dump = os.environ.get("PB3H_STAMP_DUMP", "").strip()  # (diagnostic: the raw stamps, offline reading)
                if dump and clocks:
                    import json

                    used = sorted({v for reg_ in log for k_, _, v in reg_ if k_ not in ("ids",)})
                    with open(f"{dump}.{case}.{k}.json", "w") as f:
                        json.dump({"log": log, "ghz": ghz, "clocks": {i: [int(c[i]) for c in clocks] for i in used}}, f)
                if settings.get("PACKED_GATHER_HIDE") == "on":  # (the reference: the same chunk's gathers, serial)
                    with _pc.switch("PACKED_GATHER_HIDE", "off"):
                        run(True)
                        prof_off, trace_off, _ = profiled()
                    runs_ref.append(_serial_gathers(prof_off, trace_off, set()))
                else:
                    runs_ref.append(runs_on[-1][1])
            gs = _gather_metrics_runs(runs_on, runs_ref, ghz)
            seg_of = {(k + it[0]) % 4: k for k in live}  # (the profiled chunk's lanes: segment k in lane (k + it) % 4)
            it[0] += 1
            peak = None
            if dram is not None:  # PB.5C: the allocator's peak a chip while the chunk runs, every slot held
                run(True)
                peak = _peak_run(dram, lambda on_layer: run(False, on_layer=on_layer))
                it[0] += 1
        new = n_new * len(live)
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
        dr = None
        if peak is not None:
            extra, _how = proj.bytes(states[live[0]], len(live)) if proj is not None else (0, {})
            meas, dep, capg = peak[0] / GIB, (peak[0] + extra) / GIB, dram.capacity / GIB
            dr = (meas, dep, peak[1])
            metrics.record("packed_dram_gib" + sfx, round(dep, 3))  # (the serving model's: 40 layers, PB.5C)
            metrics.record("packed_dram_measured_gib" + sfx, round(meas, 3))  # (the layers this load holds)
            metrics.record("packed_dram_unloaded_gib" + sfx, round(extra / GIB, 3))
            metrics.record("packed_dram_capacity_gib" + sfx, round(capg, 3))
            if case == "4x2k_p1m":
                _record_gate_name("packed_dram_headroom_gb" + sfx, round(32.0 - dep, 3))
                metrics.record("packed_dram_headroom_alloc_gib" + sfx, round(capg - dep, 3))
            print(
                f"CASE {case} DRAM a chip: peak {meas:.3f} GiB measured ({peak[1]}; {len(live)} slots of capacity "
                f"{cap} held), + {extra / GIB:.3f} GiB for the layers not loaded ({proj.describe() if proj else '-'}) "
                f"= {dep:.3f} GiB; allocator capacity {capg:.3f} GiB, headroom {capg - dep:.3f} (of 32: {32.0 - dep:.3f})",
                flush=True,
            )
        exp = gs["exposed"]
        frac = exp / gs["ms"] if gs["ms"] > 0 else 0.0
        metrics.record("packed_gather_ms" + sfx, round(gs["ms"], 3))
        metrics.record("packed_gather_hidden_ms" + sfx, round(gs["ms"] - exp, 3))
        metrics.record("packed_gather_exposed_ms" + sfx, round(exp, 3))
        _record_gate_name("packed_gather_exposed_frac" + sfx, round(frac, 4))
        metrics.record("packed_gather_prep_ms" + sfx, round(gs["prep"], 3))
        metrics.record("packed_gather_regions" + sfx, gs["regions"])
        metrics.record("packed_gather_exposed_worst_chip_ms" + sfx, round(gs["worst"], 3))
        names = {lane: ("tails" if lane == -1 else seg_of.get(lane, f"lane{lane}")) for lane in gs["lanes"]}
        for lane, (ms, e) in gs["lanes"].items():  # (segment k = the case's k-th position; "tails": every segment's)
            metrics.record(f"packed_gather_ms{sfx}_seg{names[lane]}", round(ms, 3))
            metrics.record(f"packed_gather_hidden_ms{sfx}_seg{names[lane]}", round(ms - e, 3))
            metrics.record(f"packed_gather_exposed_ms{sfx}_seg{names[lane]}", round(e, 3))
        lanes_s = ", ".join(f"seg {names[k]} (lane {k}): {v[0]:.3f} / {v[1]:.3f}" for k, v in gs["lanes"].items())
        print(
            f"CASE {case} history gathers (each at its least chip): reference (serial) {gs['ms']:.3f} ms, exposed {exp:.3f}, "
            f"hidden {gs['ms'] - exp:.3f} (exposed frac {frac:.3f}); per segment (reference / exposed ms) {lanes_s}; "
            f"{gs['regions']} regions; worst chip's exposed sum {gs['worst']:.3f} ms; prep ops {gs['prep']:.3f} ms; clock "
            f"{gs['ghz']:.4f} GHz; slot capacity {cap}",
            flush=True,
        )
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
            "gs": (gs["ms"], gs["exposed"]),
            "segs": [(names[k], v) for k, v in gs["lanes"].items()],
            "dram": dr,
        }
    finally:
        for st in states.values():
            st.free()


# ---------------------------------------------------------------------------------------------- PB.9C the ladder
def _ladder(mesh, layers, monkeypatch) -> None:
    """PB.9C / PB.9G: the packed ladder (tests/packed/ladder_common.packed_ladder) through the serving runtime's slots
    (adapter.build_runtime through the model's one load path; its KV cache of one slot a segment at the walls'
    capacity). PB.9G (chip-aligned lanes, PACKED_CHIPS=on): LADDER_CASES 4x2k (path lanes4x2k: four 2048-token
    requests, 8 chips each) and 4x1600s (path lanes4x1600s: four 1600-token requests, 7 chips each, + the 5th request's
    first 1024 rows, 4 chips: a split piece); PB.LAD: 4x1792s (path lanes4x1792s: four 1792-token requests, 7 whole
    chips each, + the 5th request's first 1024 tokens, 4 chips: 8192 real tokens, no pad row); every chunk runs the
    one-call glued path (no fallback)."""
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as L
    from models.demos.deepseek_v41_flash_d_p.tt import packed_device as PD
    from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk

    L = importlib.reload(L)
    cases = L.cases(["4x2k"])
    assert set(cases) <= {
        "4x2k",
        "4x1600s",
        "4x1792s",
    }, f"test_packed_walls' ladder runs 4x2k / 4x1600s / 4x1792s (got {cases})"
    chips = settings.get("PACKED_CHIPS")
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = _cap(deploy)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    nmax = max(len(L.CASES[c]) for c in cases)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(nmax))  # (the runtime's KV cache: one slot a segment)
    runtime, adapter, hf, params = SP._runtime(mesh, layers, CHUNK, want)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    pos = L.positions()
    text = PositionGolden(S, (nmax - 1) * SHIFT + max(pos) + 2048 + CHUNK, CHUNK).tokens()
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
    all_rows = []
    try:
        for case in cases:
            path = f"lanes{case}"
            lens = L.CASES[case]
            print(f"LADDER {path}: PACKED_CHIPS={chips}, histories {pos}, {len(lens)} slots at {want}", flush=True)
            PD.STATS["fallback"] = 0
            with packed("on"), _pc.switch("PACKED_GLUED", "on"):
                rows = L.packed_ladder(
                    runtime,
                    kv,
                    mesh,
                    layers,
                    path,
                    lens,
                    (lambda segs: ChunkDescriptor.packed(CHUNK, segs))
                    if case == "4x2k"
                    else (lambda segs: GluedChunk(CHUNK, tuple(segs))),
                    text,
                    SHIFT,
                    cap,
                )
            L.record(f"packed_fallback_chunks_{path}", int(PD.STATS["fallback"]))
            what = {
                "4x2k": "four 2048-token segments, each at history h",
                "4x1600s": "four 1600-token segments + a 1024-row split piece, each at history h",
                "4x1792s": "four 1792-token segments (7 whole chips each) + a 1024-token split piece (4 chips), "
                "8192 real tokens (no pad row), each at history h",
            }[case]
            L.ladder_table(path, f"{what} (chip-aligned lanes, PACKED_CHIPS={chips})", rows)
            all_rows += rows
    finally:
        for key in [k for k in runtime._states if k[0] == id(kv)]:
            try:
                runtime._states.pop(key).free()
            except Exception as e:  # noqa: BLE001
                print(f"state free: {e}")
        for t in kv.contract.by_config.values():
            if t.is_allocated():
                import ttnn

                ttnn.deallocate(t)
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    assert any(r is not None for _, r, _ in all_rows), "no history of the ladder ran"


# ----------------------------------------------------------------------------------------------- PB.9F fresh walls
FRESH_LANE_CASES = ("lanes4x2k", "lanes4x1600")  # PB.9X: + the padded 4 x ~1.6k lanes


def _fresh_walls(mesh, layers, monkeypatch) -> None:
    """PB.9F / PB.9X: the lane cases' fresh walls THROUGH THE SERVING ENTRY: the serving runtime exactly as
    prefill_runner builds it (adapter.build_runtime through the model's one load path: the batch's held model), the
    engine-owned KV cache (adapter.allocate_kv_cache, WALLS_FRESH_SLOTS slots at the deployment context),
    runtime.compile, then every chunk through runtime.prefill_packed with the engine's input made before the clock
    (adapter.make_packed_input). Under the shipped switches (PACKED_ROUTE=glued) a lane header runs the glued path."""
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP
    from models.demos.deepseek_v41_flash_d_p.tests.packed import fresh_common as F

    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = _cap(deploy)
    want = max(cap, int(S.data["target"]["seq"]))
    monkeypatch.setitem(S.data["target"], "seq", want)
    rt = SP._runtime(mesh, layers, CHUNK, want)
    runtime, adapter, hf, params = rt
    model = runtime.model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    hi = min(cap, model.max_seq) - LANE_ROWS
    text = PositionGolden(S, 3 * SHIFT + hi, CHUNK).tokens()  # (lane k reads from k x SHIFT)
    t0 = time.time()
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)  # (the engine-owned caches)
    runtime.compile(kv)  # (the engine's warm-up at startup)
    print(f"FRESH KV cache ({params.num_users} slots at {want}) + runtime.compile in {time.time() - t0:.1f} s")
    slots = F.Slots(runtime, kv)
    try:
        for case in F.cases(list(FRESH_LANE_CASES)):
            try:
                row = _fresh_lanes(runtime, kv, mesh, case, hi, text, slots)
            except Exception as e:  # (owner: a path that fails on fresh positions is recorded, not fixed)
                if F.is_hang(e):
                    raise
                import traceback

                traceback.print_exc()
                row = F.failed(case, _lane_path(), e)
            F.record(case, row)
    finally:
        slots.free()
        if loaded != "decode":
            model.use_decoder_skip(loaded)


def _lane_path() -> str:
    route = "glued"  # (PB.9G: PACKED_ROUTE removed; every packed chunk runs glued)
    return (
        "4 lanes sent as glued chunks (PACKED_ROUTE=glued, prefill_packed)"
        if route == "glued"
        else "fixed 4 x 2048 lanes (PACKED_ROUTE=lanes, PACKED_GLUED=off, prefill_packed)"
    )


def _fresh_lanes(runtime, kv, mesh, case: str, hi: int, text, slots) -> dict:
    """One fresh lane case through runtime.prefill_packed. lanes4x2k: four 2048-token requests; lanes4x1600 (PB.9X):
    four ~1.6k requests (1536..1664 new tokens, the lanes' pad rows), each at its own AgentX history draw in a new
    serving slot, in a random lane order; the real previous chunk (each request at p - 2048) untimed first."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.packed import fresh_common as F

    nwarm, nrun = F.knob("WALLS_FRESH_WARM", 4), F.knob("WALLS_FRESH_RUNS", 8)
    nprobe, nprof = F.knob("WALLS_FRESH_PROBE", 3), F.knob("WALLS_FRESH_PROF", 3)
    base = 400000 + CHUNK + (0 if case == "lanes4x2k" else 50000)
    seeds = {
        "warm": [base + 900 + w for w in range(nwarm)],
        "timed": [base + r for r in range(nrun)],
        "probe": [base + 500 + j for j in range(nprobe)],
    }
    print(f"FRESH {case}: seeds {seeds}", flush=True)
    route = "glued"  # (PB.9G: PACKED_ROUTE removed; every packed chunk runs glued)

    def draw(seed: int, phase: str) -> dict:
        d = F.Draw(seed)
        pos = [d.history(lo=LANE_ROWS, hi=hi) for _ in range(4)]
        sl = slots.pick(4, phase, d.rnd)
        lanes = list(range(4))
        d.rnd.shuffle(lanes)  # lanes[k]: the lane of request k
        lens = [LANE_ROWS] * 4 if case == "lanes4x2k" else [1536 + d.rnd.randrange(129) for _ in range(4)]
        return {"seed": seed, "pos": pos, "slots": sl, "lanes": lanes, "lens": lens, "new": sum(lens)}

    def run(dr: dict, prev: bool, prof: bool = False):
        segs, ids = [None] * 4, [None] * 4
        for k in range(4):
            n = LANE_ROWS if prev else dr["lens"][k]  # (the previous chunk: a whole lane, ending at a kv block)
            s = dr["pos"][k] - (LANE_ROWS if prev else 0)
            segs[dr["lanes"][k]] = Segment(dr["slots"][k], s, s + n)
            ids[dr["lanes"][k]] = text[k * SHIFT + s : k * SHIFT + s + n].long()
        desc = ChunkDescriptor.packed(CHUNK, segs)
        inp = runtime.make_packed_input(desc, ids)  # (the engine's H2D: not timed)
        runtime.model.sync()
        if prof:
            profiler.enable(mesh, rt=True)
            profiler.set_layer(None)
        t0 = time.time()
        dev = None
        try:
            runtime.prefill_packed(inp, kv, desc)
            runtime.model.sync()
            wall = (time.time() - t0) * 1e3
            if prof:
                profiler.signpost("end")
                profiler.finish_rt()
                dev = _device_split(profiler.result())["slowest"]
        finally:
            if prof:
                profiler.disable()
            ttnn.deallocate(inp)
        return wall, dev

    with packed("on"), _pc.switch("PACKED_GLUED", "on" if route == "glued" else "off"):
        for sd in seeds["warm"]:
            dr = draw(sd, "warm")
            run(dr, True)
            w, _ = run(dr, False)
            print(f"FRESH {case} warm-up {dr}: {w:.1f} ms", flush=True)
        fresh, repeat, prof, builds, draws = [], [], [], 0, []
        pc, preps = F.PrepCount(), [0, 0.0]  # (prepared programs built in the fresh runs: no cache entry of their own)
        for i, sd in enumerate(seeds["timed"]):
            dr = draw(sd, "timed")
            e0 = cache_entries(mesh)
            with pc:
                p0, m0 = pc.n, pc.ms
                run(dr, True)
                wf, _ = run(dr, False)
                preps[0] += pc.n - p0
                preps[1] += pc.ms - m0
            e1 = cache_entries(mesh)
            built = -1 if e0 is None or e1 is None else e1 - e0
            builds += max(built, 0)
            run(dr, True)  # the warm-repeated run: the same chunks again
            wr, _ = run(dr, False)
            fresh.append((wf, dr["new"]))
            repeat.append((wr, dr["new"]))
            dev = None
            if i < nprof:
                run(dr, True)
                _, dev = run(dr, False, prof=True)
                prof.append((wf, wr, dev))
            draws.append(
                {
                    **dr,
                    "fresh_ms": round(wf, 1),
                    "repeat_ms": round(wr, 1),
                    "built": built,
                    "prepared": pc.n - p0,
                    "device_ms": dev,
                }
            )
            print(f"FRESH {case} timed {dr}: fresh {wf:.1f} ms (+{built} programs), repeated {wr:.1f} ms, device {dev}")
        with F.BuildProbe(mesh) as pb:
            for sd in seeds["probe"]:
                dr = draw(sd, "timed")
                run(dr, True)
                run(dr, False)
        pb.print(case)
    extra = {"seeds": seeds, "draws": draws, "probe_chunks": 2 * len(seeds["probe"])}
    return F.summarize(case, _lane_path(), fresh, repeat, prof, builds, pb, extra, tuple(preps))
