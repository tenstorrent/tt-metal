# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.3 (owner 2026-11-19): the packed prefill's correctness oracle on the device (bringup/PACKED_DESIGN.md s9 item 2).
Every check is bit-identity (a max abs diff that must be 0); nothing here has a tolerance.

Segments (real text: the golden prompt's ids at different offsets; each its own slot, its history written first by
today's path, PACKED=off, in the same 8192 geometry):
    F   slot 1, start 0,    2048 rows (a fresh request filling its lane)
    P   slot 2, start 2048, 1600 rows (a continuing request, padded lane)
    O   slot 3, start 4096, 777 rows  (continuing, odd length: a ratio-2 group left pending at its end)
    Q   slot 4, start 2048, 40 rows   (PREFILL_END=cap: cap 2048 <= start: nothing computed, its snapshot and acks)
    G   slot 5, start 16384, 2048 rows (PB.3H: two whole chunks of history: every source's history gathered, the
        prefetch and the tail; PACKED_GATHER_HIDE=on hides them in packed3)
    H   slot 6, start 9216, 1500 rows  (PB.3H: one chunk + 1024 rows of history, padded lane)
Runs (every run on fresh slot states, the prefixes replayed: the device path is deterministic):
    today[S]   S as today's single-request chunk (PACKED=off)              S in F, P, O
    alone[S]   S alone in lane 0 through PACKED=on                          S in F, P, O, Q
    packed1    [F, P, O, Q]       (lanes 0-3)                               PACKED=on
    packed2    [O, F, -, P]       (every segment in another lane, an empty lane)
    packed3    [G, O, H, F]       (PB.3H: two segments with whole chunks of history)
    packed1'   packed1 again, packed3' packed3 again (determinism)
Every slot's caches have one capacity (MAX_SEQ: serving's slots are one size).
Metrics (gated == 0):
    packed_alone_vs_today_max_abs_diff   alone[S] vs today[S], every layer output (S's rows: streams and pre)
    packed_lane_equiv_max_abs_diff       packed1[S], packed2[S] vs alone[S], every layer output
    packed_slot_digest_mismatches        sha256 of every served record of each slot (window ring rows and snapshot,
                                         compressed KV, index keys; every layer) packed1 / packed2 vs alone[S]
    packed_determinism_max_abs_diff      packed1' vs packed1 (layer outputs and slot digests)
A layer's diff is 0 when the digests of S's rows match; else the max abs diff over 64 sampled rows (>= 1e-30).

Layers: the loaded model's (the gate: every layer, one load with the other accuracy tests, OV-61); DS41_PACKED_LAYERS
(comma list) for the iteration rungs (one layer, the characteristic layers) on the resident runner.

    BRINGUP_CHUNK=8192 scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_packed_oracle.py
"""

from __future__ import annotations

import importlib
import os
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
CHUNK, PAD_ID, Capture, boundary_rows, packed, served_digests = (
    _pc.CHUNK,
    _pc.PAD_ID,
    _pc.Capture,
    _pc.boundary_rows,
    _pc.packed,
    _pc.served_digests,
)
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("DS41_PACKED_LAYERS",)  # (the iteration rungs: the layers computed; the model is the gate's)
MAX_SEQ = 32768  # (state caches, one size: positions < 16384 + 8192)
SEGS = {
    "F": dict(slot=1, start=0, n=2048, off=0),
    "P": dict(slot=2, start=2048, n=1600, off=8192),
    "O": dict(slot=3, start=4096, n=777, off=16384),
    "Q": dict(slot=4, start=2048, n=40, off=24576),
    "G": dict(slot=5, start=16384, n=2048, off=4096),
    "H": dict(slot=6, start=9216, n=1500, off=12288),
}
COMPUTED = ("F", "P", "O", "G", "H")
COMPS = {"packed1": ["F", "P", "O", "Q"], "packed2": ["O", "F", None, "P"], "packed3": ["G", "O", "H", "F"]}


def _layers(model) -> list:
    v = os.environ.get("DS41_PACKED_LAYERS", "").strip()
    if v:
        want = [int(x) for x in v.split(",") if x.strip()]
        assert all(i in model.layers for i in want), (want, model.layers)
        return want
    return list(model.layers)


class Run:
    def __init__(self, tokens):
        self.tokens = tokens

    def toks(self, name):
        d = SEGS[name]
        return self.tokens[d["off"] : d["off"] + d["start"] + d["n"]].long()

    def seg(self, name):
        d = SEGS[name]
        return Segment(d["slot"], d["start"], d["start"] + d["n"])


def _fresh_state(model, layers, run, name):
    """A slot state holding segment ``name``'s history [0, start) (today's path, one chunk a CHUNK rows)."""
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    st = model.new_state(MAX_SEQ)
    d = SEGS[name]
    for c0 in range(0, d["start"], CHUNK):
        end = min(c0 + CHUNK, d["start"])
        ids = torch.full((CHUNK,), PAD_ID, dtype=torch.long)
        ids[: end - c0] = run.toks(name)[c0:end]
        desc = ChunkDescriptor.single(CHUNK, d["slot"], c0, end)
        with packed("off"):
            run_chunk(model, desc, ids, [st], layers)
    return st


def _run(model, layers, run, lanes: list, mode: str):
    """One chunk with ``lanes`` (segment names / None) -> (Capture of every computed segment's layer outputs,
    {segment: served digests})."""
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    names = [n for n in lanes if n is not None]
    states = {n: _fresh_state(model, layers, run, n) for n in names}
    desc = ChunkDescriptor.packed(CHUNK, [None if n is None else run.seg(n) for n in lanes])
    ids_lane = [None if n is None else run.toks(n)[SEGS[n]["start"] :] for n in lanes]
    ids = desc.pack_ids(ids_lane)
    cap = Capture()
    from models.demos.deepseek_v41_flash_d_p.tt.attention import prefill_end

    def on_layer(i, h):
        if not hasattr(h, "streams"):
            return
        for lane, n in enumerate(lanes):
            if n is None or n not in COMPUTED:
                continue
            s = run.seg(n)
            k = max(0, prefill_end(s.end) - s.start)
            cap.add((n, i), boundary_rows(model, h, desc, lane, k))

    t0 = time.time()
    with packed(mode):
        run_chunk(
            model, desc, ids, [states.get(n) if n is not None else None for n in lanes], layers, on_layer=on_layer
        )
    model.sync()
    dig = {n: served_digests(states[n], layers, SEGS[n]["start"] + SEGS[n]["n"]) for n in names}
    for st in states.values():
        st.free()
    print(f"  run {mode} {lanes}: {time.time() - t0:.1f} s")
    return cap, dig


def _dig_mismatch(a: dict, b: dict) -> int:
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


@mesh_parametrize
def test_packed_oracle(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    assert model.chunk == CHUNK, model.chunk
    if model.skip_mode != "off":
        model.use_decoder_skip("off")  # (every layer computes; FIX.28: host lists only)
    layers = _layers(model)
    print(f"packed oracle: layers {layers}")
    g = Golden.for_rung(S, "s56320")
    run = Run(g.tokens())

    keys = lambda n: [(n, i) for i in layers]  # noqa: E731
    today, alone = {}, {}
    for n in COMPUTED:
        today[n] = _run(model, layers, run, [n, None, None, None], "off")
    for n in SEGS:
        alone[n] = _run(model, layers, run, [n, None, None, None], "on")
    comps = {c: _run(model, layers, run, lanes, "on") for c, lanes in COMPS.items()}
    again = _run(model, layers, run, COMPS["packed1"], "on")
    again3 = _run(model, layers, run, COMPS["packed3"], "on")

    a_t = max(Capture.diff(alone[n][0], today[n][0], keys(n)) for n in COMPUTED)
    a_t_dig = sum(_dig_mismatch(alone[n][1][n], today[n][1][n]) for n in COMPUTED)
    lane_eq, digs, worst = 0.0, 0, []
    for c, (cap, dig) in comps.items():
        for n in COMPS[c]:
            if n is None:
                continue
            if n in COMPUTED:
                d = Capture.diff(cap, alone[n][0], keys(n))
                lane_eq = max(lane_eq, d)
                if d:
                    bad = [i for i in layers if cap.dig.get((n, i)) != alone[n][0].dig.get((n, i))]
                    worst.append(f"{c} {n}: first layer differing {bad[:1]}, layers {bad}")
            m = _dig_mismatch(dig[n], alone[n][1][n])
            digs += m
            if m:
                bad = sorted({k for k in set(dig[n]) | set(alone[n][1][n]) if dig[n].get(k) != alone[n][1][n].get(k)})
                worst.append(f"{c} {n}: served digests differ {bad[:6]}")
    det = 0.0
    det_dig = 0
    for c, rep in (("packed1", again), ("packed3", again3)):
        det = max(det, Capture.diff(rep[0], comps[c][0], [k for n in COMPS[c] if n in COMPUTED for k in keys(n)]))
        det_dig += sum(_dig_mismatch(rep[1][n], comps[c][1][n]) for n in rep[1])
    if det_dig and not det:
        det = 1e-30
    metrics.record("packed_alone_vs_today_max_abs_diff", a_t)
    metrics.record("packed_alone_vs_today_digest_mismatches", a_t_dig)
    metrics.record("packed_lane_equiv_max_abs_diff", lane_eq)
    metrics.record("packed_slot_digest_mismatches", digs)
    metrics.record("packed_determinism_max_abs_diff", det)
    metrics.record("packed_oracle_layers", len(layers))
    print(
        f"packed oracle: alone vs today {a_t} (digests {a_t_dig}), lane equivalence {lane_eq}, slot digest "
        f"mismatches {digs}, determinism {det} (digests {det_dig})"
    )
    from models.demos.deepseek_v41_flash_d_p.tt import gather_hide, history_gather

    print(f"  history gathers since load: {history_gather.STATS}; PACKED_GATHER_HIDE regions: {gather_hide.STATS}")
    metrics.record("packed_gather_hide_regions", gather_hide.STATS["regions"])
    for w in worst:
        print("  " + w)
    assert a_t == 0 and lane_eq == 0 and digs == 0 and det == 0, "packed oracle: not bit-identical (see above)"
