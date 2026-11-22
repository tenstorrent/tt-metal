# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7 (owner 2026-11-20): the glued packed prefill's correctness oracle on the device. Every check is bit-identity
(a max abs diff that must be 0); nothing here has a tolerance.

Glued chunks (tt/packed.GluedChunk, PACKED_GLUED=on): up to 6 segments back to back in 8192 rows, any lengths; model
rows = each segment's computed rows [start, cap) back to back. On 8x4 (256 rows a chip) most segments start and end
inside a chip, so chips hold rows of two segments and segments span chips.

Segments (real text: the golden prompt's ids at different offsets; each its own slot, its history written first by
today's path, PACKED=off, in the same 8192 geometry):
    F  slot 1, start 0,     2048 rows        P  slot 2, start 2048, 1600 rows      O  slot 3, start 4096, 777 rows
    Q  slot 4, start 2048,  40 rows (cap <= start: nothing computed: its snapshot and acks only)
    G  slot 5, start 16384, 2048 rows        H  slot 6, start 9216, 1500 rows      I  slot 7, start 6144, 160 rows
    J  slot 9, start 2048,  2048 rows        K  slot 10, start 8192, 2048 rows
    X  slot 8, start 4096,  3000 rows (split: its first part ends a glued chunk, the rest starts the next)
Runs (fresh slot states every run, prefixes replayed: the device path is deterministic):
    today[S]   S as today's single-request chunk (PACKED=off)                            S computed
    alone[S]   S alone at model row 0 through PACKED=on, PACKED_GLUED=on                  every S
    glued1     [P, O, H, F, Q]          (O at row 1600: chip 6 holds P and O; H at 2368, F at 3840)
    glued2     [G, H, O, P, F, I]       (6 segments, 8064 model rows, every segment at another row than in glued1)
    glued4     [F, J, G, K]             (4 x 2048: the lane chunk, the accepted lane path)
    glued2'    glued2 again (determinism)
    splitA     [G, H, F, X1]  X1 = X's first part (ends on the last kv block that fits)    splitB  [X2, O, P]
    splitX     X1 alone, then X2 alone on the same slot (the same request chunked at the same boundary)
Every slot's caches have one capacity (MAX_SEQ). Metrics (gated == 0):
    glued_alone_vs_today_max_abs_diff   alone[S] vs today[S], every layer output (S's rows: streams and pre) + digests
    glued_segment_equiv_max_abs_diff    glued1 / glued2 / glued4 [S] vs alone[S], every layer output
    glued_split_vs_alone_max_abs_diff   splitA[X1] vs splitX's first chunk, splitB[X2] vs its second, every layer
    glued_slot_digest_mismatches        sha256 of every served record of each slot (window ring rows and snapshot,
                                        compressed KV, index keys; every layer) glued vs alone; X after splitB vs
                                        after splitX
    glued_determinism_max_abs_diff      glued2' vs glued2 (layer outputs and slot digests)

Layers: the loaded model's (the gate's accuracy load: every layer, OV-61); DS41_PACKED_LAYERS (comma list) for the
iteration rungs on the resident runner.

    BRINGUP_CHUNK=8192 DS41_PACKED=on DS41_PACKED_GLUED=on scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_oracle.py
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
CHUNK, PAD_ID, Capture, packed, glued, served_digests = (
    _pc.CHUNK,
    _pc.PAD_ID,
    _pc.Capture,
    _pc.packed,
    _pc.glued,
    _pc.served_digests,
)
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, GluedChunk, Segment, split

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("DS41_PACKED_LAYERS",)  # (the iteration rungs: the layers computed; the model is the gate's)
MAX_SEQ = 32768  # (state caches, one size: as test_packed_oracle)
SEGS = {
    "F": dict(slot=1, start=0, n=2048, off=0),
    "P": dict(slot=2, start=2048, n=1600, off=8192),
    "O": dict(slot=3, start=4096, n=777, off=16384),
    "Q": dict(slot=4, start=2048, n=40, off=24576),
    "G": dict(slot=5, start=16384, n=2048, off=4096),
    "H": dict(slot=6, start=9216, n=1500, off=12288),
    "I": dict(slot=7, start=6144, n=160, off=20480),
    "J": dict(slot=9, start=2048, n=2048, off=34000),
    "K": dict(slot=10, start=8192, n=2048, off=40000),
    "X": dict(slot=8, start=4096, n=3000, off=28000),
}
COMPUTED = ("F", "P", "O", "G", "H", "I", "J", "K")
COMPS = {
    "glued1": ["P", "O", "H", "F", "Q"],
    "glued2": ["G", "H", "O", "P", "F", "I"],
    "glued4": ["F", "J", "G", "K"],
}
SPLIT_A = ["G", "H", "F"]  # + X's first part
SPLIT_B = ["O", "P"]  # after X's second part


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


def _chunk(model, layers, items, states, mode: str, tag: str):
    """One chunk of ``items`` [(key, Segment, ids [length])] on ``states`` [per item] -> Capture of every computed
    item's layer outputs (key, layer). mode "off": today's single-request chunk (one item); "on": a glued chunk."""
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    desc = GluedChunk(CHUNK, tuple(s for _, s, _ in items))
    cap = Capture()

    def on_layer(i, h):
        if not hasattr(h, "streams"):
            return
        for k, (key, _s, _ids) in enumerate(items):
            n = desc.rows(k)
            if n > 0:
                cap.add((key, i), _pc.glued_boundary_rows(model, h, desc.lane_off(k), n))

    t0 = time.time()
    if mode == "off":
        ((_, s, ids),) = items
        d1 = ChunkDescriptor.single(CHUNK, s.slot, s.start, s.end)
        tok = torch.full((CHUNK,), PAD_ID, dtype=torch.long)
        tok[: s.length] = ids
        with packed("off"):
            run_chunk(model, d1, tok, states, layers, on_layer=on_layer)
    else:
        tok = desc.pack_ids([ids for *_, ids in items])
        with packed("on"), glued("on"):
            run_chunk(model, desc, tok, states, layers, on_layer=on_layer)
    model.sync()
    print(
        f"  {tag} {mode} {[k for k, *_ in items]} rows {[desc.rows(k) for k in range(desc.lanes)]}: {time.time() - t0:.1f} s"
    )
    return cap


def _run(model, layers, run, names: list, mode: str):
    """One chunk of the named segments (fresh states) -> (Capture, {name: served digests})."""
    states = {n: _fresh_state(model, layers, run, n) for n in names}
    items = [(n, run.seg(n), run.toks(n)[SEGS[n]["start"] :]) for n in names]
    cap = _chunk(model, layers, items, [states[n] for n in names], mode, "run")
    dig = {n: served_digests(states[n], layers, SEGS[n]["start"] + SEGS[n]["n"]) for n in names}
    for st in states.values():
        st.free()
    return cap, dig


def _dig_mismatch(a: dict, b: dict) -> int:
    keys = set(a) | set(b)
    return sum(1 for k in keys if a.get(k) != b.get(k))


def _split(model, layers, run):
    """splitA / splitB vs splitX -> (max abs diff over X's two parts, digest mismatches of X's slot after both)."""
    xs = run.seg("X")
    used = sum(SEGS[n]["n"] for n in SPLIT_A)
    x1, x2 = split(xs, CHUNK - used)
    assert x1 is not None and x2.start == x1.end, (x1, x2)
    xt = run.toks("X")
    ids1, ids2 = xt[x1.start : x1.end], xt[x2.start : x2.end]
    # glued: X1 at the end of chunk A, X2 at the start of chunk B (other segments' states fresh each chunk)
    stx = _fresh_state(model, layers, run, "X")
    others = {n: _fresh_state(model, layers, run, n) for n in SPLIT_A}
    items = [(n, run.seg(n), run.toks(n)[SEGS[n]["start"] :]) for n in SPLIT_A] + [("X1", x1, ids1)]
    ca = _chunk(model, layers, items, [others[n] for n in SPLIT_A] + [stx], "on", "splitA")
    for st in others.values():
        st.free()
    others = {n: _fresh_state(model, layers, run, n) for n in SPLIT_B}
    items = [("X2", x2, ids2)] + [(n, run.seg(n), run.toks(n)[SEGS[n]["start"] :]) for n in SPLIT_B]
    cb = _chunk(model, layers, items, [stx] + [others[n] for n in SPLIT_B], "on", "splitB")
    for st in others.values():
        st.free()
    dg = served_digests(stx, layers, xs.end)
    stx.free()
    # alone: X1 then X2, each alone at model row 0 (the same request chunked at the same boundary)
    sty = _fresh_state(model, layers, run, "X")
    a1 = _chunk(model, layers, [("X1", x1, ids1)], [sty], "on", "splitX1")
    a2 = _chunk(model, layers, [("X2", x2, ids2)], [sty], "on", "splitX2")
    da = served_digests(sty, layers, xs.end)
    sty.free()
    d = max(
        Capture.diff(ca, a1, [("X1", i) for i in layers]),
        Capture.diff(cb, a2, [("X2", i) for i in layers]),
    )
    return d, _dig_mismatch(dg, da), (x1, x2)


@mesh_parametrize
def test_glued_oracle(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    assert model.chunk == CHUNK, model.chunk
    if model.skip_mode != "off":
        model.use_decoder_skip("off")  # (every layer computes; FIX.28: host lists only)
    layers = _layers(model)
    print(f"glued oracle: layers {layers}")
    g = Golden.for_rung(S, "s56320")
    run = Run(g.tokens())
    sp, tp = tuple(model.mesh.shape)
    for c, names in COMPS.items():
        desc = GluedChunk(CHUNK, tuple(run.seg(n) for n in names))
        shared = [q for q, cs in enumerate(desc.chip_segments(sp, tp)) if len(cs) > 1]
        print(
            f"  {c}: model rows {[desc.lane_off(k) for k in range(desc.lanes)]}, lane chunk "
            f"{desc.as_lanes() is not None}, chips holding two segments {shared}"
        )

    keys = lambda n: [(n, i) for i in layers]  # noqa: E731
    today = {n: _run(model, layers, run, [n], "off") for n in COMPUTED}
    alone = {n: _run(model, layers, run, [n], "on") for n in COMPUTED + ("Q",)}
    comps = {c: _run(model, layers, run, names, "on") for c, names in COMPS.items()}
    again = _run(model, layers, run, COMPS["glued2"], "on")

    a_t = max(Capture.diff(alone[n][0], today[n][0], keys(n)) for n in COMPUTED)
    a_t_dig = sum(_dig_mismatch(alone[n][1][n], today[n][1][n]) for n in COMPUTED)
    if a_t_dig and not a_t:
        a_t = 1e-30
    eq, digs, worst = 0.0, 0, []
    for c, (cap, dig) in comps.items():
        for n in COMPS[c]:
            if n in COMPUTED:
                d = Capture.diff(cap, alone[n][0], keys(n))
                eq = max(eq, d)
                if d:
                    bad = [i for i in layers if cap.dig.get((n, i)) != alone[n][0].dig.get((n, i))]
                    worst.append(f"{c} {n}: layers differing {bad}")
            m = _dig_mismatch(dig[n], alone[n][1][n])
            digs += m
            if m:
                bad = sorted({k for k in set(dig[n]) | set(alone[n][1][n]) if dig[n].get(k) != alone[n][1][n].get(k)})
                worst.append(f"{c} {n}: served digests differ {bad[:6]}")
    det = Capture.diff(again[0], comps["glued2"][0], [k for n in COMPS["glued2"] if n in COMPUTED for k in keys(n)])
    det_dig = sum(_dig_mismatch(again[1][n], comps["glued2"][1][n]) for n in again[1])
    if det_dig and not det:
        det = 1e-30
    sp_d, sp_dig, (x1, x2) = _split(model, layers, run)
    if sp_dig:
        worst.append(f"split X: served digests differ in {sp_dig} records")
    digs += sp_dig
    metrics.record("glued_alone_vs_today_max_abs_diff", a_t)
    metrics.record("glued_alone_vs_today_digest_mismatches", a_t_dig)
    metrics.record("glued_segment_equiv_max_abs_diff", eq)
    metrics.record("glued_split_vs_alone_max_abs_diff", sp_d)
    metrics.record("glued_slot_digest_mismatches", digs)
    metrics.record("glued_determinism_max_abs_diff", det)
    metrics.record("glued_oracle_layers", len(layers))
    print(
        f"glued oracle: alone vs today {a_t} (digests {a_t_dig}), segment equivalence {eq}, split ({x1} + {x2}) vs "
        f"alone {sp_d}, slot digest mismatches {digs}, determinism {det} (digests {det_dig})"
    )
    for w in worst:
        print("  " + w)
    assert a_t == 0 and eq == 0 and sp_d == 0 and digs == 0 and det == 0, "glued oracle: not bit-identical (see above)"
