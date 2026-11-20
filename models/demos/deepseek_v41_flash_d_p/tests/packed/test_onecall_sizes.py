# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.3A: PACKED_ATTN_ONECALL=on with slots whose caches differ in size (not gated; the oracle's slots are all one
size). test_packed_walls sizes each slot's caches to its position, so its lanes' history gather buffers and index-key
buffers differ in rows: the one-call paths then take the sdpa fork's kv_lanes over buffers of different rows with a
per-chip owner stride, and the one ring over key buffers of different capacities. Here the same bit-identity as the
oracle's lane equivalence: each segment packed with others == the segment alone in lane 0 (today's chunk), every
layer output (its rows) and every served record, with each slot sized differently.

Segments as tests/packed/test_packed_oracle.py (real text, each its own slot, history written by today's path):
F start 0 (2048 rows), P start 2048 (1600), O start 4096 (777); slot sizes 16384 / 24576 / 40960 positions.
Layers: DS41_PACKED_LAYERS (default the characteristic layers the model holds)."""

from __future__ import annotations

import importlib
import os

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc

_pc = importlib.reload(_pc)
from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("DS41_PACKED_LAYERS", "PB3A_SIZES_OFF")  # PB3A_SIZES_OFF: "ring" / "lanekv" (bisecting)
CHUNK, PAD_ID = _pc.CHUNK, _pc.PAD_ID
SEGS = {
    "F": dict(slot=1, start=0, n=2048, off=0, size=16384),
    "P": dict(slot=2, start=2048, n=1600, off=8192, size=24576),
    "O": dict(slot=3, start=4096, n=777, off=16384, size=40960),
}
COMPS = [["F", "P", "O", None], ["O", None, "F", "P"]]
CHAR = [0, 1, 2, 3, 14, 20, 21, 24]


def _layers(model) -> list:
    v = os.environ.get("DS41_PACKED_LAYERS", "").strip()
    want = [int(x) for x in v.split(",") if x.strip()] if v else [i for i in CHAR if i in model.layers]
    assert all(i in model.layers for i in want), (want, model.layers)
    return want


def _toks(tokens, name):
    d = SEGS[name]
    return tokens[d["off"] : d["off"] + d["start"] + d["n"]].long()


def _state(model, layers, tokens, name):
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    d = SEGS[name]
    st = model.new_state(d["size"])
    if d["start"] > 0:
        ids = torch.full((CHUNK,), PAD_ID, dtype=torch.long)
        ids[: d["start"]] = _toks(tokens, name)[: d["start"]]
        with _pc.packed("off"):
            run_chunk(model, ChunkDescriptor.single(CHUNK, d["slot"], 0, d["start"]), ids, [st], layers)
    return st


def _run(model, layers, tokens, lanes: list):
    from models.demos.deepseek_v41_flash_d_p.tt.attention import prefill_end
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    names = [n for n in lanes if n is not None]
    states = {n: _state(model, layers, tokens, n) for n in names}
    seg = lambda n: Segment(SEGS[n]["slot"], SEGS[n]["start"], SEGS[n]["start"] + SEGS[n]["n"])  # noqa: E731
    desc = ChunkDescriptor.packed(CHUNK, [None if n is None else seg(n) for n in lanes])
    ids = desc.pack_ids([None if n is None else _toks(tokens, n)[SEGS[n]["start"] :] for n in lanes])
    cap = _pc.Capture()

    def on_layer(i, h):
        if not hasattr(h, "streams"):
            return
        for lane, n in enumerate(lanes):
            if n is not None:
                k = max(0, prefill_end(seg(n).end) - seg(n).start)
                cap.add((n, i), _pc.boundary_rows(model, h, desc, lane, k))

    with _pc.packed("on"):
        run_chunk(model, desc, ids, [states.get(n) for n in lanes], layers, on_layer=on_layer)
    model.sync()
    dig = {n: _pc.served_digests(states[n], layers, SEGS[n]["start"] + SEGS[n]["n"]) for n in names}
    for st in states.values():
        st.free()
    return cap, dig


@mesh_parametrize
def test_onecall_sizes(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    assert settings.get("PACKED_ATTN_ONECALL") == "on", "run with DS41_PACKED_ATTN_ONECALL=on"
    from models.demos.deepseek_v41_flash_d_p.tt import packed_onecall

    off = os.environ.get("PB3A_SIZES_OFF", "")
    packed_onecall.ONE_RING[0] = packed_onecall._has_one_ring() and "ring" not in off
    packed_onecall.LANE_KV[0] = packed_onecall._has_lane_kv() and "lanekv" not in off
    try:
        model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
        if model.skip_mode != "off":
            model.use_decoder_skip("off")
        layers = _layers(model)
        tokens = Golden.for_rung(S, "s56320").tokens()
        alone = {n: _run(model, layers, tokens, [n, None, None, None]) for n in SEGS}
        worst, lane_eq, digs = [], 0.0, 0
        for lanes in COMPS:
            cap, dig = _run(model, layers, tokens, lanes)
            for n in lanes:
                if n is None:
                    continue
                keys = [(n, i) for i in layers]
                d = _pc.Capture.diff(cap, alone[n][0], keys)
                lane_eq = max(lane_eq, d)
                m = sum(1 for k in set(dig[n]) | set(alone[n][1][n]) if dig[n].get(k) != alone[n][1][n].get(k))
                digs += m
                if d or m:
                    bad = [i for i in layers if cap.dig.get((n, i)) != alone[n][0].dig.get((n, i))]
                    worst.append(f"{lanes} {n}: layers differing {bad}, digest mismatches {m}")
        metrics.record("onecall_sizes_lane_equiv_max_abs_diff", lane_eq)
        metrics.record("onecall_sizes_digest_mismatches", digs)
        print(
            f"PB.3A one-call, slots of different sizes, layers {layers} (off: {off or 'none'}): lane equivalence {lane_eq}, "
            f"digests {digs}"
        )
    finally:  # (the process's switches back: the resident runner serves later requests)
        packed_onecall.ONE_RING[0] = packed_onecall._has_one_ring()
        packed_onecall.LANE_KV[0] = packed_onecall._has_lane_kv()
    for w in worst:
        print("  " + w)
    assert lane_eq == 0 and digs == 0, "one-call with slots of different sizes: not bit-identical to alone"
