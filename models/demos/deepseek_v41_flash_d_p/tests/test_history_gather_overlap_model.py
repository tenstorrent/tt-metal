# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.73 (HISTORY_GATHER, was HISTORY_GATHER_OVERLAP): the whole model with STATIC_TABLES=off, the KV sources' history gathered as rows (on)
vs tiled + untilized (off), in one process on one load (device; run with DS41_STATIC_TABLES=off
DS41_HISTORY_GATHER=rows; the off side flips the sparse attentions' ``gather_rows`` back, host only).

Under DECODER_SKIP=decode (as test_skip_positions), at each point p (P73_POS, default 51200 and 1024000; the position
tables sized for the longest, as test_skip_positions, DS41-279): a fresh state, the real previous chunk at p - chunk,
then the chunk at p, last=False, random ids (testing/profile.PositionGolden, seed 0). Check: layer 19's output (the
last encoder layer: every KV source's attention feeds it) bit-identical on vs off. Prints the synced wall of the chunk on each side (median of 5 after 3
warm-ups; one process, no profiler: an indication only, the A/B is the orchestrator's)."""

from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: a bring-up batch's model (core/batch.shareable); needs STATIC_TABLES=off at its load
LAST_ENC = 19  # (DECODER_SKIP=decode: layers 0-19 run in full)


def _points() -> list[int]:
    v = os.environ.get("P73_POS", "51200,1024000")
    return [int(x) for x in v.split(",") if x.strip()]


@mesh_parametrize(timeout_s=0)
def test_history_gather_overlap_model(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    if settings.get("STATIC_TABLES") != "off" or settings.get("HISTORY_GATHER") == "tiles":
        pytest.skip("run with DS41_STATIC_TABLES=off DS41_HISTORY_GATHER=rows or rows_axes (load-time choices)")
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:  # (as test_skip_positions: the shipped default)
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    points = _points()
    layers = list(S.layers())
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    want = max([p + chunk for p in points if p + chunk <= deploy] + [int(S.data["target"]["seq"])])
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    blocks = [b.attention for b in model.blocks.values() if getattr(b.attention, "gather_rows", False)]
    assert blocks, "HISTORY_GATHER=tiles: no sparse attention gathers rows"
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    try:
        for p in points:
            assert p + chunk <= model.max_seq, (p, model.max_seq)
            ids = PositionGolden(S, p, chunk).tokens()
            prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
            res, walls = {}, {}
            for side in ("on", "off"):
                for b in blocks:
                    b.gather_rows = side == "on"
                state = model.new_state(p + chunk)
                try:

                    def run(tokens, start, keep=False):
                        model.begin_chunk(tokens, start, last=False)
                        h = model.embed(tokens)
                        out = None
                        for i in layers:
                            h2 = model.layer(i, h, start, state)
                            model.free(h)
                            h = h2
                            if keep and i == LAST_ENC:
                                out = model.to_host(h)
                        model.free(h)
                        return out

                    run(prev_t, p - chunk)
                    res[side] = run(cur_t, p, keep=True)
                    model.sync()
                    ts = []
                    for k in range(8):
                        run(prev_t, p - chunk)
                        model.sync()
                        t0 = time.time()
                        run(cur_t, p)
                        model.sync()
                        if k >= 3:
                            ts.append((time.time() - t0) * 1e3)
                    walls[side] = (statistics.median(ts), [round(t, 1) for t in ts])
                finally:
                    _free(state)
            a, b = res["on"], res["off"]
            same = torch.equal(a, b)
            diff = 0.0 if same else float((a - b).abs().max())
            print(
                f"[P.73] p{p}: layer {LAST_ENC} output on vs off bit-identical {same} (max abs diff {diff}); chunk wall on {walls['on'][0]:.1f} ms "
                f"{walls['on'][1]}, off {walls['off'][0]:.1f} ms {walls['off'][1]}"
            )
            assert same, f"p{p}: HISTORY_GATHER rows != tiles (max abs diff {diff})"
    finally:
        for b in blocks:
            b.gather_rows = True
        if loaded != "decode":
            model.use_decoder_skip(loaded)
