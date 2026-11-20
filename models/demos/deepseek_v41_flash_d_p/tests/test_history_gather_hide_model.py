# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.75 (HISTORY_GATHER_HIDE): the whole model with STATIC_TABLES=off, HISTORY_GATHER=rows, the KV sources' history
gathered at the first engram wait and only the chunk's own periods at the source's layer (on) vs the whole history
gathered at the source's layer (off), in one process on one load (device; run with DS41_STATIC_TABLES=off
DS41_HISTORY_GATHER_HIDE=on; the off side switches the model's prefetch off, host only: every source's layer then
gathers [0, nb S) into the same per-source buffer, P.74's op).

Under DECODER_SKIP=decode (as test_skip_positions), at each point p (P75_POS, default 51200, 256000, 1024000) and each
start offset o (P75_OFF, default 0 and 2048: a start inside a chunk period, the tail then 2 periods an owner): a
fresh state, the previous chunk at p + o - chunk, then the chunk at p + o, last=False, random ids
(testing/profile.PositionGolden, seed 0). Checks: layer 19's output (the last encoder layer: every KV source's
attention feeds it) bit-identical on vs off; on the on side every gathered source's layer took the tail path
(history_gather.STATS) after one prefetch each. Prints the synced wall of the chunk on each side (median of 5 after 3
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
from models.demos.deepseek_v41_flash_d_p.tt import history_gather
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29: a bring-up batch's model (core/batch.shareable); needs STATIC_TABLES=off at its load
LAST_ENC = 19  # (DECODER_SKIP=decode: layers 0-19 run in full)


def _ints(name: str, default: str) -> list[int]:
    return [int(x) for x in os.environ.get(name, default).split(",") if x.strip()]


@mesh_parametrize(timeout_s=0)
def test_history_gather_hide_model(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    if (
        settings.get("STATIC_TABLES") != "off"
        or settings.get("HISTORY_GATHER") != "rows"
        or settings.get("HISTORY_GATHER_HIDE") != "on"
    ):
        pytest.skip("run with DS41_STATIC_TABLES=off DS41_HISTORY_GATHER=rows DS41_HISTORY_GATHER_HIDE=on (load-time)")
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:  # (as test_skip_positions: the shipped default)
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    points, offs = _ints("P75_POS", "51200,256000,1024000"), _ints("P75_OFF", "0,2048")
    layers = list(S.layers())
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cases = [(p, o) for p in points for o in offs if p + o + chunk <= deploy]
    want = max([p + o + chunk for p, o in cases] + [int(S.data["target"]["seq"])])
    monkeypatch.setitem(S.data["target"], "seq", want)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    assert model._hide, "HISTORY_GATHER_HIDE=on did not take (STATIC_TABLES=off, HISTORY_GATHER=rows)"
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    n_src = len(model.history_sources())
    bad = []
    try:
        for p, o in cases:
            s0 = p + o
            assert s0 + chunk <= model.max_seq, (s0, model.max_seq)
            ids = PositionGolden(S, p, chunk).tokens()
            prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
            res, walls, stats = {}, {}, {}
            for side in ("on", "off"):
                model._hide = side == "on"
                state = model.new_state(s0 + chunk)
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

                    run(prev_t, s0 - chunk)
                    before = dict(history_gather.STATS)
                    res[side] = run(cur_t, s0, keep=True)
                    stats[side] = {k: history_gather.STATS[k] - before[k] for k in before}
                    model.sync()
                    ts = []
                    for k in range(8):
                        run(prev_t, s0 - chunk)
                        model.sync()
                        t0 = time.time()
                        run(cur_t, s0)
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
                f"[P.75] p{p}+{o}: layer {LAST_ENC} output on vs off bit-identical {same} (max abs diff {diff}); "
                f"gathers on {stats['on']} off {stats['off']} ({n_src} sources); chunk wall on {walls['on'][0]:.1f} "
                f"ms {walls['on'][1]}, off {walls['off'][0]:.1f} ms {walls['off'][1]}"
            )
            if not same:
                bad.append(f"p{p}+{o}: on != off (max abs diff {diff})")
            want_on = {"prefetch": n_src, "tail": n_src, "full": 0}
            if s0 >= chunk and stats["on"] != want_on:
                bad.append(f"p{p}+{o}: on side gathers {stats['on']} != {want_on}")
            if stats["off"]["prefetch"] or stats["off"]["tail"]:
                bad.append(f"p{p}+{o}: off side prefetched {stats['off']}")
    finally:
        model._hide = True
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    assert not bad, bad
