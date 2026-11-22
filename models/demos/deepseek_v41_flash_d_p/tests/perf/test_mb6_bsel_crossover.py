# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.6 (OWNER 2026-11-21): where the indexer's block-max pre-selection (INDEXER_BLOCK_SELECT, PB.BM) starts to pay on
the single 5120 path: the crossover that sets INDEXER_BLOCK_SELECT_MIN. A measurement (no model change).

At each history p of MB6_POS the ladder's STEADY chunk (test_skip_positions' _point: the real previous chunk untimed
first, then the timed chunk; same ids, same state) on both paths in ONE load (the gate ladder's load: the serving
runtime's adapter.load_model, max_seq the deployment context in whole chunks): ``block`` = every plain indexer call on
the block path (force_block_select(True): whatever INDEXER_BLOCK_SELECT_MIN), ``plain`` = every call on the plain
top-k (force_block_select(False)). Per side: the wall (warm_up, then MB6_RUNS timed runs with program-cache misses
refused, the median, as the ladder) and one real-time profiled run (the slowest chip's total and the sections; the
indexer's sections summed). The keys a plain index layer scores at p: kv_len = (p + chunk) / 2 (ratio 2).

Metrics: mb6_<block|plain>_wall_ms_p<p>, mb6_<block|plain>_device_ms_p<p>, mb6_<block|plain>_indexer_ms_p<p>,
mb6_block_minus_plain_<wall|device>_ms_p<p>.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=5120
        DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_mb6_bsel_crossover.py"
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden, misses_allowed, warm_up
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("MB6_POS", "MB6_RUNS", "MB6_OUT", "WALLS_FRESH_SLOTS")
BATCH_POS_VARS = ("MB6_POS",)
BATCH_LAYERS = "executed"
POINTS = (102400, 204800, 307200, 409600, 512000, 716800, 1024000)


@mesh_parametrize(timeout_s=3600)
def test_mb6_bsel_crossover(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    monkeypatch.setenv(settings.env_name("INDEXER_BLOCK_SELECT"), "on")  # (the forced sides need the switch on)
    if settings.snap is not None:
        monkeypatch.setitem(settings.snap, "INDEXER_BLOCK_SELECT", "on")
    from models.demos.deepseek_v41_flash_d_p.tests import ladder_common as LC
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as TSP
    from models.demos.deepseek_v41_flash_d_p.tt import indexer as IX

    chunk = int(S.get("target.chunk"))
    pts = [int(x) for x in os.environ.get("MB6_POS", ",".join(map(str, POINTS))).split(",") if x.strip()]
    runs = int(os.environ.get("MB6_RUNS", "") or 9)
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    want = max(-(-deploy // chunk) * chunk, int(S.data["target"]["seq"]))  # (the gate ladder's load)
    monkeypatch.setitem(S.data["target"], "seq", want)
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(1 + LC.SPLIT_RUNS))
    rt = TSP._runtime(mesh_device, layers, chunk, want)
    model = rt[0].model
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")
    # (DS41-311 / DS41-329: the held indexers' module globals may be an earlier module object than the import above)
    ixs = [b.indexer for b in model.blocks.values() if type(getattr(b, "indexer", None)).__name__ == "TtIndexer"]
    assert ixs, "no plain indexer in the load"
    gl = [type(ixs[0]).__call__.__globals__, vars(sys.modules[IX.__name__])]
    gl = list({id(g): g for g in gl}.values())
    out = {}
    try:
        for p in pts:
            g = PositionGolden(S, p, chunk)
            ids = g.tokens()
            prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
            state = model.new_state(p + chunk)
            res_p = {"kv_len": (p + chunk) // 2}
            try:

                def run(tokens, start, section=False):
                    if section:
                        profiler.signpost("chunk_input")
                    model.begin_chunk(tokens, start, last=False)
                    h = model.embed(tokens)
                    for i in layers:
                        h2 = model.layer(i, h, start, state)
                        model.free(h)
                        h = h2
                    model.free(h)

                def one_run():
                    run(prev_t, p - chunk)
                    model.sync()
                    t0 = time.time()
                    run(cur_t, p)
                    model.sync()
                    return round((time.time() - t0) * 1e3, 2)

                for side, force in (("plain", False), ("block", True)):
                    with contextlib.ExitStack() as forced:
                        for gg in gl:
                            forced.enter_context(gg["force_block_select"](force))
                        w = warm_up(mesh_device, one_run)
                        assert w["warm"], (p, side, w["entries"][-5:])
                        walls = []
                        misses_allowed(mesh_device, False)
                        try:
                            for _ in range(runs):
                                walls.append(one_run())
                        finally:
                            misses_allowed(mesh_device, True)
                        wall = sorted(walls)[len(walls) // 2]
                        run(prev_t, p - chunk)
                        model.sync()
                        profiler.enable(mesh_device, rt=True)
                        try:
                            profiler.set_layer(None)
                            run(cur_t, p, section=True)
                            profiler.signpost("end")
                            profiler.finish_rt()
                            dev = TSP._device_split(profiler.result())
                        finally:
                            profiler.disable()
                    secs = {**dev["phases"], **dev["subs"]}
                    idx = sum(v for k, v in dev["phases"].items() if k.startswith("indexer"))
                    r = {
                        "wall": wall,
                        "walls": walls,
                        "spread": round(max(walls) - min(walls), 2),
                        "device": round(dev["slowest"], 2),
                        "indexer": round(idx, 3),
                        "sections": {k: round(v, 3) for k, v in secs.items() if "index" in k},
                    }
                    res_p[side] = r
                    metrics.record(f"mb6_{side}_wall_ms_p{p}", wall)
                    metrics.record(f"mb6_{side}_device_ms_p{p}", r["device"])
                    metrics.record(f"mb6_{side}_indexer_ms_p{p}", r["indexer"])
                    print(
                        f"MB6 p{p} (kv_len {res_p['kv_len']}) {side}: wall {wall} ms (runs {walls}), device "
                        f"{r['device']} ms, indexer {r['indexer']} ms; sections {json.dumps(r['sections'])}",
                        flush=True,
                    )
                dw = round(res_p["block"]["wall"] - res_p["plain"]["wall"], 2)
                dd = round(res_p["block"]["device"] - res_p["plain"]["device"], 2)
                di = round(res_p["block"]["indexer"] - res_p["plain"]["indexer"], 3)
                res_p.update(block_minus_plain_wall=dw, block_minus_plain_device=dd, block_minus_plain_indexer=di)
                metrics.record(f"mb6_block_minus_plain_wall_ms_p{p}", dw)
                metrics.record(f"mb6_block_minus_plain_device_ms_p{p}", dd)
                print(f"MB6 p{p}: block - plain wall {dw} ms, device {dd} ms, indexer {di} ms", flush=True)
            finally:
                state.free()
            out[p] = res_p
    finally:
        if loaded != "decode":
            model.use_decoder_skip(loaded)
    dst = os.environ.get("MB6_OUT", "/tmp/mb6_crossover.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"MB6 crossover written: {dst}")
