# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.30 (owner 2026-11-03): the time of a chunk under DECODER_SKIP=exact, timing only (not a gate).

At each start of DECODER_SKIP_POS (default 51200,512000; one model load, a fresh state per start, random ids as
BRINGUP_POS / testing/profile.PositionGolden: the work does not depend on the values), warm, each run synced:

    nonlast   begin_chunk(.., last=False): layers 0-19, layer 20's KV projection, layers 21-39 skipped
    last      begin_chunk(.., last=True) after skipped chunks (every decoder layer's window rows before the chunk
              noted missing: its carry masked, none read)
    full      begin_chunk(.., last=True) on a state that skipped nothing: every layer, today's chunk (= DECODER_SKIP=off)

Each: TIMED_RUNS wall runs (median; the first run of each discarded) after steady_state(start) (FIX.16: the kept-tail
carry path a consecutive chunk takes), then one run under the real-time profiler (device ms: the sum of every program,
slowest chip per call). Recorded as decoder_skip_<kind>_{wall,device}_ms_p<start>.

    PYTHONPATH=$PWD DS41_ENGRAM_TABLE=resident scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_decoder_skip_time.py
"""

from __future__ import annotations

import os
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden

S = spec()
pytestmark = device_timeout(S)
TIMED_RUNS = 5


def _median(v):
    v = sorted(v)
    return v[len(v) // 2]


@mesh_parametrize
def test_decoder_skip_time(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    monkeypatch.setenv("DS41_DECODER_SKIP", "exact")
    chunk = int(S.get("target.chunk"))
    starts = [int(p) for p in os.environ.get("DECODER_SKIP_POS", "51200,512000").split(",") if p.strip()]
    seq = max(starts) + chunk
    S.data["target"]["seq"] = max(int(S.data["target"]["seq"]), seq)  # (the model's max_seq: every start fits)
    layers = list(S.layers())
    assert layers == list(range(S.num_layers)), layers
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    assert model.decoder, "DECODER_SKIP=exact did not take"
    print(
        f"model load {model.load_seconds:.1f} s; decoder {model.decoder[0]}..{model.decoder[-1]}, kv-only {model.kv_only}"
    )

    for start in starts:
        g = PositionGolden(S, start, chunk)
        tokens = g.tokens()[start : start + chunk]
        state = model.new_state(start + chunk)

        def run_chunk(last: bool):
            model.begin_chunk(tokens, start, last=last)
            h = model.embed(tokens)
            for i in layers:
                h2 = model.layer(i, h, start, state)
                model.free(h)
                h = h2
            model.free(model.final_norm(h))
            model.free(h)

        def prepare(kind: str):
            for i in model.decoder:  # (host only) last: the chunks before were skipped; full: nothing was
                c = state.window[i]
                c.missing = (max(0, start - chunk), start) if kind == "last" and start else None
            model.steady_state(start)
            model.sync()

        def device_ms(kind: str) -> float:
            prepare(kind)
            profiler.enable(mesh_device, rt=True)
            try:
                profiler.set_layer(None)
                profiler.signpost("run")
                run_chunk(kind != "nonlast")
                profiler.signpost("end")
                profiler.finish_rt()
                return sum(profiler.result()["kernel_ns"].values()) / 1e6
            finally:
                profiler.disable()

        out = {}
        for kind in ("nonlast", "last", "full"):
            walls = []
            for _ in range(TIMED_RUNS + 1):
                prepare(kind)
                t0 = time.time()
                run_chunk(kind != "nonlast")
                model.sync()
                walls.append(round((time.time() - t0) * 1e3, 1))
            walls = walls[1:]
            dev = device_ms(kind)
            out[kind] = (_median(walls), dev, walls)
            metrics.record(f"decoder_skip_{kind}_wall_ms_p{start}", _median(walls))
            metrics.record(f"decoder_skip_{kind}_device_ms_p{start}", round(dev, 2))
        state.free()
        print(
            f"DECODER_SKIP timing at {start}: "
            + "; ".join(f"{k} wall {w} ms (runs {r}) device {d:.2f} ms" for k, (w, d, r) in out.items())
        )
