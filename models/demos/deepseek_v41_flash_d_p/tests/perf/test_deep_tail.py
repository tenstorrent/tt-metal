# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""FIX.17 (owner 2026-10-31): DEEP_LAYERS=tail against the golden, and the tail pass's time.

The profile rung's prompt (rung ``last``: the golden state of [0, 51200) loaded for layers 0-20 only, then the chunk
[51200, 56320)) in tail mode: layers 0-20 run the chunk, layers 21-39 are skipped (their window caches stay zero, as
in a real tail-mode prompt), then TtDS41Model.tail_pass runs layers 21-39 over the rows the last token needs.
Checked with the ladder's limits (testing/e2e_limits.py, the rung's drift): every layer 0-20 over the whole chunk and
its state; layers 21-39 over the tail rows [end - tail_rows, end); the final hidden over the tail rows; top-5 over the
ladder's sampled rows inside the tail. Then warm timing: the normal chunk (layers 0-20, synced) and the tail pass
(synced), TIMED_RUNS runs, median; and one run under the real-time profiler for device ms of each.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_deep_tail.py
    DEEP_TAIL_POS=512000 ...   timing only: one chunk at that start, fresh state, random ids (as BRINGUP_POS)

Not a gate. DEEP_LAYERS=tail is forced for this test (the model reads it at load).
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import begin_chunk, device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.ladder import check_states, rel, sampled_rows
from models.demos.common.bringup.testing.profile import PositionGolden, profile_rung

S = spec()
pytestmark = device_timeout(S)
TIMED_RUNS = 5


def _median(v):
    v = sorted(v)
    return v[len(v) // 2]


@mesh_parametrize
def test_deep_tail(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    monkeypatch.setenv("DS41_DEEP_LAYERS", "tail")
    pos = os.environ.get("DEEP_TAIL_POS", "").strip()
    chunk = int(S.get("target.chunk"))
    if pos:  # timing only (PositionGolden: fresh state, random ids)
        start = int(pos)
        seq = start + chunk
        S.data["target"]["seq"] = max(int(S.data["target"]["seq"]), seq)
        g, rung_name = PositionGolden(S, start, chunk), None
    else:
        rung_name = profile_rung(S)
        rung = S.rung(rung_name)
        assert rung.get("prefix_from_golden"), rung
        seq, chunk = rung["seq"], rung["chunk"]
        start = seq - chunk
        g = Golden.for_rung(S, rung_name)
    layers = [i for i in S.layers() if i in g.layers]
    assert layers == list(range(S.num_layers)), layers
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=not pos)
    assert model.deep and model.deep_src is not None, "DEEP_LAYERS=tail did not take"
    full = [i for i in layers if i <= model.deep_src]
    tail_rows = model.tail_rows
    print(f"deep layers {model.deep[0]}..{model.deep[-1]}, tail {tail_rows} rows, chunk [{start}, {seq})")
    state = model.new_state(seq)
    prefix = {} if pos else {i: g.state(i, at=start) for i in full}

    def reload():
        for i in full if not pos else ():
            state.load_prefix(i, prefix[i], start)

    tokens = g.tokens()[start : start + chunk]

    def run_chunk(trail=None):
        begin_chunk(model, tokens, start)
        h = model.embed(tokens)
        for i in layers:
            h2 = model.layer(i, h, start, state)
            if trail is not None and i in full:  # (layer deep_src returns the Skipped marker: its output is kept)
                trail[i] = model.to_host(state.deep_in[start][0] if i == model.deep_src else h2).float()
            model.free(h)
            h = h2
        model.free(h)  # (the Skipped marker: layer 20's output is in state.deep_in)

    # ---- accuracy (also the compile run)
    reload()
    model.steady_state(None)
    trail = {}
    run_chunk(trail if not pos else None)

    def capture(i, s, h):
        if not pos and s == start:
            trail[i] = model.to_host(h).float()

    h = model.tail_pass(state, seq, on_layer=capture)
    hidden = model.final_norm(h)
    model.free(h)
    model.sync()
    failed = []
    if not pos:
        drift = E.load(g.dir, rung_name)
        last = g.n_chunks - 1
        t0 = chunk - tail_rows
        for i in layers:
            want = g.layer(last, i)["out"].float()
            got = trail[i]
            if i > model.deep_src:
                got, want = got[t0:], want[t0:]
            e = rel(got, want)
            lim = E.err_limit(drift, f"layer_L{i:02d}")
            E.check(drift, f"layer_L{i:02d}", f"err_layer_L{i:02d}", e, failed)
            rows = "tail rows" if i > model.deep_src else "chunk"
            print(
                f"L{i:02d} ({rows}) rel {e:.5f} limit {lim:.5f} pcc {float(torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1]):.5f}"
            )
        gm = g.model(last)
        final = model.to_host(hidden).float()
        e = rel(final[t0:], gm["final_norm"].float()[t0:])
        E.check(drift, "final_hidden", "err_final_hidden", e, failed)
        print(f"final hidden (tail rows) rel {e:.5f} limit {E.err_limit(drift, 'final_hidden'):.5f}")
        rows = sampled_rows(chunk)
        rows = rows[rows >= t0]
        lg = model.logits(hidden, rows).float()
        tt5 = lg.topk(5, dim=-1).indices
        gtop = gm["top32_ids"][rows].long()
        top1 = (tt5[:, 0] == gtop[:, 0]).float().mean().item()
        top5 = (tt5 == gtop[:, :1]).any(-1).float().mean().item()
        E.check(drift, "top5", "top5_overlap", top5, failed)
        print(
            f"top1 {top1:.4f} top5 {top5:.4f} (limit {E.overlap_limit(drift, 'top5'):.4f}) over {rows.numel()} rows; "
            f"last token top1 {'match' if int(tt5[-1, 0]) == int(gtop[-1, 0]) else 'MISS'}"
        )
        check_states(state, g, full, seq, drift, failed, S.get("tests.waivers"))
    model.free(hidden)

    # ---- warm timing: the normal chunk (layers 0-deep_src) and the tail pass, each synced
    chunk_ms, tail_ms = [], []
    for _ in range(TIMED_RUNS + 1):
        reload()
        model.steady_state(start)
        model.sync()
        t0 = time.time()
        run_chunk()
        model.sync()
        t1 = time.time()
        h = model.tail_pass(state, seq)
        model.free(h)
        model.sync()
        t2 = time.time()
        chunk_ms.append(round((t1 - t0) * 1e3, 1))
        tail_ms.append(round((t2 - t1) * 1e3, 1))
    chunk_ms, tail_ms = chunk_ms[1:], tail_ms[1:]  # the first: new carry-path programs
    # device ms of each: one run of each under the real-time profiler (sum of every program, slowest chip per call)

    def device_ms(fn):
        profiler.enable(mesh_device, rt=True)
        try:
            profiler.set_layer(None)
            profiler.signpost("run")
            fn()
            profiler.signpost("end")
            profiler.finish_rt()
            return sum(profiler.result()["kernel_ns"].values()) / 1e6
        finally:
            profiler.disable()

    reload()
    model.steady_state(start)
    dev = {"chunk": device_ms(run_chunk)}
    dev["tail"] = device_ms(lambda: model.free(model.tail_pass(state, seq)))
    print(
        f"DEEP_TAIL timing at {start}: normal chunk (layers 0-{model.deep_src}) wall {_median(chunk_ms)} ms "
        f"(runs {chunk_ms}); tail pass (layers {model.deep[0]}-{model.deep[-1]}) wall {_median(tail_ms)} ms "
        f"(runs {tail_ms}); device ms: chunk {dev['chunk']:.2f}, tail pass {dev['tail']:.2f}"
    )
    assert not failed, "\n".join(failed)
