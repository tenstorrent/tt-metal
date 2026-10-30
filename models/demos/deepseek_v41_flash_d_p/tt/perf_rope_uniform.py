# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.5 (ROPE_UNIFORM_PROGRAM): the all-device model's chunk with the switch on and off, alternating in one process, on
the profile rung's last chunk (the golden prefix loaded before each run, as testing/profile.py does). Not a gate:
evidence that the switch changes no number (bit-identical final hidden state, deterministic) and its warm wall a chunk
(begin_chunk .. sync, prefix load excluded), median per side, paired in time so host load drifts hit both sides.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_rope_uniform.py -s

The switch is read per call in tt/rope.rope_stock (settings.get patched here).
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import begin_chunk, spec
from models.demos.common.bringup.testing.profile import profile_rung
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

PAIRS = 6  # timed (on, off) pairs after one warm run a side


@pytest.mark.timeout(1500)  # (loading the 40-layer model takes ~5 min)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_rope_uniform_ab(mesh_device, monkeypatch):
    s = spec()
    name = profile_rung(s)
    rung = s.rung(name)
    g = Golden.for_rung(s, name)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    start = rung["seq"] - chunk
    model = s.hooks().device_model(mesh_device, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    prefix = {i: g.state(i, at=start) for i in layers}
    tokens = g.tokens()[start:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}
    want = g.layer(g.n_chunks - 1, layers[-1])["out"].float()

    get = settings.get

    def run(side: str) -> tuple[torch.Tensor, float]:
        monkeypatch.setattr(settings, "get", lambda n: side if n == "ROPE_UNIFORM_PROGRAM" else get(n))
        for i in layers:
            state.load_prefix(i, prefix[i], start)
        model.sync()
        t0 = time.perf_counter()
        h = None
        begin_chunk(model, tokens, start)
        for i in layers:
            if i in starts:
                if h is not None:
                    model.free(h)
                h = model.embed(tokens) if i == 0 else model.from_host(g.layer(g.n_chunks - 1, i)["in"].float())
            h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        model.sync()
        ms = (time.perf_counter() - t0) * 1e3
        out = model.to_host(h).float()
        model.free(h)
        return out, ms

    outs = {"on": [], "off": []}
    walls = {"on": [], "off": []}
    for k in range(PAIRS + 1):
        for side in ("on", "off") if k % 2 == 0 else ("off", "on"):
            o, ms = run(side)
            outs[side].append(o)
            if k > 0:  # the first pair compiles / warms
                walls[side].append(ms)
            print(f"ROPE_UNIFORM run {k} {side}: {ms:.1f} ms", flush=True)
    for side in ("on", "off"):
        w = walls[side]
        print(
            f"ROPE_UNIFORM {side}: wall median {statistics.median(w):.1f} ms, min {min(w):.1f}, max {max(w):.1f} "
            f"(n {len(w)}); pcc vs golden {metrics.pcc(outs[side][-1], want):.6f}",
            flush=True,
        )
    det = all(torch.equal(outs["on"][0], o) for o in outs["on"] + outs["off"])
    print(f"ROPE_UNIFORM on == off == every run: {det}", flush=True)
    assert all(torch.equal(outs["on"][0], o) for o in outs["on"]), "ROPE_UNIFORM_PROGRAM=on is not deterministic"
    assert det, "ROPE_UNIFORM_PROGRAM on and off differ"
