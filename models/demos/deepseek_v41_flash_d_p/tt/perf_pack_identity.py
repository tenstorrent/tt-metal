# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.11 (CKV_PACK_BEFORE_GATHER): the all-device model with the switch on and off, alternating in one process, on the
profile rung's golden prefix (loaded before each run), then two chunks with the last chunk's tokens: first at
[last - S, last) (a re-write of the prefix's last chunk positions; its carry read from the served cache), then at
[last, seq) (its window carry from the previous chunk: the kept blocks on, the kept bf16 rows off). The rope tables end
at the rung's seq, so the run cannot go past it. Compared on vs off (and run to run), not against the golden. Not a gate: evidence
that the switch changes no number (bit-identical hidden state after each chunk, deterministic) and its warm wall a
chunk (begin_chunk .. sync, prefix load excluded), median per side, paired in time.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_pack_identity.py -s

The switch is read per call (tt/attention.WindowTables.packs; settings.get patched here); the load-time buffers exist
because the model was built with the default (on).
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

PAIRS = 2  # timed (on, off) pairs after one warm run a side (10-minute foreground cap)
NAME = "CKV_PACK_BEFORE_GATHER"


@pytest.mark.timeout(1800)  # (loading the 40-layer model takes ~5 min)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_pack_identity(mesh_device, monkeypatch):
    s = spec()
    name = profile_rung(s)
    rung = s.rung(name)
    g = Golden.for_rung(s, name)
    layers = [i for i in s.layers() if i in g.layers]
    chunk = rung["chunk"]
    last = rung["seq"] - chunk
    model = s.hooks().device_model(mesh_device, s, layers, lm_head=False)
    state = model.new_state(rung["seq"])
    prefix = {i: g.state(i, at=last) for i in layers}
    tok = g.tokens()[last:]
    starts = {i for k, i in enumerate(layers) if k == 0 or layers[k - 1] != i - 1}
    get = settings.get

    def chunk_run(c: int, start: int):
        tokens = tok
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

    def run(side: str):
        monkeypatch.setattr(settings, "get", lambda n: side if n == NAME else get(n))
        for i in layers:
            state.load_prefix(i, prefix[i], last)
        o1, ms1 = chunk_run(0, last - chunk)
        o2, ms2 = chunk_run(1, last)
        return (o1, o2), (ms1, ms2)

    want = g.layer(g.n_chunks - 1, layers[-1])["out"].float()
    outs = {"on": [], "off": []}
    walls = {"on": [], "off": []}
    for k in range(PAIRS + 1):
        for side in ("on", "off") if k % 2 == 0 else ("off", "on"):
            o, ms = run(side)
            outs[side].append(o)
            if k > 0:  # the first pair compiles / warms
                walls[side].append(ms)
            print(f"PACK_IDENTITY run {k} {side}: chunk 1 {ms[0]:.1f} ms, chunk 2 {ms[1]:.1f} ms", flush=True)
    for side in ("on", "off"):
        for j in (0, 1):
            w = [m[j] for m in walls[side]]
            print(
                f"PACK_IDENTITY {side} chunk {j + 1}: wall median {statistics.median(w):.1f} ms, min {min(w):.1f}, "
                f"max {max(w):.1f} (n {len(w)})"
                + (f"; pcc vs golden {metrics.pcc(outs[side][-1][1], want):.6f}" if j == 1 else ""),
                flush=True,
            )
    ref = outs["on"][0]
    same = lambda o: all(torch.equal(a, b) for a, b in zip(ref, o))  # noqa: E731
    det = all(same(o) for o in outs["on"])
    eq = all(same(o) for o in outs["off"])
    print(f"PACK_IDENTITY on deterministic: {det}; on == off: {eq}", flush=True)
    if not eq:
        for j in (0, 1):
            d = (ref[j] - outs["off"][0][j]).abs().max()
            print(f"PACK_IDENTITY chunk {j + 1}: max |on - off| {float(d):.3e}", flush=True)
    assert det, "CKV_PACK_BEFORE_GATHER=on is not deterministic"
    assert eq, "CKV_PACK_BEFORE_GATHER on and off differ"
