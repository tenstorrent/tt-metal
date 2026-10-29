# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.12 (LOWBIT_HOST): the all-device model's chunk output with LOWBIT_HOST on, off and on again, in one process, on
the profile rung's last chunk (the golden prefix loaded before each run, as testing/profile.py does). Not a gate:
evidence that the switch does not change the model's numbers (bit-identical final hidden state).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \
        models/demos/deepseek_v41_flash_d_p/tt/perf_lowbit_identity.py -s

The switch (settings.get patched for LOWBIT_HOST) is read where it is used (tt/attention._window_rows each call; tt/indexer.TtIndexer at construction, so the
built indexers' ``lowbit_host`` is set to the side too).
"""

from __future__ import annotations

import gc

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import begin_chunk, spec
from models.demos.common.bringup.testing.profile import profile_rung
from models.demos.deepseek_v41_flash_d_p.tt.indexer import TtIndexer
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings


@pytest.mark.timeout(1500)  # (loading the 40-layer model takes ~5 min)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_lowbit_identity(mesh_device, monkeypatch):
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

    def run(side: str) -> torch.Tensor:
        monkeypatch.setattr(settings, "get", lambda n: side if n == "LOWBIT_HOST" else get(n))
        for o in gc.get_objects():
            if isinstance(o, TtIndexer):
                o.lowbit_host = side == "on"
        for i in layers:
            state.load_prefix(i, prefix[i], start)
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
        out = model.to_host(h).float()
        model.free(h)
        return out

    outs = [(side, run(side)) for side in ("on", "off", "on")]
    for side, o in outs:
        print(f"LOWBIT_IDENTITY {side}: pcc vs golden {metrics.pcc(o, want):.6f}", flush=True)
    (_, on1), (_, off), (_, on2) = outs
    print(f"LOWBIT_IDENTITY on == on (determinism): {torch.equal(on1, on2)}", flush=True)
    print(
        f"LOWBIT_IDENTITY on == off: {torch.equal(on1, off)}, max |on - off| {float((on1 - off).abs().max()):.3e}",
        flush=True,
    )
    assert torch.equal(on1, on2), "LOWBIT_HOST=on is not deterministic"
    assert torch.equal(on1, off), "LOWBIT_HOST on and off differ"
