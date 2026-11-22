# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7C diagnostic (not gated): where the host time a NEW glued layout costs goes.

Runs GLUED_PROBE_WARM warm-up AgentX mixes, then GLUED_PROBE_MIXES new mixes, each once fresh and once more (its
layouts repeated). Per run: the host enqueue ms of every chunk (run_chunk until it returns, before the sync), the wall,
the host -> device uploads (ttnn.from_torch with a device / copy_host_to_device_tensor calls) and a cProfile of the
chunks. Prints fresh - repeat per chunk (enqueue, wall, uploads) and the functions whose own time differs most.

    <the walls env> GLUED_PROBE_MIXES=3 scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_host_probe.py
"""

from __future__ import annotations

import cProfile
import importlib
import os
import pstats
import time

import pytest

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

_pc = importlib.reload(_pc)
CHUNK, packed, glued = _pc.CHUNK, _pc.packed, _pc.glued

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("GLUED_PROBE_MIXES", "GLUED_PROBE_WARM")
BATCH_LAYERS = "executed"


def _count_uploads(counter: dict):
    import ttnn

    f0, c0 = ttnn.from_torch, ttnn.copy_host_to_device_tensor

    def ft(*a, **k):
        if k.get("device") is not None:
            counter["n"] += 1
        return f0(*a, **k)

    def ch(*a, **k):
        counter["n"] += 1
        return c0(*a, **k)

    ttnn.from_torch, ttnn.copy_host_to_device_tensor = ft, ch
    return lambda: (setattr(ttnn, "from_torch", f0), setattr(ttnn, "copy_host_to_device_tensor", c0))


@mesh_parametrize(timeout_s=3 * 3600)
def test_glued_host_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tests.packed import test_glued_walls as GW

    GW = importlib.reload(GW)
    assert settings.get("DECODER_SKIP") == "decode"
    W = GW._walls_mod()
    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = W._cap(deploy)
    monkeypatch.setitem(S.data["target"], "seq", max(cap, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    if model.skip_mode != "decode":
        model.use_decoder_skip("decode")
    nwarm = int(os.environ.get("GLUED_PROBE_WARM", "") or 5)
    nmix = int(os.environ.get("GLUED_PROBE_MIXES", "") or 3)
    warm = [GW.agentx_mix(7000 + w) for w in range(nwarm)]
    mixes = [GW.agentx_mix(7200 + m) for m in range(nmix)]
    need = max(GW._shift(k) + h + n for reqs in warm + mixes for k, (h, n) in enumerate(reqs)) + CHUNK
    text = PositionGolden(S, need, CHUNK).tokens()

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    def run(plan, prof=None, ups=None):
        reqs, toks, chunks, _ = GW._plan_chunks(plan, text, "glued")
        last = {}
        for j, d in enumerate(chunks):
            for _, s in d.live():
                last[s.slot] = j
        states, out = {}, []
        try:
            for j, desc in enumerate(chunks):
                live = [s for _, s in desc.live()]
                for s in live:
                    if s.slot not in states:
                        states[s.slot] = model.new_state(cap)
                inp = make_input(mesh_device, desc, [toks[s.slot][s.start : s.end] for s in live])
                sts = [None if s is None else states[s.slot] for s in desc.segments]
                model.sync()
                u0 = ups["n"] if ups is not None else 0
                e0 = mesh_device.num_program_cache_entries()
                t0 = time.time()
                if prof is not None:
                    prof.enable()
                nat = ids_from_input(mesh_device, inp, desc)
                run_chunk(model, desc, nat, sts, layers, last=False, inp=inp)
                if prof is not None:
                    prof.disable()
                t1 = time.time()
                model.sync()
                t2 = time.time()
                out.append(
                    (
                        (t1 - t0) * 1e3,
                        (t2 - t0) * 1e3,
                        (ups["n"] - u0) if ups is not None else 0,
                        mesh_device.num_program_cache_entries() - e0,
                    )
                )
                ttnn.deallocate(inp)
                for s in live:
                    if last[s.slot] == j:
                        states.pop(s.slot).free()
        finally:
            for st in states.values():
                st.free()
        return out

    with packed("on"), glued("on"):
        for p in warm:
            run(p)
        ups = {"n": 0}
        undo = _count_uploads(ups)
        pf, pr = cProfile.Profile(), cProfile.Profile()
        tot = {"fresh": [], "repeat": []}
        try:
            for m, p in enumerate(mixes):
                f = run(p, pf, ups)
                r = run(p, pr, ups)
                tot["fresh"] += f
                tot["repeat"] += r
                for k, (a, b) in enumerate(zip(f, r)):
                    print(
                        f"PROBE mix {m} chunk {k}: enqueue {a[0]:.1f} / {b[0]:.1f} ms (fresh / repeat), wall "
                        f"{a[1]:.1f} / {b[1]:.1f}, uploads {a[2]} / {b[2]}, programs built {a[3]} / {b[3]}",
                        flush=True,
                    )
        finally:
            undo()
    n = len(tot["fresh"])
    for k in ("fresh", "repeat"):
        v = tot[k]
        print(
            f"PROBE {k}: {n} chunks, enqueue {sum(x[0] for x in v) / n:.2f} ms, wall {sum(x[1] for x in v) / n:.2f} ms, "
            f"uploads {sum(x[2] for x in v) / n:.2f} a chunk, programs built {sum(x[3] for x in v)}"
        )
    sf, sr = pstats.Stats(pf).stats, pstats.Stats(pr).stats
    diff = []
    for fn, v in sf.items():
        rv = sr.get(fn, (0, 0, 0, 0, {}))
        diff.append(((v[2] - rv[2]) * 1e3 / n, (v[3] - rv[3]) * 1e3 / n, v[1] - rv[1], fn))
    print("PROBE functions by own-time difference a chunk (fresh - repeat): own ms, cum ms, calls, function")
    for own, cum, calls, fn in sorted(diff, key=lambda x: -x[0])[:40]:
        print(f"PROBE OWN {own:8.3f} {cum:8.3f} {calls:6d}  {fn[0].split('/tt-metal/')[-1]}:{fn[1]}:{fn[2]}")
    print("PROBE functions by cumulative difference a chunk")
    for own, cum, calls, fn in sorted(diff, key=lambda x: -x[1])[:50]:
        print(f"PROBE CUM {cum:8.3f} {own:8.3f} {calls:6d}  {fn[0].split('/tt-metal/')[-1]}:{fn[1]}:{fn[2]}")
