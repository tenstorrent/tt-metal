# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7C diagnostic (not gated): which ops build a program for a glued chunk's NEW layout.

The walls' fresh-layout check (tests/packed/test_glued_walls.py, glued_program_builds_after_warmup) counts program-cache
entries; this names them. Runs GLUED_PROBE_WARM warm-up AgentX mixes, then wraps every ttnn / ttnn.bringup /
ttnn.experimental callable with a check of the mesh's program-cache entry count and runs GLUED_PROBE_MIXES new mixes:
each op that added an entry is printed with the model frames that called it (tt/*.py), counted over the mixes.

    <the walls env> GLUED_PROBE_MIXES=4 scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_dynamic_probe.py
"""

from __future__ import annotations

import collections
import importlib
import os
import sys

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


def _wrap_all(mesh, hits: collections.Counter):
    import ttnn

    undo = []

    def wrap(mod, name, f):
        def g(*a, **k):
            e0 = mesh.num_program_cache_entries()
            out = f(*a, **k)
            e1 = mesh.num_program_cache_entries()
            if e1 > e0:
                fr, where = sys._getframe(1), []
                while fr is not None and len(where) < 3:
                    fn = fr.f_code.co_filename
                    if "deepseek_v41_flash_d_p/tt/" in fn:
                        where.append(f"{fn.split('/tt/')[-1]}:{fr.f_lineno}:{fr.f_code.co_name}")
                    fr = fr.f_back
                hits[(f"{mod.__name__}.{name}", " < ".join(where))] += e1 - e0
            return out

        return g

    for mod in (ttnn, ttnn.bringup, ttnn.experimental):
        for name in dir(mod):
            if name.startswith("_"):
                continue
            f = getattr(mod, name, None)
            if f is None or isinstance(f, type) or not callable(f) or type(f).__name__ == "module":
                continue
            if name in ("num_program_cache_entries", "deallocate", "is_tensor_storage_on_device"):
                continue
            try:
                setattr(mod, name, wrap(mod, name, f))
                undo.append((mod, name, f))
            except (AttributeError, TypeError):
                pass
    return undo


@mesh_parametrize(timeout_s=3 * 3600)
def test_glued_dynamic_probe(mesh_device, monkeypatch):
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
    nwarm = int(os.environ.get("GLUED_PROBE_WARM", "") or 3)
    nmix = int(os.environ.get("GLUED_PROBE_MIXES", "") or 4)
    warm = [GW.agentx_mix(7000 + w) for w in range(nwarm)]
    mixes = [GW.agentx_mix(7100 + m) for m in range(nmix)]
    need = max(GW._shift(k) + h + n for reqs in warm + mixes for k, (h, n) in enumerate(reqs)) + CHUNK
    text = PositionGolden(S, need, CHUNK).tokens()

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    def run(plan):
        reqs, toks, chunks, _ = GW._plan_chunks(plan, text, "glued")
        last = {}
        for j, d in enumerate(chunks):
            for _, s in d.live():
                last[s.slot] = j
        states = {}
        try:
            for j, desc in enumerate(chunks):
                live = [s for _, s in desc.live()]
                for s in live:
                    if s.slot not in states:
                        states[s.slot] = model.new_state(cap)
                inp = make_input(mesh_device, desc, [toks[s.slot][s.start : s.end] for s in live])
                sts = [None if s is None else states[s.slot] for s in desc.segments]
                nat = ids_from_input(mesh_device, inp, desc)
                run_chunk(model, desc, nat, sts, layers, last=False, inp=inp)
                model.sync()
                ttnn.deallocate(inp)
                for s in live:
                    if last[s.slot] == j:
                        states.pop(s.slot).free()
        finally:
            for st in states.values():
                st.free()
        return len(chunks)

    with packed("on"), glued("on"):
        for p in warm:
            run(p)
        hits = collections.Counter()
        e0 = mesh_device.num_program_cache_entries()
        undo = _wrap_all(mesh_device, hits)
        try:
            n = sum(run(p) for p in mixes)
        finally:
            for mod, name, f in undo:
                setattr(mod, name, f)
        e1 = mesh_device.num_program_cache_entries()
    print(f"\nPROBE {nmix} fresh mixes ({n} chunks): {e1 - e0} program-cache entries added")
    for (op, where), c in hits.most_common():
        print(f"PROBE MISS {c:4d}  {op}  <-  {where}")
