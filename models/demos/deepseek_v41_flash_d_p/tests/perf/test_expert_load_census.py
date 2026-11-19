# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.79 (owner 2026-11-18): what bounds routed_experts_combine at the 142k measurement point. Diagnostic only, no
model change.

The point and its tokens are the gate's / the A/B's (test_skip_positions, DECODER_SKIP=decode): p = CENSUS_POS
(default 145408), random ids seed 0 (testing/profile.PositionGolden), the real previous chunk first. For every MoE
layer the test records, per chip, the routed token counts of its 12 local experts (the head's counts and the call's
expert ids, read back after the chunk: a diagnostic host read in the test, never in the model) and the
routed_experts_combine program's device time per chip (real-time profiler, one separate run, no reads in it).

Derived per (layer, chip): active experts, row tiles (sum of ceil(count / 32)), rows; per layer: the busiest chip's
tiles and the program's time. Written to $CENSUS_OUT (default /tmp/p79_census.json) for the fit (P.79 BREADCRUMBS).

    DS41_DECODER_SKIP=decode PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_expert_load_census.py
"""

from __future__ import annotations

import json
import os
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29


@mesh_parametrize(timeout_s=3600)
def test_expert_load_census(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import torch

    import ttnn
    from tt_py_test_utils_common.realtime_profiler_utils import profile_realtime_program

    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode (the measurement's mode)"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    p = int(os.environ.get("CENSUS_POS", "145408"))
    monkeypatch.setitem(S.data["target"], "seq", max(p + chunk, int(S.data["target"]["seq"])))
    layers = list(S.layers())
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")

    mb = ttnn.bringup.moe_block
    real = mb.routed_experts_combine
    cap = {"on": False, "calls": []}

    def wrapped(*args, **kw):
        out = real(*args, **kw)
        if cap["on"]:
            comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            counts = ttnn.to_torch(args[1], mesh_composer=comp).reshape(len(mesh_device.get_device_ids()), -1)
            ids = ttnn.to_torch(args[6], mesh_composer=comp).reshape(len(mesh_device.get_device_ids()), -1)
            cap["calls"].append((counts.to(torch.int64), ids.to(torch.int64)))
        return out

    monkeypatch.setattr(mb, "routed_experts_combine", wrapped)

    g = PositionGolden(S, p, chunk)
    ids = g.tokens()
    prev_t, cur_t = ids[p - chunk : p], ids[p : p + chunk]
    state = model.new_state(p + chunk)

    def run(tokens, start):
        model.begin_chunk(tokens, start, last=False)
        h = model.embed(tokens)
        for i in layers:
            h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        model.free(h)

    try:
        for _ in range(2):  # (warm: compiles)
            run(prev_t, p - chunk)
            run(cur_t, p)
            model.sync()
        run(prev_t, p - chunk)
        model.sync()
        cap["on"] = True
        run(cur_t, p)
        model.sync()
        cap["on"] = False
        per_layer_counts = list(cap["calls"])
        # the profiled run: no host reads inside it
        run(prev_t, p - chunk)
        model.sync()
        dev_to_chip = {int(d): c for c, d in enumerate(mesh_device.get_device_ids())}
        _, recs = profile_realtime_program(mesh_device, lambda: run(cur_t, p), collect_all=True)
    finally:
        state.free()
        if loaded != "decode":
            model.use_decoder_skip(loaded)

    progs: dict = {}
    for r in recs:
        if any("routed_experts_combine" in s.replace("\\", "/") for s in r["kernel_sources"]):
            progs.setdefault(r["runtime_id"], {})[dev_to_chip.get(r["chip_id"], r["chip_id"])] = r["duration_ns"] / 1e3
    order = sorted(progs)
    n_chips = len(dev_to_chip)
    out = {"pos": p, "chunk": chunk, "layers": []}
    assert len(order) == len(per_layer_counts), (len(order), len(per_layer_counts))
    for li, (rid, (counts, eids)) in enumerate(zip(order, per_layer_counts)):
        chips = []
        for c in range(n_chips):
            loc = [int(counts[c, int(e)]) for e in eids[c].tolist()]
            chips.append(
                {
                    "counts": loc,
                    "active": sum(1 for v in loc if v),
                    "tiles": sum(-(-v // 32) for v in loc),
                    "rows": sum(loc),
                    "us": round(progs[rid].get(c, -1.0), 1),
                }
            )
        busiest = max(chips, key=lambda x: x["tiles"])
        out["layers"].append(
            {
                "i": li,
                "program_us_max": round(max(progs[rid].values()), 1),
                "program_us_min": round(min(progs[rid].values()), 1),
                "tiles_max": busiest["tiles"],
                "tiles_mean": round(sum(x["tiles"] for x in chips) / n_chips, 1),
                "rows_max": max(x["rows"] for x in chips),
                "rows_total": sum(x["rows"] for x in chips),
                "expert_max": max(max(x["counts"]) for x in chips),
                "chips": chips,
            }
        )
        L = out["layers"][-1]
        print(
            f"CENSUS L{li}: program {L['program_us_max']:.0f} us (min chip {L['program_us_min']:.0f}), tiles max "
            f"{L['tiles_max']} mean {L['tiles_mean']}, rows max {L['rows_max']} of {L['rows_total']}, hottest expert "
            f"{L['expert_max']} rows",
            flush=True,
        )
    tot = sum(x["program_us_max"] for x in out["layers"]) / 1e3
    print(f"CENSUS total routed_experts_combine {tot:.2f} ms a chunk ({len(order)} layers)")
    metrics.record("census_experts_ms", round(tot, 2))
    path = os.environ.get("CENSUS_OUT", "/tmp/p79_census.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"CENSUS written to {path} at {time.ctime()}")
