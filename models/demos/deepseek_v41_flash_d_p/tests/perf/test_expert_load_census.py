# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.79 / X.9 (owner 2026-11-18 / 11-19): the routed-expert load at the measurement point, over several texts and
consecutive chunks. Diagnostic only, no model change.

The point is the gate's / the A/B's (test_skip_positions, DECODER_SKIP=decode): CENSUS_POS (default 145408), the text
BRINGUP_POS_TOKENS (F276, reference/texts.py: prose by default; every metric then carries _t<text>). X.9: CENSUS_CHUNKS
= k (default 1) consecutive chunks [p + j C, p + (j + 1) C), j < k, each after the one before, the real previous chunk
[p - C, p) first. For every MoE layer and chunk the test records, per chip, the routed rows of its local experts (the
head's counts and the call's expert ids, read back after the chunk: a diagnostic host read in the test, never in the
model) and the routed_experts_combine program's device time per chip (real-time profiler, one separate pass with no
reads in it, each chunk profiled on its own).

Derived per (layer, chunk): per chip the active experts, row tiles (sum of ceil(rows / 32)) and rows; the busiest
chip's tiles (tiles_max) and the busiest chip's tiles if the rows were spread evenly over the chips (tiles_balanced =
ceil(rows / chips / 32)); the program time (max over chips) and DS41-299's fit (205 us + 6.77 us x tiles_max); the
layer's 8 hottest experts. Metrics (mean over the k chunks of the sum over layers):

    census_experts_ms             measured routed_experts_combine ms a chunk (slowest chip a program, summed)
    census_experts_ms_fit         DS41-299's fit of it from tiles_max; census_experts_ms_balanced_fit: from tiles_balanced
    census_tiles_max_sum          the busiest chip's tiles, summed over layers; census_tiles_balanced_sum: balanced
    census_hot8_overlap_chunks    |hot8(j) & hot8(j + 1)| / 8, mean over consecutive chunk pairs and layers (k >= 2)
    census_hot8_overlap_texts     (when other texts' side records at the same p and k exist) |hot8 & hot8'| / 8 of the
                                  texts' summed counts, mean over the other texts and layers

Side records: the per-text routing counts in <art>/census/census_p<p>_k<k>_t<text>.json (kept across gates, read by the
cross-text overlap and tests/perf/census_texts.py) and its per-layer summary as results/<task>_census_t<text>.json;
CENSUS_OUT, when set, gets a copy.

    BRINGUP_POS_TOKENS=prose CENSUS_CHUNKS=4 DS41_DECODER_SKIP=decode PYTHONPATH=$PWD scripts/run_safe_pytest.sh \\
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_expert_load_census.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True  # FIX.29
BATCH_ITEM_VARS = ("CENSUS_POS", "CENSUS_CHUNKS", "CENSUS_OUT")  # F279: per-run, not the model's configuration
FIT_US = (205.0, 6.77)  # DS41-299: routed_experts_combine us a layer = a + b x the busiest chip's 32-row tiles
HOT = 8


def fit_us(tiles: int) -> float:
    return FIT_US[0] + FIT_US[1] * tiles


def hot_set(rows: dict, n: int = HOT) -> list[int]:
    """The n experts with the most rows (ties by expert id, so the set is a pure function of the counts)."""
    return [e for e, _ in sorted(rows.items(), key=lambda kv: (-kv[1], int(kv[0])))[:n]]


def overlap(a, b) -> float:
    return len(set(map(int, a)) & set(map(int, b))) / float(max(len(a), 1))


def side_dir() -> Path:
    return Path(S.art) / "census"


def side_path(p: int, k: int, text: str) -> Path:
    return side_dir() / f"census_p{p}_k{k}_t{text}.json"


def layer_rows(rec: dict) -> dict:
    """{layer: {expert: rows summed over the record's chunks}} of one text's side record."""
    out: dict = {}
    for ch in rec["chunks"]:
        for L in ch["layers"]:
            d = out.setdefault(int(L["layer"]), {})
            for e, v in L["expert_rows"].items():
                d[int(e)] = d.get(int(e), 0) + int(v)
    return out


@mesh_parametrize(timeout_s=3600)
def test_expert_load_census(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import torch

    import ttnn
    from models.demos.common.bringup.reference import texts
    from tt_py_test_utils_common.realtime_profiler_utils import profile_realtime_program

    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode (the measurement's mode)"
    if os.environ.get(settings.env_name("ENGRAM_TABLE")) is None:
        monkeypatch.setenv(settings.env_name("ENGRAM_TABLE"), str(settings.table["ENGRAM_TABLE"].default))
    chunk = int(S.get("target.chunk"))
    p = int(os.environ.get("CENSUS_POS", "145408"))
    k = int(os.environ.get("CENSUS_CHUNKS", "1"))
    assert k >= 1 and p >= chunk, (p, k)
    end = p + k * chunk
    monkeypatch.setitem(S.data["target"], "seq", max(end, int(S.data["target"]["seq"])))
    # F282: every chunk here is last=False: under decode only layers 0-20 run, so only they are loaded
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    loaded = model.skip_mode
    if loaded != "decode":
        model.use_decoder_skip("decode")

    mb = ttnn.bringup.moe_block
    real = mb.routed_experts_combine
    n_chips = len(mesh_device.get_device_ids())
    cap = {"on": False, "layer": None, "calls": []}

    def wrapped(*args, **kw):
        out = real(*args, **kw)
        if cap["on"]:
            comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
            counts = ttnn.to_torch(args[1], mesh_composer=comp).reshape(n_chips, -1)
            ids = ttnn.to_torch(args[6], mesh_composer=comp).reshape(n_chips, -1)
            cap["calls"].append((cap["layer"], counts.to(torch.int64), ids.to(torch.int64)))
        return out

    monkeypatch.setattr(mb, "routed_experts_combine", wrapped)

    g = PositionGolden(S, end - chunk, chunk)  # ids [0, end): the text the gate names (F276)
    text = texts.explicit() or g.text["name"]
    ids = g.tokens()
    prev_t = ids[p - chunk : p]
    cur = [(ids[p + j * chunk : p + (j + 1) * chunk], p + j * chunk) for j in range(k)]
    state = model.new_state(end)

    def run(tokens, start):
        model.begin_chunk(tokens, start, last=False)
        h = model.embed(tokens)
        for i in layers:
            cap["layer"] = i
            h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        model.free(h)

    def seq():
        run(prev_t, p - chunk)
        for t, s in cur:
            run(t, s)
        model.sync()

    dev_to_chip = {int(d): c for c, d in enumerate(mesh_device.get_device_ids())}
    calls_per_chunk, recs_per_chunk = [], []
    try:
        for _ in range(2):  # (warm: compiles every chunk's programs)
            seq()
        run(prev_t, p - chunk)
        model.sync()
        for t, s in cur:  # the counts, read back after each chunk's layers
            cap["on"], cap["calls"] = True, []
            run(t, s)
            model.sync()
            cap["on"] = False
            calls_per_chunk.append(list(cap["calls"]))
        # the profiled pass: the same chunks again from the real previous chunk, no host reads inside
        run(prev_t, p - chunk)
        model.sync()
        for t, s in cur:
            _, recs = profile_realtime_program(mesh_device, lambda: run(t, s), collect_all=True)
            model.sync()
            recs_per_chunk.append(recs)
    finally:
        state.free()
        if loaded != "decode":
            model.use_decoder_skip(loaded)

    out = {"pos": p, "chunk": chunk, "chunks": [], "text": g.text, "n_chips": n_chips, "fit_us": FIT_US, "k": k}
    for j, (calls, recs) in enumerate(zip(calls_per_chunk, recs_per_chunk)):
        progs: dict = {}
        for r in recs:
            if any("routed_experts_combine" in s.replace("\\", "/") for s in r["kernel_sources"]):
                c = dev_to_chip.get(r["chip_id"], r["chip_id"])
                progs.setdefault(r["runtime_id"], {})[c] = r["duration_ns"] / 1e3
        order = sorted(progs)
        assert len(order) == len(calls), (j, len(order), len(calls))
        ch_out = {"start": cur[j][1], "layers": []}
        for rid, (layer, counts, eids) in zip(order, calls):
            chips, expert_rows = [], {}
            for c in range(n_chips):
                loc_ids = [int(e) for e in eids[c].tolist()]
                loc = [int(counts[c, e]) for e in loc_ids]
                for e, v in zip(loc_ids, loc):
                    expert_rows[e] = expert_rows.get(e, 0) + v
                chips.append(
                    {
                        "experts": loc_ids,
                        "counts": loc,
                        "active": sum(1 for v in loc if v),
                        "tiles": sum(-(-v // 32) for v in loc),
                        "rows": sum(loc),
                        "us": round(progs[rid].get(c, -1.0), 1),
                    }
                )
            rows_total = sum(x["rows"] for x in chips)
            tiles_max = max(x["tiles"] for x in chips)
            tiles_bal = -(-rows_total // (32 * n_chips))
            ch_out["layers"].append(
                {
                    "layer": layer,
                    "program_us_max": round(max(progs[rid].values()), 1),
                    "program_us_min": round(min(progs[rid].values()), 1),
                    "fit_us": round(fit_us(tiles_max), 1),
                    "fit_balanced_us": round(fit_us(tiles_bal), 1),
                    "tiles_max": tiles_max,
                    "tiles_balanced": tiles_bal,
                    "tiles_mean": round(sum(x["tiles"] for x in chips) / n_chips, 1),
                    "rows_max": max(x["rows"] for x in chips),
                    "rows_total": rows_total,
                    "expert_max": max(expert_rows.values()),
                    "hot8": hot_set(expert_rows),
                    "expert_rows": {str(e): v for e, v in sorted(expert_rows.items()) if v},
                    "chips": chips,
                }
            )
            L = ch_out["layers"][-1]
            print(
                f"CENSUS c{j} L{layer}: program {L['program_us_max']:.0f} us (min chip {L['program_us_min']:.0f}, fit "
                f"{L['fit_us']:.0f}), tiles max {tiles_max} balanced {tiles_bal} mean {L['tiles_mean']}, rows max "
                f"{L['rows_max']} of {rows_total}, hottest expert {L['expert_max']} rows, hot8 {L['hot8']}",
                flush=True,
            )
        out["chunks"].append(ch_out)

    def per_chunk(key):
        return [sum(L[key] for L in ch["layers"]) for ch in out["chunks"]]

    ms = [v / 1e3 for v in per_chunk("program_us_max")]
    fit = [v / 1e3 for v in per_chunk("fit_us")]
    fit_bal = [v / 1e3 for v in per_chunk("fit_balanced_us")]
    tmax, tbal = per_chunk("tiles_max"), per_chunk("tiles_balanced")
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    for j in range(k):
        print(
            f"CENSUS chunk {j} (start {cur[j][1]}): routed_experts_combine {ms[j]:.2f} ms (fit {fit[j]:.2f}, balanced "
            f"fit {fit_bal[j]:.2f}), tiles max sum {tmax[j]} balanced {tbal[j]}"
        )
    metrics.record("census_experts_ms", round(mean(ms), 2), per_chunk=[round(v, 2) for v in ms])
    metrics.record("census_experts_ms_fit", round(mean(fit), 2), per_chunk=[round(v, 2) for v in fit])
    metrics.record("census_experts_ms_balanced_fit", round(mean(fit_bal), 2), per_chunk=[round(v, 2) for v in fit_bal])
    metrics.record("census_tiles_max_sum", round(mean(tmax), 1), per_chunk=tmax)
    metrics.record("census_tiles_balanced_sum", round(mean(tbal), 1), per_chunk=tbal)
    metrics.record("census_chunks", k)
    print(
        f"CENSUS {text} p{p} k{k}: routed_experts_combine {mean(ms):.2f} ms a chunk (fit {mean(fit):.2f}, balanced fit "
        f"{mean(fit_bal):.2f}), tiles max sum {mean(tmax):.1f} balanced {mean(tbal):.1f}"
    )

    if k >= 2:
        per_layer = []
        for li in range(len(out["chunks"][0]["layers"])):
            hs = [ch["layers"][li]["hot8"] for ch in out["chunks"]]
            v = mean([overlap(hs[j], hs[j + 1]) for j in range(k - 1)])
            per_layer.append(v)
            out["chunks"][0]["layers"][li]["hot8_overlap_chunks"] = round(v, 3)
        metrics.record(
            "census_hot8_overlap_chunks", round(mean(per_layer), 3), per_layer=[round(v, 3) for v in per_layer]
        )
        print(f"CENSUS hot8 overlap between consecutive chunks: {mean(per_layer):.3f} (per layer {per_layer})")

    side = side_path(p, k, text)
    side.parent.mkdir(parents=True, exist_ok=True)
    tmp = side.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, side)
    # the results copy is the per-layer summary only (the repo's 500 KB file limit: the full record was 1.6 MB)
    drop = ("expert_rows", "chips")
    summary = {
        **out,
        "chunks": [
            {**ch, "layers": [{a: v for a, v in L.items() if a not in drop} for L in ch["layers"]]}
            for ch in out["chunks"]
        ],
    }
    metrics.record_json(f"census_t{text}", {**summary, "full": str(side)})
    if os.environ.get("CENSUS_OUT"):
        with open(os.environ["CENSUS_OUT"], "w") as f:
            json.dump(out, f)
    print(f"CENSUS written to {side} at {time.ctime()}")

    # the cross-text overlap from the other texts' records at the same p and k (the last text of a gate sees them all)
    mine = layer_rows(out)
    others = {}
    for q in sorted(side_dir().glob(f"census_p{p}_k{k}_t*.json")):
        t = q.stem.split("_t", 1)[1]
        if t != text:
            others[t] = layer_rows(json.loads(q.read_text()))
    if others:
        vals = {}
        for t, rows in others.items():
            common = sorted(set(mine) & set(rows))
            vals[t] = mean([overlap(hot_set(mine[li]), hot_set(rows[li])) for li in common]) if common else 0.0
        metrics.record("census_hot8_overlap_texts", round(mean(list(vals.values())), 3), per_text=vals)
        print(f"CENSUS hot8 overlap with the other texts (summed over chunks): {vals}")
