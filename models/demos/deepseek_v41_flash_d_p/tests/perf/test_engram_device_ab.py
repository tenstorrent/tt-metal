# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.87: ENGRAM_DEVICE off vs on on ONE loaded model (the switch is read per chunk), DECODER_SKIP=decode, the
steady / resumed chunks of test_skip_positions (the real previous chunk first, warm-up, WALL_RUNS clean walls, one
real-time-profiled run) at each P87_POS point, the sides alternated P87_ROUNDS times; also every layer output of the
timed chunk compared on vs off (P87_COMPARE=1: the rows are bit-identical, so must the layers be).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "DS41_DECODER_SKIP=decode
        DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic P87_POS=145408 scripts/run_safe_pytest.sh --run-all
        --no-precompile models/demos/deepseek_v41_flash_d_p/tests/perf/test_engram_device_ab.py"
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as tsp
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("P87_POS",)
BATCH_POS_VARS = ("P87_POS",)
BATCH_LAYERS = "executed"


def _side(monkeypatch, v: str) -> None:
    """ENGRAM_DEVICE only (the model's load-time snapshot of every other switch kept: a refresh() would re-read them
    all from this request's environment)."""
    monkeypatch.setenv(settings.env_name("ENGRAM_DEVICE"), v)
    if settings.snap is not None:
        settings.snap["ENGRAM_DEVICE"] = v
    assert settings.get("ENGRAM_DEVICE") == v


@mesh_parametrize(timeout_s=4 * 3600)
def test_engram_device_ab(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    assert settings.get("DECODER_SKIP") == "decode", "run with DS41_DECODER_SKIP=decode"
    chunk = int(S.get("target.chunk"))
    points = [int(x) for x in os.environ.get("P87_POS", "145408").split(",") if x.strip()]
    rounds = int(os.environ.get("P87_ROUNDS", "2"))
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    want = max([p + chunk for p in points] + [int(S.data["target"]["seq"])])
    monkeypatch.setitem(S.data["target"], "seq", want)
    t0 = time.time()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    print(
        f"model ready in {time.time() - t0:.1f} s; device engram tables: "
        f"{'loaded' if model.engram_dev is not None else 'NOT loaded'} (load {model.engram_dev_load_s} s)",
        flush=True,
    )
    assert model.engram_dev is not None, "the device tables did not load (ENGRAM_DEVICE_MAX_GB / mesh)"
    if model.skip_mode != "decode":
        model.use_decoder_skip("decode")
    dram = tsp._dram(mesh_device)
    if dram is not None:
        print(f"DRAM after load: {dram.max() / tsp.GIB:.2f} of {dram.capacity / tsp.GIB:.2f} GiB a chip", flush=True)
        metrics.record("p87_dram_load_gib_max", round(dram.max() / tsp.GIB, 3))
    print(
        "switches (model snapshot): "
        + str(
            {
                k: settings.get(k)
                for k in (
                    "ENGRAM_TABLE",
                    "HOST_HIDE",
                    "ENGRAM_ASYNC",
                    "ENGRAM_DEVICE",
                    "DECODER_SKIP",
                    "HISTORY_GATHER_HIDE",
                )
            }
        ),
        flush=True,
    )
    text = os.environ.get("BRINGUP_POS_TOKENS", "prose")
    res = {}
    try:
        if os.environ.get("P87_COMPARE", "1") == "1":
            for p in points:
                _compare(model, layers, p, chunk, monkeypatch, ttnn)
        for k in range(rounds):
            for v in ("off", "on") if k % 2 == 0 else ("on", "off"):
                _side(monkeypatch, v)
                for p in points:
                    out = tsp._point(model, mesh_device, layers, p, chunk, dram)
                    res.setdefault((p, v), []).append(out)
    finally:
        _side(monkeypatch, str(settings.table["ENGRAM_DEVICE"].default))
    lines = [
        "| point | side | kind | walls ms (rounds) | device ms | chunk_input ms | host ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for (p, v), outs in sorted(res.items()):
        for kind in ("steady", "resumed"):
            walls = [o[kind]["wall"] for o in outs]
            dev = [round(o[kind]["slowest"], 2) for o in outs]
            ci = [round(o[kind]["phases"].get("chunk_input", 0.0), 2) for o in outs]
            med = sorted(walls)[len(walls) // 2]
            metrics.record(f"p87_{kind}_wall_ms_p{p}_t{text}_{v}", med)
            metrics.record(f"p87_{kind}_device_ms_p{p}_t{text}_{v}", dev[-1])
            lines.append(
                f"| p{p} {text} | {v} | {kind} | {walls} | {dev} | {ci} | {round(med - sorted(dev)[len(dev) // 2], 1)} |"
            )
    table = "\n".join(lines)
    metrics.record("p87_ab_table", table)
    print("\n" + table, flush=True)


def _compare(model, layers, p, chunk, monkeypatch, ttnn) -> None:
    """Every layer output of the chunk at p (after the real previous chunk) on vs off: equal bytes."""
    g = PositionGolden(S, p, chunk)
    ids = g.tokens()
    outs = {}
    for v in ("off", "on"):
        _side(monkeypatch, v)
        state = model.new_state(p + chunk)
        try:
            got = []
            for start, toks in ((p - chunk, ids[p - chunk : p]), (p, ids[p : p + chunk])):
                model.begin_chunk(toks, start, last=False)
                h = model.embed(toks)
                for i in layers:
                    try:
                        h2 = model.layer(i, h, start, state)
                    except Exception:
                        al = {n: getattr(h, n).is_allocated() for n in ("streams", "pre") if hasattr(h, n)}
                        print(
                            f"P87 compare: {v} start {start} layer {i} failed; input allocated {al}; engram dev "
                            f"{ {k: t.is_allocated() for k, t in model._engram_dev.items()} } full "
                            f"{ {k: t.is_allocated() for k, t in model._engram_full.items()} }",
                            flush=True,
                        )
                        raise
                    model.free(h)
                    h = h2
                    if start == p:
                        got.append(model.to_host(h2) if hasattr(h2, "streams") else None)  # (not a skipped layer)
                model.free(h)
            outs[v] = got
            tails = {i: state.tail[i].tolist() for i in model.engram}
            print(f"P87 compare p{p} {v}: tails {tails}", flush=True)
        finally:
            state.free()
    bad = []
    for k, (a, b) in enumerate(zip(outs["off"], outs["on"])):
        if a is None or b is None:
            continue
        if not torch.equal(a, b):
            bad.append((layers[k], float((a.float() - b.float()).abs().max())))
    print(f"P87 compare p{p}: {len(outs['on'])} layers, differing {bad}", flush=True)
    metrics.record(f"p87_layers_differing_p{p}", len(bad))
    assert not bad, f"ENGRAM_DEVICE on != off at p{p}: {bad}"
