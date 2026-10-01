# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F46: with a prior bring-up (spec ``prior``) the dashboard shows this run next to the prior, in both styles:
per-layer and gate accuracy per matching task, the current profile's chunk time (total, per section, per block type),
the 0->seq TTFT, and the runs.compare task rows. Values missing on one side stay empty; no prior, no view."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.dashboard.export import build, load_prior, main
from models.demos.common.bringup.selftest.test_prior import _prior

SHIM = Path(__file__).with_name("dom_shim.js")
# Loads the teletext page script without a DOM (the screen part is skipped) and prints page 107's screens as text.
TT_TEXT = """
globalThis.window = {};
const src = require("fs").readFileSync(process.argv[1], "utf8");
new Function(src.slice(src.indexOf("<script>") + 8, src.lastIndexOf("</script>")))();
const T = window.__TT, B = T.buildAll();
console.log(JSON.stringify({pages: T.PAGES.map(p => p.n), p107: (B[107] || []).map(T.toText)}));
"""


def task(tid, deps=()):
    return {"id": tid, "title": f"task {tid}", "deps": list(deps), "gate": {"cmd": "true", "metrics": {}}}


def put(bdir: Path, tid: str, **metrics):
    res = bdir / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{tid}.json").write_text(
        json.dumps({"task": tid, "metrics": {k: {"value": v, "t": "x"} for k, v in metrics.items()}})
    )


def profile(bdir: Path, name: str, scale: float):
    sec = {"attn": 10.0 * scale, "mlp": 5.0 * scale}
    op = lambda ms: [{"op": "linear", "shape": "x", "calls": 1, "programs": 1, "ms": ms, "ms_per_chip": {"0": ms}}]
    (bdir / "results" / name).write_text(
        json.dumps(
            {
                "chunk": [51200, 56320],
                "layers": [0, 1, 2],
                "wall_ms": 20.0 * scale,
                "sections_ms": sec,
                "sections_ms_per_chip": {k: {"0": v} for k, v in sec.items()},
                "programs": {},
                "ops": {f"L{i}.{k}": op(v / 3) for i in range(3) for k, v in sec.items()},
            }
        )
    )


@pytest.fixture
def pair(fx):
    """This run (mesh 2x2) just started; its prior (mesh 1x4) finished, with a later perf profile."""
    s = Spec.load(fx(box={"mesh": [2, 2]}))
    _prior(s.repo, block_types=s.data["block_types"], ladder=s.data["ladder"])
    s = Spec.load(fx(box={"mesh": [2, 2]}, prior="models/demos/priorfix"))
    p = s.prior_spec()
    tids = [task("R.1"), task("L.s256", ["R.1"]), task("X.1", ["L.s256"]), task("P.1", ["X.1"])]
    Ledger(p.bringup_dir).write_tasks({"tasks": tids})
    Ledger(s.bringup_dir).write_tasks({"tasks": tids[:2]})
    for tid in ("R.1", "L.s256", "X.1", "P.1"):
        Ledger(p.bringup_dir).update(tid, status="PASS", attempts=1)
    Ledger(s.bringup_dir).update("R.1", status="PASS", attempts=2, debugger_attempts=1)
    put(p.bringup_dir, "L.s256", pcc_layer_L00=0.999, pcc_layer_L01=0.998, pcc_state_min=0.99)
    put(s.bringup_dir, "L.s256", pcc_layer_L00=0.9995)
    put(p.bringup_dir, "X.1", prefill_ms_full=2600, prefill_seq=56320, prefill_chunk=5120, device_model_hybrid=0)
    profile(p.bringup_dir, "X.1_profile.json", 2.0)
    profile(p.bringup_dir, "P.1_profile.json", 1.5)  # newer: the one the dashboard treats as current
    os.utime(p.bringup_dir / "results" / "X.1_profile.json", (1, 1))
    profile(s.bringup_dir, "X.1_profile.json", 1.0)
    return s


def test_prior_data(pair):
    P = load_prior(pair)
    assert (P["this"]["label"], P["prior"]["label"], P["chunk"]) == ("2x2", "1x4", "50k->55k")
    lay = {(r["task"], r["layer"]): r for r in P["layers"]}
    assert lay[("L.s256", 0)]["delta"] == pytest.approx(0.0005)
    assert lay[("L.s256", 1)]["this"] is None and lay[("L.s256", 1)]["delta"] is None
    assert [(r["task"], r["metric"], r["this"], r["prior"]) for r in P["gate"]] == [
        ("L.s256", "pcc_state_min", None, 0.99)
    ]
    dev, wall, ttft = P["perf"]
    assert (dev["this"], dev["prior"], dev["delta"]) == (15.0, 22.5, -7.5)  # prior: P.1, not X.1
    assert dev["src"] == ["X.1_profile.json", "P.1_profile.json"] and wall["delta"] == -10.0
    assert (ttft["what"].split()[0], ttft["this"], ttft["prior"], ttft["src"]) == ("0->55k", None, 2600, [None, "X.1"])
    assert [(r["what"], r["this"], r["prior"]) for r in P["sections"]] == [("attn", 10.0, 15.0), ("mlp", 5.0, 7.5)]
    assert [(r["short"], r["this"], r["prior"]) for r in P["blocks"]] == [("blk", 5.0, 7.5)]  # per layer
    t = {r["task"]: r for r in P["tasks"]}
    assert list(t) == ["R.1", "L.s256", "X.1", "P.1"]
    assert t["R.1"]["status"] == ["PASS", "PASS"] and t["R.1"]["attempts"] == [2, 1] and t["R.1"]["debugger"] == [1, 0]
    assert t["L.s256"]["status"] == ["TODO", "PASS"] and t["P.1"]["status"] == [None, "PASS"]


def test_no_prior_no_view(fx):
    s = Spec.load(fx())
    Ledger(s.bringup_dir).write_tasks({"tasks": [task("R.1")]})
    assert load_prior(s) is None and build(s)["prior"] is None


def test_missing_prior_is_reported(fx):
    s = Spec.load(fx(prior="models/demos/nope"))
    Ledger(s.bringup_dir).write_tasks({"tasks": [task("R.1")]})
    assert "no bringup/spec.yaml" in load_prior(s)["error"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
@pytest.mark.parametrize("with_prior", [True, False])
def test_both_styles_render_the_view(pair, fx, tmp_path, with_prior):
    spec = pair if with_prior else Spec.load(fx(box={"mesh": [2, 2]}))
    outs = main(["--spec", str(spec.path), "--style", "both", "--out", str(tmp_path)])
    r = subprocess.run(["node", str(SHIM), str(outs[0])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    els = json.loads(r.stdout)
    assert els["#prior-sec"]["hidden"] is (not with_prior)
    r = subprocess.run(["node", "-e", TT_TEXT, str(outs[1])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tt = json.loads(r.stdout)
    if not with_prior:
        assert 107 not in tt["pages"] and not tt["p107"]
        return
    assert "vs prior: 2x2 (this run) vs 1x4" in els["#s-prior"]["text"] and els["#prior-body"]["html"] > 0
    assert 107 in tt["pages"] and tt["pages"][-1] == 109  # Final tests is the last page
    screens = "\n".join(tt["p107"])
    assert "PAGE ERROR" not in screens and "VS PRIOR 2X2/1X4" in screens
    for want in ("chunk device", "TTFT 0->55k", "blk", "L.s256", "state_min", "R.1"):
        assert want in screens, want
