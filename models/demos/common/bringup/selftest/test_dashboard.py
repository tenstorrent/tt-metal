# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F7: the generic dashboard renders every section from a model's records (fixture model, synthetic records).
The page script is executed under node with a DOM stand-in, so a runtime error in any section fails the test."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from models.demos.common.bringup.core import metrics as M
from models.demos.common.bringup.core.ledger import Ledger
from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.dashboard.export import bound_of, build, main
from models.demos.common.bringup.plan.ledger_gen import generate
from models.demos.common.bringup.plan.memory import check_plan
from models.demos.common.bringup.reference.check_reference import write_graphs
from models.demos.common.bringup.selftest.fixture_model import Reference
from models.demos.common.bringup.selftest.test_plan import TENSORS, comps, plan

SHIM = Path(__file__).with_name("dom_shim.js")


@pytest.fixture
def records(fx, monkeypatch):
    s = Spec.load(fx(box={"mesh": [1, 4], "chip_dram_gb": 0.01}))
    led = Ledger(s.bringup_dir)
    led.write_tasks(generate(s, Reference()))
    res = led.results_dir
    monkeypatch.setenv(M.RESULTS_ENV, str(res))
    write_graphs(Reference(), {"blk": 0})
    for i in range(3):
        M.record(f"pcc_hidden_L{i:02d}", 1 - 1e-7 * (i + 1), task="R.2")
        M.record(f"pcc_layer_L{i:02d}", 0.999 - 0.001 * i, task="L.s256")
    for c in range(4):
        M.record(f"chunk_seconds_c{c:02d}", 0.5 + 0.1 * c, task="L.s256")
        M.record(f"prefill_chunk_ms_c{c:02d}", 500 + 100 * c, task="X.1")
    for k, v in dict(
        prefill_ms_full=2600, prefill_seq=256, prefill_chunk=64, chunk_wall_ms=800, chunk_start=192, chunk_len=64
    ).items():
        M.record(k, v, task="X.1")
    for tid in ("R.1", "R.2", "R.3", "C.blk.attn_norm", "L.s256"):
        led.update(tid, status="PASS", metrics={k: v["value"] for k, v in M.load(tid, res).items()})
    led.update(
        "C.blk.attention",
        status="STOPPED",
        reason=["stopped after 3 attempts"],
        waiting=None,
        agent_runs=[{"role": "implement", "session_id": "s1"}],
        debugger_attempts=3,
    )
    led.update("PL.1", waiting="approve the plan")
    (res / "plan_memory.json").write_text(json.dumps(check_plan(s, plan(), TENSORS, {"num_key_value_heads": 4})))
    (s.bringup_dir / "components.yaml").write_text(yaml.safe_dump(comps(Reference())))
    (s.bringup_dir / "plan.yaml").write_text(
        yaml.safe_dump(
            {
                **plan(),
                "ccl_per_layer": ["all_reduce after attention"],
                "chips": [{"chip": c, "KV head": c} for c in range(4)],
            }
        )
    )
    (s.bringup_dir / "findings.yaml").write_text(
        yaml.safe_dump(
            {"findings": [{"id": "f1", "task": "C.blk.attention", "kind": "API", "title": "x", "detail": "y"}]}
        )
    )
    prof = {
        "rung": "last",
        "chunk": [384, 512],
        "layers": [0, 1, 2],
        "wall_ms": 40.0,
        "sections_ms": {"attn.sdpa": 20.0, "attn.all_reduce": 5.0, "mlp.ffn": 10.0, "other.norm": 2.0},
        "sections_ms_per_chip": {
            k: {str(c): 1.0 + c for c in range(4)} for k in ("attn.sdpa", "attn.all_reduce", "mlp.ffn", "other.norm")
        },
        "programs": {"attn.sdpa": 3},
    }
    (res / "X.1_profile.json").write_text(json.dumps(prof))
    return s


def test_data_model(records):
    d = build(records)
    assert d["page_title"] == "fixture Prefill Bring-up" and d["layer_types"] == ["blk"] * 3
    t = {x["id"]: x for x in d["tasks"]}
    assert t["C.blk.attention"]["status"] == "STOPPED" and "debugger 3" in t["C.blk.attention"]["agent"]
    assert t["PL.1"]["waiting"] == "approve the plan"
    g = {st["name"]: st for st in d["graphs"]["blk"]}
    assert g["attn_norm"]["state"] == "device" and g["attention"]["state"] == "cpu"
    assert [x["task"] for x in d["trails"]] == ["R.2", "L.s256"] and d["trails"][1]["device"]
    # timing is performance only: warm runs; the ladder (accuracy, reads every layer back) never appears (F36)
    assert [(t["task"], t["headline"]) for t in d["timing"]] == [("X.1", "0->256"), ("X.1", "192->256")]
    assert tuple(d["timing"][0]["chunks"][0]) == (0, 0.5) and d["timing"][0]["seconds"] == 2.6
    assert not [t for t in d["timing"] if t["task"].startswith("L.") or t["how"].startswith("ladder")]
    assert d["plan"]["fits"] and len(d["plan_chips"]) == 4
    steps = {s["key"]: s for s in d["profile"]["steps"]}
    assert steps["attn.all_reduce"]["bound"] == "comm" and steps["attn.all_reduce"]["pat"] == "ring"
    assert steps["attn.sdpa"]["bound"] == "compute" and steps["attn.sdpa"]["per_chip"] == [1.0, 2.0, 3.0, 4.0]
    comp = {c["key"]: c for c in d["components"]}
    assert comp["blk/attn_norm"]["state"] == "device" and comp["model/embed"]["state"] == "device"


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_page_renders_every_section(records, tmp_path):
    out = tmp_path / "index.html"
    main(["--spec", str(records.path), "--out", str(out)])
    html = out.read_text()
    assert "<title>fixture Prefill Bring-up</title>" in html and "/*__DATA__*/null" not in html
    r = subprocess.run(["node", str(SHIM), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    els = json.loads(r.stdout)
    for sel in (
        "#facts",
        "#meter",
        "#ladder tbody",
        "#dag",
        "#strip",
        "#ops tbody",
        "#findings",
        "#timing",
        "#chips",
        "#mem-legend",
        "#p-bar",
        "#p-steps",
        "#pcc",
        "#pcc-table",
    ):
        assert els[sel]["html"] > 0, sel
    assert els["#passn"]["text"].endswith(f"/{len(Ledger(records.bringup_dir).tasks())}")
    assert "Where the time goes" in els["#s-prof"]["text"] and not els["#prof-sec"]["hidden"]
    assert "fits" in els["#shard-sub"]["text"]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_page_renders_an_empty_bring_up(fx, tmp_path):
    s = Spec.load(fx())
    Ledger(s.bringup_dir).write_tasks(generate(s, early=True))
    out = tmp_path / "index.html"
    main(["--spec", str(s.path), "--out", str(out)])
    r = subprocess.run(["node", str(SHIM), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["#prof-sec"]["hidden"]


@pytest.mark.parametrize(
    "sec,b",
    [
        ("attn.sdpa", "compute"),
        ("moe.all_reduce", "comm"),
        ("attn.kv_write", "memory"),
        ("moe.dispatch", "memory"),
        ("other.residual", "other"),
    ],
)
def test_bound_guess(sec, b):
    assert bound_of(sec) == b


def test_styles(records, tmp_path):
    from models.demos.common.bringup.dashboard.export import styles_of

    assert styles_of(records, None) == ["standard"]
    assert styles_of(records, "both") == ["standard", "teletext"]
    outs = main(["--spec", str(records.path), "--style", "both", "--out", str(tmp_path)])
    assert sorted(p.name for p in outs) == ["index.html", "teletext.html"]
    ttx = (tmp_path / "teletext.html").read_text()
    assert ttx.startswith("<title>fixture Ceefax</title>") and "/*__DATA__*/null" not in ttx
    with pytest.raises(SystemExit):
        styles_of(records, "neon")


TELETEXT_SHIM = Path(__file__).with_name("teletext_shim.js")


def screen_timeline(page, seconds, *keys):
    """The page announced at each second of the teletext screen script, run under node on a fake clock, as
    [(announcement, seconds)]; a key "@<t>:<key>" is pressed at second t."""
    import itertools

    r = subprocess.run(["node", str(TELETEXT_SHIM), str(page), str(seconds), *keys], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return [(k, len(list(g))) for k, g in itertools.groupby(json.loads(r.stdout))]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_teletext_carousel(records, tmp_path):
    """F24: Index and Model graph take turns every 20 s; hold stops it, a key restarts the count, other pages stay."""
    page = main(["--spec", str(records.path), "--style", "teletext", "--out", str(tmp_path)])
    idx, graph = "Page 100, Index", "Page 102, Model graph"
    assert screen_timeline(page, 50) == [(idx, 20), (graph, 20), (idx, 10)]
    assert screen_timeline(page, 50, "@10:h") == [(idx, 9), ("Hold on", 41)]
    assert screen_timeline(page, 50, "@15:ArrowDown") == [(idx, 35), (graph, 15)]
    assert screen_timeline(page, 50, "@5:ArrowRight")[-1] == ("Page 101, Gate ladder", 45)
