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
    for k, v in dict(pos_chunk=64, pos_ms_0=400, pos_ms_192=600, pos_ms_384=800, device_model_hybrid=0).items():
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
    assert "findings" not in d  # findings stay in findings.yaml; the dashboard does not show them
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
    # F37: one warm chunk at several start positions, for the chart under the timing table
    assert d["positions"] == {
        "task": "X.1",
        "chunk": 64,
        "points": [(0, 400), (192, 600), (384, 800)],
        "model": "all-device",
    }
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
        "#timing",
        "#pos",
        "#pos-table",
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


def test_op_rows_keep_execution_order():
    """F43: op mode books each outermost ttnn call to its layer's section, merging only back-to-back repeats."""
    from models.demos.common.bringup.testing import profiler as P

    P._P.update(
        mesh=None, current_full="L3.attention.rope", op_ns={}, seq=[], last_n_dev={0: 1}, dev_to_chip={0: 0, 1: 1}
    )
    for op, shape in [("slice", "a"), ("slice", "a"), ("rotary_embedding", "b"), ("slice", "a")]:
        P._book_op(op, shape, {0: 1e6, 1: 2e6}, 1)
    rows = P._P["op_ns"]["L3.attention.rope"]
    assert [(r["op"], r["calls"]) for r in rows] == [("slice", 2), ("rotary_embedding", 1), ("slice", 1)]
    assert rows[0]["ns"] == 4e6 and rows[0]["ns_dev"] == {0: 2e6, 1: 4e6}  # slowest chip per call, summed


def test_profile_views_per_block_type(fx, tmp_path):
    """F43: with op rows the profile gets one tab per block type (per layer, mean over its layers) and every section
    lists its ttnn ops in execution order."""
    from models.demos.common.bringup.dashboard.export import load_profile

    s = Spec.load(fx())  # fixture: 3 layers, block types from its spec
    bts = s.data["block_types"]
    op = lambda o, ms: {"op": o, "shape": "8x8 bf16", "calls": 1, "programs": 1, "ms": ms, "ms_per_chip": {"0": ms}}
    ops = {f"L{i}.attn": [op("linear", 1.0 + i), op("add", 0.5)] for i in range(3)}
    prof = {
        "chunk": [0, 64],
        "layers": [0, 1, 2],
        "wall_ms": 9.0,
        "sections_ms": {"attn": 4.5 + 1.5},
        "sections_ms_per_chip": {"attn": {"0": 7.5}},
        "programs": {"attn": 6},
        "ops": ops,
    }
    (tmp_path / "X.3_profile.json").write_text(json.dumps(prof))
    P = load_profile(s, tmp_path, {})
    assert P["views"][0]["id"] == "all" and [o["op"] for o in P["steps"][0]["ops"]] == ["linear", "add"]
    assert P["steps"][0]["ops"][0]["ms"] == 6.0  # summed over the 3 layers
    assert [v["id"] for v in P["views"][1:]] == list(bts)
    for v in P["views"][1:]:
        lays = v["layers"]
        assert v["steps"][0]["ms"] == round(sum(1.5 + i for i in lays) / len(lays), 2)  # per layer


def test_timeline_alignment_splits_programs_by_op_mode_counts():
    """F44: the pipelined run's programs are split per call with the op-mode run's per-chip counts; gap = idle before
    a call on the critical chip, slot = end - previous end, so kernels + gaps = the device timeline."""
    from models.demos.common.bringup.testing.profiler import align_timeline

    op_seq = [
        {"key": None, "op": "embedding", "shape": "e", "n_dev": {0: 1, 1: 1}},
        {"key": "L0.a", "op": "linear", "shape": "x", "n_dev": {0: 2, 1: 2}},
        {"key": "L0.a", "op": "(other)", "shape": "", "n_dev": {0: 1, 1: 1}},
        {"key": "L0.b", "op": "sync_only", "shape": "", "n_dev": {}},
    ]
    tl_seq = [
        dict(key=c["key"], op=c["op"], shape=c["shape"], host_ns=h)
        for c, h in zip([op_seq[0], op_seq[1], op_seq[3]], (5, 7, 1))
    ]
    M = 1e6  # times in ms
    progs = {  # (start, end, kernel) ns; chip 1 is the longer timeline
        0: [(0, 10 * M, 10 * M), (10 * M, 20 * M, 10 * M), (20 * M, 30 * M, 10 * M), (30 * M, 35 * M, 5 * M)],
        1: [(0, 10 * M, 10 * M), (15 * M, 25 * M, 10 * M), (25 * M, 40 * M, 15 * M), (50 * M, 60 * M, 10 * M)],
    }
    tl = align_timeline(op_seq, tl_seq, progs, {0: 0, 1: 1})
    assert tl["summary"]["critical_chip"] == 1 and tl["summary"]["device_timeline_ms"] == 60
    lin = tl["calls"][1]
    assert (lin["kernel_ns"], lin["gap_ns"], lin["slot_ns"], lin["host_ns"]) == (25 * M, 5 * M, 30 * M, 7)
    other = tl["calls"][2]
    assert (other["gap_ns"], other["slot_ns"], other["host_ns"]) == (10 * M, 20 * M, 0.0)
    assert tl["summary"]["kernel_ms"] + tl["summary"]["gap_ms"] == 60
    assert "error" in align_timeline(op_seq, tl_seq[:2], progs, {0: 0, 1: 1})  # sequences differ
    assert "error" in align_timeline(op_seq, tl_seq, {0: progs[0][:3], 1: progs[1]}, {0: 0, 1: 1})  # lost programs


@pytest.fixture
def final_records(fx, monkeypatch):
    """The "Final tests" section, last on both pages: the model smoke and the runner smoke (their recorded answers),
    the full-target ladder rung and the contract tests; whatever never ran shows as not run."""
    from models.demos.common.bringup.testing.smoke import record_answer

    s = Spec.load(fx(intake={"smoke": {"prompt": "What is the capital of France?", "expect": "Paris"}}))
    tdir = s.repo / "models/demos/fixture/tests/bringup/contract"
    tdir.mkdir(parents=True)
    tests = []
    for name, gates, kind in (("kv_write", "attention", None), ("runner_smoke", "adapter", "runner_smoke")):
        (tdir / f"test_{name}.py").write_text("def test_x():\n    pass\n")
        tests.append(
            {"test": f"models/demos/fixture/tests/bringup/contract/test_{name}.py", "section": "x", "gates": gates}
            | ({"kind": kind} if kind else {})
        )
    s.bringup_dir.mkdir(parents=True, exist_ok=True)
    (s.bringup_dir / "contract_tests.yaml").write_text(yaml.safe_dump({"tests": tests}))
    led = Ledger(s.bringup_dir)
    led.write_tasks(generate(s, Reference()))
    res = led.results_dir
    monkeypatch.setenv(M.RESULTS_ENV, str(res))

    F = build(s)["final"]
    assert [(r["mode"], r["task"], r["ran"], r["expected"]) for r in F["smokes"]] == [
        ("model", "L.smoke", False, "Paris"),
        ("runner", "K.1", False, "Paris"),
    ]
    assert not F["ladder"]["ran"] and F["ladder"]["rung"] == "last"
    assert (F["contract"]["total"], F["contract"]["passed"], F["contract"]["not_run"]) == (2, 0, 2)

    monkeypatch.setenv(M.TASK_ENV, "L.smoke")
    record_answer("smoke", "model", s, " Paris", [12366], True, 41.3, prompt_len=16)
    monkeypatch.setenv(M.TASK_ENV, "K.1")
    record_answer(
        "runner_smoke", "runner", s, "Paris.", [12366, 13], True, 312.0, prompt_len=135, boundary=128, records=80
    )
    for i in range(3):
        M.record(f"pcc_layer_L{i:02d}", 0.99 - 0.01 * i, task="L.last")
    for k, v in dict(rung_seq=512, rung_chunk=128, rung_start=384, top1_match=0.95, top5_overlap=1.0).items():
        M.record(k, v, task="L.last")
    for tid in ("L.last", "L.smoke", "K.1"):
        led.update(tid, status="PASS", last_run="2026-10-01T10:00:00")
    assert (res / "L.smoke_smoke.json").exists() and (res / "K.1_runner_smoke.json").exists()

    F = build(s)["final"]
    m, r = F["smokes"]
    assert (m["ran"], m["ok"], m["answer"], m["seconds"], m["task"]) == (True, True, " Paris", 41.3, "L.smoke")
    assert (r["ran"], r["ok"], r["boundary"], r["records"], r["prompt_len"], r["task"]) == (
        True,
        True,
        128,
        80,
        135,
        "K.1",
    )
    L = F["ladder"]
    assert (L["ran"], L["task"], L["min_pcc"], L["top1"], L["top5"]) == (True, "L.last", 0.97, 0.95, 1.0)
    C = F["contract"]
    assert (C["passed"], C["not_run"]) == (1, 1) and C["tests"][1]["task"] == "K.1"
    return s


def test_final_tests_data(final_records):
    assert [r["ok"] for r in build(final_records)["final"]["smokes"]] == [True, True]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_final_tests_section_renders(final_records, tmp_path):
    s = final_records
    out = tmp_path / "index.html"
    main(["--spec", str(s.path), "--out", str(out)])
    r = subprocess.run(["node", str(SHIM), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    els = json.loads(r.stdout)
    assert els["#final tbody"]["html"] > 0 and els["#final-lines"]["html"] > 0
    html = out.read_text()
    assert html.index('id="final-sec"') > html.index('id="s-pcc"')  # the last section

    # teletext: page 109, the last page; its screens as text (the page's own debugging aid)
    page = main(["--spec", str(s.path), "--style", "teletext", "--out", str(tmp_path)])
    src = page.read_text()
    i = src.rindex("</script>")
    probe = 'process.stderr.write("@@" + pFinal().map(toText).join("\\n") + "@@");'
    (tmp_path / "probe.html").write_text(src[:i] + probe + src[i:])
    r = subprocess.run(["node", str(TELETEXT_SHIM), str(tmp_path / "probe.html"), "1"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = r.stderr.split("@@")[1]
    assert "FINAL TESTS" in text and "PAGE ERROR" not in text
    assert "MODEL SMOKE" in text and "RUNNER SMOKE" in text and "bound 128 rec 80" in text
    assert "NOT RUN" in text and "kv_write" in text  # the attention contract test never ran
