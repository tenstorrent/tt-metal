# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F82: re-verifying passed tasks runs their gates without touching state.json or results/."""

import json

from models.demos.common.bringup.core import verify as V
from models.demos.common.bringup.selftest.test_device_side import gspec  # noqa: F401 (fixture)
from models.demos.common.bringup.selftest.test_orchestrator import CHECK, impl_task


def test_verify_reruns_passed_gates_without_touching_the_run(sandbox, tmp_path):
    (sandbox.repo / "tests").mkdir()
    (sandbox.repo / "tests/check.py").write_text(CHECK)
    (sandbox.repo / "src").mkdir()
    (sandbox.repo / "src/impl.txt").write_text("0.999")
    sandbox.tasks(impl_task(), impl_task(id="C.2", deps=["C.1"]), impl_task(id="S.1", deps=["C.1"]))
    led = sandbox.spec.bringup_dir
    st = {"C.1": {"status": "PASS"}, "C.2": {"status": "PASS"}, "S.1": {"status": "TODO"}}
    (led / "state.json").write_text(json.dumps(st))
    before = (led / "state.json").read_text()
    res = V.verify(sandbox.spec, ["C.*", "S.*"], out=tmp_path / "v")
    assert [r["task"] for r in res["tasks"]] == ["C.1", "C.2"] and not res["failed"]  # only the passed ones
    assert (tmp_path / "v/results/C.1.json").exists() and not (led / "results/C.1.json").exists()
    (sandbox.repo / "src/impl.txt").write_text("0.5")  # a limit the code no longer meets
    res = V.verify(sandbox.spec, ["C.1"], out=tmp_path / "v2")
    assert res["failed"] == ["C.1"] and any("pcc_out" in line for line in res["tasks"][0]["lines"])
    assert (led / "state.json").read_text() == before
    assert json.loads((tmp_path / "v2/summary.json").read_text())["failed"] == ["C.1"]


def test_a_batched_verify_judges_every_task_as_its_own_gate(gspec, monkeypatch, tmp_path):  # noqa: F811
    """F142: the C.* / S.* frozen tests of one block type in one pytest process give each task the verdict and the
    metrics its own gate run gives; other tasks run alone, in the --tasks order."""
    import sys

    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.testing.templates import render_component_test, render_swap_test

    s = gspec()
    monkeypatch.setenv("BRINGUP_IMPL", "reference")
    rel = lambda p: str(p.relative_to(s.repo))  # noqa: E731
    py = f"{sys.executable} -m pytest -q -p no:cacheprovider"
    tasks = []
    for step in ("attn_norm", "mlp"):
        f = rel(render_component_test(s, "blk", step))
        need = {f"err_{step}_L00": "<= limit"} | ({"never_recorded": "== 1"} if step == "mlp" else {})
        tasks.append(
            {
                "id": f"C.blk.{step}",
                "title": step,
                "step": "implement",
                "tests": [f],
                "gate": {"cmd": f"{py} {f}", "metrics": need},
                "brief": {"block_type": "blk", "step": step, "layer": 0},
            }
        )
    f = rel(render_swap_test(s, "blk", ["attn_norm", "attention"]))
    tasks.append(
        {
            "id": "S.blk.02",
            "title": "swap",
            "step": "implement",
            "tests": [f],
            "gate": {"cmd": f"{py} {f}", "metrics": {"err_swap_out": "<= limit"}},
            "brief": {"block_type": "blk"},
        }
    )
    tasks.append({"id": "X.1", "title": "other", "gate": {"cmd": "true", "metrics": {}}})
    led = Ledger(s.bringup_dir)
    led.write_tasks({"tasks": tasks})
    (s.bringup_dir / "state.json").write_text(json.dumps({t["id"]: {"status": "PASS"} for t in tasks}))
    pats = ["X.1", "C.*", "S.*"]
    one = V.verify(s, pats, out=tmp_path / "one")
    import shutil

    from models.demos.common.bringup.core.gate import run_name

    shutil.rmtree(s.run_dir(run_name(led)) / "stamps")  # F170: only the batched run's records below
    many = V.verify(s, pats, out=tmp_path / "many", batch=True, force=True)  # F150: not the kept runs of `one`
    again = V.verify(s, pats, out=tmp_path / "again", batch=True)  # F170 (audit item 11): the batch's runs are kept
    assert {r["task"] for r in again["tasks"] if r.get("reused")} == {"X.1", "C.blk.attn_norm", "S.blk.02"}
    assert [r["task"] for r in one["tasks"]] == ["X.1", "C.blk.attn_norm", "C.blk.mlp", "S.blk.02"]
    assert [(r["task"], r["verdict"]) for r in many["tasks"]] == [(r["task"], r["verdict"]) for r in one["tasks"]]
    assert one["failed"] == many["failed"] == ["C.blk.mlp"] and {r.get("batch") for r in many["tasks"]} == {None, 0}
    assert "3 passed" in (tmp_path / "many" / "batch_0.log").read_text()  # one process ran the three test files
    for t in ("C.blk.attn_norm", "C.blk.mlp", "S.blk.02"):
        a, b = (json.loads((tmp_path / d / "results" / f"{t}.json").read_text())["metrics"] for d in ("one", "many"))
        assert {k: v["value"] for k, v in a.items()} == {k: v["value"] for k, v in b.items()}, t
