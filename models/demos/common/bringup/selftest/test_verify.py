# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F82: re-verifying passed tasks runs their gates without touching state.json or results/."""

import json

from models.demos.common.bringup.core import verify as V
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
