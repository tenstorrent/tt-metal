# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The derived-op test pass (task O.1): captured ttnn.bringup calls vs the forks' test cases."""

import json
import subprocess

import pytest

from models.demos.common.bringup.testing import fork_cases


def test_every_fork_op_maps_to_its_folder():
    m = fork_cases.op_to_fork()
    assert m["ttnn.bringup.unified_routed_expert_moe"] == "unified_routed_expert_ffn"
    assert m["ttnn.bringup.dispatch"] == "dispatch" and m["ttnn.bringup.offset_cumsum"] == "offset_cumsum"
    assert m["ttnn.bringup.rms_norm"] == "rms_norm_ttnn"  # its C++ binding (rms_norm_ttnn_nanobind.cpp)


def test_uncovered_calls_are_listed_per_fork(tmp_path, monkeypatch):
    cap = tmp_path / "fork_calls.json"
    calls = [
        {"sig": "aaa", "count": 2, "op": "ttnn.bringup.dispatch", "args": [], "kwargs": {}},
        {"sig": "bbb", "count": 1, "op": "ttnn.bringup.combine", "args": [], "kwargs": {}},
    ]
    cap.write_text(json.dumps({"calls": calls}))
    have = {"dispatch": [{"model": "m", "sig": "aaa"}, {"model": "other", "sig": "bbb"}], "combine": []}
    monkeypatch.setattr(fork_cases, "cases", lambda fork: have.get(fork, []))
    res, missing = fork_cases.check(cap, "m")
    assert res["forks"] == ["combine", "dispatch"]
    assert res["stats"] == {"forks_used": 2, "fork_calls": 2, "fork_calls_uncovered": 1}
    assert [c["sig"] for c in missing["combine"]] == ["bbb"] and "dispatch" not in missing


def test_a_call_to_an_unknown_op_stops(tmp_path):
    cap = tmp_path / "fork_calls.json"
    cap.write_text(
        json.dumps({"calls": [{"sig": "x", "count": 1, "op": "ttnn.bringup.nope", "args": [], "kwargs": {}}]})
    )
    with pytest.raises(SystemExit, match="no fork binds"):
        fork_cases.check(cap, "m")


def test_gate_commit_stages_the_task_extras_and_the_fork_calls(fx):
    from models.demos.common.bringup.core.gate import stage_paths
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    s = Spec.load(fx())
    led = Ledger(s.bringup_dir)
    led.results_dir.mkdir(parents=True, exist_ok=True)
    for f in ("O.1.json", "O.1_extra.json", "fork_calls.json", "X.3_profile.json"):
        (led.results_dir / f).write_text("{}")
    names = lambda task: {p.rsplit("/", 1)[-1] for p in stage_paths(s, led, task)}  # noqa: E731
    got = names({"id": "O.1", "step": "optests"})
    assert {"O.1.json", "O.1_extra.json", "fork_calls.json"} <= got and "X.3_profile.json" not in got
    got = names({"id": "X.3", "step": "perf"})
    assert "X.3_profile.json" in got and "fork_calls.json" not in got


def test_source_swap_points_the_original_name_at_the_fork(monkeypatch):
    """fork_source's plugin swaps attributes for the session and puts them back."""
    import sys
    import types

    from models.demos.common.bringup.testing import fork_source as S

    orig_mod = types.ModuleType("fakepkg_orig")
    orig_mod.op = "original"
    fork_mod = types.ModuleType("fakepkg_fork")
    fork_mod.op = "fork"
    monkeypatch.setitem(sys.modules, "fakepkg_orig", orig_mod)
    monkeypatch.setitem(sys.modules, "fakepkg_fork", fork_mod)
    monkeypatch.setenv(S.ENV, json.dumps({"fakepkg_orig.op": "fakepkg_fork.op"}))
    S.pytest_configure(None)
    assert orig_mod.op == "fork"
    S.pytest_unconfigure(None)
    assert orig_mod.op == "original"


def test_source_swap_converts_enum_arguments_at_the_fork_op(monkeypatch):
    """A swap entry with convert_enums wraps the fork op: original-enum arguments become the fork enum's same-named
    members; the original enum itself stays in place for the other original ops."""
    import enum
    import sys
    import types

    from models.demos.common.bringup.testing import fork_source as S

    class Orig(enum.Enum):
        A = 0
        B = 1

    class Fork(enum.Enum):
        A = 0
        B = 1
        C = 2

    orig_mod = types.ModuleType("fakepkg_eorig")
    orig_mod.op, orig_mod.Act = (lambda *a, **k: "original"), Orig
    fork_mod = types.ModuleType("fakepkg_efork")
    fork_mod.Act = Fork

    def fork_op(x, activation=Fork.A, other=None):
        assert isinstance(activation, Fork), activation
        return (x, activation, other)

    fork_mod.op = fork_op
    monkeypatch.setitem(sys.modules, "fakepkg_eorig", orig_mod)
    monkeypatch.setitem(sys.modules, "fakepkg_efork", fork_mod)
    spec = {"op": "fakepkg_efork.op", "convert_enums": {"fakepkg_eorig.Act": "fakepkg_efork.Act"}}
    monkeypatch.setenv(S.ENV, json.dumps({"fakepkg_eorig.op": spec}))
    S.pytest_configure(None)
    try:
        assert orig_mod.Act is Orig
        assert orig_mod.op(1, activation=Orig.B, other="z") == (1, Fork.B, "z")
        assert orig_mod.op(Orig.B) == (Fork.B, Fork.A, None)  # positional too
    finally:
        S.pytest_unconfigure(None)
    assert orig_mod.op() == "original"


def test_source_outcomes_from_junit(tmp_path):
    from models.demos.common.bringup.testing import fork_source as S

    x = tmp_path / "r.xml"
    x.write_text(
        "<testsuites><testsuite>"
        '<testcase classname="m" name="a"/>'
        '<testcase classname="m" name="b"><failure message="x"/></testcase>'
        '<testcase classname="m" name="c"><skipped/></testcase>'
        "</testsuite></testsuites>"
    )
    assert S._outcomes(x) == {"m::a": "passed", "m::b": "failed", "m::c": "skipped"}


def test_gate_commit_stages_changed_forks_and_knowledge_only(fx):
    """F48: a fork the agent made (or changed) and its knowledge-file entries are committed with the gate; other
    forks' files are not staged (the formatting pass must not touch them)."""
    import subprocess

    from models.demos.common.bringup.core.gate import stage_paths
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    s = Spec.load(fx())
    repo = s.repo
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    kn = repo / "models/demos/common/bringup/knowledge"
    kn.mkdir(parents=True)
    (kn / "known_issues.md").write_text("# issues\n")
    (kn / "repo_map.md").write_text("# map\n")
    old = repo / "ttnn/ttnn/bringup/old_fork"
    old.mkdir(parents=True)
    (old / "op.cpp").write_text("int a;\n")
    (repo / "ttnn/ttnn/bringup/INDEX.md").write_text("| Fork |\n")
    git("add", "-A")
    git("-c", "user.email=x@y", "-c", "user.name=x", "commit", "-qm", "base")
    led = Ledger(s.bringup_dir)
    task = {"id": "C.1", "step": "implement", "paths": []}
    assert not [p for p in stage_paths(s, led, task) if p.startswith(("ttnn/", "models/demos/common/"))]
    new = repo / "ttnn/ttnn/bringup/new_fork"
    (new / "__pycache__").mkdir(parents=True)
    (new / "op.cpp").write_text("int b;\n")
    (new / "__pycache__" / "x.pyc").write_text("")
    (repo / "ttnn/ttnn/bringup/INDEX.md").write_text("| Fork |\n| new_fork |\n")
    (kn / "known_issues.md").write_text("# issues\n- new entry\n")
    got = [p for p in stage_paths(s, led, task) if p.startswith(("ttnn/", "models/demos/common/"))]
    assert got == [
        "models/demos/common/bringup/knowledge/known_issues.md",
        "ttnn/ttnn/bringup/INDEX.md",
        "ttnn/ttnn/bringup/new_fork/op.cpp",
    ]


def test_gate_commit_carries_new_fork_files_and_knowledge(fx):
    """glm53 F49 (now F51), adapted to F48: in a repo with no commit yet, a fork the agent created (new files) and the
    knowledge files are staged file by file and end up in the gate commit."""
    from models.demos.common.bringup.core.gate import git_commit, stage_paths
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    s = Spec.load(fx())
    led = Ledger(s.bringup_dir)
    led.results_dir.mkdir(parents=True, exist_ok=True)
    (led.results_dir / "C.x.json").write_text("{}")
    repo = s.repo
    for args in (["init", "-q"], ["config", "user.email", "t@example.com"], ["config", "user.name", "t"]):
        subprocess.run(["git", *args], cwd=repo, check=True)
    fork = repo / "ttnn/ttnn/bringup/sdpa"
    (fork / "tests/unit").mkdir(parents=True)
    (fork / "CHANGELOG.md").write_text("- option\n")
    (fork / "tests/unit/test_new.py").write_text("def test_x():\n    pass\n")
    know = repo / "models/demos/common/bringup/knowledge"
    know.mkdir(parents=True)
    (know / "known_issues.md").write_text("# Known issues\n")
    (know / "repo_map.md").write_text("# Repo map\n")
    got = stage_paths(s, led, {"id": "C.x", "step": "implement"})
    assert "ttnn/ttnn/bringup" not in got  # F48: changed files, never the whole directory
    assert {
        "ttnn/ttnn/bringup/sdpa/CHANGELOG.md",
        "ttnn/ttnn/bringup/sdpa/tests/unit/test_new.py",
        "models/demos/common/bringup/knowledge/known_issues.md",
        "models/demos/common/bringup/knowledge/repo_map.md",
    } <= set(got)
    assert git_commit(s, got, "gate", "")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).split()
    assert "ttnn/ttnn/bringup/sdpa/tests/unit/test_new.py" in tracked
    assert "ttnn/ttnn/bringup/sdpa/CHANGELOG.md" in tracked
    assert "models/demos/common/bringup/knowledge/known_issues.md" in tracked


def test_gate_commit_skips_shared_paths_that_do_not_exist(fx):
    """glm53 F49 (now F51): with no fork or knowledge file present, nothing shared is staged."""
    from models.demos.common.bringup.core.gate import stage_paths
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec

    s = Spec.load(fx())
    subprocess.run(["git", "init", "-q"], cwd=s.repo, check=True)
    led = Ledger(s.bringup_dir)
    got = stage_paths(s, led, {"id": "C.x", "step": "implement"})
    assert not [p for p in got if p.startswith(("ttnn/", "models/demos/common/"))]


def test_gate_outputs_are_allowed_for_the_agent(fx):
    """F55: files the task's gate writes (e.g. O.1's results/fork_calls.json) never count as an agent path violation."""
    from models.demos.common.bringup.core.gate import gate_outputs
    from models.demos.common.bringup.core.ledger import Ledger
    from models.demos.common.bringup.core.spec import Spec
    from models.demos.common.bringup.orchestrator import Orchestrator, allowed

    s = Spec.load(fx())
    o = Orchestrator(s)
    task = {"id": "O.1", "step": "optests", "role": "optests", "paths": []}
    pats = o.allowed_paths(task, "optests")
    b = str(Ledger(s.bringup_dir).results_dir.relative_to(s.repo))
    assert allowed(f"{b}/fork_calls.json", pats) and allowed(f"{b}/O.1.json", pats)
    assert [p.name for p in gate_outputs(Ledger(s.bringup_dir), task)][-1] == "fork_calls.json"
