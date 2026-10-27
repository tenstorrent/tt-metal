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
    """F48: a fork the agent made (or changed) is committed with the gate; other forks' files are not staged (the
    formatting pass must not touch them). The shared knowledge files are not agent-writable, so an edit there is never
    swept into a task's commit; the model's own known_issues.md is."""
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
    s.bringup_dir.mkdir(parents=True, exist_ok=True)
    (s.bringup_dir / "known_issues.md").write_text("# Known issues: m\n- new entry\n")
    staged = stage_paths(s, led, task)
    got = [p for p in staged if p.startswith(("ttnn/", "models/demos/common/"))]
    assert got == ["ttnn/ttnn/bringup/INDEX.md", "ttnn/ttnn/bringup/new_fork/op.cpp"]
    assert str((s.bringup_dir / "known_issues.md").relative_to(repo)) in staged


def test_gate_commit_carries_new_fork_files_and_knowledge(fx):
    """glm53 F49 (now F51), adapted to F48: in a repo with no commit yet, a fork the agent created (new files) and the
    model's own knowledge files are staged file by file and end up in the gate commit."""
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
    (s.bringup_dir / "known_issues.md").write_text("# Known issues: m\n")
    (s.bringup_dir / "repo_map.md").write_text("# Repo map: m\n")
    ki, rm = (str((s.bringup_dir / f).relative_to(repo)) for f in ("known_issues.md", "repo_map.md"))
    got = stage_paths(s, led, {"id": "C.x", "step": "implement"})
    assert "ttnn/ttnn/bringup" not in got  # F48: changed files, never the whole directory
    assert {"ttnn/ttnn/bringup/sdpa/CHANGELOG.md", "ttnn/ttnn/bringup/sdpa/tests/unit/test_new.py", ki, rm} <= set(got)
    assert git_commit(s, got, "gate", "")
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True).split()
    assert "ttnn/ttnn/bringup/sdpa/tests/unit/test_new.py" in tracked
    assert "ttnn/ttnn/bringup/sdpa/CHANGELOG.md" in tracked
    assert ki in tracked


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


def test_a_math_case_is_limited_by_the_ops_noise_floor_not_its_own_error():
    """F139: the limit is calib x the torch reference at the op's real formats vs fp64; a correct device-like result
    (bf16 inputs, fp32 sums, bf16 output) passes, a 1 % scale does not."""
    import torch

    from models.demos.common.bringup.testing import fork_cases as FC

    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(256, 512, generator=g), torch.randn(512, 128, generator=g) / 20
    lim = FC.noise_limit(lambda a, b: a @ b, [x, w], ["bf16", "bf16"])
    want = (x.double() @ w.double()).float()
    dev = FC.round_to(FC.round_to(x, "bf16") @ FC.round_to(w, "bf16"), "bf16")
    rel = lambda a: float((a.double() - want.double()).norm() / want.double().norm())  # noqa: E731
    assert rel(dev) <= lim["rel"] < 0.01 and rel(dev * 1.01) > lim["rel"]
    assert FC.noise_limit(lambda a, b: a @ b, [x, w], ["bf16", "bfp8"])["rel"] > lim["rel"]  # bfp8 weights: wider
    e = [rel(FC.round_to(want, d)) for d in ("bf16", "bfp8", "bfp4")]
    assert e[0] <= e[1] < e[2], e


def test_the_case_lint_refuses_a_numeric_tolerance_without_a_noise_floor():
    from models.demos.common.bringup.testing.fork_cases import case_lint

    assert case_lint({"id": "a", "pcc": 0.9999, "rtol": 0.02})
    assert case_lint({"id": "b", "max_rel": {"y": 0.01}})
    assert not case_lint({"id": "c", "exact": True})
    assert not case_lint({"id": "d", "noise": {"inputs": ["bf16", "bfp8"], "out": "bf16"}})


@pytest.mark.xfail(strict=True, reason="F140: the forks' existing cases still carry tolerances read off the device")
def test_no_fork_case_carries_a_tolerance_of_its_own():
    import glob
    import runpy

    from models.demos.common.bringup.testing.fork_cases import REPO, case_lint

    bad = [
        e
        for f in sorted(glob.glob(str(REPO / "ttnn/ttnn/bringup/*/tests/cases.py")))
        for c in runpy.run_path(f).get("CASES", [])
        for e in case_lint(c)
    ]
    assert not bad, f"{len(bad)} cases, e.g. {bad[:3]}"


def test_a_volatile_cache_key_does_not_change_a_calls_signature():
    """F179: DeepSeek-V4.1 O.1: a flat_routed_expert call carried extra_key (a program-cache key from buffer addresses)
    that changed between two runs, so its signature did too and no case could cover it. The signature leaves volatile
    arguments out; the pre-F179 signature is kept as sig_full so existing cases still match."""
    from models.demos.common.bringup.testing import fork_capture as FC

    FC._CALLS.clear()
    FC.record("ttnn.bringup.flat_routed_expert", [1, 2], {"extra_key": 9507296, "k": 3})
    FC.record("ttnn.bringup.flat_routed_expert", [1, 2], {"extra_key": 9582688, "k": 3})
    assert len(FC._CALLS) == 1 and next(iter(FC._CALLS.values()))["count"] == 2
    FC.record("ttnn.bringup.flat_routed_expert", [1, 2], {"extra_key": 1, "k": 4})
    assert len(FC._CALLS) == 2  # a real difference still separates calls
    c = next(iter(FC._CALLS.values()))
    assert c["sig_full"] != c["sig"]
    FC._CALLS.clear()


def _fake_runner(outcomes: dict[str, str], rc: int = 0):
    """A run_safe_pytest stand-in: writes the junit report pytest would, {node id: passed|failed|error|skipped|absent}."""

    def run(cmd, env):
        xml = next(c.split("=", 1)[1] for c in cmd if c.startswith("--junitxml="))
        cases = []
        for n in [c for c in cmd if "::" in c]:
            kind = outcomes.get(n, "passed")
            if kind == "absent":  # (the group died before reporting it)
                continue
            f, *rest = n.split("::")
            cls = ".".join(["ttnn.ttnn.bringup.x.tests." + f.rsplit("/", 1)[-1][:-3], *rest[:-1]])
            body = {"failed": "<failure message='boom'/>", "error": "<error message='setup'/>",
                    "skipped": "<skipped message='needs 8 cards'/>"}.get(kind, "")  # fmt: skip
            cases.append(f'<testcase classname="{cls}" name="{rest[-1]}">{body}</testcase>')
        open(xml, "w").write(f"<testsuites><testsuite>{''.join(cases)}</testsuite></testsuites>")
        return rc

    return run


def test_every_model_case_is_counted_and_a_fork_without_cases_does_not_stop_the_others(monkeypatch, tmp_path):
    """F198: O.1 used to return before running anything when one fork had no case file, and folded skips into passes."""
    from models.demos.common.bringup.testing import fork_tests as FT

    nodes = {
        "a": ["p/test_a.py::test_x[c1]", "p/test_a.py::test_x[c2]", "p/test_a.py::T::test_y"],
        "b": ["p/test_b.py::test_z[c1]", "p/test_b.py::test_z[c2]"],
    }
    grouper = lambda f: ({(2, 2): nodes[f]}, ["ERROR p/test_bad.py - ImportError"] if f == "b" else [])  # noqa: E731
    out = {
        "p/test_a.py::test_x[c2]": "skipped",
        "p/test_a.py::T::test_y": "failed",
        "p/test_b.py::test_z[c2]": "absent",
    }
    real = fork_cases.FORKS
    fake = tmp_path / "forks"
    for f in ("a", "b"):
        (fake / f / "tests").mkdir(parents=True)
        (fake / f / "tests" / "test_x.py").write_text("")
    (fake / "c" / "tests").mkdir(parents=True)  # a fork the model calls, with no case file
    monkeypatch.setattr(fork_cases, "FORKS", fake)
    monkeypatch.setattr("models.demos.common.bringup.testing.cards.visible_devices", lambda mesh: "0,1,2,3")
    t = fork_cases.run_tests(["a", "b", "c"], runner=_fake_runner(out), grouper=grouper)
    monkeypatch.setattr(fork_cases, "FORKS", real)
    assert (t["a"]["passed"], t["a"]["failed"], t["a"]["skipped"]) == (1, 1, 1)
    assert "skipped: p/test_a.py::test_x[c2]: needs 8 cards" in t["a"]["problems"]
    assert (t["b"]["passed"], t["b"]["not_run"], t["b"]["collect_errors"]) == (1, 1, 1)
    assert t["c"]["collect_errors"] == 1 and "no tests/test_*.py" in t["c"]["problems"][0]
    st = fork_cases.tally_stats(t)
    assert st == {"fork_tests_passed": 2, "fork_tests_failed": 4, "fork_tests_skipped": 1, "fork_tests_not_run": 3}
    assert FT.bad(FT.empty_tally()) == 0


def test_a_test_file_that_does_not_import_is_a_collection_error(tmp_path, monkeypatch):
    from models.demos.common.bringup.testing import fork_tests as FT

    good, broken = tmp_path / "test_good.py", tmp_path / "test_broken.py"
    good.write_text("def test_ok():\n    pass\n")
    broken.write_text("import no_such_module_anywhere\n\ndef test_never():\n    pass\n")
    monkeypatch.setattr(FT, "REPO", tmp_path)
    nodes, errors = FT.collect([str(good), str(broken)])
    assert [n.rsplit("::", 1)[-1] for n in nodes] == ["test_ok"]
    assert errors and "test_broken.py" in errors[0]
    assert FT.collect([str(tmp_path / "test_empty.py")])[1]  # nothing collected is an error too


def test_junit_names_match_node_ids_with_and_without_classes():
    from models.demos.common.bringup.testing.fork_tests import same_test

    assert same_test("a/b/test_x.py::test_y[p-1]", "a.b.test_x::test_y[p-1]")
    assert same_test("a/b/test_x.py::C::test_y", "a.b.test_x.C::test_y")
    assert not same_test("a/b/test_x.py::test_y", "a.b.test_z::test_y")
    assert not same_test("a/b/test_x.py::test_y[1]", "a.b.test_x::test_y[10]")


def test_a_source_test_failing_at_baseline_too_is_still_a_failure(capsys):
    """F198: the source check used to fail only on regressions, so tests failing on the original and the fork (or a
    selection that never produced a result) were accepted for good."""
    from models.demos.common.bringup.testing.fork_source import report

    base = {
        "a": {"original": "passed", "fork": "passed"},
        "b": {"original": "failed", "fork": "failed"},
        "c": {"original": "passed", "fork": "passed"},
        "d": {"original": "skipped", "fork": "skipped"},
        "x::<no results>": {"original": "failed", "fork": "failed"},
    }
    fork = {"a": "passed", "b": "failed", "c": "failed", "d": "skipped", "x::<no results>": "failed", "e": "failed"}
    assert report(base, fork, "f") == 4
    out = capsys.readouterr().out
    assert "FAIL b: failed, failing at baseline too (original failed)" in out
    assert "FAIL c: failed, regression (passed at baseline)" in out and "FAIL e: failed, new test" in out
    assert "skipped d" in out and "passed 1, failing 4, skipped 1" in out
    assert report({"a": base["a"]}, {"a": "passed"}, "f") == 0


def test_a_group_lost_to_a_box_error_runs_once_more(monkeypatch):
    """F198: one dead Ethernet core left a device open and 175 all-gather tests errored at setup, reported as results."""
    from models.demos.common.bringup.testing import fork_tests as FT

    nodes = ["p/test_a.py::test_x[1]", "p/test_a.py::test_x[2]"]
    calls = []
    good = _fake_runner({})

    def runner(cmd, env):
        calls.append(cmd)
        if len(calls) == 1:
            return _fake_runner({n: "error" for n in nodes}, rc=1)(cmd, env)
        return good(cmd, env)

    monkeypatch.setattr("models.demos.common.bringup.testing.cards.visible_devices", lambda mesh: None)
    t = FT.run(["a"], runner=runner, grouper=lambda f: ({None: nodes}, []))["a"]
    assert len(calls) == 2 and t["passed"] == 2 and t["error"] == 0
    assert any("box error" in p for p in t["problems"])
    assert not FT.box_error(1, {"x": ("failed", "")}) and FT.box_error(1, {}) and not FT.box_error(0, {})


def test_the_fork_suite_counts_every_part_and_only_the_owner_waives(monkeypatch):
    """F199: the closing all-fork run; a failure passes only through an owner waiver, reported as WAIVED."""
    from models.demos.common.bringup.testing import fork_suite as FS
    from models.demos.common.bringup.testing import fork_tests as FT

    def tally(**kw):
        t = FT.empty_tally()
        for k, v in kw.items():
            if k == "problems":
                t["problems"] = list(v)
            else:
                t[k] = v
        return t

    parts = {
        "unit": lambda f: tally(
            passed=5, failed=1, skipped=2, problems=["failed: u.py::t_bad: boom", "skipped: u.py::s"]
        ),
        "source": lambda f: tally(passed=3, collect_errors=1, problems=["collection error: x::<no results>"]),
    }
    res = FS.run_suite(["a"], {"u.py::t_bad": "owner: known upstream bug #1"}, parts=parts)
    assert res["a"]["unit"]["failed"] == 0 and res["a"]["unit"]["waived"] == 1
    assert res["a"]["unit"]["problems"][0].startswith("WAIVED failed: u.py::t_bad")
    assert FS.totals(res) == {
        "fork_suite_failed": 1,
        "fork_suite_passed": 8,
        "fork_suite_skipped": 2,
        "fork_suite_waived": 1,
    }
    res = FS.run_suite(["a"], {}, parts=parts)
    assert FS.totals(res)["fork_suite_failed"] == 2


def test_the_skills_name_the_strict_fork_rule():
    from models.demos.common.bringup.core.spec import FRAMEWORK

    for f in ("skill/rebase-bringup/SKILL.md", "skill/bringup/SKILL.md", "skill/bringup-fork-op/SKILL.md"):
        text = (FRAMEWORK / f).read_text()
        assert "fork_suite" in text or "F.1" in text, f
        assert "must match the baseline" not in text, f
