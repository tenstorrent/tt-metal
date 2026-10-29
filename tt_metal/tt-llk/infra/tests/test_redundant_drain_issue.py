#!/usr/bin/env python3
"""Tests for the issue filer, with every process it starts stubbed."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
INFRA = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, INFRA)
import redundant_drain_issue as rdi  # noqa: E402

HEAD = "a" * 40
PR = {"number": 1234, "title": "t", "url": "https://x/pull/1234", "author": "someone"}
SITES = [
    {
        "file": "ttnn/k.cpp",
        "line": 9,
        "kind": "ADVISORY",
        "drain": "tensix_sync",
        "covered_by": None,
        "reason": "r1",
        "function": "void g()",
    },
    {
        "file": "ttnn/k.cpp",
        "line": 5,
        "kind": "REDUNDANT",
        "drain": "mop_sync",
        "covered_by": "tensix_sync",
        "reason": "r2",
        "function": "void MAIN() {",
    },
]


def stub(monkeypatch, plan):
    """Route try_run through `plan(program, args)`; return the recorded calls."""
    calls = []

    def try_run(program, args):
        calls.append((program, list(args)))
        return plan(program, list(args))

    monkeypatch.setattr(rdi, "try_run", try_run)
    return calls


def creates(calls):
    return [a for p, a in calls if p == "gh" and a[:2] == ["issue", "create"]]


def edits(calls):
    return [a for p, a in calls if p == "gh" and a[:2] == ["issue", "edit"]]


def arg(args, flag):
    return args[args.index(flag) + 1]


def test_one_issue_per_pull_request_lists_every_site(monkeypatch, capsys):
    calls = stub(monkeypatch, lambda p, a: (0, "https://x/issues/9\n", ""))
    rdi.file_issue(SITES, PR, HEAD, existing={}, dry=False)
    c = creates(calls)
    assert len(c) == 1 and not edits(calls)
    body = arg(c[0], "--body")
    assert "<!-- llk-redundant-drain:pr:1234 -->" in body
    assert "`ttnn/k.cpp:5`" in body and "`ttnn/k.cpp:9`" in body
    assert body.index("k.cpp:5") < body.index("k.cpp:9")  # sorted by file and line
    assert arg(c[0], "--assignee") == "someone"
    assert (
        arg(c[0], "--title")
        == "[LLK] Remove or review the drains #1234 added at k.cpp:5 and 1 more"
    )
    assert "a `mop_sync()` that a following `tensix_sync()` already covers" in body
    assert "move the drain to that caller" in body
    assert "created https://x/issues/9" in capsys.readouterr().out


def test_an_open_issue_with_the_marker_is_updated_not_duplicated(monkeypatch):
    calls = stub(monkeypatch, lambda p, a: (0, "", ""))
    existing = {
        7: "old\n<!-- llk-redundant-drain:pr:1234 -->\nold",
        8: "<!-- llk-redundant-drain:pr:99 -->",
    }
    assert rdi.file_issue(SITES, PR, HEAD, existing, dry=False) == 7
    assert not creates(calls)
    e = edits(calls)
    assert len(e) == 1 and e[0][2] == "7"
    assert "`ttnn/k.cpp:5`" in arg(e[0], "--body")


def test_a_failed_assignment_is_retried_without_the_assignee(monkeypatch, capsys):
    def plan(p, a):
        if a[:2] == ["issue", "create"] and "--assignee" in a:
            return (1, "", "could not assign")
        return (0, "https://x/issues/10\n", "")

    calls = stub(monkeypatch, plan)
    assert (
        rdi.file_issue(SITES, PR, HEAD, existing={}, dry=False) == "https://x/issues/10"
    )
    c = creates(calls)
    assert len(c) == 2 and "--assignee" in c[0] and "--assignee" not in c[1]
    assert "@someone" in arg(c[1], "--body")
    assert "could not assign" in capsys.readouterr().err


def test_a_second_creation_failure_fails_the_run(monkeypatch, capsys):
    stub(
        monkeypatch,
        lambda p, a: (1, "", "boom") if a[:2] == ["issue", "create"] else (0, "", ""),
    )
    with pytest.raises(SystemExit) as e:  # allow-pytest.raises: no conftest here
        rdi.file_issue(SITES, PR, HEAD, existing={}, dry=False)
    assert e.value.code == 1
    assert "boom" in capsys.readouterr().err


def test_dry_run_files_nothing(monkeypatch, capsys):
    calls = stub(monkeypatch, lambda p, a: (0, "", ""))
    rdi.file_issue(SITES, None, HEAD, existing={}, dry=True)
    assert not calls
    out = capsys.readouterr().out
    assert (
        "would create: [LLK] Remove or review the drains commit aaaaaaaaaaa added at k.cpp:5 and 1 more"
        in out
    )
    assert "<!-- llk-redundant-drain:commit:aaaaaaaaaaa -->" in out


DIFF = (
    "diff --git a/ttnn/k.cpp b/ttnn/k.cpp\n--- a/ttnn/k.cpp\n+++ b/ttnn/k.cpp\n"
    "@@ -1,0 +5,2 @@\n+    tensix_sync();\n+    TTI_NOP;\n"
    "@@ -20,1 +22,1 @@\n-    tensix_sync();\n+    ckernel::tensix_sync();\n"
    "diff --git a/tt_metal/third_party/x.h b/tt_metal/third_party/x.h\n"
    "--- a/tt_metal/third_party/x.h\n+++ b/tt_metal/third_party/x.h\n"
    "@@ -1,0 +2,1 @@\n+    tensix_sync();\n"
    "diff --git a/tt_metal/tt-llk/tt_llk_quasar/q.h b/tt_metal/tt-llk/tt_llk_quasar/q.h\n"
    "--- a/tt_metal/tt-llk/tt_llk_quasar/q.h\n+++ b/tt_metal/tt-llk/tt_llk_quasar/q.h\n"
    "@@ -1,0 +2,1 @@\n+    tensix_sync();\n"
    "diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
    "@@ -1,0 +2,1 @@\n+tensix_sync()\n"
)


def end_to_end_plan(checked_out=HEAD, blamed=HEAD):
    def plan(p, a):
        if p == "git" and a[:2] == ["rev-parse", "--verify"]:
            return (0, (checked_out if a[-1] == "HEAD^{commit}" else HEAD) + "\n", "")
        if p == "git" and a[:2] == ["diff", "-U0"]:
            assert "-M" in a and "-w" in a
            return (0, DIFF, "")
        if p == "git" and a[0] == "rev-list":
            return (0, HEAD + "\n", "")
        if p == "python":
            site = {
                "file": "ttnn/k.cpp",
                "line": 5,
                "kind": "ADVISORY",
                "drain": "tensix_sync",
                "covered_by": None,
                "reason": "r",
                "function": "void MAIN() {",
            }
            return (0, json.dumps(site) + "\n", "")
        if p == "git" and a[0] == "blame":
            return (0, f"{blamed} 1 5 1\nauthor x\n", "")
        if p == "gh" and a[0] == "api":
            return (
                0,
                json.dumps(
                    [
                        {
                            "number": 1234,
                            "title": "t",
                            "html_url": "https://x/pull/1234",
                            "user": {"login": "someone"},
                        }
                    ]
                ),
                "",
            )
        if p == "gh" and a[:2] == ["label", "list"]:
            return (0, "llk-redundant-drain\n", "")
        if p == "gh" and a[:2] == ["issue", "list"]:
            return (0, "[]", "")
        if p == "gh" and a[:2] == ["issue", "create"]:
            return (0, "https://x/issues/9\n", "")
        return (0, "", "")

    return plan


def test_main_filters_files_passes_the_baseline_and_files_once(monkeypatch):
    calls = stub(monkeypatch, end_to_end_plan())
    monkeypatch.setattr(
        sys, "argv", ["redundant_drain_issue.py", "--base", "b", "--head", "h"]
    )
    rdi.main()
    checker = [a for p, a in calls if p == "python"][0]
    assert (
        "--baseline" in checker
        and checker[checker.index("--baseline") + 1] == rdi.BASELINE
    )
    assert checker[-1:] == [
        "ttnn/k.cpp"
    ]  # vendored, Quasar and non-source files are not checked
    c = creates(calls)
    assert len(c) == 1 and arg(c[0], "--assignee") == "someone"
    assert "`ttnn/k.cpp:5`" in arg(c[0], "--body")


def test_added_lines_are_net_of_removed_and_filtered(monkeypatch):
    stub(monkeypatch, end_to_end_plan())
    added = rdi.added_drain_lines("b", "h")
    assert added == {"ttnn/k.cpp": {5}}  # the respelled drain at line 22 is not added


def test_a_site_blamed_outside_the_range_is_dropped(monkeypatch):
    calls = stub(monkeypatch, end_to_end_plan(blamed="c" * 40))
    monkeypatch.setattr(
        sys, "argv", ["redundant_drain_issue.py", "--base", "b", "--head", "h"]
    )
    rdi.main()
    assert not creates(calls)


def test_a_checker_crash_fails_the_run(monkeypatch):
    def plan(p, a):
        if p == "python":
            return (1, "", "Traceback")
        return end_to_end_plan()(p, a)

    stub(monkeypatch, plan)
    monkeypatch.setattr(
        sys, "argv", ["redundant_drain_issue.py", "--base", "b", "--head", "h"]
    )
    with pytest.raises(SystemExit) as e:  # allow-pytest.raises: no conftest here
        rdi.main()
    assert e.value.code == 1


def test_main_refuses_a_checkout_that_is_not_at_the_head(monkeypatch, capsys):
    stub(monkeypatch, end_to_end_plan(checked_out="b" * 40))
    monkeypatch.setattr(
        sys, "argv", ["redundant_drain_issue.py", "--base", "b", "--head", "h"]
    )
    with pytest.raises(SystemExit) as e:  # allow-pytest.raises: no conftest here
        rdi.main()
    assert e.value.code == 1
    assert "check it out first" in capsys.readouterr().err
