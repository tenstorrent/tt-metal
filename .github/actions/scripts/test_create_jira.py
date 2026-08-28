#!/usr/bin/env python3
"""Tests for the RTL sim check-detail parser and the relevance matcher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from create_jira import format_test, match_entry, parse_failed  # noqa: E402

MAP_PATH = SCRIPTS_DIR / "ai_ip_tests.json"


@pytest.fixture(scope="module")
def relevance_map():
    return json.loads(MAP_PATH.read_text())


def test_parses_gtest_line():
    detail = "- `[1x3] unit_tests_api --gtest_filter=Foo.Bar`"
    assert parse_failed(detail) == [("1x3", "unit_tests_api", "Foo.Bar", "gtest")]


def test_parses_pytest_node_id():
    detail = "- `[2x3_DISPATCH] tests/tt_metal/tools/profiler/test_device_profiler.py::test_full_buffer`"
    assert parse_failed(detail) == [
        (
            "2x3_DISPATCH",
            "tests/tt_metal/tools/profiler/test_device_profiler.py",
            "test_full_buffer",
            "pytest",
        )
    ]


def test_parses_whole_file_pytest():
    detail = "- `[2x3] models/demos/x/test_add.py`"
    assert parse_failed(detail) == [("2x3", "models/demos/x/test_add.py", "", "pytest")]


def test_pytest_row_rendered_with_gtest_separator():
    """The sim reporter hardcodes --gtest_filter= for every runner."""
    detail = "- `[2x3] models/demos/x/test_add.py --gtest_filter=test_foo`"
    assert parse_failed(detail) == [("2x3", "models/demos/x/test_add.py", "test_foo", "pytest")]


def test_ignores_prose_and_dedups():
    detail = (
        "RTL sim: 2 test(s) failed:\n"
        "- `[1x3] unit_tests_api --gtest_filter=Foo.Bar`\n"
        "- `[1x3] unit_tests_api --gtest_filter=Foo.Bar`\n"
        "- … and 3 more (truncated)\n"
        "No RTL sim test failures were recorded.\n"
    )
    assert len(parse_failed(detail)) == 1


def test_omitted_field_is_a_wildcard():
    mapping = {"relevant_tests": [{"group": "unit_tests_api", "requirement": "R"}]}
    assert match_entry("1x3", "unit_tests_api", "Anything", "gtest", mapping)["requirement"] == "R"
    assert match_entry("1x3", "unit_tests_legacy", "Anything", "gtest", mapping) is None


def test_runner_is_matched():
    mapping = {"relevant_tests": [{"runner": "pytest", "requirement": "R"}]}
    assert match_entry("1x3", "a.py", "t", "pytest", mapping) is not None
    assert match_entry("1x3", "unit_tests_api", "Foo.Bar", "gtest", mapping) is None


def test_back2back_batch_matches_any_component():
    """select_quasar_tests.py merges back2back entries into one ':'-joined filter."""
    mapping = {"relevant_tests": [{"group": "unit_tests_legacy", "filter": "*DmLoopback*"}]}
    batch = "*SingleDmL1Write*:*DmLoopback*:*QuasarComputeKernelSingleThread*"
    assert match_entry("2x3_DISPATCH", "unit_tests_legacy", batch, "gtest", mapping) is not None
    other = "*SingleDmL1Write*:*QuasarComputeKernelSingleThread*"
    assert match_entry("2x3_DISPATCH", "unit_tests_legacy", other, "gtest", mapping) is None


def test_format_test_round_trips_into_the_parser():
    for row in [
        ("1x3", "unit_tests_api", "Foo.Bar", "gtest"),
        ("2x3", "models/demos/x/test_add.py", "test_foo", "pytest"),
        ("2x3", "models/demos/x/test_add.py", "", "pytest"),
    ]:
        rendered = format_test(*row)
        if row[3] == "pytest" and not row[2]:
            # "(whole file)" is a human label, not a parseable suffix
            assert parse_failed(rendered)[0] == row
        else:
            assert parse_failed(rendered) == [row]


def test_shipped_map_is_valid_and_ordered(relevance_map):
    """The config-only 2x3_DISPATCH wildcard must not shadow specific entries."""
    entries = relevance_map["relevant_tests"]
    wildcard_idx = [
        i for i, e in enumerate(entries) if e.get("config") == "2x3_DISPATCH" and "group" not in e and "filter" not in e
    ]
    assert len(wildcard_idx) == 1, "expected exactly one config-only 2x3_DISPATCH entry"
    specific_at_dispatch = [i for i, e in enumerate(entries) if e.get("group") and e.get("config") != "1x3"]
    assert all(i < wildcard_idx[0] for i in specific_at_dispatch)


@pytest.mark.parametrize(
    "row,expected",
    [
        (("1x3", "unit_tests_legacy", "*DmLoopback*", "gtest"), "AIIPSW-2"),
        (("1x3", "unit_tests_api", "*TensixSingleCoreDirectDramReaderDatacopyWriter", "gtest"), "AIIPSW-6"),
        (("2x3", "unit_tests_dispatch", "*QuasarDispatchSInstantiatedAndRunning*", "gtest"), "AIIPSW-6"),
        (("2x3_DISPATCH", "unit_tests_legacy", "*SingleDmL1Write*", "gtest"), "AIIPSW-6"),
        (
            ("2x3_DISPATCH", "tests/tt_metal/tools/profiler/test_device_profiler.py", "test_full_buffer", "pytest"),
            "AIIPSW-13",
        ),
        (
            ("2x3", "models/demos/vision/classification/resnet50/quasar/tests/ops/test_add.py", "", "pytest"),
            "AIIPSW-4",
        ),
    ],
)
def test_shipped_map_attributes_known_tests(relevance_map, row, expected):
    entry = match_entry(*row, relevance_map)
    assert entry is not None, f"no entry matched {format_test(*row)}"
    assert entry.get("requirement") == expected


def test_assignee_env_var_reaches_the_payload(monkeypatch, capsys):
    """The env name is read by code, so a rename must fail here, not silently."""
    import jira_client

    for k, v in {
        "JIRA_BASE_URL": "https://example.invalid",
        "JIRA_USER_EMAIL": "a@b.c",
        "JIRA_API_TOKEN": "t",
        "JIRA_PROJECT_KEY": "RELEASE",
        "JIRA_SUMMARY": "s",
        "JIRA_DRY_RUN": "1",
        "JIRA_ASSIGNEE_ACCOUNT_ID": "acct-123",
    }.items():
        monkeypatch.setenv(k, v)

    jira_client.main()
    assert '"accountId": "acct-123"' in capsys.readouterr().out

    monkeypatch.delenv("JIRA_ASSIGNEE_ACCOUNT_ID")
    jira_client.main()
    assert "assignee" not in capsys.readouterr().out


def _hrefs(line):
    from jira_client import _line_nodes

    return [(n["text"], (n.get("marks") or [{}])[0].get("attrs", {}).get("href")) for n in _line_nodes(line)]


def test_bare_url_becomes_a_link():
    assert _hrefs("Run: https://x.test/a") == [("Run: ", None), ("https://x.test/a", "https://x.test/a")]


def test_labelled_link_keeps_its_label():
    assert _hrefs("Commit: [abc123](https://x.test/c/abc123)") == [
        ("Commit: ", None),
        ("abc123", "https://x.test/c/abc123"),
    ]


def test_trailing_punctuation_is_not_part_of_the_url():
    assert _hrefs("see https://x.test/a.")[-1] == (".", None)
    assert _hrefs("see https://x.test/a.")[1] == ("https://x.test/a", "https://x.test/a")


def test_plain_text_is_left_alone():
    assert _hrefs("no links here") == [("no links here", None)]


def test_adf_never_emits_an_empty_text_node():
    """Jira rejects a text node with an empty string."""
    from jira_client import _adf

    doc = _adf("https://x.test/a\nplain\n[l](https://x.test/b)")
    assert all(n["text"] for p in doc["content"] for n in p["content"])


def test_commit_link_falls_back_to_the_bare_sha_off_github():
    from jira_client import _commit_link

    assert _commit_link("deadbeefcafe1234", repo="") == "deadbeefcafe1234"
    assert _commit_link("deadbeefcafe1234", repo="o/r") == "[deadbeefcafe](https://github.com/o/r/commit/deadbeefcafe1234)"
    assert _commit_link("", repo="o/r") == "unknown"


def test_label_may_contain_brackets():
    """Job names end in a runner tag: "Gemma-4-31B e2e tests [bh_quietbox_2]"."""
    nodes = _hrefs("- [Gemma-4-31B e2e tests [bh_quietbox_2]](https://x.test/job/1)")
    assert nodes == [("- ", None), ("Gemma-4-31B e2e tests [bh_quietbox_2]", "https://x.test/job/1")]


def test_two_labelled_links_on_one_line_do_not_merge():
    assert _hrefs("[a](https://x.test/1) and [b](https://x.test/2)") == [
        ("a", "https://x.test/1"),
        (" and ", None),
        ("b", "https://x.test/2"),
    ]


def test_adf_renders_headings_and_bullets():
    """"### " and "- " give the RELEASE-7 shape without an ADF-aware producer."""
    from jira_client import _adf

    doc = _adf("### Impact\nBad.\n- one\n- two\nplain\n- solo")
    assert [b["type"] for b in doc["content"]] == ["heading", "paragraph", "bulletList", "paragraph", "bulletList"]
    heading, _, first_list = doc["content"][:3]
    assert heading["attrs"]["level"] == 3 and heading["content"][0]["text"] == "Impact"
    assert len(first_list["content"]) == 2


def test_adf_links_render_inside_bullets_and_headings():
    from jira_client import _adf

    doc = _adf("### See https://x.test/h\n- [job](https://x.test/j)")
    heading_marks = doc["content"][0]["content"][-1]["marks"][0]
    assert heading_marks == {"type": "link", "attrs": {"href": "https://x.test/h"}}
    item_para = doc["content"][1]["content"][0]["content"][0]
    assert item_para["content"][0]["text"] == "job"


def test_done_transition_prefers_the_conventional_name():
    from jira_client import _pick_done_transition

    ts = [
        {"id": "1", "name": "Won't Do", "to": {"statusCategory": {"key": "done"}}},
        {"id": "2", "name": "In Progress", "to": {"statusCategory": {"key": "indeterminate"}}},
        {"id": "3", "name": "Done", "to": {"statusCategory": {"key": "done"}}},
    ]
    assert _pick_done_transition(ts)["id"] == "3"
    assert _pick_done_transition(ts[:2])["id"] == "1"  # any done-category beats none
    assert _pick_done_transition(ts[1:2]) is None


def test_close_issues_comments_then_transitions(monkeypatch):
    import jira_client

    calls = []

    def fake_api(base, email, token, method, path, body=None):
        calls.append((method, path))
        if path.startswith("/rest/api/3/search/jql"):
            return {"issues": [{"key": "RELEASE-9"}]}
        if path.endswith("/transitions") and method == "GET":
            return {"transitions": [{"id": "31", "name": "Done", "to": {"statusCategory": {"key": "done"}}}]}
        return {}

    monkeypatch.setattr(jira_client, "_api", fake_api)
    out = jira_client.close_issues("https://j.test", "e", "t", "RELEASE", "package-release-ref:stable", "green again")
    assert out == ["closed RELEASE-9 (Done): https://j.test/browse/RELEASE-9"]
    assert ("POST", "/rest/api/3/issue/RELEASE-9/comment") in calls
    assert ("POST", "/rest/api/3/issue/RELEASE-9/transitions") in calls


def test_close_issues_without_a_done_transition_keeps_the_issue_open(monkeypatch):
    import jira_client

    def fake_api(base, email, token, method, path, body=None):
        if path.startswith("/rest/api/3/search/jql"):
            return {"issues": [{"key": "RELEASE-9"}]}
        if path.endswith("/transitions") and method == "GET":
            return {"transitions": []}
        return {}

    monkeypatch.setattr(jira_client, "_api", fake_api)
    out = jira_client.close_issues("https://j.test", "e", "t", "RELEASE", "lbl", "c")
    assert out == ["commented on RELEASE-9 but found no Done transition; left open"]


def test_close_action_is_selected_by_env(monkeypatch, capsys):
    import jira_client

    def fake_api(base, email, token, method, path, body=None):
        if path.startswith("/rest/api/3/search/jql"):
            return {"issues": []}
        raise AssertionError(f"unexpected call {method} {path}")

    monkeypatch.setattr(jira_client, "_api", fake_api)
    for k, v in {
        "JIRA_ACTION": "close",
        "JIRA_BASE_URL": "https://j.test",
        "JIRA_USER_EMAIL": "e",
        "JIRA_API_TOKEN": "t",
        "JIRA_PROJECT_KEY": "RELEASE",
        "JIRA_DEDUP_LABEL": "package-release-ref:stable",
    }.items():
        monkeypatch.setenv(k, v)
    jira_client.main()
    assert "nothing to close" in capsys.readouterr().out
