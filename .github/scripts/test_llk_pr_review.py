# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests of the fixed steps of the LLK PR review workflow.

Run with: python3 -m pytest .github/scripts/test_llk_pr_review.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import llk_pr_review as review  # noqa: E402

REPO = "tenstorrent/tt-metal"
HEAD = "a" * 40
DIFF = """\
diff --git a/src/a.h b/src/a.h
index 1111111..2222222 100644
--- a/src/a.h
+++ b/src/a.h
@@ -10,4 +10,5 @@ void f()
 context ten
-removed
+added eleven
+added twelve
 context thirteen
@@ -40,2 +41,3 @@ void g()
 context forty-one
+++ an added line that starts with two more plus signs
 context forty-three
diff --git a/src/gone.h b/src/gone.h
deleted file mode 100644
--- a/src/gone.h
+++ /dev/null
@@ -1,2 +0,0 @@
-one
-two
"""


def _finding(path="src/a.h", line=11, body="This overflows.", **more):
    return {"path": path, "line": line, "body": body, **more}


# ── anchors ───────────────────────────────────────────────────────────────────
def test_the_new_side_lines_of_each_hunk_are_found():
    hunks = review.right_side_lines(DIFF)

    assert hunks == {"src/a.h": [[10, 11, 12, 13], [41, 42, 43]]}


@pytest.mark.parametrize(
    "finding, inline",
    [
        (_finding(line=11), True),
        (_finding(line=12, start_line=11), True),
        (_finding(line=42), True),  # the added line that looks like a header
        (_finding(line=20), False),  # not in the diff
        (_finding(line=41, start_line=13), False),  # spans two hunks
        (_finding(path="src/other.h"), False),  # not a changed file
        (_finding(path="src/gone.h", line=1), False),  # a deleted file has no new side
    ],
)
def test_a_finding_is_inline_only_on_new_side_lines_of_one_hunk(finding, inline):
    finding = {"start_line": None, **finding}
    assert review.anchored(finding, review.right_side_lines(DIFF)) is inline


# ── what may be posted ────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "text",
    [
        "token ghs_" + "A" * 36,
        "ghp_" + "b" * 36,
        "github_pat_" + "C" * 40,
        "key sk-ant-api03-" + "d" * 20,
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwcyJ9",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
    ],
)
def test_output_that_looks_like_a_credential_is_refused_whole(text):
    document = {"findings": [_finding(), _finding(line=12, body=f"See {text} here.")]}

    with pytest.raises(review.ReviewError, match="looks like a credential"):
        review.checked_findings(document, REPO)


@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        {},
        {"findings": "none"},
        {"findings": [{"path": "src/a.h", "line": 11}]},  # no body
        {"findings": [_finding(line=0)]},
        {"findings": [_finding(line="11")]},
        {"findings": [_finding(path="/etc/passwd")]},
        {"findings": [_finding(path="src/../../x")]},
        {"findings": [_finding(start_line=12)]},  # after the line
        {"findings": [_finding(body="x" * (review.MAX_BODY_CHARS + 1))]},
        {"findings": [_finding()] * (review.MAX_FINDINGS + 1)},
    ],
)
def test_output_of_the_wrong_shape_is_refused(document):
    with pytest.raises(review.ReviewError):
        review.checked_findings(document, REPO)


def test_links_out_of_the_repository_become_plain_text():
    body = (
        "Rule: https://github.com/tenstorrent/tt-metal/blob/abc/README.md "
        "![x](https://evil.example/p.png?d=abc) and http://other.example/a "
        "and https://github.com/someone/else"
    )
    found = review.checked_findings({"findings": [_finding(body=body)]}, REPO)[0]["body"]

    assert "https://github.com/tenstorrent/tt-metal/blob/abc/README.md" in found
    assert "https[:]//evil.example/p.png?d=abc" in found
    assert "http[:]//other.example/a" in found and "https[:]//github.com/someone/else" in found
    assert len(re.findall(r"https?://", found)) == 1


def test_a_finding_cannot_carry_the_reviewers_signature_or_a_one_line_range():
    body = f"Text.\n\n{review.MARKER}\n{review.SIGNATURE}"
    found = review.checked_findings({"findings": [_finding(body=body, start_line=11)]}, REPO)[0]

    assert found["body"] == "Text." and found["start_line"] is None


# ── posting ───────────────────────────────────────────────────────────────────
class FakeGitHub:
    """Answers `gh api` from a script and records every call."""

    def __init__(self, heads, *, reject=(), compare=None):
        self.heads, self.reject, self.compare = list(heads), reject, compare
        self.calls, self.posts = [], []

    def __call__(self, argv, *, timeout=300):
        assert argv[:2] == ["gh", "api"], argv
        self.calls.append(argv)
        if "--method" in argv:
            endpoint = argv[4]
            fields = dict(part.split("=", 1) for part in argv[5:] if "=" in part)
            if any(marker in fields.get("path", "") for marker in self.reject):
                raise review.ReviewError("gh api failed: HTTP 422")
            self.posts.append((endpoint, fields))
            return f"https://github.com/{REPO}/pull/7#c{len(self.posts)}\n"
        endpoint = argv[2]
        if "/compare/" in endpoint:
            if self.compare is None:
                raise review.ReviewError("gh api failed: HTTP 404")
            return json.dumps(self.compare)
        assert endpoint == f"repos/{REPO}/pulls/7"
        head = self.heads.pop(0) if len(self.heads) > 1 else self.heads[0]
        return json.dumps({"head": {"sha": head}})


def _post(monkeypatch, findings, github):
    monkeypatch.setattr(review, "_run", github)
    meta = {"repository": REPO, "number": 7, "reviewed_head_sha": HEAD}
    poster = review.Poster(REPO, 7)
    checked = review.checked_findings({"findings": findings}, REPO)
    failed = review.post_review(meta, checked, review.right_side_lines(DIFF), poster)
    return failed, poster, github.posts


def test_findings_go_inline_at_the_reviewed_commit_and_are_signed(monkeypatch):
    failed, poster, posts = _post(
        monkeypatch, [_finding(), _finding(line=12, start_line=11, body="Range.")], FakeGitHub([HEAD])
    )

    assert failed == 0 and len(poster.urls) == 2
    assert [endpoint for endpoint, _ in posts] == [f"repos/{REPO}/pulls/7/comments"] * 2
    first, second = (fields for _, fields in posts)
    assert first["commit_id"] == HEAD and first["line"] == "11" and first["side"] == "RIGHT"
    assert "start_line" not in first and first["body"].endswith(review.FOOTER)
    assert second["start_line"] == "11" and second["line"] == "12"


def test_every_write_goes_to_the_one_pull_request(monkeypatch):
    github = FakeGitHub([HEAD])
    _post(monkeypatch, [_finding(), _finding(path="src/other.h")], github)

    allowed = (f"repos/{REPO}/pulls/7", f"repos/{REPO}/pulls/7/comments", f"repos/{REPO}/issues/7/comments")
    for argv in github.calls:
        endpoint = argv[4] if "--method" in argv else argv[2]
        assert endpoint in allowed, endpoint


def test_a_finding_without_a_valid_anchor_keeps_its_location_in_an_issue_comment(monkeypatch):
    failed, _poster, posts = _post(monkeypatch, [_finding(line=20)], FakeGitHub([HEAD]))

    assert failed == 0 and posts[0][0] == f"repos/{REPO}/issues/7/comments"
    assert posts[0][1]["body"].startswith("`src/a.h:20`\n\nThis overflows.")


def test_an_inline_comment_github_rejects_falls_back_to_an_issue_comment(monkeypatch):
    failed, poster, posts = _post(monkeypatch, [_finding()], FakeGitHub([HEAD], reject=("src/a.h",)))

    assert failed == 0 and len(poster.urls) == 1
    assert posts[0][0] == f"repos/{REPO}/issues/7/comments"
    assert posts[0][1]["body"].startswith("`src/a.h:11`")


def test_no_findings_posts_the_summary(monkeypatch):
    failed, poster, posts = _post(monkeypatch, [], FakeGitHub([HEAD]))

    assert failed == 0 and len(poster.urls) == 1
    assert "No issues found." in posts[0][1]["body"] and posts[0][1]["body"].endswith(review.FOOTER)


def test_a_head_that_moved_during_the_review_is_said_on_the_changed_files_only(monkeypatch):
    moved = "b" * 40
    compare = {"status": "ahead", "files": [{"filename": "src/a.h"}]}
    findings = [_finding(), _finding(path="src/other.h", body="Elsewhere.")]
    failed, _poster, posts = _post(monkeypatch, findings, FakeGitHub([moved], compare=compare))

    assert failed == 0 and len(posts) == 3
    assert "The PR head moved to `bbbbbbbbbbb` while the review was running" in posts[0][1]["body"]
    assert posts[1][1]["commit_id"] == HEAD  # still anchored to what was reviewed
    assert posts[1][1]["body"].startswith("_Reviewed at `aaaaaaaaaaa`; the PR has since moved")
    assert posts[2][1]["body"].startswith("`src/other.h:11` at `aaaaaaaaaaa`\n\nElsewhere.")


def test_a_rewritten_history_marks_every_finding(monkeypatch):
    compare = {"status": "diverged", "files": []}
    _failed, _poster, posts = _post(monkeypatch, [_finding(line=42)], FakeGitHub(["b" * 40], compare=compare))

    assert posts[1][1]["body"].startswith("_Reviewed at `aaaaaaaaaaa`")


def test_a_head_that_moves_while_posting_gets_a_note(monkeypatch):
    _failed, _poster, posts = _post(monkeypatch, [_finding()], FakeGitHub([HEAD, "c" * 40]))

    assert len(posts) == 2
    assert "while the review was being posted" in posts[1][1]["body"]


def test_a_dry_run_writes_nothing(monkeypatch, capsys):
    github = FakeGitHub([HEAD])
    monkeypatch.setattr(review, "_run", github)
    poster = review.Poster(REPO, 7, dry_run=True)
    meta = {"repository": REPO, "number": 7, "reviewed_head_sha": HEAD}
    review.post_review(
        meta, review.checked_findings({"findings": [_finding()]}, REPO), review.right_side_lines(DIFF), poster
    )

    assert github.posts == [] and "[dry run] inline comment on src/a.h:11" in capsys.readouterr().out


# ── inputs ────────────────────────────────────────────────────────────────────
@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with a base commit and a head commit that edits, renames and adds."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.h").write_text("one\ntwo\nthree\n")
    (tmp_path / "old.h").write_text("".join(f"line {n}\n" for n in range(30)))
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    (tmp_path / "a.h").write_text("one\nTWO\nthree\n")
    subprocess.run([*git, "mv", "old.h", "new.h"], check=True)
    (tmp_path / "b.h").write_text("new\n")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "change a, rename old, add b"], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    return tmp_path, base, head


def test_the_inputs_hold_the_exact_diff_the_metadata_and_the_discussion(repo, monkeypatch):
    tmp_path, base, head = repo
    answers = {
        f"repos/{REPO}/pulls/7": {
            "html_url": "u",
            "title": "T",
            "body": "B" * 30000,
            "user": {"login": "author"},
            "state": "open",
            "draft": True,
            "labels": [{"name": "llk:x"}],
            "base": {"ref": "main"},
            "head": {"sha": head},
        },
        f"repos/{REPO}/issues/7/comments": [[{"user": {"login": "c1"}, "body": "hello", "created_at": "2026-11-18"}]],
        f"repos/{REPO}/pulls/7/comments": [
            [{"user": {"login": "r1"}, "body": "nit", "path": "a.h", "line": 2, "created_at": "2026-11-18"}]
        ],
        f"repos/{REPO}/pulls/7/reviews": [
            [
                {"user": {"login": "r1"}, "body": "", "state": "COMMENTED"},
                {"user": {"login": "r2"}, "body": "", "state": "APPROVED"},
            ]
        ],
    }
    monkeypatch.setattr(review, "_gh_json", lambda *args: answers[args[-1]])
    out = tmp_path / "inputs"
    meta = review.write_inputs(REPO, 7, base, head, base, out)

    assert meta["reviewed_head_sha"] == head and meta["comparison_base_sha"] == base
    assert meta["state"] == "open" and meta["draft"] is True and meta["labels"] == ["llk:x"]
    assert meta["body"].endswith("[cut]") and len(meta["body"]) < 30000
    by_path = {entry["path"]: entry for entry in meta["changed_files"]}
    assert set(by_path) == {"a.h", "b.h", "new.h"}
    assert by_path["new.h"]["previous_path"] == "old.h" and by_path["new.h"]["status"] == "R"
    assert "+TWO" in (out / by_path["a.h"]["diff"]).read_text()
    assert "change a, rename old, add b" in (out / by_path["a.h"]["history"]).read_text()
    assert "+TWO" in (out / "diff" / "full.diff").read_text()
    assert "change a, rename old, add b" in (out / "commits.txt").read_text()
    assert json.loads((out / "pr.json").read_text()) == meta
    discussion = json.loads((out / "discussion.json").read_text())
    assert discussion["issue_comments"][0]["body"] == "hello"
    assert discussion["review_comments"][0]["path"] == "a.h"
    assert [entry["state"] for entry in discussion["reviews"]] == ["APPROVED"]
    # The per-file diffs give the same anchors as the whole diff.
    assert review.right_side_lines((out / "diff" / "full.diff").read_text())["a.h"] == [[1, 2, 3]]


def test_the_command_line_refuses_short_commits_and_an_empty_review(repo, monkeypatch, capsys):
    tmp_path, base, head = repo
    assert (
        review.main(
            [
                "inputs",
                "--repo",
                REPO,
                "--pr",
                "7",
                "--base-sha",
                "abc",
                "--head-sha",
                head,
                "--base-tip-sha",
                base,
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 1
    )
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "pr.json").write_text(json.dumps({"repository": REPO, "number": 7, "reviewed_head_sha": head}))
    monkeypatch.delenv("LLK_REVIEW_FINDINGS", raising=False)
    assert review.main(["post", "--input", str(tmp_path / "in")]) == 1
    assert "returned no output" in capsys.readouterr().out


# ── a session that did not review ─────────────────────────────────────────────
def _execution(tmp_path, agents, nested=0):
    events = [{"type": "system", "subtype": "init"}]
    for index in range(agents):
        events.append(
            {
                "type": "assistant",
                "parent_tool_use_id": None,
                "message": {
                    "content": [
                        {"type": "text", "text": "launching"},
                        {"type": "tool_use", "name": "Agent" if index % 2 else "Task", "input": {}},
                    ]
                },
            }
        )
    for _ in range(nested):  # a subagent's own calls do not count
        events.append(
            {
                "type": "assistant",
                "parent_tool_use_id": "toolu_1",
                "message": {"content": [{"type": "tool_use", "name": "Agent", "input": {}}]},
            }
        )
    events.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Read", "input": {}}]}})
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(events))
    return path


def test_the_agents_a_session_launched_are_counted_from_its_record(tmp_path):
    assert review.agent_calls(_execution(tmp_path, 7, nested=3)) == 7


@pytest.mark.parametrize("agents, posted", [(review.MIN_AGENT_CALLS, True), (2, False)])
def test_a_session_that_skipped_the_procedure_is_not_posted(tmp_path, monkeypatch, capsys, agents, posted):
    github = FakeGitHub([HEAD])
    monkeypatch.setattr(review, "_run", github)
    inputs = tmp_path / "in"
    (inputs / "diff").mkdir(parents=True)
    (inputs / "diff" / "full.diff").write_text(DIFF)
    (inputs / "pr.json").write_text(json.dumps({"repository": REPO, "number": 7, "reviewed_head_sha": HEAD}))
    monkeypatch.setenv("LLK_REVIEW_FINDINGS", json.dumps({"findings": []}))
    code = review.main(["post", "--input", str(inputs), "--execution-file", str(_execution(tmp_path, agents))])

    assert (code == 0) is posted and bool(github.posts) is posted
    if not posted:
        assert "launched 2 agent(s)" in capsys.readouterr().out


def test_file_history_stops_where_a_shallow_fetch_cut_it(repo):
    _tmp_path, base, head = repo

    assert "change a, rename old, add b" in review.file_history(head, ["a.h"], set())
    assert "base" in review.file_history(head, ["a.h"], set())
    cut = review.file_history(head, ["a.h"], {base})
    assert cut.splitlines()[-1] == "(older history was not fetched)" and ": base" not in cut
