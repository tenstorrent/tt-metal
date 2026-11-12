# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Unit tests for ci_no_retry.py against a fake GitHub API. Run: python3 -m pytest test_ci_no_retry.py"""

import importlib.util
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

spec = importlib.util.spec_from_file_location("ci_no_retry", Path(__file__).with_name("ci_no_retry.py"))
ci_no_retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_no_retry)

REPO, RUN = "o/r", "42"
# The message format classify-failure/action.yml writes.
MARK = {
    "title": "ci-no-retry",
    "message": "1 deterministic failure(s), a re-run of this commit cannot pass: TPS GATE FAIL",
}
OTHER = {"title": "Prefill perf gate", "message": "TPS GATE FAIL"}


def job(id_, name, conclusion, annotations=()):
    return {"id": id_, "name": name, "conclusion": conclusion, "check_run_url": f"check-runs/{id_}", "_a": annotations}


class FakeApi:
    """attempts: {attempt: [job, ...]}. Serves jobs, single jobs and annotations, paginated like GitHub."""

    def __init__(self, attempts, fail=None):
        self.attempts, self.fail = attempts, fail
        self.by_id = {j["id"]: j for jobs in attempts.values() for j in jobs}

    def __call__(self, url):
        if self.fail:
            raise self.fail
        path, _, query = url.partition("?")
        params = dict(kv.split("=") for kv in query.split("&")) if query else {}
        page, per = int(params.get("page", 1)), int(params.get("per_page", 100))
        window = slice((page - 1) * per, page * per)
        if "/attempts/" in path:
            jobs = self.attempts[int(path.rsplit("/attempts/", 1)[1].split("/")[0])]
            return {"jobs": [{k: v for k, v in j.items() if k != "_a"} for j in jobs[window]]}
        if path.endswith("/annotations"):
            return list(self.by_id[int(path.split("/")[-2])]["_a"])[window]
        return {k: v for k, v in self.by_id[int(path.rsplit("/", 1)[1])].items() if k != "_a"}


def decide(capsys, monkeypatch, attempts, attempt=1, fail=None, tmp_path=None):
    if tmp_path is not None:
        monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setattr(ci_no_retry, "api", FakeApi(attempts, fail))
    monkeypatch.setattr(sys, "argv", ["x", "run-decision", "--repo", REPO, "--run-id", RUN, "--attempt", str(attempt)])
    assert ci_no_retry.main() == 0
    return capsys.readouterr().out.strip().splitlines()


def test_all_marked_skips(capsys, monkeypatch, tmp_path):
    attempts = {1: [job(1, "a", "failure", [OTHER, MARK]), job(2, "b", "success"), job(3, "c", "skipped")]}
    assert decide(capsys, monkeypatch, attempts, tmp_path=tmp_path)[-1] == "skip"
    assert (tmp_path / "out").read_text() == "all-non-retryable=true\n"


def test_mixed_reruns(capsys, monkeypatch, tmp_path):
    attempts = {1: [job(1, "a", "failure", [MARK]), job(2, "b", "failure", [OTHER])]}
    out = decide(capsys, monkeypatch, attempts, tmp_path=tmp_path)
    assert out[-1] == "rerun" and "  retryable: b" in out
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("conclusion", ["timed_out", "cancelled", None, "action_required"])
def test_unmarked_non_failure_conclusions_keep_the_run_retryable(capsys, monkeypatch, conclusion):
    attempts = {1: [job(1, "a", "failure", [MARK]), job(2, "b", conclusion)]}
    assert decide(capsys, monkeypatch, attempts)[-1] == "rerun"


def test_marker_beyond_first_annotation_page(capsys, monkeypatch):
    attempts = {1: [job(1, "a", "failure", [OTHER] * 150 + [MARK])]}
    assert decide(capsys, monkeypatch, attempts)[-1] == "skip"


def test_jobs_beyond_first_page(capsys, monkeypatch):
    attempts = {1: [job(i, f"j{i}", "failure", [MARK]) for i in range(150)] + [job(999, "late", "failure")]}
    assert decide(capsys, monkeypatch, attempts)[-1] == "rerun"


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: urllib.error.HTTPError("u", 401, "Unauthorized", {}, None),
        lambda: urllib.error.URLError("down"),
        lambda: KeyError("jobs"),
    ],
    ids=["http-401", "network", "bad-payload"],
)
def test_api_errors_fail_open(capsys, monkeypatch, make_error):
    out = decide(capsys, monkeypatch, {1: []}, fail=make_error())
    assert out[-1] == "rerun" and "check-failed" in out[0]


def check(capsys, monkeypatch, attempts, attempt, job_id, actor=ci_no_retry.AUTO_RETRY_ACTOR):
    monkeypatch.setattr(ci_no_retry, "api", FakeApi(attempts))
    argv = ["x", "job-check", "--job-id", str(job_id), "--repo", REPO, "--run-id", RUN, "--attempt", str(attempt)]
    argv += ["--triggering-actor", actor]
    monkeypatch.setattr(sys, "argv", argv)
    return ci_no_retry.main(), capsys.readouterr().out


def test_first_attempt_runs(capsys, monkeypatch):
    assert check(capsys, monkeypatch, {1: [job(1, "a", None)]}, 1, 1)[0] == 0


def test_marked_previous_attempt_short_circuits_and_remarks(capsys, monkeypatch):
    attempts = {1: [job(1, "leg", "failure", [MARK]), job(2, "other", "failure")], 2: [job(3, "leg", None)]}
    rc, out = check(capsys, monkeypatch, attempts, 2, 3)
    assert rc == ci_no_retry.SKIP_EXIT
    assert out.startswith("::error title=ci-no-retry::Not re-run: attempt 1 failed deterministically: ")
    assert "TPS GATE FAIL" in out and "re-run it by hand" in out


def test_short_circuit_reason_does_not_nest(capsys, monkeypatch):
    # Attempt 2 was itself short-circuited; attempt 3 must carry the original reason once.
    first = check(capsys, monkeypatch, {1: [job(1, "leg", "failure", [MARK])], 2: [job(2, "leg", None)]}, 2, 2)[1]
    remark = {"title": "ci-no-retry", "message": first.split("::", 2)[2].strip()}
    attempts = {2: [job(2, "leg", "failure", [remark])], 3: [job(3, "leg", None)]}
    rc, out = check(capsys, monkeypatch, attempts, 3, 3)
    assert rc == ci_no_retry.SKIP_EXIT
    assert out.strip() == (
        "::error title=ci-no-retry::Not re-run: attempt 2 failed deterministically: "
        f"{MARK['message']} -- auto-retry cannot change that; re-run it by hand to measure again"
    )


@pytest.mark.parametrize(
    "previous",
    [
        [job(1, "leg", "failure", [OTHER])],  # failed, not marked
        [job(1, "leg", "success", [MARK])],  # passed
        [job(1, "leg", "failure", [MARK]), job(9, "leg", "failure", [MARK])],  # ambiguous name
        [job(1, "renamed", "failure", [MARK])],  # not found
    ],
)
def test_job_runs_unless_its_previous_attempt_was_marked(capsys, monkeypatch, previous):
    rc, _ = check(capsys, monkeypatch, {1: previous, 2: [job(3, "leg", None)]}, 2, 3)
    assert rc == 0


@pytest.mark.parametrize("actor", ["someone-else", ""])
def test_rerun_not_started_by_auto_retry_runs_marked_job(capsys, monkeypatch, actor):
    # A manual re-run (or an unknown actor) measures again: no short-circuit, and no API call is needed.
    attempts = {1: [job(1, "leg", "failure", [MARK])], 2: [job(3, "leg", None)]}
    monkeypatch.setattr(ci_no_retry, "api", FakeApi(attempts, fail=AssertionError("no API call expected")))
    argv = ["x", "job-check", "--job-id", "3", "--repo", REPO, "--run-id", RUN, "--attempt", "2"]
    monkeypatch.setattr(sys, "argv", argv + ["--triggering-actor", actor])
    assert ci_no_retry.main() == 0
    out = capsys.readouterr().out
    assert "not by auto-retry" in out and "::error" not in out


def test_triggering_actor_defaults_to_the_actions_env(capsys, monkeypatch):
    monkeypatch.setenv("GITHUB_TRIGGERING_ACTOR", ci_no_retry.AUTO_RETRY_ACTOR)
    attempts = {1: [job(1, "leg", "failure", [MARK])], 2: [job(3, "leg", None)]}
    monkeypatch.setattr(ci_no_retry, "api", FakeApi(attempts))
    argv = ["x", "job-check", "--job-id", "3", "--repo", REPO, "--run-id", RUN, "--attempt", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    assert ci_no_retry.main() == ci_no_retry.SKIP_EXIT


def test_triggering_actor_match_ignores_case(capsys, monkeypatch):
    attempts = {1: [job(1, "leg", "failure", [MARK])], 2: [job(3, "leg", None)]}
    rc, _ = check(capsys, monkeypatch, attempts, 2, 3, actor=ci_no_retry.AUTO_RETRY_ACTOR.upper())
    assert rc == ci_no_retry.SKIP_EXIT
