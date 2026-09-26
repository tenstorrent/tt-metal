#!/usr/bin/env python3
"""Tests for the one-ticket-per-failed-job release filing."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import release_failure_tickets as rft  # noqa: E402

PRE = "build-test-publish (Ubuntu 22.04) / release-demo-tests / release-demo-tests / "
RUN = "https://github.com/o/r/actions/runs/1"


def job(name, conclusion="failure", steps=None, url="https://github.com/o/r/actions/runs/1/job/9"):
    return {"name": name, "conclusion": conclusion, "html_url": url, "steps": steps or []}


def test_release_refs_are_main_stable_and_version_tags():
    assert rft.is_release_ref("main") and rft.is_release_ref("stable") and rft.is_release_ref("v0.78.0-rc3")
    assert not rft.is_release_ref("dgnidash/feature") and not rft.is_release_ref("")


def test_only_failed_and_timed_out_jobs_are_selected():
    jobs = [job("a", "success"), job("b", "skipped"), job("c", "failure"), job("d", "timed_out")]
    kept, dropped = rft.select_failed(jobs, "stable")
    assert [j["name"] for j in kept] == ["c", "d"] and dropped == 0


def test_a_runner_that_never_started_is_dropped():
    never = job(PRE + "Gemma e2e tests [r]", steps=[{"name": "Set up runner", "conclusion": "failure"}])
    real = job(PRE + "GPT e2e tests [r]", steps=[{"name": "Run tests", "conclusion": "failure"}])
    kept, dropped = rft.select_failed([never, real], "stable")
    assert [rft.leaf(j["name"]) for j in kept] == ["GPT e2e tests [r]"] and dropped == 1


def test_precheck_is_dropped_on_main_only():
    pre = job("release-precheck")
    assert rft.select_failed([pre], "main") == ([], 1)
    assert rft.select_failed([pre], "stable")[0] == [pre]


def test_one_ticket_per_job_with_owner_commit_and_run_link():
    j = job(PRE + "Llama 3.1-8B e2e tests [bh_p150]")
    title, desc, labels, dedup = rft.ticket(j, "stable", "de546d3b1467" + "0" * 28, "o/r", "1", RUN)
    assert title == "Release stable — Llama 3.1-8B e2e tests [bh_p150] failed"
    assert "- [Llama 3.1-8B e2e tests [bh_p150]](https://github.com/o/r/actions/runs/1/job/9) — owner: " in desc
    assert "(models)" in desc
    assert "- Commit: [de546d3b1467](https://github.com/o/r/commit/de546d3b1467" in desc
    assert f"- Run link: [1]({RUN})" in desc
    assert "- Pipeline: build-test-publish (Ubuntu 22.04) / release-demo-tests / release-demo-tests" in desc
    assert labels == ["ci-failure", "package-and-release", "package-release-ref:stable"]
    assert dedup == "package-release-job:stable:llama-3-1-8b-e2e-tests-bh-p150"


def test_unknown_job_has_no_owner_note_and_no_pipeline_line():
    title, desc, _, _ = rft.ticket(job("publish-docs"), "main", "abc", "o/r", "1", RUN)
    assert title == "Release main — publish-docs failed"
    assert "owner:" not in desc and "Pipeline:" not in desc


def test_root_cause_is_matched_by_job_url_then_by_leaf_name():
    j = job(PRE + "Gemma-4-31B e2e tests [bh_quietbox_2]", url="https://x/job/2")
    by_url = {"failed": [{"job_name": "other", "job_url": "https://x/job/2", "root_cause": "OOM"}]}
    assert rft.root_cause_for(j, by_url)["root_cause"] == "OOM"
    by_name = {"failed": [{"job_name": "Gemma-4-31B e2e tests [bh_quietbox_2]", "job_url": "", "root_cause": "PCC"}]}
    assert rft.root_cause_for(j, by_name)["root_cause"] == "PCC"
    assert rft.root_cause_for(j, {"failed": [{"job_name": "nope", "job_url": "https://x/job/3"}]}) is None
    assert rft.root_cause_for(j, None) is None


def test_only_this_jobs_root_cause_lands_in_its_ticket():
    a = job(PRE + "A e2e tests [r]", url="https://x/job/a")
    summary = {
        "failed": [
            {
                "job_name": "A e2e tests [r]",
                "job_url": "https://x/job/a",
                "root_cause": "",
                "error_message": "Job hit the GitHub timeout limit and was killed.",
                "category": "infra:timeout",
                "log_complete": False,
            },
            {"job_name": "B e2e tests [r]", "job_url": "https://x/job/b", "root_cause": "B broke"},
        ]
    }
    _, desc, _, _ = rft.ticket(a, "stable", "abc", "o/r", "1", RUN, summary)
    assert "### Other information" in desc
    assert "- Job hit the GitHub timeout limit and was killed. [infra:timeout] (log incomplete)" in desc
    assert "B broke" not in desc


def _ai(**over):
    base = {
        "job_name": "x",
        "job_url": "https://github.com/o/r/actions/runs/1/job/9",
        "category": "tt-metal:pcc",
        "subcategory": "",
        "error_message": "assert 0.97 > 0.99",
        "root_cause": "PCC drift after SFPI bump",
        "log_complete": True,
    }
    base.update(over)
    return {"failed": [base]}


def test_the_verbatim_error_rides_along_with_the_root_cause():
    _, desc, _, _ = rft.ticket(job("x"), "stable", "abc", "o/r", "1", RUN, _ai())
    assert "- PCC drift after SFPI bump [tt-metal:pcc] — error: assert 0.97 > 0.99" in desc


def test_error_backfilling_the_cause_is_not_repeated():
    _, desc, _, _ = rft.ticket(job("x"), "stable", "abc", "o/r", "1", RUN, _ai(root_cause=""))
    assert "- assert 0.97 > 0.99 [tt-metal:pcc]" in desc and "— error:" not in desc


def test_subcategory_extends_the_category_tag():
    _, desc, _, _ = rft.ticket(job("x"), "stable", "abc", "o/r", "1", RUN, _ai(subcategory="pcc_drop"))
    assert "[tt-metal:pcc/pcc_drop]" in desc
    _, desc, _, _ = rft.ticket(job("x"), "stable", "abc", "o/r", "1", RUN, _ai(category=None, subcategory="pcc_drop"))
    assert "[pcc_drop]" in desc


def test_no_root_cause_means_no_other_information_section():
    _, desc, _, _ = rft.ticket(job("x"), "stable", "abc", "o/r", "1", RUN, {"failed": []})
    assert "Other information" not in desc


def test_title_never_exceeds_the_jira_limit():
    title, *_ = rft.ticket(job("x" * 400), "stable", "abc", "o/r", "1", RUN)
    assert len(title) <= 255 and title.endswith("...")


def _wire(monkeypatch, run, jobs, summary=None):
    monkeypatch.setattr(rft, "fetch_run", lambda repo, run_id: run)
    monkeypatch.setattr(rft, "fetch_jobs", lambda repo, run_id: jobs)
    monkeypatch.setattr(rft, "_fetch_summary", lambda repo, run_id: summary)
    for k, v in {
        "REPO": "o/r",
        "RUN_ID": "1",
        "JIRA_BASE_URL": "https://j.test",
        "JIRA_USER_EMAIL": "e",
        "JIRA_API_TOKEN": "t",
        "JIRA_PROJECT_KEY": "RELEASE",
        "JIRA_DRY_RUN": "",
    }.items():
        monkeypatch.setenv(k, v)


def test_main_files_one_issue_per_failed_job(monkeypatch):
    filed = []
    monkeypatch.setattr(rft, "file_issue", lambda **kw: filed.append(kw) or f"created {kw['summary']}")
    _wire(
        monkeypatch,
        {"head_branch": "stable", "head_sha": "abc"},
        [job(PRE + "A e2e tests [r]"), job(PRE + "B e2e tests [r]"), job("ok", "success")],
    )
    rft.main()
    assert [f["summary"] for f in filed] == [
        "Release stable — A e2e tests [r] failed",
        "Release stable — B e2e tests [r] failed",
    ]
    assert {f["dedup_label"] for f in filed} == {
        "package-release-job:stable:a-e2e-tests-r",
        "package-release-job:stable:b-e2e-tests-r",
    }


def test_main_skips_non_release_refs_without_listing_jobs(monkeypatch, capsys):
    monkeypatch.setattr(
        rft, "fetch_jobs", lambda repo, run_id: (_ for _ in ()).throw(AssertionError("must not list jobs"))
    )
    monkeypatch.setattr(rft, "file_issue", lambda **kw: (_ for _ in ()).throw(AssertionError("must not file")))
    _wire(monkeypatch, {"head_branch": "dgnidash/feature", "head_sha": "abc"}, [])
    monkeypatch.setattr(
        rft, "fetch_jobs", lambda repo, run_id: (_ for _ in ()).throw(AssertionError("must not list jobs"))
    )
    rft.main()
    assert "not a release ref" in capsys.readouterr().out


def test_one_filing_failure_does_not_drop_the_rest(monkeypatch, capsys):
    calls = []

    def flaky(**kw):
        calls.append(kw["summary"])
        if "A " in kw["summary"]:
            sys.exit("error: Jira POST -> 500")
        return "created RELEASE-1"

    monkeypatch.setattr(rft, "file_issue", flaky)
    _wire(monkeypatch, {"head_branch": "main", "head_sha": "abc"}, [job("A"), job("B")])
    try:
        rft.main()
    except SystemExit as e:
        assert "1 issue(s) could not be filed" in str(e)
    else:
        raise AssertionError("expected a non-zero exit")
    assert calls == ["Release main — A failed", "Release main — B failed"]
    assert "could not file a ticket for A" in capsys.readouterr().out


def test_missing_jira_secrets_skip_filing_but_dry_run_still_prints(monkeypatch, capsys):
    _wire(monkeypatch, {"head_branch": "stable", "head_sha": "abc"}, [job("A")])
    monkeypatch.setenv("JIRA_USER_EMAIL", "")
    monkeypatch.setattr(rft, "file_issue", lambda **kw: (_ for _ in ()).throw(AssertionError("must not file")))
    rft.main()
    assert "secrets not configured" in capsys.readouterr().out
    monkeypatch.setenv("JIRA_DRY_RUN", "1")
    monkeypatch.setattr(rft, "file_issue", lambda **kw: f"DRY RUN {kw['summary']} dry={kw['dry_run']}")
    rft.main()
    assert "DRY RUN Release stable — A failed dry=True" in capsys.readouterr().out
