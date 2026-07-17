#!/usr/bin/env python3
"""File one RELEASE Jira issue per *relevant* failed RTL sim test.

Reads the per-test failure detail the sim polling agent embeds in the
"RTL Sim CI test" GitHub check output, keeps only the tests listed in a
relevance mapping, and files (or de-dupes onto) one Jira issue per relevant
failed test -- so if three relevant tests fail, three tickets are filed/updated.

Environment:
  RTL_SIM_DETAIL    check output.summary (+ text). Per-test lines look like:
                      - `[1x3] unit_tests_api --gtest_filter=Foo.Bar`
  RTL_SIM_SHA       commit the check ran on                        (optional)
  RTL_SIM_URL       link to the sim results (check html_url)       (optional)
  RTL_SIM_RUN_URL   link to the release workflow run               (optional)
  RTL_SIM_MAP       path to the relevance-mapping JSON
                    (default: <this dir>/rtl_sim_jira_tests.json)
  JIRA_BASE_URL / JIRA_USER_EMAIL / JIRA_API_TOKEN / JIRA_PROJECT_KEY  (required)
  JIRA_ISSUE_TYPE   issue type name                                (default: Bug)
  JIRA_DRY_RUN      when truthy, print instead of calling Jira

A failed test is identified by (config, group, filter) and is "relevant" when the
mapping has an entry whose stated fields all match (an omitted field is a
wildcard). Each relevant test gets a stable per-test dedup label so it owns one
issue and later releases comment on it instead of opening duplicates.
"""
import json
import os
import re

from create_jira_issue import _env, file_issue

# - `[1x3] unit_tests_api --gtest_filter=RtlSimCheckOutput.DoesNotExist_ForcedFailure`
LINE_RE = re.compile(r"\[([^\]]+)\]\s+(\S+)\s+--gtest_filter=(\S+)")


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def parse_failed(detail):
    """Return de-duplicated (config, group, filter) tuples from the check detail."""
    seen, out = set(), []
    for m in LINE_RE.finditer(detail or ""):
        key = tuple(g.strip("`") for g in m.groups())
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def is_relevant(config, group, filt, mapping):
    for e in mapping.get("relevant_tests", []):
        if (e.get("config", config) == config
                and e.get("group", group) == group
                and e.get("filter", filt) == filt):
            return True
    return False


def main():
    base = _env("JIRA_BASE_URL", required=True)
    email = _env("JIRA_USER_EMAIL", required=True)
    token = _env("JIRA_API_TOKEN", required=True)
    project = _env("JIRA_PROJECT_KEY", required=True)
    issue_type = _env("JIRA_ISSUE_TYPE", "Bug")
    dry = bool(_env("JIRA_DRY_RUN"))

    detail = _env("RTL_SIM_DETAIL", "")
    sha = _env("RTL_SIM_SHA", "unknown")
    url = _env("RTL_SIM_URL", "-")
    run_url = _env("RTL_SIM_RUN_URL", "-")

    here = os.path.dirname(os.path.abspath(__file__))
    map_path = _env("RTL_SIM_MAP", os.path.join(here, "rtl_sim_jira_tests.json"))
    with open(map_path) as f:
        mapping = json.load(f)

    failed = parse_failed(detail)
    print(f"parsed {len(failed)} failed test(s) from the check detail")

    filed = 0
    for config, group, filt in failed:
        test = f"[{config}] {group} --gtest_filter={filt}"
        if not is_relevant(config, group, filt, mapping):
            print(f"skip (not in relevance map): {test}")
            continue
        result = file_issue(
            base=base, email=email, token=token, project=project,
            summary=f"RTL sim test failed during release: {test}",
            issue_type=issue_type,
            description=(
                "The RTL sim regression test below failed during Package and release.\n\n"
                f"Test:        {test}\n"
                f"Commit:      {sha}\n"
                f"Sim results: {url}\n"
                f"Release run: {run_url}\n"
            ),
            labels=["rtl-sim", "ci-failure", "release"],
            # Stable per-test label: this test owns one issue; reruns comment.
            dedup_label="rtl-sim:" + _slug(f"{config}-{group}-{filt}"),
            dry_run=dry,
        )
        print(result)
        filed += 1

    # Safety net: the check was red but carried no parseable per-test detail
    # (e.g. the sim pipeline has not forwarded it, or an infra failure). File one
    # aggregate ticket so the failure is never silently dropped.
    if not failed:
        print("no per-test lines in check detail; filing one aggregate ticket")
        print(file_issue(
            base=base, email=email, token=token, project=project,
            summary=f"RTL sim regression failed during release (commit {sha}, no per-test detail)",
            issue_type=issue_type,
            description=(
                "The 'RTL Sim CI test' check failed but reported no per-test detail.\n\n"
                f"Commit:      {sha}\n"
                f"Sim results: {url}\n"
                f"Release run: {run_url}\n"
            ),
            labels=["rtl-sim", "ci-failure", "release"],
            dedup_label="rtl-sim-release-failure",
            dry_run=dry,
        ))
        filed += 1

    print(f"filed/updated {filed} issue(s)")


if __name__ == "__main__":
    main()
