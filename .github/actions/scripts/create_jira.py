#!/usr/bin/env python3
"""File one RELEASE Jira issue per *relevant* failed RTL sim test.

Reads the per-test detail from the "RTL Sim CI test" check output, keeps only the
tests listed in the relevance mapping, and files (or de-dupes onto) one issue per
relevant failed test. A test is (config, group, filter, runner); an omitted field
in a mapping entry is a wildcard. Each gets a stable dedup label so reruns
comment instead of opening duplicates.

Environment:
  RTL_SIM_DETAIL    check output.summary (+ text). Per-test lines look like:
                      - `[1x3] unit_tests_api --gtest_filter=Foo.Bar`
                      - `[2x3] models/demos/.../test_add.py::test_foo`
  RTL_SIM_SHA / RTL_SIM_URL / RTL_SIM_RUN_URL                        (optional)
  RTL_SIM_MAP       relevance mapping   (default: ./ai_ip_tests.json)
  JIRA_BASE_URL / JIRA_USER_EMAIL / JIRA_API_TOKEN / JIRA_PROJECT_KEY (required)
  JIRA_ISSUE_TYPE   issue type name                              (default: Bug)
  JIRA_DRY_RUN      print instead of calling Jira
"""
import json
import os
import re
import sys

from jira_client import _env, _truthy, file_issue

# The sim reporter renders every row with --gtest_filter= regardless of runner,
# so the group's extension, not the separator, decides the runner.
LINE_RE = re.compile(r"\[([^\]]+)\]\s+(\S+?)(?:::(\S+)|\s+--gtest_filter=(\S+))?(?=[\s`]|$)")


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def parse_failed(detail):
    """De-duplicated (config, group, filter, runner) tuples from the detail."""
    seen, out = set(), []
    for m in LINE_RE.finditer(detail or ""):
        config, group, node_id, gtest_filter = (g.strip("`") if g else g for g in m.groups())
        runner = "pytest" if group.endswith(".py") else "gtest"
        # No --gtest_filter= means it is prose, not a test line.
        if runner == "gtest" and not gtest_filter:
            continue
        key = (config, group, node_id or gtest_filter or "", runner)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def format_test(config, group, filt, runner):
    """Human-readable identity of a failed test, as it appears in the ticket."""
    if runner == "pytest":
        return f"[{config}] {group}::{filt}" if filt else f"[{config}] {group} (whole file)"
    return f"[{config}] {group} --gtest_filter={filt}"


def _filter_matches(want, filt):
    """Match the mapped filter, or any component of a ':'-joined back2back batch."""
    return want == filt or want in filt.split(":")


def match_entry(config, group, filt, runner, mapping):
    """Return the relevance-map entry matching this failed test, or None."""
    for e in mapping.get("relevant_tests", []):
        if (
            e.get("config", config) == config
            and e.get("group", group) == group
            and _filter_matches(e.get("filter", filt), filt)
            and e.get("runner", runner) == runner
        ):
            return e
    return None


def main():
    base = _env("JIRA_BASE_URL", required=True)
    email = _env("JIRA_USER_EMAIL", required=True)
    token = _env("JIRA_API_TOKEN", required=True)
    project = _env("JIRA_PROJECT_KEY", required=True)
    issue_type = _env("JIRA_ISSUE_TYPE", "Bug")
    dry = _truthy(_env("JIRA_DRY_RUN"))

    detail = _env("RTL_SIM_DETAIL", "")
    sha = _env("RTL_SIM_SHA", "unknown")
    url = _env("RTL_SIM_URL", "-")
    run_url = _env("RTL_SIM_RUN_URL", "-")

    here = os.path.dirname(os.path.abspath(__file__))
    map_path = _env("RTL_SIM_MAP", os.path.join(here, "ai_ip_tests.json"))
    with open(map_path) as f:
        mapping = json.load(f)

    failed = parse_failed(detail)
    print(f"parsed {len(failed)} failed test(s) from the check detail")

    filed = 0
    failed_to_file = 0
    for config, group, filt, runner in failed:
        test = format_test(config, group, filt, runner)
        entry = match_entry(config, group, filt, runner, mapping)
        if entry is None:
            print(f"skip (not in relevance map): {test}")
            continue
        requirement = entry.get("requirement")
        team = entry.get("team")
        labels = ["rtl-sim", "ci-failure", "release"]
        if requirement:
            labels.append(requirement)
        desc = ["The RTL sim regression test below failed during Package and release.\n", f"Test:        {test}"]
        if requirement:
            desc.append(f"Requirement: {requirement}")
        if team:
            desc.append(f"Team:        {team}")
        if runner == "gtest" and entry.get("filter") and entry["filter"] != filt:
            # Matched one component of a back2back batch; the batch is what ran.
            desc.append(f"Matched on:  {entry['filter']} (ran back-to-back with the other filters above)")
        desc += [f"Commit:      {sha}", f"Sim results: {url}", f"Release run: {run_url}"]
        try:
            result = file_issue(
                base=base,
                email=email,
                token=token,
                project=project,
                summary=f"RTL sim test failed during release: {test}",
                issue_type=issue_type,
                description="\n".join(desc) + "\n",
                labels=labels,
                # Stable per-test label: this test owns one issue; reruns comment.
                dedup_label="rtl-sim:" + _slug(f"{config}-{group}-{filt}"),
                assignee=entry.get("assignee"),
                dry_run=dry,
            )
        except (SystemExit, Exception) as e:  # jira_client exits on API error
            # One unusable assignee or a transient 5xx must not drop the rest.
            print(f"::warning::could not file {test}: {e}")
            failed_to_file += 1
            continue
        print(result)
        filed += 1

    # Red but no parseable detail: one aggregate ticket so it is not lost.
    if not failed:
        print("no per-test lines in check detail; filing one aggregate ticket")
        print(
            file_issue(
                base=base,
                email=email,
                token=token,
                project=project,
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
            )
        )
        filed += 1

    print(f"filed/updated {filed} issue(s)")
    if failed_to_file:
        # Exit non-zero so a silent "zero tickets, green workflow" is impossible.
        sys.exit(f"error: {failed_to_file} issue(s) could not be filed")


if __name__ == "__main__":
    main()
