#!/usr/bin/env python3
"""File one RELEASE Jira issue per *relevant* failed RTL sim test.

Reads the per-test failure detail the sim polling agent embeds in the
"RTL Sim CI test" GitHub check output, keeps only the tests listed in a
relevance mapping, and files (or de-dupes onto) one Jira issue per relevant
failed test -- so if three relevant tests fail, three tickets are filed/updated.

Environment:
  RTL_SIM_DETAIL    check output.summary (+ text). Per-test lines look like:
                      - `[1x3] unit_tests_api --gtest_filter=Foo.Bar`
                      - `[2x3] models/demos/.../test_add.py::test_foo`
  RTL_SIM_SHA       commit the check ran on                        (optional)
  RTL_SIM_URL       link to the sim results (check html_url)       (optional)
  RTL_SIM_RUN_URL   link to the release workflow run               (optional)
  RTL_SIM_MAP       path to the relevance-mapping JSON
                    (default: <this dir>/ai_ip_tests.json)
  JIRA_BASE_URL / JIRA_USER_EMAIL / JIRA_API_TOKEN / JIRA_PROJECT_KEY  (required)
  JIRA_ISSUE_TYPE   issue type name                                (default: Bug)
  JIRA_DRY_RUN      when truthy, print instead of calling Jira

A failed test is identified by (config, group, filter, runner) and is "relevant"
when the mapping has an entry whose stated fields all match (an omitted field is
a wildcard). Each relevant test gets a stable per-test dedup label so it owns one
issue and later releases comment on it instead of opening duplicates.
"""
import json
import os
import re

from create_jira_issue import _env, _truthy, file_issue

# One bullet per failed test. The config is always bracketed; what follows is
# either a gtest binary + --gtest_filter=, or a pytest file path (+ optional
# node id). The sim reporter currently renders every row with --gtest_filter=
# regardless of runner, so a `.py` group may carry either separator -- treat the
# group's extension, not the separator, as the runner signal.
# - `[1x3] unit_tests_api --gtest_filter=RtlSimCheckOutput.DoesNotExist_ForcedFailure`
# - `[2x3] models/demos/.../test_add.py::test_foo`
LINE_RE = re.compile(r"\[([^\]]+)\]\s+(\S+?)(?:::(\S+)|\s+--gtest_filter=(\S+))?(?=[\s`]|$)")


def _slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def parse_failed(detail):
    """Return de-duplicated (config, group, filter, runner) tuples from the detail.

    `filter` is "" for a whole-file pytest row (the yaml omits `filter`, so the
    whole file runs). `runner` is "pytest" when the group is a .py path.
    """
    seen, out = set(), []
    for m in LINE_RE.finditer(detail or ""):
        config, group, node_id, gtest_filter = (g.strip("`") if g else g for g in m.groups())
        runner = "pytest" if group.endswith(".py") else "gtest"
        # A gtest row without --gtest_filter= is not a test line (e.g. prose that
        # happens to start with a bracketed word); skip it.
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
    """True when the mapped filter matches the failed row's filter.

    `back2back: true` entries in the quasar yaml are merged by the sim CI into a
    single ':'-joined gtest filter, so one failed row can stand for a batch of
    tests. Treat the batch as matching if any component matches: at release time
    a red batch containing a watched test has to be looked at either way.
    """
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
        print(result)
        filed += 1

    # Safety net: the check was red but carried no parseable per-test detail
    # (e.g. the sim pipeline has not forwarded it, or an infra failure). File one
    # aggregate ticket so the failure is never silently dropped.
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


if __name__ == "__main__":
    main()
