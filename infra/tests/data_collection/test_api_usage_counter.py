# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-execution REST-call attribution in download_cicd_logs_and_artifacts.sh.

produce-data shares one GitHub installation token bucket with many other workflows, so the
collection script counts only its OWN REST calls (by category) and emits one `[api-usage]` line
per execution. These tests pin:
  1. the counters survive the command-substitution / pipe subshells the collectors run in, and
     sum correctly into the `[api-usage]` line (unit),
  2. an arbitrary run name stays parseable (unit), and
  3. a full first-attempt collection against a stubbed `gh` reports the expected per-category
     counts (integration).

The bash snippets are written to a file and run as `bash <file>`, with all dynamic values passed
through the environment rather than interpolated into a `bash -c` command string -- both to keep
the snippets readable and to avoid constructing shell commands from formatted strings.
"""

import os
import re
import shutil
import subprocess
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "infra" / "data_collection" / "github" / "download_cicd_logs_and_artifacts.sh"


def _api_usage_fields(stdout: str) -> dict:
    lines = [ln for ln in stdout.splitlines() if ln.startswith("[api-usage]")]
    assert lines, f"no [api-usage] line in output:\n{stdout}"
    body = lines[-1].split("]", 1)[1]
    fields = {}
    # workflow is a quoted free-form value (may contain spaces/brackets); pull it out first.
    m = re.search(r'workflow="([^"]*)"', body)
    if m:
        fields["workflow"] = m.group(1)
        body = body[: m.start()] + body[m.end() :]
    fields.update(dict(re.findall(r"(\w+)=(\S+)", body)))
    return fields


def _run_snippet(tmp_path: pathlib.Path, snippet: str, env_extra: dict) -> subprocess.CompletedProcess:
    """Run a static bash snippet from a file with dynamic values supplied via the environment.

    The snippet is a constant (no interpolation) and reads its inputs from env vars, so no shell
    command is built from formatted strings.
    """
    snippet_file = tmp_path / "snippet.sh"
    snippet_file.write_text(snippet)
    env = {**os.environ, "SCRIPT_PATH": str(SCRIPT), **env_extra}
    return subprocess.run(["bash", str(snippet_file)], capture_output=True, text=True, env=env)


# Exercises the two subshell shapes the real collectors run inside: command substitution
# (get_jobs_with_pagination_fallback) and a pipe-into-while (the per-job loop).
_SUBSHELL_SNIPPET = """
set -eo pipefail
source "$SCRIPT_PATH"
: > "$API_COUNT_FILE"
workflow_name="$WORKFLOW_NAME"; workflow_run_id="$RUN_ID"; attempt_number=1
api_count jobs_list 1                                      # direct
x=$(api_count artifact_download 56; echo ok)              # command-substitution subshell
echo hi | { api_count artifact_list 2; cat >/dev/null; } # pipe subshell
api_count log_archive 1
emit_api_usage
"""

_NASTY_NAME_SNIPPET = """
set -eo pipefail
source "$SCRIPT_PATH"
: > "$API_COUNT_FILE"
workflow_name="$WORKFLOW_NAME"; workflow_run_id="$RUN_ID"; attempt_number=1
api_count jobs_list 2
emit_api_usage
"""

_NOOP_SNIPPET = """
set -eo pipefail
source "$SCRIPT_PATH"
unset API_COUNT_FILE
api_count jobs_list 3   # no-op, must not fail
emit_api_usage          # no-op, prints nothing
echo done
"""


def test_counters_survive_subshells_and_sum(tmp_path):
    r = _run_snippet(
        tmp_path,
        _SUBSHELL_SNIPPET,
        {
            "API_COUNT_FILE": str(tmp_path / "counts"),
            "WORKFLOW_NAME": "Sanity tests (push) SKUs[WH,Sim]",
            "RUN_ID": "123",
        },
    )
    assert r.returncode == 0, r.stderr
    f = _api_usage_fields(r.stdout)
    assert f["workflow"] == "Sanity tests (push) SKUs[WH,Sim]"  # spaces/brackets survive quoting
    assert f["run_id"] == "123" and f["attempt"] == "1"
    assert f["jobs_list"] == "1"
    assert f["artifact_download"] == "56"
    assert f["artifact_list"] == "2"
    assert f["log_archive"] == "1"
    assert f["per_job_log"] == "0" and f["annotations"] == "0" and f["attempt_meta"] == "0"
    assert f["total"] == str(1 + 56 + 2 + 1)


def test_arbitrary_workflow_name_stays_parseable(tmp_path):
    # Names are arbitrary user text; a double quote / newline must not break the quoted field or
    # the numeric fields that follow it.
    nasty = 'weird " name\nwith [brackets] and 🛠️'
    r = _run_snippet(
        tmp_path,
        _NASTY_NAME_SNIPPET,
        {"API_COUNT_FILE": str(tmp_path / "counts"), "WORKFLOW_NAME": nasty, "RUN_ID": "7"},
    )
    assert r.returncode == 0, r.stderr
    # exactly one [api-usage] line (the newline in the name did not split the record)
    assert len([ln for ln in r.stdout.splitlines() if ln.startswith("[api-usage]")]) == 1
    f = _api_usage_fields(r.stdout)
    assert f["run_id"] == "7" and f["jobs_list"] == "2" and f["total"] == "2"
    assert '"' not in f["workflow"]  # embedded quote was neutralised
    assert "[brackets]" in f["workflow"] and "🛠" in f["workflow"]


def test_emit_is_noop_without_counter_file(tmp_path):
    # Using the helpers without API_COUNT_FILE set must not error or print a line.
    r = _run_snippet(tmp_path, _NOOP_SNIPPET, {})
    assert r.returncode == 0, r.stderr
    assert "[api-usage]" not in r.stdout
    assert "done" in r.stdout


_GH_STUB = r"""#!/bin/bash
# Minimal gh stub for the api-usage integration test. First-attempt scenario:
#   * gh run download            -> creates two test_reports_ artifact dirs
#   * gh api .../attempts/1/jobs -> two skipped jobs (so no per-job logs / annotations)
#   * gh api .../attempts/1/logs -> a (deliberately non-zip) body; the gh call succeeds, the
#                                   downstream extractor handles the bad archive gracefully
sub=$1; shift
case "$sub" in
  run)
    dest=""
    while [[ $# -gt 0 ]]; do case "$1" in -D) dest=$2; shift ;; esac; shift; done
    mkdir -p "$dest/test_reports_u1" "$dest/test_reports_u2"
    echo "<testsuite/>" > "$dest/test_reports_u1/r.xml"
    echo "<testsuite/>" > "$dest/test_reports_u2/r.xml"
    exit 0
    ;;
  api)
    path=""
    for a in "$@"; do [[ "$a" == /* ]] && path=$a; done
    case "$path" in
      */attempts/*/jobs)
        echo '{"total_count":2,"jobs":[{"id":1,"conclusion":"skipped","name":"a"},{"id":2,"conclusion":"skipped","name":"b"}]}'
        ;;
      */attempts/*/logs)
        echo "not-a-real-zip"
        ;;
      *)
        echo '{}'
        ;;
    esac
    exit 0
    ;;
  *)
    exit 0
    ;;
esac
"""


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq required by the collection script")
def test_first_attempt_counts_against_stub_gh(tmp_path):
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    gh = stub_dir / "gh"
    gh.write_text(_GH_STUB)
    gh.chmod(0o755)

    env = {**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}"}
    r = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--repo",
            "tenstorrent/tt-metal",
            "--workflow-run-id",
            "999",
            "--attempt-number",
            "1",
            "--workflow-name",
            "Sanity tests",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    f = _api_usage_fields(r.stdout)
    # A workflow name with a space round-trips intact thanks to the quoting.
    assert f["workflow"] == "Sanity tests", r.stdout
    # two report artifacts (one internal listing), one jobs page, one log-archive request,
    # no per-job logs or annotations (both jobs skipped), no re-run attempt-meta fetch.
    assert f["artifact_download"] == "2", r.stdout
    assert f["artifact_list"] == "1", r.stdout
    assert f["jobs_list"] == "1", r.stdout
    assert f["log_archive"] == "1", r.stdout
    assert f["per_job_log"] == "0", r.stdout
    assert f["annotations"] == "0", r.stdout
    assert f["attempt_meta"] == "0", r.stdout
    assert f["total"] == str(2 + 1 + 1 + 1), r.stdout


# A partial/rate-limited download: gh run download extracts one artifact, then exits non-zero.
_GH_STUB_PARTIAL_DOWNLOAD = r"""#!/bin/bash
sub=$1; shift
case "$sub" in
  run)
    dest=""
    while [[ $# -gt 0 ]]; do case "$1" in -D) dest=$2; shift ;; esac; shift; done
    mkdir -p "$dest/test_reports_u1"
    echo "<testsuite/>" > "$dest/test_reports_u1/r.xml"
    echo "error: API rate limit exceeded" >&2   # one artifact done, then a failing request
    exit 1
    ;;
  api)
    path=""
    for a in "$@"; do [[ "$a" == /* ]] && path=$a; done
    case "$path" in
      */attempts/*/jobs) echo '{"total_count":1,"jobs":[{"id":1,"conclusion":"skipped","name":"a"}]}' ;;
      */attempts/*/logs) echo "not-a-real-zip" ;;
      *) echo '{}' ;;
    esac
    exit 0
    ;;
  *) exit 0 ;;
esac
"""


@pytest.mark.skipif(shutil.which("jq") is None, reason="jq required by the collection script")
def test_partial_download_failure_counts_the_failed_request(tmp_path):
    # A rate-limited download that extracts 1 artifact then fails must count BOTH the succeeded
    # ZIP (the directory) and the request that failed -- 2, not 1 -- so throttled collections
    # are not undercounted.
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    gh = stub_dir / "gh"
    gh.write_text(_GH_STUB_PARTIAL_DOWNLOAD)
    gh.chmod(0o755)

    env = {**os.environ, "PATH": f"{stub_dir}:{os.environ['PATH']}"}
    r = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--repo",
            "tenstorrent/tt-metal",
            "--workflow-run-id",
            "999",
            "--attempt-number",
            "1",
            "--workflow-name",
            "X",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    f = _api_usage_fields(r.stdout)
    assert f["artifact_download"] == "2", r.stdout  # 1 downloaded dir + 1 failed request
