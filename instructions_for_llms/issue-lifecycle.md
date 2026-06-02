# CI Issue Lifecycle Protocol

This protocol replaces `tools/ci/lifecycle.py` from `tenstorrent/tt-auto-triage`.
Execute it when asked to run the issue lifecycle on `CI auto triage` issues in `tenstorrent/tt-metal`.

---

## Overview

For each open issue labelled `CI auto triage` in `tenstorrent/tt-metal`:

1. Parse the issue to find the tracked workflow/job
2. Gate on having enough new data since the issue was last modified
3. Determine the current job status via the GitHub Actions API
4. If still failing: update the "Failing job URLs" section in the issue body
5. Analyze logs to decide: keep open / post update comment / close
6. **Always dry-run first** — present decisions to the triage channel C0B1F8Z7GQM before executing any closes

---

## Step 0 — Load All Issues

Fetch all open issues with label `CI auto triage`:

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/issues?state=open&labels=CI+auto+triage&per_page=100"
```

Repeat with `page=2`, `page=3`, etc. until fewer than 100 results are returned.

---

## Step 1 — Parse Each Issue

From the issue body, extract these hidden marker fields:

```
Auto-triage-workflow:       <workflow display name>
Auto-triage-job-name:       <exact job name as shown in GitHub Actions>
Auto-triage-workflow-file:  <optional: .github/workflows/foo.yaml>
```

Use regex: `Auto-triage-workflow:\s*\`?([^\`\n]+)\`?`

**Skip the issue entirely if:**
- The issue has the label `do-not-auto-close`
- The markers are absent (issue is too old or manually created)

Record: `issue_number`, `workflow_name`, `job_name`, `workflow_file` (may be empty), `created_at`.

---

## Step 2 — Dynamic Threshold

Compute how many consecutive successful/failing runs are required before acting.

**Get runs per day** for this workflow (last 7 days):

```bash
# Resolve workflow filename
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/workflows?per_page=100" \
  | python3 -c "import sys,json; [print(w['path'].split('/')[-1]) for w in json.load(sys.stdin)['workflows'] if w['name'].lower() == '<workflow_name>'.lower()]"

# Then fetch recent completed runs
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/workflows/<file>/runs?status=completed&per_page=35"
```

Count runs in the last 7 days, divide by 7 → `runs_per_day`.

**Threshold:**
- `runs_per_day >= 5.0` → threshold = **5**
- `runs_per_day < 5.0`  → threshold = **2**

---

## Step 3 — Gate: Enough New Runs Since Issue Creation?

Count completed workflow runs that occurred **after** `issue.updated_at` (last modified time).

If `new_runs_since_last_modified < threshold`:
→ **Skip** this issue with reason `"only N/threshold new runs since issue was last modified"`

---

## Step 4 — Determine Job Status

Fetch the last `threshold` completed runs for this workflow and find conclusions for `job_name`:

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/workflows/<file>/runs?status=completed&per_page=<threshold>"
# For each run_id, fetch jobs:
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/runs/<run_id>/jobs?per_page=100"
# Find job where name == job_name, record .conclusion
```

**Decision table** (from the `threshold` most recent conclusions):

| Conclusions | Status |
|---|---|
| All `"failure"` | `STILL_FAILING` |
| All `"success"` | `RESOLVED` |
| Any `"skipped"` | `SKIPPED` |
| Mixed (some pass, some fail) | `STILL_FAILING` (conservative) |
| Job absent from all runs → check YAML | see below |

**If the job produced zero conclusions in recent runs**, fetch the workflow YAML and search for `job_name` as a substring:
- Found (even in a comment) → `DISABLED`
- Not found anywhere → `REMOVED`

**For `DISABLED`, `SKIPPED`, `UNKNOWN`:** Keep open, do not analyze further.

---

## Step 5 — Update Failing Job URLs (STILL_FAILING only)

If status is `STILL_FAILING`, find the `threshold` most recent failing job URLs (from Step 4 job fetches) and update the issue body.

The issue body contains a section like:
```
### Failing job URLs (last N runs)
- https://github.com/tenstorrent/tt-metal/actions/runs/.../jobs/...
```

Replace it with the current failing job URLs. Use the GitHub API to edit the issue body:
```bash
curl -s -X PATCH -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/issues/<number>" \
  -d '{"body": "<updated body>"}'
```

Then post a comment: `"Failing job URLs updated. Last updated: YYYY-MM-DD HH:MM UTC"`

**If the section is not found in the body:** skip the update (old issue format).
**If the URLs are unchanged:** skip the update (no-op).

---

## Step 6 — Analyze and Decide

This is the judgment step. For each status, follow the rules below.

### STILL_FAILING

Download failed job logs for the `threshold` most recent failing runs:

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -L "https://api.github.com/repos/tenstorrent/tt-metal/actions/jobs/<job_id>/logs"
```

Fetch the last ~200 lines of each log (logs can be huge — focus on the end where errors appear).

Compare the **current error** against the **original error** described in the issue body (the "Error excerpt" and "Error signature" fields).

**Decision rules:**

| Scenario | Action |
|---|---|
| Same root cause (same test, same assertion type, same subsystem) | Keep open, no comment |
| Same category of failure, minor symptom variation (e.g. same assertion type but different threshold/token/position) | Keep open, post update comment noting the symptom change and linking new run URLs |
| Completely different root cause (different test name, different assertion type, different subsystem, or failure mode fundamentally changed e.g. was OOM, now is a hang) | **CLOSE** — but only if all 3 conditions hold (see below) |

**The 3 conditions required to close a STILL_FAILING issue:**

**Condition A — Consistently different:** The new failure must show a clearly different root cause in ALL `threshold` logs. If any run shows the original error, something ambiguous, or an infra failure → keep open.

**Condition B — Not infra noise:** The new failure must be caused by test/code, not CI infrastructure. Infrastructure failures that block closing:
- Network disconnects, connection refused, socket timeouts
- Runner disconnected / lost connection / shutdown signal
- Docker pull failures, runner setup failures
- SSH disconnection during test
- OOM from runner itself (not test-induced)
- Azure/AWS/GitHub Actions internal errors
- Timeout waiting for a runner or hardware

**Condition C — Confident:** If uncertain whether old and new failure are related → keep open. Default to conservative.

### RESOLVED

Download logs from the `threshold` most recent **passing** runs. Verify:
- Did the test genuinely run and pass (not silently skipped or removed)?
- Was there meaningful test output showing assertions actually executed?

If genuinely passing: **CLOSE**
If the job "passes" by skipping all assertions or never running: **Keep open**, explain.

### REMOVED

Fetch the current workflow YAML (from main branch). Confirm `job_name` is absent.

If removal looks intentional (job was clearly erroneous, too flaky, or explicitly deleted in a PR): **CLOSE**
If no clear rationale is visible: **Keep open**

---

## Step 7 — Comment Formats

When closing, post this comment before closing the issue:

```markdown
## Auto-triage lifecycle close

**Original failure:** <one-sentence summary from the issue body>

**Reason for closing:** <one of:>
- The job has passed consistently across the last N runs. <evidence: quote log lines or link runs>
- The failure has changed to a completely different root cause: <describe new failure, link logs>
- The job has been removed from the workflow. <confirm removal>

**Fix PR (if identified):** <URL or "not identified">

*Closed by BrAIn lifecycle review.*
```

### Update Action (signature same, keep open)

Post a comment:

```markdown
Issue still relevant. Last updated: YYYY-MM-DD HH:MM UTC.
```

### Update Action (signature changed, keep open)

When posting an update:

1. **Edit the issue body** to replace the "Failing job URLs" section with the new failing run URLs (same mechanism as Step 5).

2. **Post a comment** to record that the update happened:

```markdown
## Auto-triage lifecycle update

**Original failure:** <brief summary from issue body>

**Current symptom change:** <describe how the current failure differs symptomatically>

Issue body updated with new failing run URLs at YYYY-MM-DD HH:MM UTC.

*Updated by BrAIn lifecycle review.*
```

---

## Step 8 — Dry-Run Output Format

Before executing any closes or updates, produce a summary table:

```
Issue    | Workflow / Job                          | Status          | Decision
---------|----------------------------------------|-----------------|------------------
#45850   | galaxy-perf-tests / Galaxy DiT Qwen    | STILL_FAILING   | keep open (same root cause)
#45740   | fabric-tests / Fabric CCL              | STILL_FAILING   | UPDATE (symptom changed)
#45736   | models-t3000 / DeepSeek vLLM           | RESOLVED        | CLOSE (genuinely passing)
#45723   | fabric-perf / SingleSender             | STILL_FAILING   | CLOSE (different root cause)
```

With per-close reasoning:
- For each `CLOSE`: quote the key evidence (log snippet or YAML excerpt) that triggered it
- For each `UPDATE`: describe the symptom change
- For skipped/kept: brief reason

**Present this to the triage channel C0B1F8Z7GQM before executing anything. Wait for approval.**

---

## Step 9 — Execute (after approval)

For each approved close:
1. POST the closing comment
2. PATCH the issue state to `closed`

For each approved update:
1. POST the update comment

For each approved URL update (from Step 5):
1. PATCH the issue body

---

## Key Conservative Rules

- **When in doubt, keep open.** Wrong opens are annoying. Wrong closes lose signal.
- **Never close on infra noise.** If a log shows runner crash/network error, ignore that run.
- **Never close a mixed-conclusion issue.** Require ALL `threshold` logs to agree.
- **Never guess fix PRs.** Only provide a URL if you found it literally in the log evidence.
- **Respect `do-not-auto-close`.** If the label is present, skip the issue entirely.
- **Minimum data.** If fewer than `threshold` runs have occurred since the issue was last modified, skip it.

---

## API Rate Limiting

- The GitHub API allows 5000 requests/hour with a token.
- With many issues × many runs × log fetches, this can get tight. Add `sleep 0.2` between job-list calls.
- Log fetches are the most expensive. Fetch the last 300 lines only when possible by using byte-range or grepping the first error pattern after download.
- If rate limited (HTTP 403/429), stop and report how far you got.
