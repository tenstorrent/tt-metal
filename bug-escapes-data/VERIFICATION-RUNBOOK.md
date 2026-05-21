# Bug Escape Verification Runbook (AI-Driven)

**Version**: 1.0
**Status**: ⚠️ DEPRECATED — use RUNBOOK.md Steps 4–5 for all bisect and verification work
**Maintainer**: BrAIn

> This file describes an older verification procedure with different branch naming conventions
> (`verify-{sha}-before` / `verify-{sha}-after`) and a different YAML pruning approach.
> It conflicts with RUNBOOK.md. Do NOT follow this procedure.
> The authoritative procedure is in RUNBOOK.md Steps 4a, 4b, 4.5, 4.6, and 5.
> This file is kept for historical reference only.

---

## Overview

A *vertical bug escape* is confirmed when a test in a **higher** layer (e.g. models)
was failing due to a bug in a **lower** layer (e.g. tt-metalium), and:
- The test **failed** on the commit *before* the fix landed (**BEFORE**)
- The test **passed** on the commit *after* the fix landed (**AFTER**)

Layer hierarchy (lowest → highest): `tt-llk → tt-metalium → ttnn → models`

---

## Verification Procedure

### Step 0: Prerequisites

```bash
GH_TOKEN=$(env | grep GITHUB_TOKEN | cut -d= -f2)
OWNER_REPO="tenstorrent/tt-metal"
```

Confirm token has push access before starting:
```bash
curl -s -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/repos/$OWNER_REPO" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['permissions'])"
# Must include "push": true
```

### Step 1: Resolve BEFORE and AFTER commits

Given `fix_commit_sha` (the merge commit of the fix PR):

```bash
FIX_SHA="<fix_commit_sha>"

# AFTER = the fix commit itself
AFTER_SHA="$FIX_SHA"

# BEFORE = the first parent of the fix commit
BEFORE_SHA=$(curl -s -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/repos/$OWNER_REPO/commits/$FIX_SHA" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['parents'][0]['sha'])")
```

### Step 2: Create BEFORE and AFTER branches

Branch naming: `verify-<short_sha>-before` / `verify-<short_sha>-after`

```bash
SHORT_SHA="${FIX_SHA:0:8}"
BEFORE_BRANCH="verify-${SHORT_SHA}-before"
AFTER_BRANCH="verify-${SHORT_SHA}-after"

for branch in "$BEFORE_BRANCH $BEFORE_SHA" "$AFTER_BRANCH $AFTER_SHA"; do
  bname=$(echo $branch | cut -d' ' -f1)
  bsha=$(echo $branch | cut -d' ' -f2)
  curl -s -X POST \
    -H "Authorization: token $GH_TOKEN" \
    -H "Content-Type: application/json" \
    "https://api.github.com/repos/$OWNER_REPO/git/refs" \
    -d "{\"ref\": \"refs/heads/$bname\", \"sha\": \"$bsha\"}"
done
```

### Step 3: Prune the test matrix on each branch

Goal: modify the pipeline YAML on each branch so only the specific failing test runs.
This keeps hardware CI time short (~10-15 min instead of 2+ hours).

**3a. Discover the tests YAML path**

From the candidate's `workflow` field (e.g. `.github/workflows/blackhole-demo-tests.yaml`),
find the `TESTS_YAML_PATH` env var in the corresponding `-impl.yaml`:

```bash
WF_BASENAME=$(basename "$WORKFLOW" .yaml)   # e.g. blackhole-demo-tests
IMPL_FILE=".github/workflows/${WF_BASENAME}-impl.yaml"

# Fetch impl file from BEFORE_BRANCH (or main if not there)
TESTS_YAML_PATH=$(curl -s -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/repos/$OWNER_REPO/contents/$IMPL_FILE?ref=$BEFORE_BRANCH" \
  | python3 -c "
import sys, json, base64, re
d = json.load(sys.stdin)
content = base64.b64decode(d['content']).decode()
m = re.search(r'TESTS_YAML_PATH:\s*[\'\"]*([^\s\'\"]+)', content)
print(m.group(1) if m else '')
")
```

**3b. Find the matching test entry**

```bash
# Fetch the YAML, find the entry whose 'name' matches TEST_JOB
# Use substring matching: strip [runner-label] suffixes added by GitHub Actions
python3 - <<PYEOF
import json, base64, yaml, sys, subprocess, os

token = os.environ.get('GH_TOKEN') or subprocess.check_output(
    "env | grep GITHUB_TOKEN | cut -d= -f2", shell=True).decode().strip()
owner_repo = "$OWNER_REPO"
tests_yaml_path = "$TESTS_YAML_PATH"
branch = "$BEFORE_BRANCH"
test_job = "$TEST_JOB"

import urllib.request
req = urllib.request.Request(
    f"https://api.github.com/repos/{owner_repo}/contents/{tests_yaml_path}?ref={branch}",
    headers={"Authorization": f"token {token}"})
with urllib.request.urlopen(req) as r:
    d = json.load(r)
content = base64.b64decode(d['content']).decode()
tests = yaml.safe_load(content)

# Match: exact → strip runner label → substring
def find_entry(tests, target):
    for t in tests:
        if t.get('name','') == target: return t
    stripped = target.rsplit(' [', 1)[0]  # strip [bh_llmbox] etc.
    for t in tests:
        if t.get('name','') == stripped: return t
    matches = [(len(t['name']), t) for t in tests
               if t.get('name','') in target or target in t.get('name','')]
    return sorted(matches)[-1][1] if matches else None

entry = find_entry(tests, test_job)
if entry:
    print(json.dumps(entry, indent=2))
else:
    print("NO MATCH", file=sys.stderr)
    sys.exit(1)
PYEOF
```

**3c. Build pruned YAML**

Create a single-entry list YAML:
- For pytest tests: replace the `cmd` field with just the specific test
- For job-level tests (no specific pytest path): keep entry unchanged (run the whole job)

```python
# pruned = [entry]   (single entry, possibly with cmd modified for pytest)
pruned_yaml = yaml.dump([entry], default_flow_style=False, allow_unicode=True)
```

**3d. Write pruned YAML to both branches via GitHub Contents API**

```bash
# For each branch, PUT the pruned YAML content
_put_file() {
  local branch="$1"
  local content_b64=$(echo "$PRUNED_YAML" | base64 -w0)

  # Get current SHA of the file on the branch
  local file_sha
  file_sha=$(curl -s -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$OWNER_REPO/contents/$TESTS_YAML_PATH?ref=$branch" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sha',''))" 2>/dev/null || echo "")

  local payload
  if [ -n "$file_sha" ]; then
    payload="{\"message\":\"verify: prune test matrix\",\"content\":\"$content_b64\",\"sha\":\"$file_sha\",\"branch\":\"$branch\"}"
  else
    payload="{\"message\":\"verify: create pruned test matrix\",\"content\":\"$content_b64\",\"branch\":\"$branch\"}"
  fi

  curl -s -X PUT \
    -H "Authorization: token $GH_TOKEN" \
    -H "Content-Type: application/json" \
    "https://api.github.com/repos/$OWNER_REPO/contents/$TESTS_YAML_PATH" \
    -d "$payload" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('OK', r.get('commit',{}).get('sha','?'))"
}
```

### Step 4: Dispatch workflow runs

Dispatch the main workflow (not the impl) on each branch, enabling only the relevant SKU:

```bash
_dispatch() {
  local branch="$1"
  curl -s -X POST \
    -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$OWNER_REPO/actions/workflows/$(basename $WORKFLOW)/dispatches" \
    -d "{\"ref\": \"$branch\"}"

  sleep 10
  # Get the run ID of the just-dispatched run
  curl -s -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$OWNER_REPO/actions/runs?branch=$branch&per_page=3" \
    | python3 -c "import sys,json; runs=json.load(sys.stdin)['workflow_runs']; print(runs[0]['id'] if runs else 'NONE')"
}

BEFORE_RUN_ID=$(_dispatch "$BEFORE_BRANCH")
AFTER_RUN_ID=$(_dispatch "$AFTER_BRANCH")
```

### Step 5: Poll for completion

Poll every 5 minutes. Do NOT busy-wait.

```bash
_check_run() {
  local run_id="$1"
  curl -s -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$OWNER_REPO/actions/runs/$run_id" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], d['conclusion'] or '')"
}
```

Continue polling until both are `completed`. Record start time — if either exceeds 90 min, treat as inconclusive-timeout.

### Step 6: Fetch logs and classify

For each run, get the job logs and classify:

```bash
_get_job_logs() {
  local run_id="$1" job_name_fragment="$2"
  local job_id
  job_id=$(curl -s -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$OWNER_REPO/actions/runs/$run_id/jobs" \
    | python3 -c "
import sys,json
jobs=json.load(sys.stdin)['jobs']
for j in jobs:
    if '$job_name_fragment' in j['name']:
        print(j['id']); break
")
  curl -sL -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$OWNER_REPO/actions/jobs/$job_id/logs"
}
```

**Classification logic** (apply to each run independently):

| Signal | Classification |
|--------|---------------|
| Run conclusion = `success` | PASS |
| Run conclusion = `failure` and logs contain test failure signature | FAIL (real) |
| Run conclusion = `failure` and logs show infra error (OOM, runner lost, device error) | FAIL (infra) |
| Run timed out / cancelled | INCONCLUSIVE |

**Verdict mapping:**

| BEFORE | AFTER | Verdict |
|--------|-------|---------|
| FAIL (real) | PASS | **CONFIRMED** ✅ |
| FAIL (real) | FAIL (real) | **REFUTED** — both fail |
| PASS | any | **REFUTED** — wasn't failing before fix |
| PASS | PASS | **REFUTED** — wasn't broken |
| FAIL (infra) | any | **INCONCLUSIVE** — infra noise on BEFORE |
| any | FAIL (infra) | **INCONCLUSIVE** — infra noise on AFTER |
| INCONCLUSIVE | any | **INCONCLUSIVE** |

### Step 7: Record result and cleanup

**On CONFIRMED**: append to `/workspace/group/bug-escapes-data/master-ledger.jsonl`

```json
{
  "rank": N,
  "fix_commit_sha": "...",
  "fix_pr_number": 43XXX,
  "fix_pr_title": "...",
  "fix_layer": "tt-metalium",
  "test_layer": "models",
  "workflow": ".github/workflows/...",
  "job": "...",
  "test_name": "...",
  "before_run_id": ...,
  "after_run_id": ...,
  "verdict": "confirmed",
  "verified_at": "2026-05-13T..."
}
```

**Update** `/workspace/group/bug-escapes-data/verification-queue.json` with verdict + run IDs.

**Cleanup branches** (always, even on inconclusive):
```bash
for branch in "$BEFORE_BRANCH" "$AFTER_BRANCH"; do
  curl -s -X DELETE \
    -H "Authorization: token $GH_TOKEN" \
    "https://api.github.com/repos/$OWNER_REPO/git/refs/heads/$branch"
done
```

---

## Edge Cases

### Test name has runner label suffix `[bh_llmbox]`
Strip the `[...]` suffix when matching against YAML entry names.

### Tests YAML doesn't exist on the branch
This can happen when the experimental branch diverges from main. Fallback: fetch from `main` instead and use that as the pruned base.

### No specific pytest path (job-level test)
When `TEST_NAME` equals `TEST_JOB` (no `::` separator), the whole job is the test.
Don't try to modify the `cmd` field — just prune to a single-entry list.

### Multiple candidates for same workflow
Each verification is independent. Don't share branches between candidates.

### Infra failure on one side
Retry once. If it fails again, mark inconclusive and move on.

---

## Candidate Queue

Source: `/workspace/group/bug-escapes-data/verification-queue.json`

Process in rank order. One verification at a time — wait for verdict before dispatching next.

Statuses:
- `pending` — not started
- `running` — dispatched, waiting for completion
- `confirmed` / `refuted` / `inconclusive` — done

---

## Output Format (channel message)

```
Rank #N verified: <VERDICT>

Fix: PR#XXXXX — <title>
Fix layer: tt-metalium → Test layer: models
BEFORE (<short_sha>): FAIL — <failure signature snippet>
  Job: https://github.com/tenstorrent/tt-metal/actions/runs/<before_run_id>/job/<before_job_id>
AFTER  (<short_sha>): PASS
  Job: https://github.com/tenstorrent/tt-metal/actions/runs/<after_run_id>/job/<after_job_id>
```

**⚠️ Always link to job-level URLs** (`/runs/{run_id}/job/{job_id}`), not run-level.
Get job IDs via `GET /repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs`, match on job name.
Never output bare run IDs or job IDs — always embed them in a full GitHub URL.

---

## Notes

- Keep verifications strictly sequential (1 at a time)
- Max acceptable wall time per verification: 90 min
- If a run is queued for >30 min without starting, check hardware availability
- Log all verdicts to master-ledger.jsonl regardless of outcome (confirmed only for confirmed escapes)
