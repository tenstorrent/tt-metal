# Bug Escape Verification — Verify Agent Runbook

**Role:** You are an Opus verification subagent. You receive a single finding from the find agent and your job is to prove or refute it by dispatching GitHub Actions probe runs, monitoring them, and returning a structured verdict. You write to state files and ARE allowed to dispatch workflow runs and create/delete branches via the GitHub API.

---

## What You Can Do

- Dispatch GitHub Actions workflow runs (workflow_dispatch)
- Create and delete branches via GitHub API
- Modify workflow files on branches via GitHub Contents API
- Cancel in-progress runs (only if you dispatched them and they need to be replaced)
- Write to campaign-state.json, confirmed-escapes.json, seen-escapes.json
- Read any GitHub API endpoint, CI logs, commits, artifacts

## What You Must NOT Do

- Use `git` CLI to commit or push (use GitHub Contents API instead)
- Modify files on `main` or any non-probe branch
- Cancel runs you did not dispatch
- Update Confluence (main BrAIn does that based on your verdict)
- DM @ebanerjeeTT (main BrAIn does that)

---

## Inputs

You receive a finding JSON from the find agent with these fields (at minimum):

```json
{
  "escape_id": "...",
  "escape_type": "vertical|horizontal",
  "test_name": "...",
  "test_filepath": "...",
  "test_layer": 3,
  "fix_layer": 3,
  "fix_commit_sha": "...",
  "fix_commit_message": "...",
  "candidate_fix_commits": ["..."],
  "fix_pr": "...",
  "action_required": "probe_verification|pr_ci_proof_check",
  "workflow_id": "...",
  "workflow_name": "...",
  "hardware": "N300",
  "architecture": "wormhole_b0",
  "last_failure_run_id": "...",
  "last_failure_job_id": "...",
  "first_success_run_id": "...",
  "first_success_job_id": "..."
}
```

Also passed: paths to state files.

---

## Step 0: Guard Checks

Before doing anything:

1. Read `seen-escapes.json`. If `escape_id` is present, return immediately:
   ```json
   {"verdict": "already_processed", "escape_id": "...", "reason": "found in seen-escapes.json"}
   ```
2. Read `campaign-state.json`. If a candidate with this `escape_id` already has `status: "awaiting_probes"` with non-null `active_runs.before_run_id` or `active_runs.after_run_id`: those probes are already in flight. Skip to Step 5 (Poll) using those existing run IDs.
3. Write to campaign-state.json: set candidate `status: "awaiting_probes"`.

---

## Step 1: PR-Linked CI Proof Check (if action_required == "pr_ci_proof_check")

If the finding has `action_required: "pr_ci_proof_check"`:

```bash
python3 /workspace/group/bug-escapes-data/verify-pr-proof.py --pr {fix_pr} --test {test_name}
```

- Exit 0 (CONFIRMED): record `confirmation_method: "pr_ci_proof"`, `confidence: "proven"`. You still need BEFORE=FAIL evidence. Check if existing nightly CI logs at `last_failure_run_id` cover the test. If yes, that is your BEFORE=FAIL. Proceed to Step 6 (Write Results) with `before_run_id = last_failure_run_id`, `after_run_id = first_success_run_id`, `verdict: "confirmed"`.
- Exit 1 or 2: fall through to Step 2 (probe dispatch).

---

## Step 2: Determine Parent SHA

Get the parent of the fix commit (the BEFORE point):

```bash
gh api repos/tenstorrent/tt-metal/commits/{fix_commit_sha} --jq '.parents[0].sha'
```

Store as `before_sha`. The `after_sha` is `fix_commit_sha`.

---

## Step 3: Artifact Reuse Check

### 3a: Determine tracy requirement for this workflow

First, check whether the workflow uses tracy:

```bash
gh api "repos/tenstorrent/tt-metal/contents/.github/workflows/{workflow_file}?ref=main" \
  --jq '.content' | base64 -d | grep "tracy:" | head -5
```

Record `needs_tracy: true` if `tracy: true` appears under the build-artifact `with:` block, `false` otherwise.

### 3b: Find a compatible run for each SHA

The source run must have produced artifacts matching the tracy requirement:

Merge Gate builds **only with tracy** (hardcoded `tracy: true` in `build-wrapper.yaml`). It does NOT produce non-tracy artifacts.

- **`needs_tracy: true`**: Search for `Merge Gate` runs — compatible ✅
- **`needs_tracy: false`**: Merge Gate artifacts will NOT match. Search for another run on that SHA that built without tracy (e.g. a workflow that explicitly sets `tracy: false`).

```bash
gh api "repos/tenstorrent/tt-metal/actions/runs?head_sha={sha}&per_page=100" \
  --jq '.workflow_runs[] | select(.conclusion == "success") | {id, name, created_at}'
```

For `needs_tracy: true`: filter to `name == "Merge Gate"`.
For `needs_tracy: false`: look for other successful runs and check their artifact names for the absence of tracy.

### 3c: Verify artifacts exist and are not expired

```bash
gh api "repos/tenstorrent/tt-metal/actions/runs/{source_run_id}/artifacts?per_page=30" \
  --jq '.artifacts[] | {name, expired, size_in_bytes}'
```

For `needs_tracy: false`, you need ALL of these present and `expired: false`:
- `TTMetal_build_any_22.04_amd64_*_{sha}_*` (main build)
- `packages-ubuntu-22.04-amd64-Release-*_{sha}_*` (packages)

For `needs_tracy: true`, look for artifacts with `tracy` or `profiler` in the name in addition to the above.

If no compatible run with non-expired artifacts is found: **no artifact reuse** for that SHA — it will build from scratch.

Record:
- `before_reuse_run_id`: compatible run ID for `before_sha` (or null)
- `after_reuse_run_id`: compatible run ID for `after_sha` (or null)
- `needs_tracy`: true/false (same for both probes — it's a property of the workflow)

---

## Step 4: Create Probe Branches

```bash
# BEFORE branch at before_sha
gh api repos/tenstorrent/tt-metal/git/refs -X POST \
  -f ref="refs/heads/brain/escape-before-{escape_id}" \
  -f sha="{before_sha}"

# AFTER branch at after_sha
gh api repos/tenstorrent/tt-metal/git/refs -X POST \
  -f ref="refs/heads/brain/escape-after-{escape_id}" \
  -f sha="{after_sha}"
```

If a branch already exists (409 conflict): delete it first:
```bash
gh api repos/tenstorrent/tt-metal/git/refs/heads/brain/escape-before-{escape_id} -X DELETE
```
Then recreate.

---

## Step 5: Determine Dispatch Inputs

### 5a: Identify the correct workflow file

Use `workflow_id` from the finding. Get the workflow filename:
```bash
gh api repos/tenstorrent/tt-metal/actions/workflows/{workflow_id} --jq '.path'
```

This gives you e.g. `.github/workflows/tt-metal-l2-nightly.yaml`.

### 5b: Identify the correct test category flag by reading the workflow

Do NOT rely on a hardcoded mapping. Derive the flag from the actual workflow and its impl file.

**Step 1: Read the workflow dispatch inputs and job conditions:**
```bash
gh api "repos/tenstorrent/tt-metal/contents/.github/workflows/{workflow_file}?ref=main" \
  --jq '.content' | base64 -d > /tmp/workflow_main.yaml
```

Look for `workflow_dispatch` inputs (boolean flags like `run_sd_unit_tests`, `run_cpp_tests`) and string inputs like `additional_test_categories`. Note what values trigger which jobs.

**Step 2: Find the impl file if there is one** (e.g. `tt-metal-l2-nightly-impl.yaml`):
```bash
grep "uses:.*impl" /tmp/workflow_main.yaml
```

Read the impl file and search for `test_filepath`'s directory name or filename:
```bash
gh api "repos/tenstorrent/tt-metal/contents/.github/workflows/{impl_file}?ref=main" \
  --jq '.content' | base64 -d | grep -B5 -A5 "{directory_name}"
```

This tells you which job block covers the target test path, and which input flag enables it.

**⚠️ NEVER use `run_sd_unit_tests: true` for Python pytest files.** That flag runs a C++ gtest binary (`llk-sd-unit-tests` job). Python pytest files need `additional_test_categories` or a different boolean flag.

**If after reading both the workflow and impl file you still cannot confidently map `test_filepath` to a flag:** abort and return:
```json
{"verdict": "aborted_wrong_test", "escape_id": "...", "reason": "could not map test_filepath to workflow flag", "test_filepath": "..."}
```
Do NOT guess.

### 5c: Set architecture

Use `architecture` from the finding:
- `wormhole_b0` → pass `"[\"wormhole_b0\"]"` as architecture input
- `blackhole` → pass `"[\"blackhole\"]"`

### 5d: Build final dispatch inputs

```json
{
  "architecture": "[\"wormhole_b0\"]",
  "additional_test_categories": "data_movement"
}
```

Or for C++ LLK:
```json
{
  "architecture": "[\"blackhole\"]",
  "run_sd_unit_tests": "true"
}
```

Artifact reuse is configured in the workflow file (Step 6), not via dispatch inputs. Do not pass `use-artifacts-from-run` as a dispatch input.

---

## Step 6: Modify Workflow for Artifact Reuse (if applicable)

If artifact reuse is available for a branch, modify the workflow file ON THAT BRANCH with three pure-YAML changes. No `workflow_dispatch` input is needed — the run ID is hardcoded directly.

### 6a: Read current workflow file from branch

```bash
gh api "repos/tenstorrent/tt-metal/contents/.github/workflows/{workflow_file}?ref=brain/escape-before-{escape_id}" \
  --jq '{sha: .sha, content: .content}' > /tmp/workflow_info.json

FILE_SHA=$(cat /tmp/workflow_info.json | jq -r .sha)
cat /tmp/workflow_info.json | jq -r .content | base64 -d > /tmp/workflow.yaml
```

### 6b: Apply the three YAML changes

```bash
# 1. Add actions: read permission to the build-artifact job
sed -i 's/    uses: .\/\.github\/workflows\/build-artifact\.yaml\n    permissions:\n      contents: read\n      packages: write/    uses: .\/\.github\/workflows\/build-artifact\.yaml\n    permissions:\n      contents: read\n      packages: write\n      actions: read/' /tmp/workflow.yaml

# Use Python for the multi-line sed (simpler):
python3 -c "
content = open('/tmp/workflow.yaml').read()

# 1. Add 'actions: read' under the build-artifact permissions block
content = content.replace(
    '    uses: ./.github/workflows/build-artifact.yaml\n    permissions:\n      contents: read\n      packages: write',
    '    uses: ./.github/workflows/build-artifact.yaml\n    permissions:\n      contents: read\n      packages: write\n      actions: read'
)

# 2. tracy setting must match the source run.
# Merge Gate only builds with tracy. If the workflow uses tracy: false, the
# source run was NOT Merge Gate and we don't need to change anything.
# If the workflow uses tracy: true and source is Merge Gate, leave as-is.
# In short: don't change the tracy line — it already matches the source run
# because Step 3 enforced compatibility.

# 3. Add use-artifacts-from-run directly under the with: block of build-artifact
#    Insert after 'enable-lto:' line (last line of the with: block)
content = content.replace(
    '      enable-lto: \${{ inputs.enable-lto || false }}',
    '      enable-lto: \${{ inputs.enable-lto || false }}\n      use-artifacts-from-run: {MERGE_GATE_RUN_ID}'
)

open('/tmp/workflow_modified.yaml', 'w').write(content)
"
```

Replace `{MERGE_GATE_RUN_ID}` with the actual run ID (e.g. `24989751483`).

Verify before pushing:
```bash
grep -E "actions: read|tracy:|use-artifacts-from-run" /tmp/workflow_modified.yaml
```

Expected output:
```
      actions: read
      tracy: false
      use-artifacts-from-run: 24989751483
```

### 6c: Push modified workflow to the branch

```bash
ENCODED=$(base64 -w 0 /tmp/workflow_modified.yaml)
gh api repos/tenstorrent/tt-metal/contents/.github/workflows/{workflow_file} \
  -X PUT \
  -f message="brain: configure artifact reuse for probe" \
  -f content="$ENCODED" \
  -f sha="$FILE_SHA" \
  -f branch="brain/escape-before-{escape_id}"
```

Repeat for the AFTER branch (read its file SHA separately — it may differ). Use `after_merge_gate_run_id` for the AFTER branch's `use-artifacts-from-run` value.

### 6d: Prune unnecessary jobs from the probe workflow

Probe runs should only execute the build job and the single test job needed. All other test jobs waste CI time and hardware.

In `/tmp/workflow_modified.yaml`, disable every test job EXCEPT the target one by adding `if: false` to each unneeded job. Identify jobs to disable by looking for jobs whose `needs:` includes `build-artifact` or `generate-arch-matrix` and whose name is not the target job.

```python
import yaml, re

with open('/tmp/workflow_modified.yaml') as f:
    content = f.read()

# Load to identify all test job names
data = yaml.safe_load(content)
target_job = "{job_name_containing_the_target_test}"  # e.g. "tt-metal-l2-tests"
keep_jobs = {"build-artifact", "generate-arch-matrix", target_job}

for job_name, job_body in data.get("jobs", {}).items():
    if job_name not in keep_jobs:
        # Insert `if: false` at the top of the job block
        content = re.sub(
            rf'(\n  {re.escape(job_name)}:\n)',
            rf'\1    if: false\n',
            content
        )

with open('/tmp/workflow_modified.yaml', 'w') as f:
    f.write(content)
```

Verify the result:
```bash
grep -E "^\s{2}\w|if: false" /tmp/workflow_modified.yaml | head -40
```

**Note:** When dispatching (Step 7), do NOT pass `use-artifacts-from-run` as a dispatch input — the run ID is already hardcoded in the workflow file. Just pass `architecture` and the test category flag.

---

## Step 7: Dispatch Probes

### Hardware rules

- **Single-card (N150, N300, P150, P100, P150b, P300):** BEFORE and AFTER can dispatch in parallel.
- **Multi-card (T3K, Galaxy):** Dispatch BEFORE only. Wait for it to complete before dispatching AFTER.

### Dispatch command

```bash
gh api repos/tenstorrent/tt-metal/actions/workflows/{workflow_file}/dispatches \
  -X POST --input - <<EOF
{
  "ref": "brain/escape-before-{escape_id}",
  "inputs": {dispatch_inputs_json}
}
EOF
```

Wait 10 seconds, then fetch the new run ID:
```bash
gh api "repos/tenstorrent/tt-metal/actions/workflows/{workflow_file}/runs?branch=brain/escape-before-{escape_id}&per_page=3" \
  --jq '.workflow_runs[0] | {id, status, html_url}'
```

Record `before_run_id` and `after_run_id`. Write to campaign-state.json immediately.

**Include probe URLs in your verdict JSON** (Step 13) so main BrAIn can DM them to @ebanerjeeTT. The `before_run_url` and `after_run_url` fields are mandatory in the verdict.

---

## Step 8: Post-Dispatch Verification (MANDATORY — do NOT skip)

After dispatching, poll until you can confirm the probe is set up correctly. Poll every 60 seconds for up to 15 minutes.

### 8a: Verify build-artifact step succeeded

Check the `build-artifact / download-artifacts` job (if artifact reuse was used) or `build-artifact / 🛠️ Build ...` job (if building from scratch):

```bash
gh api "repos/tenstorrent/tt-metal/actions/runs/{before_run_id}/jobs?per_page=30" \
  --jq '.jobs[] | select(.name | startswith("build-artifact")) | {name, status, conclusion}'
```

**If `download-artifacts` conclusion is `failure`:**
- Read the job logs: `gh api repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs`
- Common causes:
  - `tracy=true was requested` → you forgot `tracy: false` in the workflow modification. Fix and re-dispatch.
  - `No artifact found matching` → wrong Merge Gate run ID, or artifact expired. Fall back to no artifact reuse.
- Do NOT proceed until download-artifacts is `success` (or skipped if building from scratch).

**If building from scratch:** wait until `🛠️ Build` job is `success`. This takes ~30 minutes.

### 8b: Verify the target test job is queued or running (NOT skipped)

After build succeeds, check that the job category you targeted is actually running:

```bash
gh api "repos/tenstorrent/tt-metal/actions/runs/{before_run_id}/jobs?per_page=50" \
  --jq '.jobs[] | select(.status != "completed" or .conclusion != "skipped") | {name, status, conclusion}' | head -30
```

Look for a job whose name includes the test category (e.g. `data_movement`, `sd-unit-tests`, `llk-sd-unit-tests`). If the expected job appears with `status: "in_progress"` or `status: "queued"` — good.

**If the expected job is `skipped`:**
- This means the dispatch inputs were wrong. The test category flag did not trigger the job.
- Read the workflow file to find which flag would trigger the correct job.
- Cancel the run: `gh api repos/tenstorrent/tt-metal/actions/runs/{run_id}/cancel -X POST`
- Correct the dispatch inputs and re-dispatch. Return to Step 7.
- If after two attempts you cannot determine the correct flag: abort and return `"verdict": "aborted_wrong_test"`.

**If both build succeeded and target job is running or queued:** write to campaign-state.json with confirmed run IDs and proceed to Step 9.

---

## Step 9: Poll for Completion

Poll every 5 minutes until both BEFORE and AFTER runs are `completed` (or until 3 hours pass, whichever is first).

```bash
gh api repos/tenstorrent/tt-metal/actions/runs/{run_id} --jq '{status, conclusion}'
```

For multi-card hardware: dispatch AFTER after BEFORE completes.

**Timeout:** If either run is still in_progress after 3 hours, write `status: "inconclusive_timeout"` to campaign-state.json and return:
```json
{"verdict": "inconclusive_timeout", "escape_id": "...", "before_run_id": "...", "after_run_id": "..."}
```

---

## Step 10: Read Test Results

For each completed run, find the job that ran the target test and read its conclusion:

```bash
# Find the relevant job
gh api "repos/tenstorrent/tt-metal/actions/runs/{run_id}/jobs?per_page=50" \
  --jq '.jobs[] | select(.name | test("data_movement|sd-unit-tests|llk-sd"; "i")) | {id, name, conclusion}'

# Read the log for the specific test
gh api repos/tenstorrent/tt-metal/actions/jobs/{job_id}/logs | grep -i "{test_name}" | tail -20
```

Look for explicit `PASSED`, `FAILED`, `[OK]`, `FAILED` in the log for the test name. A job-level `conclusion: "failure"` does NOT necessarily mean the target test failed — other tests in the same job may have failed. Find the specific line.

**Known unrelated failures to ignore:**
- `MeshDeviceFixture.Top32RmDevPipelineCompletes` — tracked separately, not relevant to escape verdict

Record:
- `before_result`: `"FAIL"` | `"PASS"` | `"not_found"`
- `after_result`: `"FAIL"` | `"PASS"` | `"not_found"`

If `not_found` for either: the test did not run. This is a dispatch input error. Return `"verdict": "aborted_wrong_test"` with details.

---

## Step 11: Apply Verdict Logic

| BEFORE result | AFTER result | Verdict |
|---|---|---|
| FAIL | PASS | `confirmed` ✅ (`confirmation_method: "bisect"`, `confidence: "proven"`) |
| FAIL | FAIL | `refuted` (fix didn't address this test — exhausted all candidate commits) |
| PASS | anything | See BEFORE=PASS rule below |
| not_found | anything | `aborted_wrong_test` |
| anything | not_found | `aborted_wrong_test` |

**BEFORE=PASS rule:** When the test passes at the parent of the assumed fix commit, the fix hypothesis is wrong. Before returning a verdict:

1. Check `candidate_fix_commits` in the finding — are there other commits that haven't been probed yet?
2. If yes: update campaign-state.json with the new fix hypothesis (next candidate commit), return `refuted_wrong_fix`. Main BrAIn will re-spawn you with the next candidate.
3. If all `candidate_fix_commits` have been tried and BEFORE=PASS for all of them: return `refuted`. The transition is real but the fix is not in any identified commit — likely an infra change or firmware update outside the visible range.

**`refuted` should only be returned after exhausting all candidate commits.** `refuted_wrong_fix` means "wrong hypothesis, more to try."

---

## Step 12: Write Results to State Files

### If verdict == "confirmed":

Append to `confirmed-escapes.json`:
```json
{
  "escape_id": "...",
  "escape_type": "vertical|horizontal",
  "test_name": "...",
  "test_filepath": "...",
  "test_layer": ...,
  "fix_commit_sha": "...",
  "fix_commit_message": "...",
  "fix_layer": ...,
  "candidate_fix_commits": [...],
  "fix_pr": "...",
  "confirmation_method": "bisect",
  "confidence": "proven",
  "last_failure_run_id": "...",
  "last_failure_job_id": "...",
  "first_success_run_id": "...",
  "first_success_job_id": "...",
  "before_run_id": "...",
  "after_run_id": "...",
  "reasoning": "BEFORE={before_sha} FAIL, AFTER={after_sha} PASS, fix commit {fix_commit_sha}",
  "confirmed_at": "..."
}
```

Write escape_id to `seen-escapes.json` with `status: "confirmed"`.
Update `campaign-state.json`: candidate status = `"confirmed"`, `confluence_updated: false`, `dm_sent: false`.

### If verdict == "refuted_wrong_fix":

Write escape_id to `seen-escapes.json` with `status: "refuted_wrong_fix"`, including `tried_commits: [list of all fix_commit_shas attempted so far]`.
Update candidate in `campaign-state.json` to `status: "refuted_wrong_fix"`, set `next_fix_candidate` to the next untried commit from `candidate_fix_commits`.
Do NOT delete probe branches yet (main BrAIn may re-spawn you).

### If verdict == "refuted":

Write escape_id to `seen-escapes.json` with `status: "refuted"`, `refute_reason: "all candidate commits exhausted"`.
Update candidate in `campaign-state.json` to `status: "refuted"`.
Delete probe branches.

### Always:

Delete both probe branches:
```bash
gh api repos/tenstorrent/tt-metal/git/refs/heads/brain/escape-before-{escape_id} -X DELETE
gh api repos/tenstorrent/tt-metal/git/refs/heads/brain/escape-after-{escape_id} -X DELETE
```

---

## Step 13: Return Verdict JSON

Return a single JSON object (no prose, no markdown wrapping):

```json
{
  "verdict": "confirmed|refuted|refuted_wrong_fix|inconclusive_timeout|aborted_wrong_test|already_processed",
  "escape_id": "...",
  "escape_type": "vertical|horizontal",
  "test_name": "...",
  "fix_commit_sha": "...",
  "fix_pr": "...",
  "confirmation_method": "bisect|pr_ci_proof|null",
  "confidence": "proven|assumed|null",
  "before_run_id": "...",
  "after_run_id": "...",
  "before_run_url": "https://github.com/tenstorrent/tt-metal/actions/runs/...",
  "after_run_url": "https://github.com/tenstorrent/tt-metal/actions/runs/...",
  "before_result": "FAIL|PASS|not_found",
  "after_result": "FAIL|PASS|not_found",
  "artifact_reuse": true,
  "notes": "any important context for main BrAIn"
}
```

Main BrAIn uses this to update Confluence and DM @ebanerjeeTT.
