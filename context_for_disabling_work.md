# Context for CI Pipeline Disable Work — t3000-demo-tests

## Current Status (as of 2026-05-24)

**PR #44938** — https://github.com/tenstorrent/tt-metal/pull/44938
**Tracking issue #44937** — https://github.com/tenstorrent/tt-metal/issues/44937
**Branch:** `ci/disable-failing-tests-t3000-demo-tests-20260521`
**Worktree on host:** `/workspace/group/worktrees/t3000-demo-disable/`

### PR is ready to merge
- Rebased on latest main (2026-05-24) and force-pushed ✅
- Verification run 26295163268 completed **SUCCESS** (2026-05-22T21:04 UTC) ✅
- `t3k_sd35_large_tests` still failing on main in 3/3 consecutive recent runs ✅
- GitHub REST API token is broken in current agent environment — cannot undraft/merge via API, but the code change and branch are correct

---

## What Was Done

### 1. Tracking issue created
Issue #44937: `[ci] Disable consistently failing tests in t3000-demo-tests`

### 2. Fix applied
**File:** `models/tt_dit/tests/models/sd35/test_pipeline_sd35.py`

Converted the first parametrize entry from list syntax to `pytest.param(...)` with a skip mark:

```python
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        pytest.param(
            (2, 4),
            (2, 1),
            (2, 0),
            (2, 1),
            ttnn.Topology.Linear,
            1,
            marks=pytest.mark.skip(reason="Disabled by issue #44937"),
        ),
        [(2, 4), (2, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=["2x4cfg1sp0tp1", "2x4cfg0sp0tp1", "4x8cfg1sp0tp1"],
    indirect=["mesh_device"],
)
```

**Disabled test ID:**
`test_sd35_pipeline[wormhole_b0-no_traced-device_params0-2x4cfg1sp0tp1-large-1024-1024-3.5-28-True]`

**Root cause:**
`TT_THROW: TIMEOUT: device timeout in fetch queue wait` → SIGABRT → exit code 134.
Failed in 5/5 consecutive main runs since May 15. The `yes_traced` variant skips itself programmatically in CI (calls `pytest.skip()` internally), so only `no_traced` actually executes — both share the same broken hardware config `2x4cfg1sp0tp1`.

### 3. Other failures classified — nothing else to disable

| Job | Error | Decision |
|-----|-------|----------|
| `t3k_wan2.2_tests` | `DRAM training failed for channel 2` (UMD init) | Bad machine, infra noise — only failed 1/3 main runs. Do NOT disable. |
| `t3k_motif_tests` | GitHub action download timed out after 3 attempts | GitHub infra timeout — not disableable per protocol |

**Pipeline is as green as feasibly possible** — only one real consistent test failure, now disabled. Remaining failures are hardware/infra noise that cannot be addressed by disabling tests.

---

## Verification Run Details

| Run | Status | Notes |
|-----|--------|-------|
| 26238046274 | ❌ failure | Infra failures only: `t3k_motif_tests` (GH download timeout), `t3k_wan2.2_tests` (DRAM hardware fault on bad machine). Not caused by code changes. |
| 26295163268 | ✅ success | All jobs passed. Run completed 2026-05-22T21:04 UTC. |

---

## Full Protocol — Making a CI Pipeline Green by Disabling Failing Tests

**One pipeline at a time.** Do not start a second pipeline until the current one is merged.

### Source of Truth — Which Pipelines to Work On

The canonical list of pipelines needing work comes from the `aggregate-workflow-data` workflow in `tenstorrent/tt-metal`.

**A pipeline is in scope if ALL of the following are true:**
1. It appears in `aggregate-workflow-data`
2. It has been **red for more than 3 days**
3. It is **not** a galaxy workflow
4. It is **not** already listed in the Pipeline Status table as Done, In Progress, or Out of Scope

### Step 0 — Create a tracking issue

```bash
curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/issues" \
  -d '{
    "title": "[ci] Disable consistently failing tests in <workflow-name>",
    "body": "Tracking issue. Disabling tests that are consistently failing in `<workflow-name>` on `main`.\n\nPR: (link once created)",
    "labels": ["ci-infra"]
  }'
```

Note the issue number — every `pytest.mark.skip` reason will reference it.

### Step 1 — Create the disable branch and draft PR

```bash
# Fetch the target branch
git -C /workspace/group/tt-metal-bare fetch origin <branch-name>
git -C /workspace/group/tt-metal-bare worktree add \
  /workspace/group/worktrees/<branch-name> FETCH_HEAD
git -C /workspace/group/worktrees/<branch-name> checkout -b <branch-name>

# Set token in remote URL for pushing
git -C /workspace/group/worktrees/<branch-name> remote set-url origin \
  https://x-access-token:${GITHUB_TOKEN}@github.com/tenstorrent/tt-metal.git
```

Create draft PR immediately (even before any changes), referencing the tracking issue.

### Step 2 — Find the latest run of the target pipeline on main

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/workflows/<workflow-file>/runs?branch=main&per_page=5" \
  | python3 -c "
import sys,json
for r in json.load(sys.stdin)['workflow_runs'][:5]:
    print(r['id'], r['created_at'], r['status'], r.get('conclusion') or '—')
"
```

### Step 3 — List all failing jobs in that run

```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/runs/<RUN_ID>/jobs?per_page=50" \
  | python3 -c "
import sys,json
for j in json.load(sys.stdin)['jobs']:
    if j['conclusion'] not in ('success','skipped'):
        print(j['id'], j['conclusion'] or j['status'], j['name'])
"
```

### Step 4 — Classify each failing job: timeout vs error

Fetch the job log (note: returns 302 redirect — follow manually):
```bash
LOG_URL=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/jobs/<JOB_ID>/logs" \
  -D - 2>/dev/null | grep -i "^location:" | awk '{print $2}' | tr -d '\r')
curl -s "$LOG_URL" | tail -100
```

**Classification rules:**
- `"timed out after X minutes"` → **TIMEOUT** — job ran too long. **Do not disable any tests.** Skip it.
- `"Process completed with exit code 1"` → **ERROR** — find and disable the specific failing test(s).
- `DRAM training failed`, `Failed to download archive`, UMD init errors → **INFRA** — do not disable.

**Flakiness rule:** Before disabling, verify the test has failed in **at least 3 consecutive runs** on `main`. If it only fails sometimes, do not disable it.

For error jobs, grep the log for `FAILED` lines:
```
FAILED models/path/to/test.py::test_name[param-id] - ErrorType: message
```

### Step 5 — Disable the failing test(s)

**Parametrized test** — wrap the failing param in `pytest.param`:
```python
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        pytest.param(
            (2, 4),
            (2, 1),
            ...
            marks=pytest.mark.skip(reason="Disabled by issue #XXXXX"),
        ),
        [(4, 8), ...],   # other params stay as-is
    ],
    ids=["2x4cfg...", "4x8cfg..."],
)
```

**Whole-function failure** — add decorator above the function:
```python
@pytest.mark.skip(reason="Disabled by issue #XXXXX")
@pytest.mark.parametrize(...)
def test_foo(...):
```

**Rule:** Only disable the minimum — the specific parametrization(s) that fail. Leave passing configs untouched.

Skip reason format: `"Disabled by issue #XXXXX"` — no verbose detail.

### Step 6 — Commit, push, trigger verification run

```bash
git -C /workspace/group/worktrees/<branch> add <files>
git -C /workspace/group/worktrees/<branch> commit -m "[ci] Disable consistently failing tests in <workflow> (issue #XXXXX)"
git -C /workspace/group/worktrees/<branch> push origin <branch>

# Trigger the pipeline on the branch
curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/workflows/<workflow-file>/dispatches" \
  -d '{"ref": "<branch>"}'
```

Poll until complete:
```bash
for i in $(seq 1 120); do
  sleep 60
  R=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
    "https://api.github.com/repos/tenstorrent/tt-metal/actions/runs/<RUN_ID>" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('status'), r.get('conclusion','—'))")
  echo "$(date -u +%H:%M) - $R"
  if [[ "$R" == *"completed"* ]]; then break; fi
done
```

Get the new run ID after dispatch:
```bash
curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/tenstorrent/tt-metal/actions/runs?branch=<branch>&per_page=1" \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['workflow_runs'][0]; print(r['id'], r['status'])"
```

### Step 7 — Iterate

If the run still has failures: go back to Step 4. Continue until all remaining failures are timeouts or confirmed infra noise.

### Step 8 — Final PR checks before review

1. **Rebase on latest main:**
```bash
git -C /workspace/group/tt-metal-bare fetch origin main
git -C /workspace/group/worktrees/<branch> fetch origin main:refs/remotes/origin/main
git -C /workspace/group/worktrees/<branch> rebase origin/main
git -C /workspace/group/worktrees/<branch> push --force origin <branch>
```

2. **Verify still failing on main** — check latest completed run of the pipeline on `main` and confirm every disabled test is still in the failing set.

3. **Undraft the PR** and request reviews.

### Worktree setup notes

- Bare clone at: `/workspace/group/tt-metal-bare/`
- Worktrees at: `/workspace/group/worktrees/<branch-name>/`
- `git -C /workspace/group/tt-metal-bare worktree list` — see all worktrees
- For new branches from main: `git -C /workspace/group/tt-metal-bare worktree add /workspace/group/worktrees/<name> -b <branch-name>`
- For existing remote branches: fetch first, then `worktree add ... FETCH_HEAD`, then `checkout -b`

---

## Pipeline Status Table (all pipelines)

| Pipeline | Status | Notes |
|---|---|---|
| `t3000-perf-tests` | ✅ Done | PR #44771 |
| Runtime unit tests | ✅ Done | PR already merged |
| `t3000-demo-tests` | 🔄 In progress — PR #44938 | Ready to merge — see above |
| Nightly L2 tests | 🚫 Out of scope | Handled by separate agent |
| Galaxy workflows | 🚫 Out of scope | Never touch |

---

## Next Pipeline After t3000-demo-tests Merges

**Pipeline:** `(T3K) T3000 e2e tests`
**Failing job:** `t3k_ccl_tests [wh_llmbox]`
**Workflow file:** verify exact name in `.github/workflows/` (likely `t3k-e2e-tests.yaml`)
**Error:** `TT_THROW: trisc build failed`, exit code 1 — all 5 consecutive main runs
**Test command:** `pytest tests/nightly/t3000/ccl`

### Pre-analysis (done as of ~2026-05-21, re-verify before disabling)

**Test 1:** `test_ring_joint_sdpa_program_cache`
- File: `tests/nightly/t3000/ccl/test_ring_joint_attention.py`
- Has one parametrize entry (plus dtype variants bf16/bf8_b/bf4_b) — all variants likely affected by trisc build failure
- May need `@pytest.mark.skip` at function level if all dtype variants fail

**Test 2:** `test_all_gather_matmul_async`
- File: `tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`
- 6 parametrize combinations total; 2 consistently failing:
  - `ag_output_shape0-perf-no_barrier_with_persistent-chunking`
  - `ag_output_shape0-check-barrier_with_persistent-default`
- 4 other combos were passing — disable only the 2 failing ones with `pytest.param`

**Important:** Re-verify these against latest main runs before disabling. The pre-analysis is ~3 days old.

---

## GitHub API Notes for New Agent

- Token in `$GITHUB_TOKEN` env var — use `Authorization: token $GITHUB_TOKEN` (not `Bearer`)
- No `gh` CLI — use `curl` for REST API
- Git push works fine; REST API reads/writes have been unreliable in this session
- Rate limit endpoint: `GET https://api.github.com/rate_limit`
- For log fetching: GET `.../jobs/<JOB_ID>/logs` returns 302 redirect to signed Azure blob URL — follow the redirect manually: `curl -D - ... | grep location` then fetch that URL without auth headers
