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

## Protocol Used

Full protocol: `/workspace/group/protocols/disable-failing-pipeline-tests.md`

Key rules:
- Only disable if 3+ consecutive failures on main (not flaky)
- Timeout failures (`##[error]The action '...' has timed out after X minutes`) → never disable
- Infra errors (DRAM training, download timeouts) → never disable
- Skip reason format: `"Disabled by issue #XXXXX"` — no verbose detail
- One pipeline at a time — do not start next until current PR is merged

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
