# CI Disable Work — Status Log

**Last updated:** 2026-06-09T13:55 UTC

> **Source of truth.** This file is the canonical record of automation-tracked PRs. Wiping it resets the automation to fresh-state view; stale GitHub PRs not listed here are intentionally invisible.

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Last touched by automation | Readiness`); update fields in place rather than rewriting the section. `Last touched by automation: <UTC ISO>` is required on every PR row and drives the 4-hour throttle — update it every time the automation does any work on the PR (rebase, dispatch, log analysis, comment, removal).
- Each PR section also carries a `Disables (with main evidence)` table listing every currently-disabled test ID together with the most recent failing main-run job-link (`/runs/<id>/job/<jid>`), the commit SHA that run was built from (rendered as a link to `https://github.com/tenstorrent/tt-metal/commit/<sha>`), and the run completion timestamp. This mirrors the PR description's evidence table (see `disabling-work/ci-disable-targeted-verification.md` → `Main-run evidence model`) and is the starting point the next session re-checks before doing any work on the PR. If keeping the state log compact is preferred, the per-PR section MAY instead include the one-line pointer `Main-run evidence: see PR description.` — the PR description's evidence table is authoritative either way. Preserve any existing PR entries unchanged when extending the schema.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`. (`verification-inconclusive` = a verification was dispatched but failed to actually exercise the previously-passing jobs; eligible for re-dispatch and does NOT consume the one-run-per-PR budget.)
- **Multiple Quick Index rows MAY exist for the same workflow file** — one row per open automation PR (see `disabling-work/ci-disable-targeted-verification.md` → `## Multi-PR per workflow`). To compute the test-level disable set for a workflow, take the union of the `Disables (with main evidence)` table contents across every Quick Index row whose `Workflow` column references that workflow file. A workflow has "uncovered failing tests" iff its deterministic-main-failure set contains at least one test not in that union. Cross-PR conflict prevention: a new PR's initial batch MUST exclude every test already in any currently-OPEN automation PR's disable set for that workflow.
- **Per-PR sections** are appended below `## Recently Completed Runs` as PRs are created — one `## PR #<num> — <workflow> (<test summary>)` section per tracked PR, in the schema described above. On a freshly-cleared state log there are no per-PR sections yet; they will be created by the next session as PRs are opened.

---

## Quick Index

> Multiple PRs per workflow are allowed (see `disabling-work/ci-disable-targeted-verification.md` → `## Multi-PR per workflow`). Coverage is computed test-by-test by taking the union of every open automation PR's `Disables (with main evidence)` table for that workflow; a workflow has "uncovered failing tests" iff at least one deterministic main failure for that workflow is missing from that union.

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| #46425 | single-card-demo-tests.yaml | verified-fail | fail (expected — targeted run) | Yes | Verification run 27175683021 completed; sdxl still failing on main |
| #46427 | blackhole-demo-tests.yaml | verified-fail | fail (expected — targeted run) | Yes | Verification run 27175756052 completed; sdxl still failing on main (run 27185553601, 2026-06-09) |
| #46429 | blackhole-post-commit.yaml | out-of-scope | N/A | N/A | Fixed by PR #46397 (merged 2026-06-08T23:28Z); test now passes on main |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| _(none)_ | | | | | |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| 27175683021 | single-card-demo-tests.yaml | ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-09 | 2026-06-09T00:23Z | 2026-06-09T00:46Z | failure (expected) | PR #46425; sdxl-N150-func failed as expected |
| 27175756052 | blackhole-demo-tests.yaml | ci-disable/blackhole-demo-tests-sdxl-blackhole-2026-06-09 | 2026-06-09T00:25Z | 2026-06-09T00:54Z | failure (expected) | PR #46427; stable_diffusion_xl demos failed as expected |
| 27175855392 | blackhole-post-commit.yaml | ci-disable/blackhole-post-commit-h2d-stream-service-2026-06-09 | 2026-06-09T00:28Z | — | out-of-scope | PR #46429 closed; test fixed by #46397 |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- 2026-06-09T13:55Z — Manual review: PRs #46425 and #46427 verified-fail (expected); main evidence links updated to latest runs. PR #46429 marked out-of-scope — fixed upstream by #46397 (merged 2026-06-08T23:28Z).
- 2026-06-09T00:30Z — Session 2026-06-09: Created 3 focus PRs for deterministic failures. State log is fresh (wiped from prior sessions). All 3 dispatch slots used.
- 2026-06-09T00:28Z — PR #46429 created (blackhole-post-commit/h2d_stream_service); verification run 27175855392 dispatched.
- 2026-06-09T00:25Z — PR #46427 created (blackhole-demo-tests/sdxl blackhole); verification run 27175756052 dispatched.
- 2026-06-09T00:23Z — PR #46425 created (single-card-demo-tests/sdxl wormhole); verification run 27175683021 dispatched.

---

## PR #46425 — single-card-demo-tests.yaml (sdxl wormhole_b0 test_demo disable)

| Field | Value |
|-------|-------|
| PR | https://github.com/tenstorrent/tt-metal/pull/46425 |
| Disable issue | https://github.com/tenstorrent/tt-metal/issues/46424 |
| Timeout issue | N/A |
| Branch | ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-09 |
| Workflow file | .github/workflows/single-card-demo-tests.yaml |
| Lifecycle stage | verified-fail |
| Last rebase | 2026-06-09T00:21Z (main HEAD 3965f285330173df91ac6dac0c72bd801d87c47e) |
| Last revalidation | 2026-06-09T00:21Z (confirmed failing in runs 27120250843, 27144621588, 27159995928) |
| Verification run | 27175683021 (completed; sdxl-N150-func failed as expected; build passed; targeted run only) |
| Last touched by automation | 2026-06-09T13:55Z |
| Readiness | Ready — awaiting merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Test | Most recent failing job link | Commit | Completed at |
|------|------------------------------|--------|--------------|
| `models/demos/stable_diffusion_xl_base/demo/demo.py::test_demo[wormhole_b0-default_additional_parameters-with_trace-device_encoders-device_vae-5.0-50-negative_prompt0-An astronaut riding a green horse-False-no_cfg_parallel-1024x1024]` | https://github.com/tenstorrent/tt-metal/actions/runs/27159995928/job/80178667801 | [7097398f](https://github.com/tenstorrent/tt-metal/commit/7097398fbced3b6f1cbf0a9a5e23e7f56f2e0fc0) | 2026-06-08 18:56 UTC |
| `models/demos/stable_diffusion_xl_base/demo/demo.py::test_demo[wormhole_b0-default_additional_parameters-with_trace-device_encoders-device_vae-5.0-50-negative_prompt0-An astronaut riding a green horse-False-no_cfg_parallel-512x512]` | https://github.com/tenstorrent/tt-metal/actions/runs/27159995928/job/80178667801 | [7097398f](https://github.com/tenstorrent/tt-metal/commit/7097398fbced3b6f1cbf0a9a5e23e7f56f2e0fc0) | 2026-06-08 18:56 UTC |

---

## PR #46427 — blackhole-demo-tests.yaml (sdxl blackhole test_demo disable)

| Field | Value |
|-------|-------|
| PR | https://github.com/tenstorrent/tt-metal/pull/46427 |
| Disable issue | https://github.com/tenstorrent/tt-metal/issues/46426 |
| Timeout issue | N/A |
| Branch | ci-disable/blackhole-demo-tests-sdxl-blackhole-2026-06-09 |
| Workflow file | .github/workflows/blackhole-demo-tests.yaml |
| Lifecycle stage | verified-fail |
| Last rebase | 2026-06-09T00:24Z (main HEAD 3965f285330173df91ac6dac0c72bd801d87c47e) |
| Last revalidation | 2026-06-09T13:55Z (confirmed failing in runs 27053516661, 27083801540, 27117989562, 27185553601) |
| Verification run | 27175756052 (completed; stable_diffusion_xl demos failed as expected; build passed; targeted run only) |
| Last touched by automation | 2026-06-09T13:55Z |
| Readiness | Ready — awaiting merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Test | Most recent failing job link | Commit | Completed at |
|------|------------------------------|--------|--------------|
| `models/demos/stable_diffusion_xl_base/demo/demo.py::test_demo[blackhole-default_additional_parameters-with_trace-device_encoders-device_vae-5.0-50-negative_prompt0-An astronaut riding a green horse-False-no_cfg_parallel-1024x1024]` | https://github.com/tenstorrent/tt-metal/actions/runs/27185553601/job/80254318066 | [f4a3533f](https://github.com/tenstorrent/tt-metal/commit/f4a3533fb2ba7c8f303eaccc3f32dbc9b950291d) | 2026-06-09 05:42 UTC |

---

## PR #46429 — blackhole-post-commit.yaml (h2d_stream_service large buffer disable)

| Field | Value |
|-------|-------|
| PR | https://github.com/tenstorrent/tt-metal/pull/46429 |
| Disable issue | https://github.com/tenstorrent/tt-metal/issues/46428 |
| Timeout issue | N/A |
| Branch | ci-disable/blackhole-post-commit-h2d-stream-service-2026-06-09 |
| Workflow file | .github/workflows/blackhole-post-commit.yaml |
| Lifecycle stage | out-of-scope |
| Last rebase | 2026-06-09T00:26Z (main HEAD 3965f285330173df91ac6dac0c72bd801d87c47e) |
| Last revalidation | 2026-06-09T00:26Z (confirmed failing in runs 27140451724, 27153730343, 27165679180) |
| Verification run | 27175855392 (dispatched but superseded — test fixed upstream) |
| Last touched by automation | 2026-06-09T13:55Z |
| Readiness | N/A — test fixed by PR #46397 (merged 2026-06-08T23:28Z); close this PR |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Test | Most recent failing job link | Commit | Completed at |
|------|------------------------------|--------|--------------|
| `tests/ttnn/unit_tests/base_functionality/test_h2d_stream_service.py::test_h2d_stream_service_replicated_sweep[silicon_arch_name=blackhole-input_path=tensor-shape_list=[1, 1, 1, 65536]-scratch_cb_pages=1-fifo_pages=1]` | https://github.com/tenstorrent/tt-metal/actions/runs/27165679180/job/80209617722 | [ec55847b](https://github.com/tenstorrent/tt-metal/commit/ec55847b2c28ab0d7cce5cedb28c9a1a88a47f5b) | 2026-06-08 20:41 UTC |

