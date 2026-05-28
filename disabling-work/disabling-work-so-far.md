# CI Disable Work — Status Log

> **Source of truth.** This file is the canonical record of automation-tracked PRs. Wiping it resets the automation to fresh-state view; stale GitHub PRs not listed here are intentionally invisible.

Last updated: **2026-05-28T23:57Z**

---

## How to read/update this file

- Read this file at the start of every automation session and treat it as the authoritative current state for CI disable work.
- Scan the `## Quick Index` table first; it gives the lifecycle stage per PR before drilling into details.
- Per-PR sections use uniform field tables (`PR | Disable issue | Timeout issue | Branch | Workflow file | Lifecycle stage | Last rebase | Last revalidation | Verification run | Last touched by automation | Readiness`); update fields in place rather than rewriting the section. `Last touched by automation: <UTC ISO>` is required on every PR row and drives the 4-hour throttle — update it every time the automation does any work on the PR (rebase, dispatch, log analysis, comment, removal).
- Each PR section also carries a `Disables (with main evidence)` table listing every currently-disabled test ID together with the most recent failing main-run job-link (`/runs/<id>/job/<jid>`) and the run completion timestamp. This mirrors the PR description's evidence table (see `disabling-work/ci-disable-targeted-verification.md` → `Main-run evidence model`) and is the starting point the next session re-checks before doing any work on the PR. If keeping the state log compact is preferred, the per-PR section MAY instead include the one-line pointer `Main-run evidence: see PR description.` — the PR description's evidence table is authoritative either way. Preserve any existing PR entries unchanged when extending the schema.
- Append new entries to the top of `## Recent Activity` (most recent first); keep at most 30 entries — trim older entries to a single `- Older history truncated — see git history of this file.` line if needed.
- Commit and push any change to this file before ending the session.
- Lifecycle stages: `new`, `batch-committed`, `verifying`, `verification-inconclusive`, `verified-pass`, `verified-fail`, `merged`, `out-of-scope`. (`verification-inconclusive` = a verification was dispatched but failed to actually exercise the previously-passing jobs; eligible for re-dispatch and does NOT consume the one-run-per-PR budget.)

---

## Quick Index

| PR | Workflow | Lifecycle stage | Verification result | Ready to merge? | Notes |
|----|----------|-----------------|---------------------|-----------------|-------|
| [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) | `tt-metal-l2-nightly.yaml` | `verifying` | pending | No | Verification dispatched 2026-05-28T23:57Z, run [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) | `tt-metal-l2-nightly.yaml` | `ci-disable/tt-metal-l2-nightly-mesh-device-top32-20260528` | 2026-05-28T23:57Z | in_progress | Fresh build; sd_unit_tests only; targets llk-sd-unit-tests on N150/N300/P100/P150 |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| _(no completed runs yet)_ | | | | | | |

---

## PR #45484 — tt-metal-l2-nightly.yaml (llk-sd-unit-tests)

| Field | Value |
|-------|-------|
| PR | [#45484](https://github.com/tenstorrent/tt-metal/pull/45484) |
| Disable issue | [#45483](https://github.com/tenstorrent/tt-metal/issues/45483) |
| Timeout issue | N/A |
| Branch | `ci-disable/tt-metal-l2-nightly-mesh-device-top32-20260528` |
| Workflow file | `.github/workflows/tt-metal-l2-nightly.yaml` |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-05-28T23:57Z (created from main `577298dde0ac8bfb943e44997162ee14e9b0069b`) |
| Last revalidation | 2026-05-28T23:57Z |
| Verification run | [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) (in_progress) |
| Last touched by automation | 2026-05-28T23:57Z |
| Readiness | Not yet — awaiting verification result |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing main run (job link) | Run completed at |
|---|---|---|
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [wh_n150] | https://github.com/tenstorrent/tt-metal/actions/runs/26595275788/job/78367033195 | 2026-05-28 19:16 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [wh_n300] | https://github.com/tenstorrent/tt-metal/actions/runs/26518398862/job/78108245432 | 2026-05-27 15:23 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [bh_p100] | https://github.com/tenstorrent/tt-metal/actions/runs/26595275788/job/78367033385 | 2026-05-28 19:09 UTC |
| `MeshDeviceFixture.Top32RmDevPipelineCompletes` [bh_p150] | https://github.com/tenstorrent/tt-metal/actions/runs/26595275788/job/78367033433 | 2026-05-28 19:21 UTC |

**Evidence notes:** Test `MeshDeviceFixture.Top32RmDevPipelineCompletes` has been failing in every completed run of `llk-sd-unit-tests` for at least 3 consecutive completed runs (26496921030, 26518398862, 26560429708, 26595275788 — all on different main SHAs). Error signature: `trisc1 compile failure` in `top32_rm_dev_compute_v2` kernel → `" thrown in the test body."` gtest exception. The test is already excluded in `runtime-unit-tests.yaml` (PR #44767).

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- **2026-05-28T23:57Z** Session 2026-05-28T23:57: Created PR #45484 for `tt-metal-l2-nightly.yaml`, disabling `MeshDeviceFixture.Top32RmDevPipelineCompletes` in `llk-sd-unit-tests`. Dispatched verification run [26609412851](https://github.com/tenstorrent/tt-metal/actions/runs/26609412851) (fresh build, `run_sd_unit_tests=true`, both architectures). Created tracking issue #45483. Focus slots filled: 1/3 (reason: only 1 non-Galaxy single-card workflow found with confirmed 3-consecutive deterministic in-scope failures after extensive search; other workflow failures were either timeouts, multi-card hardware, or lacked 3 consecutive same-test completions).
