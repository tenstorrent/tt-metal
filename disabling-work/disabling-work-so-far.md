# CI Disable Work — Status Log

**Last updated:** 2026-06-03T18:32 UTC

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
| [#45979](https://github.com/tenstorrent/tt-metal/pull/45979) | `runtime-unit-tests.yaml` | `verifying` | pending | No | Verification dispatched 2026-06-03T18:19 UTC (run 26904326397) |
| [#45981](https://github.com/tenstorrent/tt-metal/pull/45981) | `t3000-unit-tests.yaml` | `verifying` | pending | No | Verification dispatched 2026-06-03T18:27 UTC (run 26904737723) |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [26904326397](https://github.com/tenstorrent/tt-metal/actions/runs/26904326397) | `runtime-unit-tests.yaml` | `ci-disable/runtime-unit-tests-data-movement-2026-06-03-verify` | 2026-06-03T18:19 UTC | queued | Fresh build; targeted verification for PR #45979; pruned to runtime_data_movement only |
| [26904737723](https://github.com/tenstorrent/tt-metal/actions/runs/26904737723) | `t3000-unit-tests.yaml` | `ci-disable/t3000-unit-tests-attention1d-qwen25-2026-06-03-verify` | 2026-06-03T18:27 UTC | in_progress | Fresh build; targeted verification for PR #45981; model filter `tttv2 modules` |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| _(none — fresh state log)_ | | | | | | |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- **2026-06-03T18:32 UTC** — PR #45981 created for `t3000-unit-tests.yaml` `t3k_tttv2_fast_unit_tests` job: disabled 2 deterministically-failing pytest parametrizations (`test_attention_1d_vs_reference[10-standard-1x2-decode-32-Qwen2.5-7B-1x2]` and `[10-paged-...]`). Fails on `wh_llmbox` only. Error: `TT_FATAL dims must be unique` in `concat_ndim`. Tracking issue #45980. Verification dispatched as run 26904737723 (fresh build; `model=tttv2 modules` filter to target only `t3k_tttv2_fast_unit_tests`). Note: `t3k_ttnn_tests [wh_llmbox]` SIGABRT failure is a device timeout/hang — **out of scope** per policy.
- **2026-06-03T18:25 UTC** — Session start. PR #45979 created for `runtime-unit-tests.yaml` `runtime_data_movement` job: disabled 5 deterministically-failing GTest cases (`TensixDirectWriteMulticast`, `TensixDataMovementOneToAllMulticastSemaphore2x2_2_0`, `TensixDataMovementOneToAllMulticastLinkedSemaphoreLoopback2x2_2_0`, `TensixDataMovementOneToAllMulticastLinkedSemaphore5x5_2_0`, `TensixDataMovementOneToAllUnicastSemaphore2x2_2_0`). All 5 fail on both `wh_n150_civ2` and `bh_p150b_civ2`. Tracking issue #45978. Verification dispatched as run 26904326397.

---

## PR #45981 — t3000-unit-tests.yaml (2 test_attention_1d_vs_reference Qwen2.5-7B disables)

| Field | Value |
|-------|-------|
| PR | [#45981](https://github.com/tenstorrent/tt-metal/pull/45981) |
| Disable issue | [#45980](https://github.com/tenstorrent/tt-metal/issues/45980) |
| Timeout issue | — |
| Branch | `ci-disable/t3000-unit-tests-attention1d-qwen25-2026-06-03` |
| Workflow file | `t3000-unit-tests.yaml` |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-06-03 (created off `origin/main` at `0485c74b235`) |
| Last revalidation | 2026-06-03 (evidence checked against run 26895624808) |
| Verification run | [26904737723](https://github.com/tenstorrent/tt-metal/actions/runs/26904737723) — started 2026-06-03T18:27 UTC (fresh build; `model=tttv2 modules` targeted filter) |
| Last touched by automation | 2026-06-03T18:32Z |
| Readiness | Not yet verified |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (both tests fail on `wh_llmbox` [job 79337583451] in run 26895624808, completed 2026-06-03T16:33 UTC, head SHA `c7ab438343c0913107e7a518e2cf2d115b07d34e`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `test_attention_1d_vs_reference[10-standard-1x2-decode-32-Qwen2.5-7B-1x2]` | `wh_llmbox` | [79337583451](https://github.com/tenstorrent/tt-metal/actions/runs/26895624808/job/79337583451) |
| `test_attention_1d_vs_reference[10-paged-1x2-decode-32-Qwen2.5-7B-1x2]` | `wh_llmbox` | [79337583451](https://github.com/tenstorrent/tt-metal/actions/runs/26895624808/job/79337583451) |

---

## PR #45979 — runtime-unit-tests.yaml (5 runtime_data_movement gtest disables)

| Field | Value |
|-------|-------|
| PR | [#45979](https://github.com/tenstorrent/tt-metal/pull/45979) |
| Disable issue | [#45978](https://github.com/tenstorrent/tt-metal/issues/45978) |
| Timeout issue | — |
| Branch | `ci-disable/runtime-unit-tests-data-movement-2026-06-03` |
| Workflow file | `runtime-unit-tests.yaml` |
| Lifecycle stage | `verifying` |
| Last rebase | 2026-06-03 (created off `origin/main` at `0485c74b235`) |
| Last revalidation | 2026-06-03 (evidence checked against run 26866206838) |
| Verification run | [26904326397](https://github.com/tenstorrent/tt-metal/actions/runs/26904326397) — queued 2026-06-03T18:19 UTC (fresh build; no SHA-matching successful source run) |
| Last touched by automation | 2026-06-03T18:25Z |
| Readiness | Not yet verified |

### Disables (with main evidence)

Main-run evidence: see PR description.

Summary (all 5 tests fail on `wh_n150_civ2` [job 79231037011] and `bh_p150b_civ2` [job 79231037021] in run 26866206838, completed 2026-06-03T06:33–06:37 UTC, head SHA `15806d0d564eff581a6eb21bea8c56f0b35867d5`):

| Disabled test | SKUs failing | Job link |
|---|---|---|
| `GenericMeshDeviceFixture.TensixDirectWriteMulticast` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastSemaphore2x2_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastLinkedSemaphoreLoopback2x2_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastLinkedSemaphore5x5_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
| `GenericMeshDeviceFixture.TensixDataMovementOneToAllUnicastSemaphore2x2_2_0` | `wh_n150_civ2`, `bh_p150b_civ2` | [79231037011](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037011) / [79231037021](https://github.com/tenstorrent/tt-metal/actions/runs/26866206838/job/79231037021) |
