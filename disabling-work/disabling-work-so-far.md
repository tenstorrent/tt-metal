# CI Disable Work — Status Log

**Last updated:** 2026-06-11T00:28Z

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
| [#46551](https://github.com/tenstorrent/tt-metal/pull/46551) | single-card-demo-tests.yaml | closed | verified-pass | Closed | sdxl N150 wormhole — tests started passing on main, disable removed, PR closed |
| [#46553](https://github.com/tenstorrent/tt-metal/pull/46553) | blackhole-post-commit.yaml | verified-pass | pass | Yes | deepseek blitz sampling+model |
| [#46555](https://github.com/tenstorrent/tt-metal/pull/46555) | t3000-unit-tests.yaml | verifying | pending (re-dispatch) | No | t3k dits PCC failure — re-dispatched after inconclusive |
| [#46696](https://github.com/tenstorrent/tt-metal/pull/46696) | single-card-demo-tests.yaml | verifying | pending | No | mnist test_demo_dataset matmul TT_FATAL |
| [#46698](https://github.com/tenstorrent/tt-metal/pull/46698) | blackhole-demo-tests.yaml | verifying | pending | No | flux.1-dev bh_loudbox + bh_quietbox_2 mesh_trace TT_FATAL |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [27314954594](https://github.com/tenstorrent/tt-metal/actions/runs/27314954594) | t3000-unit-tests.yaml | ci-disable/t3000-unit-tests-dits-t3k-2026-06-10 | 2026-06-11T00:15Z | in_progress | fresh build, model=dits, re-dispatch after inconclusive |
| [27315173882](https://github.com/tenstorrent/tt-metal/actions/runs/27315173882) | single-card-demo-tests.yaml | ci-disable/single-card-demo-tests-mnist-matmul-2026-06-11 | 2026-06-11T00:21Z | queued | fresh build, requested-models=["mnist"] |
| [27315355354](https://github.com/tenstorrent/tt-metal/actions/runs/27315355354) | blackhole-demo-tests.yaml | ci-disable/blackhole-demo-tests-flux1-loudbox-quietbox-2026-06-11 | 2026-06-11T00:26Z | in_progress | fresh build, model=flux.1-dev |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| [27244540816](https://github.com/tenstorrent/tt-metal/actions/runs/27244540816) | t3000-unit-tests.yaml | ci-disable/t3000-unit-tests-dits-t3k-2026-06-10 | 2026-06-10T00:21Z | 2026-06-10T00:40Z | verification-inconclusive | Collection error: pytest.param([(2, 4), 1], marks=...) used list as single arg; fix committed as d5fc3fceeb1; re-dispatched as 27314954594 |
| [27244463459](https://github.com/tenstorrent/tt-metal/actions/runs/27244463459) | blackhole-post-commit.yaml | ci-disable/blackhole-post-commit-deepseek-blitz-2026-06-10 | 2026-06-10T00:19Z | 2026-06-10T01:16Z | verified-pass | All ops-unit-tests (P150b) jobs passed |
| [27244284837](https://github.com/tenstorrent/tt-metal/actions/runs/27244284837) | single-card-demo-tests.yaml | ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-10 | 2026-06-10T00:14Z | 2026-06-10T00:58Z | verified-pass | sdxl-N150-func passed; tests also started passing on main (disable removed, PR closed) |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- 2026-06-11T00:28Z — Dispatched verification run 27315355354 for PR #46698 (blackhole-demo-tests flux.1-dev, fresh build, model=flux.1-dev)
- 2026-06-11T00:26Z — Created PR #46698 for blackhole-demo-tests flux.1-dev bh_loudbox+bh_quietbox_2 (issue #46697). Branch: ci-disable/blackhole-demo-tests-flux1-loudbox-quietbox-2026-06-11
- 2026-06-11T00:21Z — Dispatched verification run 27315173882 for PR #46696 (single-card-demo-tests mnist, fresh build, requested-models=["mnist"])
- 2026-06-11T00:20Z — Created PR #46696 for single-card-demo-tests mnist test_demo_dataset (issue #46695). Branch: ci-disable/single-card-demo-tests-mnist-matmul-2026-06-11
- 2026-06-11T00:15Z — Dispatched verification run 27314954594 for PR #46555 (t3000-unit-tests re-dispatch after inconclusive, fresh build, model=dits)
- 2026-06-11T00:14Z — Rebased PR #46555 branch onto 21bdebf19057 (origin/main). Updated evidence to run 27310935727/job 80681874558.
- 2026-06-11T00:12Z — Rebased PR #46553 onto 21bdebf19057. Evidence refreshed to run 27311852391/job 80686030225. Transitioned to verified-pass.
- 2026-06-11T00:10Z — PR #46551: tests now passing on main (run 27257893975, job 80497555974/80497556053). Removed disable from branch, closed PR, closed issue #46550.
- 2026-06-11T00:05Z — Session start. Checked all 3 verification runs (27244284837: verified-pass, 27244463459: verified-pass, 27244540816: verification-inconclusive). Found sdxl tests now passing on main.
- 2026-06-10T00:22Z — Dispatched verification run 27244540816 for PR #46555 (t3000-unit-tests, fresh build, model=dits)
- 2026-06-10T00:21Z — Created PR #46555 for t3000-unit-tests t3k_dits_tests (issue #46554). Branch: ci-disable/t3000-unit-tests-dits-t3k-2026-06-10
- 2026-06-10T00:19Z — Dispatched verification run 27244463459 for PR #46553 (blackhole-post-commit, fresh build, run-ops-unit-tests=true)
- 2026-06-10T00:19Z — Created PR #46553 for blackhole-post-commit deepseek blitz tests (issue #46552). Branch: ci-disable/blackhole-post-commit-deepseek-blitz-2026-06-10
- 2026-06-10T00:14Z — Dispatched verification run 27244284837 for PR #46551 (single-card-demo-tests, fresh build, model=sdxl)
- 2026-06-10T00:13Z — Created PR #46551 for single-card-demo-tests sdxl N150 wormhole (issue #46550). Branch: ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-10
- 2026-06-10T00:00Z — Session start. State log was fresh (cleared). Proceeding as fresh-start session. Surveyed blackhole-post-commit, single-card-demo-tests, t3000-unit-tests, blackhole-demo-tests, runtime-unit-tests. Identified 3 deterministic failures.

---

## PR #46551 — single-card-demo-tests (sdxl N150 wormhole CLIPEncoder) — CLOSED

| Field | Value |
|-------|-------|
| PR | [#46551](https://github.com/tenstorrent/tt-metal/pull/46551) |
| Disable issue | [#46550](https://github.com/tenstorrent/tt-metal/issues/46550) (closed) |
| Timeout issue | n/a |
| Branch | ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-10 |
| Workflow file | .github/workflows/single-card-demo-tests.yaml |
| Lifecycle stage | closed |
| Last rebase | 2026-06-11T00:10Z (onto 21bdebf19057) |
| Last revalidation | 2026-06-11T00:10Z |
| Verification run | [27244284837](https://github.com/tenstorrent/tt-metal/actions/runs/27244284837) (verified-pass) |
| Last touched by automation | 2026-06-11T00:10Z |
| Readiness | Closed — tests passing on main, disable removed |

### Disables (with main evidence)

All disables removed — tests started passing on main as of run [27257893975](https://github.com/tenstorrent/tt-metal/actions/runs/27257893975/job/80497555974) (2026-06-10 07:17 UTC). PR closed.

---

## PR #46553 — blackhole-post-commit (deepseek blitz sampling+model)

| Field | Value |
|-------|-------|
| PR | [#46553](https://github.com/tenstorrent/tt-metal/pull/46553) |
| Disable issue | [#46552](https://github.com/tenstorrent/tt-metal/issues/46552) |
| Timeout issue | n/a |
| Branch | ci-disable/blackhole-post-commit-deepseek-blitz-2026-06-10 |
| Workflow file | .github/workflows/blackhole-post-commit.yaml |
| Lifecycle stage | verified-pass |
| Last rebase | 2026-06-11T00:12Z (onto 21bdebf19057adf3824f9b540b939b8ac2324575) |
| Last revalidation | 2026-06-11T00:12Z |
| Verification run | [27244463459](https://github.com/tenstorrent/tt-metal/actions/runs/27244463459) (verified-pass) |
| Last touched by automation | 2026-06-11T00:12Z |
| Readiness | verified-pass — ready to merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py::test_sampling_topk_single_device[test_1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py::test_sampling_topk_single_device[test_2] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py::test_sampling_topk_single_device[test_4] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-1-1-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-1-64-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-1-128-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-16-1-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-16-64-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-16-128-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-32-1-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-32-64-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-32-128-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27311852391/job/80686030225 | [4a8e7ef](https://github.com/tenstorrent/tt-metal/commit/4a8e7efbf5c9a822f2a41387ef603209da4c39be) | 2026-06-10 23:45 UTC |

---

## PR #46555 — t3000-unit-tests (t3k dits PCC failure)

| Field | Value |
|-------|-------|
| PR | [#46555](https://github.com/tenstorrent/tt-metal/pull/46555) |
| Disable issue | [#46554](https://github.com/tenstorrent/tt-metal/issues/46554) |
| Timeout issue | n/a |
| Branch | ci-disable/t3000-unit-tests-dits-t3k-2026-06-10 |
| Workflow file | .github/workflows/t3000-unit-tests.yaml |
| Lifecycle stage | verifying |
| Last rebase | 2026-06-11T00:14Z (onto 21bdebf19057adf3824f9b540b939b8ac2324575) |
| Last revalidation | 2026-06-11T00:14Z |
| Verification run | [27314954594](https://github.com/tenstorrent/tt-metal/actions/runs/27314954594) (dispatched 2026-06-11T00:15Z, in_progress) |
| Last touched by automation | 2026-06-11T00:15Z |
| Readiness | Awaiting verification run completion |

### Prior inconclusive run

[27244540816](https://github.com/tenstorrent/tt-metal/actions/runs/27244540816) — collection error (pytest.param with list as single arg); fix applied as commit d5fc3fceeb1; re-dispatched as 27314954594.

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/tt_dit/tests/unit/test_embeddings.py::test_wan_time_text_image_embedding[wormhole_b0-device_params0-t3k-1-512-4096] [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27310935727/job/80681874558 | [d77a9e7](https://github.com/tenstorrent/tt-metal/commit/d77a9e7293e7482419c5a4fd18d57a486e22dd32) | 2026-06-10 23:17 UTC |

---

## PR #46696 — single-card-demo-tests (mnist test_demo_dataset matmul TT_FATAL)

| Field | Value |
|-------|-------|
| PR | [#46696](https://github.com/tenstorrent/tt-metal/pull/46696) |
| Disable issue | [#46695](https://github.com/tenstorrent/tt-metal/issues/46695) |
| Timeout issue | n/a |
| Branch | ci-disable/single-card-demo-tests-mnist-matmul-2026-06-11 |
| Workflow file | .github/workflows/single-card-demo-tests.yaml |
| Lifecycle stage | verifying |
| Last rebase | 2026-06-11T00:20Z (on 21bdebf19057adf3824f9b540b939b8ac2324575) |
| Last revalidation | 2026-06-11T00:20Z |
| Verification run | [27315173882](https://github.com/tenstorrent/tt-metal/actions/runs/27315173882) (dispatched 2026-06-11T00:21Z, queued) |
| Last touched by automation | 2026-06-11T00:21Z |
| Readiness | Awaiting verification run completion |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/demos/vision/classification/mnist/demo/demo.py::test_demo_dataset[1-128-device_params0] [N150] | https://github.com/tenstorrent/tt-metal/actions/runs/27257893975/job/80497555974 | [f92eba0](https://github.com/tenstorrent/tt-metal/commit/f92eba08956d05a286d13101d1bb9a73bdd1aed5) | 2026-06-10 06:53 UTC |
| models/demos/vision/classification/mnist/demo/demo.py::test_demo_dataset[1-128-device_params0] [N300] | https://github.com/tenstorrent/tt-metal/actions/runs/27257893975/job/80497556053 | [f92eba0](https://github.com/tenstorrent/tt-metal/commit/f92eba08956d05a286d13101d1bb9a73bdd1aed5) | 2026-06-10 15:32 UTC |

---

## PR #46698 — blackhole-demo-tests (flux.1-dev bh_loudbox + bh_quietbox_2 mesh_trace TT_FATAL)

| Field | Value |
|-------|-------|
| PR | [#46698](https://github.com/tenstorrent/tt-metal/pull/46698) |
| Disable issue | [#46697](https://github.com/tenstorrent/tt-metal/issues/46697) |
| Timeout issue | n/a |
| Branch | ci-disable/blackhole-demo-tests-flux1-loudbox-quietbox-2026-06-11 |
| Workflow file | .github/workflows/blackhole-demo-tests.yaml |
| Lifecycle stage | verifying |
| Last rebase | 2026-06-11T00:26Z (on 21bdebf19057adf3824f9b540b939b8ac2324575) |
| Last revalidation | 2026-06-11T00:26Z |
| Verification run | [27315355354](https://github.com/tenstorrent/tt-metal/actions/runs/27315355354) (dispatched 2026-06-11T00:26Z, in_progress) |
| Last touched by automation | 2026-06-11T00:26Z |
| Readiness | Awaiting verification run completion |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/tt_dit/tests/models/flux1/test_performance_flux1.py::test_flux1_pipeline_performance[blackhole-device_params0-bh_2x4sp0tp1-1024-1024-3.5-28] [bh_loudbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27255332239/job/80489386807 | [f92eba0](https://github.com/tenstorrent/tt-metal/commit/f92eba08956d05a286d13101d1bb9a73bdd1aed5) | 2026-06-10 05:49 UTC |
| models/tt_dit/tests/models/flux1/test_performance_flux1.py::test_flux1_pipeline_performance[blackhole-device_params0-2x2sp0tp1-1024-1024-3.5-28] [bh_quietbox_2] | https://github.com/tenstorrent/tt-metal/actions/runs/27255332239/job/80489386880 | [f92eba0](https://github.com/tenstorrent/tt-metal/commit/f92eba08956d05a286d13101d1bb9a73bdd1aed5) | 2026-06-10 05:47 UTC |
