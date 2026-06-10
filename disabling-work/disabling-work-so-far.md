# CI Disable Work — Status Log

**Last updated:** 2026-06-10T00:22Z

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
| [#46551](https://github.com/tenstorrent/tt-metal/pull/46551) | single-card-demo-tests.yaml | verifying | pending | No | sdxl N150 wormhole CLIPEncoder |
| [#46553](https://github.com/tenstorrent/tt-metal/pull/46553) | blackhole-post-commit.yaml | verifying | pending | No | deepseek blitz sampling+model |
| [#46555](https://github.com/tenstorrent/tt-metal/pull/46555) | t3000-unit-tests.yaml | verifying | pending | No | t3k dits PCC failure |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [27244284837](https://github.com/tenstorrent/tt-metal/actions/runs/27244284837) | single-card-demo-tests.yaml | ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-10 | 2026-06-10T00:14Z | in_progress | fresh build, model=sdxl |
| [27244463459](https://github.com/tenstorrent/tt-metal/actions/runs/27244463459) | blackhole-post-commit.yaml | ci-disable/blackhole-post-commit-deepseek-blitz-2026-06-10 | 2026-06-10T00:19Z | in_progress | fresh build, ops-unit-tests only |
| [27244540816](https://github.com/tenstorrent/tt-metal/actions/runs/27244540816) | t3000-unit-tests.yaml | ci-disable/t3000-unit-tests-dits-t3k-2026-06-10 | 2026-06-10T00:21Z | in_progress | fresh build, model=dits |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| _(none yet)_ | | | | | | |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- 2026-06-10T00:22Z — Dispatched verification run 27244540816 for PR #46555 (t3000-unit-tests, fresh build, model=dits)
- 2026-06-10T00:21Z — Created PR #46555 for t3000-unit-tests t3k_dits_tests (issue #46554). Branch: ci-disable/t3000-unit-tests-dits-t3k-2026-06-10
- 2026-06-10T00:19Z — Dispatched verification run 27244463459 for PR #46553 (blackhole-post-commit, fresh build, run-ops-unit-tests=true)
- 2026-06-10T00:19Z — Created PR #46553 for blackhole-post-commit deepseek blitz tests (issue #46552). Branch: ci-disable/blackhole-post-commit-deepseek-blitz-2026-06-10
- 2026-06-10T00:14Z — Dispatched verification run 27244284837 for PR #46551 (single-card-demo-tests, fresh build, model=sdxl)
- 2026-06-10T00:13Z — Created PR #46551 for single-card-demo-tests sdxl N150 wormhole (issue #46550). Branch: ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-10
- 2026-06-10T00:00Z — Session start. State log was fresh (cleared). Proceeding as fresh-start session. Surveyed blackhole-post-commit, single-card-demo-tests, t3000-unit-tests, blackhole-demo-tests, runtime-unit-tests. Identified 3 deterministic failures.

---

## PR #46551 — single-card-demo-tests (sdxl N150 wormhole CLIPEncoder)

| Field | Value |
|-------|-------|
| PR | [#46551](https://github.com/tenstorrent/tt-metal/pull/46551) |
| Disable issue | [#46550](https://github.com/tenstorrent/tt-metal/issues/46550) |
| Timeout issue | n/a |
| Branch | ci-disable/single-card-demo-tests-sdxl-wormhole-2026-06-10 |
| Workflow file | .github/workflows/single-card-demo-tests.yaml |
| Lifecycle stage | verifying |
| Last rebase | 2026-06-10T00:13Z (on 0ec19d11499dfe33ad0c88021c1dec2d0aced6f4) |
| Last revalidation | 2026-06-10T00:13Z |
| Verification run | [27244284837](https://github.com/tenstorrent/tt-metal/actions/runs/27244284837) (dispatched 2026-06-10T00:14Z, in_progress) |
| Last touched by automation | 2026-06-10T00:14Z |
| Readiness | Awaiting verification run completion |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/demos/stable_diffusion_xl_base/demo/demo.py::test_demo[wormhole_b0-default_additional_parameters-with_trace-device_encoders-device_vae-5.0-50-negative_prompt0-An astronaut riding a green horse-False-no_cfg_parallel-1024x1024] [wormhole_b0-N150] | https://github.com/tenstorrent/tt-metal/actions/runs/27211039610/job/80389154065 | [640b88e](https://github.com/tenstorrent/tt-metal/commit/640b88e1e1fbd8b79111fd445f92dde2730460b3) | 2026-06-09 14:47 UTC |
| models/demos/stable_diffusion_xl_base/demo/demo.py::test_demo[wormhole_b0-default_additional_parameters-with_trace-device_encoders-device_vae-5.0-50-negative_prompt0-An astronaut riding a green horse-False-no_cfg_parallel-512x512] [wormhole_b0-N150] | https://github.com/tenstorrent/tt-metal/actions/runs/27211039610/job/80389154065 | [640b88e](https://github.com/tenstorrent/tt-metal/commit/640b88e1e1fbd8b79111fd445f92dde2730460b3) | 2026-06-09 14:47 UTC |

---

## PR #46553 — blackhole-post-commit (deepseek blitz sampling+model)

| Field | Value |
|-------|-------|
| PR | [#46553](https://github.com/tenstorrent/tt-metal/pull/46553) |
| Disable issue | [#46552](https://github.com/tenstorrent/tt-metal/issues/46552) |
| Timeout issue | n/a |
| Branch | ci-disable/blackhole-post-commit-deepseek-blitz-2026-06-10 |
| Workflow file | .github/workflows/blackhole-post-commit.yaml |
| Lifecycle stage | verifying |
| Last rebase | 2026-06-10T00:19Z (on 0ec19d11499dfe33ad0c88021c1dec2d0aced6f4) |
| Last revalidation | 2026-06-10T00:19Z |
| Verification run | [27244463459](https://github.com/tenstorrent/tt-metal/actions/runs/27244463459) (dispatched 2026-06-10T00:19Z, in_progress) |
| Last touched by automation | 2026-06-10T00:19Z |
| Readiness | Awaiting verification run completion |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py::test_sampling_topk_single_device[test_1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py::test_sampling_topk_single_device[test_2] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_sampling.py::test_sampling_topk_single_device[test_4] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-1-1-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-1-64-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-1-128-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-16-1-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-16-64-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-16-128-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-32-1-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-32-64-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_model.py::test_prefill_and_decode[blackhole-32-128-1] [blackhole-P150b] | https://github.com/tenstorrent/tt-metal/actions/runs/27234221952/job/80434147219 | [cd014e0](https://github.com/tenstorrent/tt-metal/commit/cd014e070ba9acead1dcec22a23cc60de035b3de) | 2026-06-09 22:04 UTC |

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
| Last rebase | 2026-06-10T00:21Z (on 0ec19d11499dfe33ad0c88021c1dec2d0aced6f4) |
| Last revalidation | 2026-06-10T00:21Z |
| Verification run | [27244540816](https://github.com/tenstorrent/tt-metal/actions/runs/27244540816) (dispatched 2026-06-10T00:21Z, in_progress) |
| Last touched by automation | 2026-06-10T00:21Z |
| Readiness | Awaiting verification run completion |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/tt_dit/tests/unit/test_embeddings.py::test_wan_time_text_image_embedding[wormhole_b0-device_params0-t3k-1-512-4096] [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27240264988/job/80443258130 | [31b12dd](https://github.com/tenstorrent/tt-metal/commit/31b12ddf85e3aec40de7f4a24d9d3bf9e6e405e2) | 2026-06-09 23:09 UTC |
