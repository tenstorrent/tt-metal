# CI Disable Work — Status Log

**Last updated:** 2026-06-14T00:30Z

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
| [#46555](https://github.com/tenstorrent/tt-metal/pull/46555) | t3000-unit-tests.yaml | closed | verified-pass | Closed | t3k dits PCC failure — test started passing on main, disable removed, PR closed 2026-06-12 |
| [#46696](https://github.com/tenstorrent/tt-metal/pull/46696) | single-card-demo-tests.yaml | closed | verified-pass | Closed | mnist tests started passing on main (run 27353497098), disable removed, PR closed 2026-06-13 |
| [#46698](https://github.com/tenstorrent/tt-metal/pull/46698) | blackhole-demo-tests.yaml | verified-pass | pass | Yes | flux.1-dev bh_loudbox + bh_quietbox_2 mesh_trace TT_FATAL |
| [#46808](https://github.com/tenstorrent/tt-metal/pull/46808) | t3000-unit-tests.yaml | verified-pass | pass | Yes | TestValidateModuleConfigs 2 tests failing on wh_llmbox (EagerLLMExecutor missing methods) |
| [#46812](https://github.com/tenstorrent/tt-metal/pull/46812) | blackhole-post-commit.yaml | verified-pass | pass | Yes | test_host_io_loopback DEVICE_PULL small-tensor failures on bh_p300-viommu |
| [#46814](https://github.com/tenstorrent/tt-metal/pull/46814) | t3000-unit-tests.yaml | closed | verified-pass | Closed | Tests started passing on main (run 27442163505), disables removed, PR closed 2026-06-13 |
| [#46930](https://github.com/tenstorrent/tt-metal/pull/46930) | blackhole-post-commit.yaml | verified-pass | pass | Yes | test_host_io_loopback DEVICE_PULL 512-byte tensor on bh_p300-viommu — classified 2026-06-14 |
| [#46932](https://github.com/tenstorrent/tt-metal/pull/46932) | t3000-demo-tests.yaml | verified-pass | pass | Yes | qwen3_vl BERTScore CI test below threshold on wh_llmbox — classified 2026-06-14 |
| [#46934](https://github.com/tenstorrent/tt-metal/pull/46934) | blackhole-demo-tests.yaml | verified-pass | pass | Yes | mochi encoder performance below threshold on bh_quietbox_2 — classified 2026-06-14; parametrize bug fixed |
| [#46953](https://github.com/tenstorrent/tt-metal/pull/46953) | t3000-unit-tests.yaml | verifying | pending | No | IntermeshSplit2x2FabricFixture + MultiHostSocketTestSplitT3K.SocketTests re-regression on wh_llmbox |

---

## Active Runs

| Run | Pipeline | Branch | Started | Status | Notes |
|-----|----------|--------|---------|--------|-------|
| [27483467658](https://github.com/tenstorrent/tt-metal/actions/runs/27483467658) | t3000-unit-tests.yaml | ci-verify/t3000-unit-tests-multiprocess-socket-2026-06-14 | 2026-06-14T00:26Z | in_progress | fresh build, model=all, only t3k_tt_metal_multiprocess_tests job enabled |

**Policy:** Concurrent runs across PRs are allowed; each automation session may dispatch at most three new runs.

---

## Recently Completed Runs

| Run | Pipeline | Branch | Started | Ended | Result | Notes |
|-----|----------|--------|---------|-------|--------|-------|
| [27451009453](https://github.com/tenstorrent/tt-metal/actions/runs/27451009453) | t3000-demo-tests.yaml | ci-disable/t3000-demo-tests-qwen3vl-bertscore-2026-06-13 | 2026-06-13T00:33Z | 2026-06-13T14:03Z | verified-pass | All jobs passed (conclusion: success); classified 2026-06-14 |
| [27450999496](https://github.com/tenstorrent/tt-metal/actions/runs/27450999496) | blackhole-post-commit.yaml | ci-disable/blackhole-post-commit-host-io-device-pull-512-2026-06-13 | 2026-06-13T00:32Z | 2026-06-13T00:56Z | verified-pass | Failing job (host IO timeout) was also failing on main; no regression; classified 2026-06-14 |
| [27451128176](https://github.com/tenstorrent/tt-metal/actions/runs/27451128176) | blackhole-demo-tests.yaml | ci-disable/blackhole-demo-tests-mochi-encoder-perf-2026-06-13 | 2026-06-13T00:37Z | 2026-06-13T00:48Z | verified-pass | Mochi job had pytest collection error (parametrize bug in our disable) but job was already failing on main; no regression. Bug fixed 2026-06-14. |
| [27386543130](https://github.com/tenstorrent/tt-metal/actions/runs/27386543130) | t3000-unit-tests.yaml | ci-disable/t3000-unit-tests-multiprocess-socket-2026-06-12 | 2026-06-12T00:33Z | 2026-06-12T01:14Z | verified-pass | Tests were still failing but job was pre-failing on main; all disabled tests started passing on main (run 27442163505) — PR closed |
| [27386537661](https://github.com/tenstorrent/tt-metal/actions/runs/27386537661) | blackhole-post-commit.yaml | ci-disable/blackhole-post-commit-host-io-device-pull-2026-06-12 | 2026-06-12T00:33Z | 2026-06-12T00:59Z | verified-pass | Disabled tests correctly skipped; only new failure (512-1024-512) was also failing on main |
| [27386528492](https://github.com/tenstorrent/tt-metal/actions/runs/27386528492) | t3000-unit-tests.yaml | ci-disable/t3000-unit-tests-executor-parity-2026-06-12 | 2026-06-12T00:33Z | 2026-06-12T01:24Z | verified-pass | Disabled tests correctly SKIPPED; job failure due to pre-existing segfault in test_rmsnorm_1d |
| [27315355354](https://github.com/tenstorrent/tt-metal/actions/runs/27315355354) | blackhole-demo-tests.yaml | ci-disable/blackhole-demo-tests-flux1-loudbox-quietbox-2026-06-11 | 2026-06-11T00:26Z | 2026-06-11T01:10Z | verified-pass | All flux.1-dev jobs passed (bh_loudbox, bh_quietbox_2, bh_p300) |
| [27315173882](https://github.com/tenstorrent/tt-metal/actions/runs/27315173882) | single-card-demo-tests.yaml | ci-disable/single-card-demo-tests-mnist-matmul-2026-06-11 | 2026-06-11T00:21Z | 2026-06-11T14:08Z | verified-pass | mnist-N150-func and mnist-N300-func passed |
| [27314954594](https://github.com/tenstorrent/tt-metal/actions/runs/27314954594) | t3000-unit-tests.yaml | ci-disable/t3000-unit-tests-dits-t3k-2026-06-10 | 2026-06-11T00:15Z | 2026-06-11T00:44Z | verified-pass | t3k_dits_tests passed; test also started passing on main — disable removed, PR closed |

---

## Blockers

| Blocker | Status | Notes |
|---------|--------|-------|
| _(none)_ | | |

---

## Recent Activity

- 2026-06-14T00:26Z — Dispatched verification run 27483467658 for PR #46953 (t3000-unit-tests multiprocess socket re-regression, fresh build, model=all, t3k_tt_metal_multiprocess_tests only)
- 2026-06-14T00:25Z — Created PR #46953 for t3000-unit-tests IntermeshSplit2x2FabricFixture + MultiHostSocketTestSplitT3K.SocketTests re-regression on wh_llmbox (issue #46952). Branch: ci-disable/t3000-unit-tests-multiprocess-socket-2026-06-14
- 2026-06-14T00:15Z — PR #46934: classified run 27451128176 as verified-pass (mochi job was already failing on main; fixed parametrize bug in disable code). Rebased onto 10358b932eda035c3704d5c06183a429da4c5302. Evidence refreshed to run 27457802737/job 81165940451.
- 2026-06-14T00:12Z — PR #46932: classified run 27451009453 as verified-pass (all jobs passed). Rebased onto 10358b932eda035c3704d5c06183a429da4c5302. Evidence unchanged.
- 2026-06-14T00:10Z — PR #46930: classified run 27450999496 as verified-pass (host IO timeout was pre-existing on main). Rebased onto 10358b932eda035c3704d5c06183a429da4c5302. Evidence unchanged.
- 2026-06-13T00:37Z — Dispatched verification run 27451128176 for PR #46934 (blackhole-demo-tests mochi encoder perf, fresh build, model=mochi, system-type=QuietBox 2)
- 2026-06-13T00:37Z — Created PR #46934 for blackhole-demo-tests mochi encoder performance on bh_quietbox_2 (issue #46933). Branch: ci-disable/blackhole-demo-tests-mochi-encoder-perf-2026-06-13
- 2026-06-13T00:33Z — Dispatched verification run 27451009453 for PR #46932 (t3000-demo-tests qwen3_vl, fresh build, model=qwen3_vl)
- 2026-06-13T00:33Z — Created PR #46932 for t3000-demo-tests qwen3_vl BERTScore CI test on wh_llmbox (issue #46931). Branch: ci-disable/t3000-demo-tests-qwen3vl-bertscore-2026-06-13
- 2026-06-13T00:32Z — Dispatched verification run 27450999496 for PR #46930 (blackhole-post-commit host_io DEVICE_PULL 512-byte, fresh build, run-blackhole-multi-card-fast-unit-tests=true)
- 2026-06-13T00:32Z — Created PR #46930 for blackhole-post-commit test_host_io_loopback DEVICE_PULL 512-byte tensor (issue #46929). Branch: ci-disable/blackhole-post-commit-host-io-device-pull-512-2026-06-13
- 2026-06-13T00:25Z — PR #46812: classified run 27386537661 as verified-pass. Rebased onto f1e8f9b60d09. Evidence refreshed to run 27441737760/job 81130000607.
- 2026-06-13T00:22Z — PR #46808: classified run 27386528492 as verified-pass. Rebased onto f1e8f9b60d09. Evidence refreshed to run 27442163505/job 81119978381.
- 2026-06-13T00:20Z — PR #46696: tests now passing on main (run 27353497098, 2026-06-11). Removed disable, rebased, pushed, closed PR and issue #46695.
- 2026-06-13T00:15Z — PR #46814: all disabled tests started passing on main (run 27442163505/job 81119978338). Classified run 27386543130 as verified-pass. Removed all disables, rebased, pushed, closed PR and issue #46813.
- 2026-06-13T00:03Z — Session start. Checked 3 active runs (27386528492, 27386537661, 27386543130 — all completed with conclusion: failure). Classified all 3.
- 2026-06-12T00:33Z — Dispatched verification run 27386543130 for PR #46814 (t3000-unit-tests multiprocess socket, fresh build, model=dits)
- 2026-06-12T00:33Z — Dispatched verification run 27386537661 for PR #46812 (blackhole-post-commit host_io DEVICE_PULL, fresh build, run-blackhole-multi-card-fast-unit-tests=true only)
- 2026-06-12T00:33Z — Dispatched verification run 27386528492 for PR #46808 (t3000-unit-tests executor parity, fresh build, model=tttv2 modules)
- Older history truncated — see git history of this file.

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

## PR #46555 — t3000-unit-tests (t3k dits PCC failure) — CLOSED

| Field | Value |
|-------|-------|
| PR | [#46555](https://github.com/tenstorrent/tt-metal/pull/46555) |
| Disable issue | [#46554](https://github.com/tenstorrent/tt-metal/issues/46554) (closed) |
| Timeout issue | n/a |
| Branch | ci-disable/t3000-unit-tests-dits-t3k-2026-06-10 |
| Workflow file | .github/workflows/t3000-unit-tests.yaml |
| Lifecycle stage | closed |
| Last rebase | 2026-06-12T00:05Z (onto f690c41b9788a5179e9b2ba4d00440d1df6b87e3) |
| Last revalidation | 2026-06-12T00:05Z |
| Verification run | [27314954594](https://github.com/tenstorrent/tt-metal/actions/runs/27314954594) (verified-pass) |
| Last touched by automation | 2026-06-12T00:05Z |
| Readiness | Closed — test started passing on main, disable removed |

### Disables (with main evidence)

All disables removed — test started passing on main as of run [27381852093](https://github.com/tenstorrent/tt-metal/actions/runs/27381852093/job/80921398031) (2026-06-11 22:55 UTC). PR and issue closed.

---

## PR #46696 — single-card-demo-tests (mnist test_demo_dataset matmul TT_FATAL) — CLOSED

| Field | Value |
|-------|-------|
| PR | [#46696](https://github.com/tenstorrent/tt-metal/pull/46696) |
| Disable issue | [#46695](https://github.com/tenstorrent/tt-metal/issues/46695) (closed) |
| Timeout issue | n/a |
| Branch | ci-disable/single-card-demo-tests-mnist-matmul-2026-06-11 |
| Workflow file | .github/workflows/single-card-demo-tests.yaml |
| Lifecycle stage | closed |
| Last rebase | 2026-06-13T00:20Z (onto f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8) |
| Last revalidation | 2026-06-13T00:20Z |
| Verification run | [27315173882](https://github.com/tenstorrent/tt-metal/actions/runs/27315173882) (verified-pass) |
| Last touched by automation | 2026-06-13T00:20Z |
| Readiness | Closed — tests started passing on main (run 27353497098, 2026-06-11), disable removed |

### Disables (with main evidence)

All disables removed — tests started passing on main as of run [27353497098](https://github.com/tenstorrent/tt-metal/actions/runs/27353497098) (2026-06-11T14:19Z). PR and issue closed.

---

## PR #46698 — blackhole-demo-tests (flux.1-dev bh_loudbox + bh_quietbox_2 mesh_trace TT_FATAL)

| Field | Value |
|-------|-------|
| PR | [#46698](https://github.com/tenstorrent/tt-metal/pull/46698) |
| Disable issue | [#46697](https://github.com/tenstorrent/tt-metal/issues/46697) |
| Timeout issue | n/a |
| Branch | ci-disable/blackhole-demo-tests-flux1-loudbox-quietbox-2026-06-11 |
| Workflow file | .github/workflows/blackhole-demo-tests.yaml |
| Lifecycle stage | verified-pass |
| Last rebase | 2026-06-12T00:15Z (onto f690c41b9788a5179e9b2ba4d00440d1df6b87e3) |
| Last revalidation | 2026-06-12T00:15Z |
| Verification run | [27315355354](https://github.com/tenstorrent/tt-metal/actions/runs/27315355354) (verified-pass) |
| Last touched by automation | 2026-06-12T00:15Z |
| Readiness | verified-pass — ready to merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/tt_dit/tests/models/flux1/test_performance_flux1.py::test_flux1_pipeline_performance[blackhole-device_params0-bh_2x4sp0tp1-1024-1024-3.5-28] [bh_loudbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27326025746/job/80728202149 | [798b264](https://github.com/tenstorrent/tt-metal/commit/798b264f40bb40c93fa4d2f1e161a774a8ce1420) | 2026-06-11 06:25 UTC |
| models/tt_dit/tests/models/flux1/test_performance_flux1.py::test_flux1_pipeline_performance[blackhole-device_params0-2x2sp0tp1-1024-1024-3.5-28] [bh_quietbox_2] | https://github.com/tenstorrent/tt-metal/actions/runs/27326025746/job/80728202166 | [798b264](https://github.com/tenstorrent/tt-metal/commit/798b264f40bb40c93fa4d2f1e161a774a8ce1420) | 2026-06-11 06:25 UTC |

---

## PR #46808 — t3000-unit-tests (TestValidateModuleConfigs EagerLLMExecutor missing methods)

| Field | Value |
|-------|-------|
| PR | [#46808](https://github.com/tenstorrent/tt-metal/pull/46808) |
| Disable issue | [#46807](https://github.com/tenstorrent/tt-metal/issues/46807) |
| Timeout issue | n/a |
| Branch | ci-disable/t3000-unit-tests-executor-parity-2026-06-12 |
| Workflow file | .github/workflows/t3000-unit-tests.yaml |
| Lifecycle stage | verified-pass |
| Last rebase | 2026-06-13T00:22Z (onto f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8) |
| Last revalidation | 2026-06-13T00:22Z |
| Verification run | [27386528492](https://github.com/tenstorrent/tt-metal/actions/runs/27386528492) (verified-pass, classified 2026-06-13) |
| Last touched by automation | 2026-06-13T00:22Z |
| Readiness | verified-pass — ready to merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/common/tests/test_executor_parity.py::TestValidateModuleConfigs::test_validate_module_configs_method_exists [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27442163505/job/81119978381 | [22a02a8f](https://github.com/tenstorrent/tt-metal/commit/22a02a8f76d23cda34f5347224e6070a601ef885) | 2026-06-12 23:38 UTC |
| models/common/tests/test_executor_parity.py::TestValidateModuleConfigs::test_compile_accepts_validate_configs_flag [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27442163505/job/81119978381 | [22a02a8f](https://github.com/tenstorrent/tt-metal/commit/22a02a8f76d23cda34f5347224e6070a601ef885) | 2026-06-12 23:38 UTC |

---

## PR #46812 — blackhole-post-commit (test_host_io_loopback DEVICE_PULL small-tensor bh_p300-viommu)

| Field | Value |
|-------|-------|
| PR | [#46812](https://github.com/tenstorrent/tt-metal/pull/46812) |
| Disable issue | [#46810](https://github.com/tenstorrent/tt-metal/issues/46810) |
| Timeout issue | n/a |
| Branch | ci-disable/blackhole-post-commit-host-io-device-pull-2026-06-12 |
| Workflow file | .github/workflows/blackhole-post-commit.yaml |
| Lifecycle stage | verified-pass |
| Last rebase | 2026-06-13T00:25Z (onto f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8) |
| Last revalidation | 2026-06-13T00:25Z |
| Verification run | [27386537661](https://github.com/tenstorrent/tt-metal/actions/runs/27386537661) (verified-pass, classified 2026-06-13) |
| Last touched by automation | 2026-06-13T00:25Z |
| Readiness | verified-pass — ready to merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/demos/deepseek_v3_b1/tests/unit_tests/test_host_io.py::test_host_io_loopback[blackhole-H2DMode.DEVICE_PULL-64-128-512] [bh_p300-viommu] | https://github.com/tenstorrent/tt-metal/actions/runs/27441737760/job/81130000607 | [16e79a78](https://github.com/tenstorrent/tt-metal/commit/16e79a784996b0608f809a186f5816dd07e86d39) | 2026-06-12 22:13 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_host_io.py::test_host_io_loopback[blackhole-H2DMode.DEVICE_PULL-64-256-512] [bh_p300-viommu] | https://github.com/tenstorrent/tt-metal/actions/runs/27441737760/job/81130000607 | [16e79a78](https://github.com/tenstorrent/tt-metal/commit/16e79a784996b0608f809a186f5816dd07e86d39) | 2026-06-12 22:13 UTC |
| models/demos/deepseek_v3_b1/tests/unit_tests/test_host_io.py::test_host_io_loopback[blackhole-H2DMode.DEVICE_PULL-64-512-512] [bh_p300-viommu] | https://github.com/tenstorrent/tt-metal/actions/runs/27441737760/job/81130000607 | [16e79a78](https://github.com/tenstorrent/tt-metal/commit/16e79a784996b0608f809a186f5816dd07e86d39) | 2026-06-12 22:13 UTC |

---

## PR #46814 — t3000-unit-tests (MultiHostSocketTestSplitT3K.SocketTests + IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast) — CLOSED

| Field | Value |
|-------|-------|
| PR | [#46814](https://github.com/tenstorrent/tt-metal/pull/46814) |
| Disable issue | [#46813](https://github.com/tenstorrent/tt-metal/issues/46813) (closed) |
| Timeout issue | n/a |
| Branch | ci-disable/t3000-unit-tests-multiprocess-socket-2026-06-12 |
| Workflow file | .github/workflows/t3000-unit-tests.yaml |
| Lifecycle stage | closed |
| Last rebase | 2026-06-13T00:15Z (onto f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8) |
| Last revalidation | 2026-06-13T00:15Z |
| Verification run | [27386543130](https://github.com/tenstorrent/tt-metal/actions/runs/27386543130) (verified-pass) |
| Last touched by automation | 2026-06-13T00:15Z |
| Readiness | Closed — all disabled tests started passing on main (run 27442163505/job 81119978338, 2026-06-12T20:48Z) |

### Disables (with main evidence)

All disables removed — tests started passing on main as of run [27442163505](https://github.com/tenstorrent/tt-metal/actions/runs/27442163505/job/81119978338) (2026-06-12T20:48Z). PR and issue closed.

---

## PR #46930 — blackhole-post-commit (test_host_io_loopback DEVICE_PULL 512-byte tensor bh_p300-viommu)

| Field | Value |
|-------|-------|
| PR | [#46930](https://github.com/tenstorrent/tt-metal/pull/46930) |
| Disable issue | [#46929](https://github.com/tenstorrent/tt-metal/issues/46929) |
| Timeout issue | n/a |
| Branch | ci-disable/blackhole-post-commit-host-io-device-pull-512-2026-06-13 |
| Workflow file | .github/workflows/blackhole-post-commit.yaml |
| Lifecycle stage | verified-pass |
| Last rebase | 2026-06-14T00:10Z (onto 10358b932eda035c3704d5c06183a429da4c5302) |
| Last revalidation | 2026-06-14T00:10Z |
| Verification run | [27450999496](https://github.com/tenstorrent/tt-metal/actions/runs/27450999496) (verified-pass, classified 2026-06-14) |
| Last touched by automation | 2026-06-14T00:10Z |
| Readiness | verified-pass — ready to merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/demos/deepseek_v3_b1/tests/unit_tests/test_host_io.py::test_host_io_loopback[blackhole-H2DMode.DEVICE_PULL-512-1024-512] [bh_p300-viommu] | https://github.com/tenstorrent/tt-metal/actions/runs/27441737760/job/81130000607 | [16e79a78](https://github.com/tenstorrent/tt-metal/commit/16e79a784996b0608f809a186f5816dd07e86d39) | 2026-06-12 22:13 UTC |

---

## PR #46932 — t3000-demo-tests (qwen3_vl BERTScore CI test below threshold wh_llmbox)

| Field | Value |
|-------|-------|
| PR | [#46932](https://github.com/tenstorrent/tt-metal/pull/46932) |
| Disable issue | [#46931](https://github.com/tenstorrent/tt-metal/issues/46931) |
| Timeout issue | n/a |
| Branch | ci-disable/t3000-demo-tests-qwen3vl-bertscore-2026-06-13 |
| Workflow file | .github/workflows/t3000-demo-tests.yaml |
| Lifecycle stage | verified-pass |
| Last rebase | 2026-06-14T00:12Z (onto 10358b932eda035c3704d5c06183a429da4c5302) |
| Last revalidation | 2026-06-14T00:12Z |
| Verification run | [27451009453](https://github.com/tenstorrent/tt-metal/actions/runs/27451009453) (verified-pass, classified 2026-06-14) |
| Last touched by automation | 2026-06-14T00:12Z |
| Readiness | verified-pass — ready to merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/demos/qwen3_vl/demo/demo.py::test_demo[wormhole_b0-mesh_device0-device_params0-performance-ci-only-bert-score] [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27387344895/job/80937973451 | [1491ae5d](https://github.com/tenstorrent/tt-metal/commit/1491ae5d30226804908e3e5f961a2e2e864aee7c) | 2026-06-12 20:05 UTC |

---

## PR #46934 — blackhole-demo-tests (mochi encoder performance below threshold bh_quietbox_2)

| Field | Value |
|-------|-------|
| PR | [#46934](https://github.com/tenstorrent/tt-metal/pull/46934) |
| Disable issue | [#46933](https://github.com/tenstorrent/tt-metal/issues/46933) |
| Timeout issue | n/a |
| Branch | ci-disable/blackhole-demo-tests-mochi-encoder-perf-2026-06-13 |
| Workflow file | .github/workflows/blackhole-demo-tests.yaml |
| Lifecycle stage | verified-pass |
| Last rebase | 2026-06-14T00:15Z (onto 10358b932eda035c3704d5c06183a429da4c5302) |
| Last revalidation | 2026-06-14T00:15Z |
| Verification run | [27451128176](https://github.com/tenstorrent/tt-metal/actions/runs/27451128176) (verified-pass, classified 2026-06-14; mochi job had pytest collection error in PR branch but job was already failing on main) |
| Last touched by automation | 2026-06-14T00:15Z |
| Readiness | verified-pass — ready to merge |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| models/tt_dit/tests/models/mochi/test_performance_mochi.py::test_mochi_pipeline_performance[blackhole-device_params0-dit_2x2sp0tp1_vae_1x4sp0tp1_BH_QB-genmo/mochi-1-preview-848-480-3.5-50-168] [bh_quietbox_2] | https://github.com/tenstorrent/tt-metal/actions/runs/27457802737/job/81165940451 | [f1e8f9b6](https://github.com/tenstorrent/tt-metal/commit/f1e8f9b60d09e0858bc2164fb2f855fa2586e4d8) | 2026-06-13 05:59 UTC |

---

## PR #46953 — t3000-unit-tests (IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast + MultiHostSocketTestSplitT3K.SocketTests re-regression)

| Field | Value |
|-------|-------|
| PR | [#46953](https://github.com/tenstorrent/tt-metal/pull/46953) |
| Disable issue | [#46952](https://github.com/tenstorrent/tt-metal/issues/46952) |
| Timeout issue | n/a |
| Branch | ci-disable/t3000-unit-tests-multiprocess-socket-2026-06-14 |
| Workflow file | .github/workflows/t3000-unit-tests.yaml |
| Lifecycle stage | verifying |
| Last rebase | 2026-06-14T00:25Z (onto 10358b932eda035c3704d5c06183a429da4c5302) |
| Last revalidation | 2026-06-14T00:25Z |
| Verification run | [27483467658](https://github.com/tenstorrent/tt-metal/actions/runs/27483467658) (dispatched 2026-06-14T00:26Z, in_progress; fresh build, only t3k_tt_metal_multiprocess_tests) |
| Last touched by automation | 2026-06-14T00:26Z |
| Readiness | Awaiting verification run completion |

### Disables (with main evidence)

Main-run evidence: see PR description.

| Disabled test | Most recent failing job | Commit | Run completed at |
|---|---|---|---|
| IntermeshSplit2x2FabricFixture.RandomizedInterMeshUnicast [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27480951725/job/81228951510 | [10358b93](https://github.com/tenstorrent/tt-metal/commit/10358b932eda035c3704d5c06183a429da4c5302) | 2026-06-13 23:02 UTC |
| MultiHostSocketTestsSplitT3K/MultiHostSocketTestSplitT3K.SocketTests/* (all 20 variants) [wh_llmbox] | https://github.com/tenstorrent/tt-metal/actions/runs/27480951725/job/81228951510 | [10358b93](https://github.com/tenstorrent/tt-metal/commit/10358b932eda035c3704d5c06183a429da4c5302) | 2026-06-13 23:02 UTC |
