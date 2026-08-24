# Milestone A 2D Module Status

This page records the completed exit-gate evidence for Milestone A of `tttv2_2d_modules_plan.md`.
Results below are transcribed from
`tttv2_2d_modules_work_log.md` and
`tttv2_2d_modules_galaxy_prefetcher_work_log.md` as of 2026-08-19.

## Scope

Milestone A adds reusable modules for the canonical Wormhole Galaxy logical mesh `(8, 4)`:

- `Embedding2D`, `RotarySetup2D`, `RMSNorm2D`, `Attention2D`, `MLP2D`, `LMHead2D`, and
  `Sampling2D`;
- shared Galaxy CCL/resource bindings and `Prefetcher2D`; and
- an immutable, topology-neutral batched-prefill policy in the common runtime.

There is no `Penalties2D`. The 2D modules expose explicit config construction and have no
`from_model_args` compatibility API.

## Verification Status

| Area | Host evidence | Recorded WH `(8, 4)` evidence | Status |
| --- | --- | --- | --- |
| Embedding2D | Focused suite: 11 passed | Llama and Qwen decode batch 32 plus prefill 128/2048, each repeated, PCC >= 0.99 | Qualified for recorded Milestone A cases |
| RotarySetup2D | Focused suite: 13 passed | Llama and Qwen decode plus prefill 128/2048, each repeated, PCC >= 0.99 | Qualified for recorded Milestone A cases |
| LMHead2D | Focused suite: 19 passed | Llama and Qwen decode/prefill final-token batches repeated, PCC >= 0.99; Qwen padding mask checked exactly | Qualified for recorded Milestone A cases |
| Sampling2D | Included in final 1259-test host gate | Qwen forced argmax repeated with exact tokens and padded-vocabulary exclusion | Qualified for the required forced-argmax hardware case; stochastic hardware is not recorded |
| RMSNorm2D | Focused contracts: 16 passed | Llama/Qwen batch-32 fused residual decode repeated; distributed prefill 128/2048 repeated; head-local Q/K repeated, all PCC >= 0.99 | Qualified for recorded Milestone A cases |
| MLP2D | Focused MLP/Galaxy/prefetch suite: 73 passed | Llama and Qwen decode plus prefill 128/2048, each repeated, PCC validated; complete file: 4 passed | Qualified for recorded Milestone A cases |
| Attention2D | Host suite: 64 passed | Llama-70B and Qwen3-32B repeated decode plus prefill 128/2048; output and K/V cache PCC >= 0.99; combined file: 2 passed | Qualified for recorded Milestone A cases |
| Galaxy CCL/resources | Concrete CCL/resource/composition host contracts included in final gate | Repeated MLP/RMS paths and fused Attention axis-1 decode pass with clean teardown | Qualified for Milestone A; non-fused Attention decode is not required or qualified |
| Prefetcher2D | Concrete composition regression: 29 passed | Repeated Llama/Qwen MLP decode consumes production-prefetched weights and tears down cleanly | Qualified for recorded Milestone A integration |
| Batched-prefill policy | Full prefill runtime: 144 passed; final integrated host gate: 1259 passed | Physical-32 capture/replay contract covers 128/1024/2048 with refreshed 31/32 rows and slots | Host lifecycle qualified; no real-device trace run recorded |

The refreshed integrated host gate recorded `1259 passed, 1 skipped, 9 warnings in 251.51s`. Host mocks
establish config, validation, ownership, and failure-path behavior; they do not substitute for
real-device numerical, cache, repeat-invocation, or teardown evidence.

## Exit-Gate Result

Milestone A passes its exit gate. The final Attention hardware qualification establishes:

- repeated Llama-70B and Qwen3-32B decode with output and K/V cache PCC >= 0.99;
- repeated 128- and 2048-token prefill with output and K/V cache PCC >= 0.99;
- Qwen head-local Q/K normalization and model-derived fused-QKV head geometry; and
- clean sequential execution of both model variants in one process: `2 passed in 53.93s`.

The former Attention axis-1 blocker was resolved by using the production fused
`all_reduce_create_qkv_heads` Ring primitive on the qualified 6U Galaxy topology. Production-aligned
row-wise SDPA core selection, explicit cache typecasts, worker-only prefill placement, and
model-derived local QKV core counts completed the numerical path. Final focused and combined runs
closed all 32 devices normally without a reset.

The failed non-fused decode experiments were specific RS/AG adapter and resource recipes. They do
not establish a defect in the general-purpose TTNN collectives, which pass in other Milestone A
paths. That alternate Attention decode composition is unqualified and unnecessary; it is not an
open Milestone A blocker or a required CCL follow-up.

## Post-Record Module Corrections

Milestone B step 1/4 (see `tttv2_2d_modules_milestone_b_work_log.md`) corrected two
module contracts after the evidence above was recorded:

- `Attention2D` now requires `wo` with source shape `(n_heads * head_dim, dim)`
  instead of `(dim, dim)`, which is the only way to express Qwen3-32B's real
  decoupled head dimension. The two shapes coincide for every geometry the
  recorded evidence covers, so no recorded numerical result changes.
- `LMHead2D` now also accepts a column-local activation width (`dim / 4`), which
  is what a device activation from the column-sharded residual stream carries.
  The recorded hardware qualification only passed host `LazyWeight` inputs.

Both changes are host-tested; neither has been re-run on hardware.

## Modularity Scorecard

| Required item | Evidence | Assessment |
| --- | --- | --- |
| New 2D/model files | Added five new functional module implementations, `Prefetcher2D`, Galaxy `ccl.py`/`resources.py`, package exports, and focused tests; MLP2D and RMSNorm2D were completed in their existing files | Within Milestone A boundaries |
| Existing shared files changed | `llm_runtime/prefill/config.py`, `plan.py`, and `runtime.py` add and consume a generic immutable batching policy; `modules/README.md` documents the inventory; existing MLP2D/RMSNorm2D files are milestone-owned | Runtime changes are limited to generic policy delegation; the final integrated regression passed |
| Why config alone was insufficient | Eligibility was previously encoded directly in planner/runtime decisions, so the resolved policy had to be threaded through the planner call and consumed mechanically | Narrow shared plumbing with focused tests |
| 1D module implementation files changed | `git status` shows no changed `models/common/modules/**/*_1d.py` file | Required value met in the current diff |
| Default runtime behavior changed | The default policy preserves prior values; final integrated runtime/module host gate passed 1259 tests | No intentional default change observed |
| 1D regressions | Full selected 1D matrix completed after separately finishing Sampling1D (`140 passed, 50 deselected`); no 1D implementation file changed | Exit-gate regression evidence recorded |
| Common-code topology assumptions | Batched-prefill eligibility and physical batch selection were fixed in common planning code | Moved behind topology-neutral immutable policy; no Galaxy/model branch was added |
| Boundary leakage | Static topology and model differences are represented in 2D configs or injected Galaxy collaborators; runtime diff contains no Galaxy, Llama, Qwen, Wormhole, 2D, or `(8, 4)` execution branch | Module/config/model boundary preserved in the reviewed diff |

## CCL Follow-Up

Galaxy CCL remains separate from `models/common/modules/tt_ccl.py`. After both reconstructed models
pass their later milestones, evaluate whether their APIs can share an owner. The overlap includes
collective topology, semaphores, persistent buffers, and subdevice identity; Galaxy additionally
requires mode-specific resource keys, exact tensor/sequence plans, adjacent semaphore windows, and
explicit sender/worker subdevice lifecycle.
