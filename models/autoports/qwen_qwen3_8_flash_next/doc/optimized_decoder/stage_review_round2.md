# Stage Review

Verdict: clean-pass

Independent follow-up round 2 of 3 for stage 3 optimized-decoder, Qwen/Qwen3.8-Flash-Next. This review checks the sole round-one Required Work item, the previous Other Concerns, and the added evidence/assessor. It does not repeat the full stage review. The live checkout remains based on `3cd9937dfdbb2917990443e2e937e4dd3b9e60c7`, with stage changes staged and uncommitted. No push is authorized.

## Required Work

- None. Previous P2, **“Exercise the changed two-row compact-expert allocation path under Watcher” — closed.**

  Evidence: `review_watcher_tail2_layer{0,1,3}.json`, their provenance sidecars, the layer-1 selection sidecar, raw logs, and `watcher/review_watcher_tail2_layer{0,1,3}/generated/watcher/watcher.log` cover final-default public prefill 34 and three changing traced decode inputs. The public implementation divides 34 into physical 32+2 (`tt/functional_decoder.py:157`); the inherited default compact limit is 2, and the optimized two-row branch calls `fused_compact_union_routes` and allocates its compact sparse/early-activation outputs in L1 (`tt/optimized_decoder.py:331`). The direct GDN/QSA runner constructs `OptimizedDecoder` without a policy override; the resident-PLE selection records `OptimizedDecoder`, `OptimizedSparseMoe`, BFP8 experts, LoFi, and an empty policy.

  All three provenance files record `TT_METAL_WATCHER=10`, `TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, and disabled device/streaming profilers. They finish, close the device, and have matching raw-log identities. Watcher streams contain 2/4/1 completed dumps for layers 0/1/3, respectively, clean detach, and no assertion/error/fatal/exception signature. All 207 recorded source identities in each new hardware run, including the control, match the current files. The optimized implementation identity is `032d0b666e87ed94972b4c6a9d8d844995519c1a44407000d0c93f862290c6e9`, matching the final repaired implementation reviewed in round one.

  GDN and QSA raw probes pass. Their prefill PCCs are 0.9999886751 and 0.9996632934; minimum decode PCCs are 0.9999773502 and 0.9995757937. The invoked test asserts two rounds of three changed-input replays, exact reset replay equality, and exact fresh-prefill equality. PLE completes all three decode comparisons, six replays of one capture, exact prefill/reset/replay equality, all four state-equality checks, and all runtime audits. Its strict legacy row predicate remains false, with its supported numerical disposition recorded below. This closes the requested coverage gap without a runtime modification or a need for new performance/capacity evidence.

## Other Concerns

- **Previous stop-rule handoffs — open, non-blocking.** Reduced KV precision, wider physical chunks/2D prefill, exact-GR and shared-down refinements, full-prefill expert L1 intermediates, and additional attention/config combinations remain future work under the already accepted stop-rule disposition. The work log retains the untried-work list. This follow-up introduces no claim that those options were experimentally rejected.
- **Previous activation-fixture limitation — open, non-blocking.** These runs use checkpoint weights and seeded Gaussian hidden inputs. `probe_optimized_real_ci.py` loads the target checkpoint; the shared test generates hidden inputs from seeds 1934/2234. Both PLE runs use the same probe, seed 1731, checkpoint, logical length, token construction, and packed-lookup cache. They support decoder correctness and safety, not natural-language generation quality.
- **Previous legacy result metadata — open, documentation-only.** The new GDN/QSA result properties inherit the same “deterministic synthetic” weight label from the shared test. The real-weight loader, checkpoint field, and source-bound runner disambiguate it. `acceptance_manifest.json` explicitly documents the label. Earlier QSA dtype metadata was outside this focused re-review and retains round one's disposition.
- **Previous local checkpoint/push note — open as the stage owner's post-pass action.** The local isolated commit and work-log SHA update follow this passing review. No commit or push was performed by this reviewer; push prohibition is taken from the goal/review contract.
- **PLE34 row-five diagnostic — controlled, non-blocking.** The optimized and matched fused probes both retain `passed:false` and exit 1 because the legacy predicate requires every row to reach 0.995. Both satisfy the user's whole-output 0.995 acceptance and diagnostic 0.99 floor. The new `ple34_row5` entry in `doc/eval_readiness/limitations.json` correctly records `original_model: "not_a_claim_of_literal_hf_failure"`; the fused control proves predecessor sensitivity, not a failure of unchanged literal HF.

## Hard-Check Gaps

- No missing required check remains in the focused follow-up. The plan-progress failures and PLE selection `completed:false` are retained accurately: the unchanged PLE runner writes its complete report, closes the device, and then raises `SystemExit(1)` for the strict row predicate. They are not evidence of an aborted hardware workload.
- `tests/assess_optimized_watcher.py` preserves raw results and checks the explicit whole-output/diagnostic predicates, tracked Watcher environment, current optimized-source identity, raw-log identities, clean Watcher detach, and matched row-five control. Its PLE checks also require strict decode, exact reset/state/replay, and runtime audits. The saved summary matches the raw prefill metrics and Watcher streams.
- The assessor is specific to these artifacts, not a general evidence validator: it does not itself validate the control's provenance/factory selection or all runner-source identities. This review checked those directly, verified the actual three decode entries, and bound the final raw PLE log JSON to the result JSON. That is sufficient evidence for this review; no new validator or rerun is required.
- `logs/review_followup_precommit_pass.log` records passing applicable hooks. The acceptance manifest still links the earlier final-stage hook run in its `source_checks` field; the follow-up log exists and was inspected. This is a documentation bookkeeping note, not another review-round requirement.
- The inherited allocation-tracker positive-control disposition from round one is unchanged. This follow-up changes neither the tracker/runtime nor allocation acknowledgments, so no new positive-control run is required.

## Anomaly Ledger

- Observed anomaly: Round-one Watcher evidence omitted the optimized physical two-row compact-union allocation path.
  Evidence: The previous P2 finding and the new `review_watcher_tail2_layer{0,1,3}` artifacts, provenance, source, and Watcher streams.
  Affected path: Compact indexed sparse expert output and early packed-activation allocation for a two-row prefill tail.
  Control or comparison: Final-default public 34 executes 32+2 for all meaningful kinds; each run follows it with changing traced decode and reset/replay checks.
  Likely subsystem: Evidence coverage; no corruption was observed.
  Investigation performed: Traced the public chunking and default compact/L1 branches, checked actual factory wiring, matched all recorded source identities, inspected raw completion and Watcher logs, and independently checked saved result predicates.
  Resolution: fixed. The missing Watcher branch is covered.

- Observed anomaly: The added PLE34 optimized probe and its matched fused control return raw false for prefill row 5.
  Evidence: Optimized whole/contribution global PCCs are 0.9981388703/0.9980098274; corresponding minimum row PCCs are 0.9946201857/0.9938173011. Fused global PCCs are 0.9982004687/0.9980757756, and its row 5 PCCs are 0.9944641349/0.9936444270. Each report has exactly one failing row per metric, no nonfinite rows, and three passing decode steps. Optimized minimum decode whole/contribution PCCs are 0.9973270775/0.9971834646.
  Affected path: PLE hidden-output prefill comparison on the seeded 34-token fixture. Row5 lies in the first 32-token chunk, before the new two-row tail. The optimized tail's row 33 whole/contribution PCCs are 0.9976523015/0.9974322952; its row 32 contribution PCC is 0.9978205927.
  Control or comparison: `review_tail2_fused_ple_control.json` uses the same checkpoint, packed-cache path, seed, hidden/token fixture, sequence length, and decode steps through the explicit fused factory. Its provenance and selection were inspected. Global optimized-minus-fused changes are approximately -0.00006160/-0.00006595; row 5 improves by approximately 0.00015605/0.00017287.
  Likely subsystem: Inherited resident-PLE numerical sensitivity with a small policy-dependent perturbation. This matched predecessor comparison alone does not isolate the arithmetic cause or demonstrate literal-HF failure.
  Investigation performed: Read the unchanged probe's metric and completion predicates, verified deterministic matching fixture construction and real factory selection, checked source/log identities, independently assessed global/row/decode/reset/state/runtime fields, and checked summary/limitations claims against the raw reports. No raw false status was relabeled as a strict-probe pass.
  Resolution: controlled under the explicit whole-output contract. All row diagnostics exceed 0.99; strict decode and exact state/replay checks pass. Token top-1 is inapplicable to isolated hidden streams. No new accuracy failure or Watcher safety failure is established.

## Scope Inspected

- Goal/skill paths: Original `/home/ttuser/qwen38-flash/multigoal_logs/03-03-optimized-decoder.prompt.txt`, `review_contract.md`, and installed 0.1.24 `stage-review`, `optimize`, `tt-device-usage`, `tt-enable-tracing`, model-bringup startup, and optimize evidence-checklist instructions.
- Artifact paths: `stage_review_round1.md`; focused `review_watcher_tail2*` and `review_tail2*` plans, progress, raw results, selection/provenance and summary; all three focused Watcher streams and corresponding raw logs; control raw log; follow-up pre-commit log; relevant README/work-log/checklist/acceptance-manifest notes and `limitations.json` entry.
- Code paths: New `tests/assess_optimized_watcher.py`; relevant portions of `probe_optimized_real_ci.py`, `test_synthetic_decoder.py`, `probe_ple_decoder.py`, `run_optimized_probe.py`, `run_optimized_job.py`, and `run_recorded.py`; public prefill chunking and optimized compact/L1 allocation/factory defaults. Existing code was inspected only to bind the new evidence to the required path and matched control.
- Commands run: Read-only `rg`, `cat`, `sed`, `git status`, `git diff --numstat`, `git rev-parse`, and short standard-library Python artifact checks. No TTNN import, device opening, hardware/server/test run, reset, benchmark, implementation modification, or nested reviewer was performed. The sole written artifact is this report.

## Residual Risk

- This is an independent inspection of recorded hardware evidence, not a reviewer hardware rerun. It closes the narrow missing branch and preserves the rest of round one's accepted scope and evidence.
- The new evidence does not extend capability to full-model generation, natural-language accuracy, multichip execution, serving trace reuse, or a larger batch/context combination. The raw PLE per-row miss remains disclosed for later model/precision evaluation.
- Stage bookkeeping, local isolated checkpoint creation, and SHA recording remain the stage owner's post-review work. No additional review round or hardware rerun is needed for the inspected follow-up while the implementation is unchanged.
