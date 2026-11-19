# Stage Review

Verdict: more-work-needed

Independent round 1 of at most 3. Stage 3 optimized-decoder, Qwen/Qwen3.8-Flash-Next. Reviewed the live `main` worktree based on `3cd9937dfdbb2917990443e2e937e4dd3b9e60c7`; stage changes are uncommitted. This review found one narrow required watcher-coverage gap, rather than an observed incorrect output in the selected policy. The numerical dispositions and final performance claims inspected below are supported.

## Required Work

- P2: Exercise the changed two-row compact-expert allocation path under Watcher.

  Evidence: `tt/optimized_decoder.py:331` distinguishes single-row compact routing from `fused_compact_union_routes` at lines 336–339. The new optimized path places compact sparse outputs and the early packed-activation destination in L1 at lines 359–388; the destination's expert dimension comes from `compact_ids.shape[-1]`. Thus the two-row union has a different indexed sparse/output allocation contract from single-token decode. All stage watcher provenance files invoke `probe_public_continuation`. Its constants are `STARTS=(31,33)`, `COUNTS=(8,7)`, and `LIVE_LENGTHS=(65,97,129)` (`tests/probe_public_continuation.py:45`). These produce physical prefill counts 31, 32, 1, 8, and 7, plus single-row decode, and never a physical two-row compact union. The final accepted watcher runs are `final_v3_watcher_continuation{0,1}` and `final_watcher_continuation3`. The non-watcher synthetic suite and `final_v3_real_ci0` do cover lengths 2/34/66, so numerical coverage exists; their provenance does not enable Watcher.

  Why this matters: `$optimize`'s evidence checklist explicitly requires the watcher battery to reach every prefill/decode path changed by the stage. One-row compact decode and the larger DRAM prefill branch do not exercise the changed L1 union-buffer shape. This is a missing required hardware-safety check, not a claim that the existing passing numerical tests are invalid.

  Required next step: Run a small final-factory watcher correctness check that reaches the two-row compact union for each meaningful kind, with `TT_METAL_WATCHER=10`, allocation tracking enabled including program-cache allocations, and the profiler disabled. A 34-token public prefill followed by changing traced decode inputs exercises the 32+2 tail and is sufficient; reuse the existing GDN/QSA test selection and a resident-PLE probe or an equivalent focused wrapper. Preserve logs/provenance and show clean completion plus existing PCC/reset/replay checks. No new geometry search, full capacity run, or performance rerun is required if implementation is unchanged. A fresh follow-up reviewer can inspect just this finding and the added evidence.

## Other Concerns

- Keep the documented stop-rule handoffs: reduced KV dtype, larger physical chunks/2D prefill, further exact-GR kernels, full-prefill expert L1 intermediates, shared-down refinements, and additional attention precision/config combinations. The work log records two consecutive retained gains below 3%, and does not pretend these untried paths were experimentally rejected. They are not blockers under the optimization stop rule.
- Target checkpoint weights were used, but the decoder activation fixtures are seeded Gaussian tensors, not recorded natural-language activations. This is disclosed, and the slower GDN precision choice is supported by actual target-weight failures rather than only random-weight CI. Later full-model/precision stages still need their own model-visible evidence; no text-quality conclusion follows from this stage.
- A few inherited result fields retain legacy metadata: `final_v3_real_ci0.json` properties say “deterministic synthetic” although `probe_optimized_real_ci.py` loads the real checkpoint; several earlier QSA result files say expert dtype BF16 although their selection sidecars identify BFP8 construction. Actual factory wiring, sidecars and final profiler rows resolve these ambiguities. Treat them as documentation notes, not a reason to rerun passing hardware evidence.
- Push state is taken from `review_contract.md`: no push is authorized, and a local isolated checkpoint is to follow a passing review. This review did not create a commit or infer push state from reflogs.

## Hard-Check Gaps

- The missing watcher branch above is the sole required gap found. The raw `passed:false` values for PLE4768 and QSA1024 are not themselves additional failures of the user's whole-output contract; independent controls and assessments were inspected rather than accepting scheduler status.
- The allocation-tracker positive control is inherited at `../functional_decoder/trace_alloc_tracking/positive_control.{json,log}`. It records rejection of a deliberately live buffer and successful replay after deallocation. Current tracked invocations explicitly set tracking to 1 and skip-program-cache to 0. No tracker/runtime implementation change is present in the stage-owned diff. The current manifest could link this reused control more directly; a new positive-control run is not required by this inspection.
- Tests and reports do not establish a complete-model roofline or token throughput. `profiler_accounting.md` correctly labels its compulsory expert-weight bound, report-estimated bandwidth, and physical tile FLOPs. Its expert-only denominator must remain distinct from whole-decoder or full-model efficiency.

## Anomaly Ledger

- Observed anomaly: Initial faster GDN policy fails real-checkpoint batch decode.
  Evidence: `final_batch0_32.json` request 3 decode PCC 0.99276167. Expert-HiFi4, shared-BF16, DRAM residual and expert-L1-off controls retain the failure. BF16 GDN/HiFi2 gives 0.99279869; BFP8/interleaved HiFi4 gives 0.99273765.
  Affected path: GDN packed z/a/b projection feeding recurrence and subsequent decoder output.
  Control or comparison: `final_batch0_32_control_attn16_hifi4.json` and `_attn16_interleaved.json` give request 3 PCC 0.99987632, reproduced by `final_v3_batch0_32.json`.
  Likely subsystem: Projection precision/fidelity sensitivity; no low-level kernel defect is proved.
  Investigation performed: Independent AutoDebug plus discriminating AutoFix controls; final GDN default is BF16/HiFi4. Final numerical, continuation, stress and performance runs use that source.
  Resolution: fixed. Earlier faster BFP8-GDN candidates are ineligible; they are not the reported optimized result.

- Observed anomaly: PLE4768 raw strict result remains false for one residual-subtracted prefill row.
  Evidence: `final_ple_capacity4768.json` whole-output PCC 0.99807674, minimum whole row 0.99518332, contribution row 7 at 0.99488053. Decode/reset/replay/runtime checks pass.
  Affected path: Resident PLE public prefill at the preserved prepared capacity boundary.
  Control or comparison: `final_ple4768_backend_control.json`, `ple4768_row7_diagnostic.json`, and `final_ple_capacity4768_contract.json`. The independent literal rerun is exact and the saved device-output identity matches; TT versus packed-HF control passes all rows, minimum contribution PCC 0.99854770. Packed HF versus literal row 7 is 0.99510699.
  Likely subsystem: Packed lookup sensitivity with a small additional optimized-policy perturbation.
  Investigation performed: Read the assessor predicates and raw/control metrics. Whole-output, diagnostic floor, strict decode, runtime audit, and exact replay/reset checks are separately enforced.
  Resolution: controlled under the explicit user contract. This is not an unchanged-literal-HF failure exemption. The stricter raw failure is preserved; 4767 also passes its original strict probe.

- Observed anomaly: QSA1024 timing plan fails its extra per-row predicate.
  Evidence: `final_perf_optimized_3_1024.json` default row remains false; repeated output/state equality passes. The strict local rows are approximately 0.993039 whole and 0.992520 contribution.
  Affected path: Long public QSA prefill with SDPA K chunk 128.
  Control or comparison: `qsa_long_diagnostic.json`, `qsa_long_assessment.json`, and `qsa_long_hf_control.json`. Independent full-HF comparison on saved exact inputs gives global PCC 0.99960095, minimum whole/contribution rows 0.99242079/0.99196941, and minimum decode PCC 0.99980420. K chunk 32 removes the strict local miss; expert HiFi4 does not.
  Likely subsystem: SDPA numerical sensitivity, rather than replay or persistent-state corruption.
  Investigation performed: Inspected the CPU control and its exact saved-comparison binding, unchanged legacy predicate, independent resets per decode input, and raw replay/state booleans. Compact and original HF JSONs are structurally identical.
  Resolution: controlled. Whole-output acceptance and the diagnostic floor pass; token top-1 is inapplicable to isolated hidden streams. This does not assert canonical HF failure.

- Observed anomaly: Inherited GDN request 16 and PLE fresh-prefill contribution diagnostics miss 0.995.
  Evidence: Final GDN batch request 16 contribution aggregate 0.99394786 and minimum row 0.99047929; corresponding independent HF GR-control sensitivity is 0.99418509/0.99061060. PLE fresh 65/129 cases retain contribution-row values 0.99491256/0.99467212.
  Affected path: Fresh prefill only; resumed and decode gates remain strict.
  Control or comparison: Final batch/continuation artifacts contain matched controls and current TT/control comparisons; `../eval_readiness/limitations.json` preserves the disclosed changes in control settings.
  Likely subsystem: Inherited arithmetic/packed-lookup sensitivity amplified downstream.
  Investigation performed: Re-enumerated sub-threshold metrics in the final reports and separated negative controls, control sensitivity, and actual TT outputs. Literal whole outputs pass; the new request-3 regression is repaired separately.
  Resolution: controlled within the explicit hidden-output contract, without claiming that changed control settings are unchanged literal HF.

- Observed anomaly: Native fused-SiLU separate experts pass their local timing comparator but fail independent HF decode.
  Evidence: `split_experts_final_matmul_activation.json` best 3.07589483 ms; `split_final_hf_activation.json` decode step 0 PCC 0.99213892.
  Affected path: Optional separate expert gate matmul with native fused SiLU.
  Control or comparison: Ordinary separate experts pass `split_final_hf.json` and measure 3.07662606 ms; final packed default measures 3.05184126 ms.
  Likely subsystem: Activation rounding/fusion numerical difference.
  Investigation performed: Inspected optional flag wiring, independent HF result, and retained rejected status. No first-API-error rejection was used.
  Resolution: controlled by rejection; neither optional control replaces the default.

- Observed anomaly: Final watcher suite has no physical two-row compact union execution.
  Evidence: The branch and watcher shape analysis in Required Work.
  Affected path: L1 compact expert output/activation allocation for two-row prefill.
  Control or comparison: Non-watcher length 2/34/66 numerical tests pass; single-row compact and larger prefill paths are watcher clean.
  Likely subsystem: Coverage gap, not an observed corruption.
  Investigation performed: Enumerated every stage watcher provenance file and derived physical chunks from the actual probe constants and public chunking code.
  Resolution: more-work-needed; perform only the focused missing watcher check.

## Scope Inspected

- Goal/skill paths: `review_contract.md`; installed `stage-review/SKILL.md`, `optimize/SKILL.md`, optimize references `method.md`, `evidence-checklist.md`, `perf-accounting.md`; `tt-device-usage/SKILL.md`, `tt-enable-tracing/SKILL.md`, and repository `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`. Installed package root: `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24`.
- Artifact paths: README, work log, checklist, acceptance/artifact manifests, selected policy, context-contract and limitations diffs; final synthetic XML, real-CI/HF, batch, continuation, watcher, 128/128 stress, 32768 context, PLE4767/4768, QSA-long controls; final timing raw JSONs and summary; geometry/tie/separate-expert searches; dense component sweeps; AutoDebug/AutoFix evidence; final per-op reports and underlying `ops.csv` rows; provenance/selection sidecars and final pre-commit log.
- Code paths: Entire `tt/optimized_decoder.py`; inherited public chunking/state API and projection dispatch; optimized factory wrappers, candidate timing harness, dense/residual probes, stress/synthetic tests, continuation probe, capacity/QSA assessors, and tracker positive-control probe. No implementation file was modified.
- Commands run: Read-only `rg`, `cat`, `sed`, `nl`, `git status`, and staged diff inspection; short standard-library Python scripts to enumerate JSON metrics and watcher environments, parse XML/CSV, compare compact JSON copies, verify source identities, and recompute headline comparisons. No TTNN import, device opening, hardware test, reset, server, or benchmark was run by this reviewer.
- Performance findings: Recomputed short final fused-to-optimized decode reductions of 39.1966%, 33.0867%, and 23.4058% for layers 0/1/3, with corresponding prefill reductions. The final BFP8/LoFi 27-geometry sweep and alternating tie confirmation support the chosen region; the tiny down-block-20 difference is a measurement tie. Final packed experts beat the legal tuned separate family. GDN/shared/QSA packed dense groups and separate PLE controls include conversions and splitting; failed larger L1 cases have adapted successful configurations.
- Runtime-policy findings: Actual final CSV rows show BF16 activations/BFP8 weights/LoFi for both expert roles, gate/down blocks 20/10, 40 cores, indexed active 10/512 decode, and variable prefill unions 104/158/187. GDN packed rows are BF16/HiFi4; ordinary DRAM-sharded QSA/shared/PLE rows are BFP8/HiFi2. Shared-down remains BFP8/HiFi4 as its deferred optimization note states. Separate BFP4 gate/up and down captures show BFP4/LoFi in the intended role. The selected policy is not inferred solely from constructor labels.
- Trace/context findings: Current tracking runs include program-cache allocations; exact output acknowledgments are narrow, and live continuation probes consume outputs before intervening work, preserve one trace across changed inputs/other-request prefills, and verify state isolation/page remapping. The current GDN source identity matches final repaired evidence. Earlier QSA evidence predates the GDN-only default repair, whose resolved QSA policy is unchanged. BF16 paged KV and advertised state context remain unchanged; 32768 component-prefix smoke scopes and the preserved PLE4768 profile are explicitly limited, without a new full-MoE long-prefill claim.

## Residual Risk

- This is an inspection review of saved device evidence, not an independent hardware rerun. The missing two-row watcher coverage must be closed before stage pass.
- Small decoder fixtures cannot establish full-stack generation, reasoning precision, multichip behavior, vLLM reuse, or simultaneous maximum batch/context capability; those were outside the authorized stage.
- Generic profiler FLOP percentages and physical tile padding do not establish useful token throughput. Clean whole-call timings, segmented-prefill caveats, and same-capture decode accounting remain the applicable claims.
- Once the focused watcher evidence passes, the existing correctness, performance, precision rejection, context, and stop-rule evidence need not be repeated merely to produce a stronger format. Local checkpoint creation and work-log SHA recording remain the stage owner's post-pass steps.
