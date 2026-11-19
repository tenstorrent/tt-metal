# Stage Review

Verdict: clean-pass

Independent round 1 review of stage 8, datatype-sweep, for `ornith-ai/Ornith-1.5-9B`. Reviewed the live `main` worktree against base `f8630a5a52013134e41b38a97352a7fed9d691c2`, before stage commits. This review ran no hardware workloads, tests, servers, or resets and changed only this report.

Paths below are relative to `models/autoports/ornith_ai_ornith_1_5_9b/`; `D` means `doc/datatype_sweep/`.

## Required Work

- None. The selected `bfp8_all_lofi` policy satisfies the stated full-model thresholds, P1, construction, context, resumed-prefill, qualitative, and post-selection benchmark requirements. No precision waiver is used.

## Other Concerns

- The natural-text prefill result is close to the threshold: 462/512, or 90.234375%. Decode is 466/512, or 91.015625%, with top-5 511/512 and top-100 512/512. The raw `D/selected_default.json` reproduces these scores. The reported binomial uncertainty is appropriate; this finite reference does not establish downstream benchmark equivalence. It is not an additional acceptance gate.
- P1 uses four complete reasoning answers totaling 10,518 positions, with 1,024 traced decode positions. This satisfies the runner's unchanged position and precision bounds but offers less topic diversity than its default 30-text collection. `D/p1_final.json` and the README disclose the sample size.
- Most shared-suite outputs stop during coherent reasoning at 256 tokens. The matched HF controls behave similarly. The thermodynamics control has ten samples per backend; its fixed rubric finds HF 0/10 and TT 1/10 material-error flags, below the predefined excess threshold. This supports the limited regression verdict, not statistical equivalence or a completed-answer quality claim.
- Historical sweep rows precede later repairs and formatting, so their recorded source hashes are not all current. The selected empty-kwargs rerun fingerprints all 22 current TT Python files, reproduces the winning scores, and remains within the recorded baseline timing spread of the selected candidate. All five P1 score caches independently match the current source-plus-policy hash. The repaired BF16 controls have their own post-repair measurements. There is no unresolved shipment/evidence mismatch.
- Run-plan push state: stage-owned changes are to receive local checkpoint commits after this review; autonomous remote push is prohibited. No vLLM implementation or serving validation is part of this verdict.

## Hard-Check Gaps

- The actual stage checker reports four passing P1 checks but does not by itself establish candidate completeness, policy consumption, default-path performance, or capability preservation. This review checked those separately against source and raw artifacts rather than treating `stage_gate.log` or `artifact_audit.json` as proof.
- Runtime summaries are sufficient for this stage's policy-consumption requirement. I verified all 19 policies against their loaded projection tensors and compute objects, including layer exceptions, and inspected the actual consumers in `tt/model.py` and `tt/multichip_decoder.py`. No new profiler-only gate is needed.
- Capacity runs establish allocation, finite full-length execution, terminal decode, and trace-safe preparation. They do not claim full-model HF agreement at every one of 262,144 positions. The numerical evidence and capacity scope are separately labeled.

## Anomaly Ledger

- Observed anomaly: BFP4 policies look acceptable on the short HF-greedy AIME reference but lose substantial natural-text top-1 agreement.
  Evidence: `D/bfp4_*.json`, `D/sweep_results.json`, `D/natural_text_provenance.json`.
  Affected path: Attention, MLP, and LM-head precision candidates.
  Control or comparison: Identical actual-book-token reference scored by HF BF16 and every full-model candidate; BFP8 baseline and BF16 control.
  Likely subsystem: Weight-quantization error.
  Investigation performed: Combined and split attention trials, gate/up and down trials, head trial, and BFP4+LoFi evidence for every considered material group. Combined MLP and gate/up also compare HiFi2.
  Resolution: controlled. Every BFP4 candidate fails the full-model top-1 bar and is rejected; none is selected through an AIME-only or synthetic-PCC argument.

- Observed anomaly: BF16 activation and weight controls initially exceed L1 capacity during prepared prefill.
  Evidence: `D/AUTODEBUG.md`, `D/AUTOFIX.md`, preserved initial failure logs/JSON, `D/placement_smoke_bfp8_prefill_act_bf16.json`, `D/placement_smoke_bf16_hifi4.json`, `D/bf16_hifi4.json`.
  Affected path: Retained normalization/projection inputs and the special 512–1,024-row projection branch.
  Control or comparison: Compressed-policy placement and K16 versus BF16-specific DRAM placement and K8.
  Likely subsystem: L1 tensor/dataflow-buffer overlap.
  Investigation performed: Source-derived allocation ledgers, dtype-dependent physical-row placement, legal K8 configuration, host boundary checks, and tracked real-layer requests at 14 lengths from 511 through 32,769. Both repaired profiles retain six captures across 42 model and 42 sampling replays with program misses forbidden. Full-model accuracy and maximum-context capacity pass afterward.
  Resolution: fixed. The source retains original BFP8 placement/geometry and does not lower context or chunk caps.

- Observed anomaly: BF16 CCL control fails when a same-dtype conversion is used to request a memory-layout change.
  Evidence: Preserved `D/bfp8_ccl_bf16_initial*` artifacts, passing `D/bfp8_ccl_bf16.json`, and `tt/multichip_decoder.py::decode_reduce`.
  Affected path: Decode collective input preparation.
  Control or comparison: Same-dtype `to_memory_config` versus dtype-changing `typecast`.
  Likely subsystem: Tensor conversion API contract.
  Investigation performed: Explicit branch repair followed by focused and complete two-reference candidate execution.
  Resolution: fixed. The passing control is slower and is not selected.

- Observed anomaly: The first short-prefix harness call hits the sampling-initialization guard.
  Evidence: `D/short_b1_bfp8_mlp_lofi_initial_harness_error.json`, `tests/datatype_cache_probe.py`, and all four final `D/short_b1_*` / `D/short_b32_*` records.
  Affected path: Harness setup before teacher-forced decode.
  Control or comparison: Explicit first-step sampling reset with authoritative histories.
  Likely subsystem: Test harness request initialization.
  Investigation performed: Corrected harness initialization, then full 32-token-prefix/256-position checks at batch 1 and batch 32 for both KV dtypes. All users agree and meet the unchanged full-model bars.
  Resolution: fixed. The runtime guard remains intact.

- Observed anomaly: A selected thermodynamics response overstates the efficiency restriction, and most shared-suite responses are capped before a final answer.
  Evidence: Actual `D/qualitative/{hf,tt}/prompt*_seed0.json` text; all twenty `D/qualitative_entropy_control/{hf,tt}/prompt04_seed*.json` outputs; blind scores, key, rubric, and unblinded results; `doc/eval_readiness/limitations.json`.
  Affected path: Sampled chat-template generation.
  Control or comparison: Same pinned HF checkpoint, prompt tokens, sampling recipe, seeds, and 256-token cap. Reused HF controls are explicitly identified.
  Likely subsystem: Model behavior and finite generation budget; no mechanical token-feedback defect is visible.
  Investigation performed: Direct inspection of the six paired shared prompts and twenty entropy-control outputs, consistency of the blind-score hashes/counts, and the predefined ten-seed rule. The shared French translation completes correctly; expected thinking delimiters are not corruption.
  Resolution: controlled. The current comparison is within its fixed rule, and original-model limitations retain evidence links. No precision gate is waived.

- Observed anomaly: Untracked timing logs warn that allocations with live traces may be unsafe.
  Evidence: `D/selected_default.log`, `D/token_out_selected.log`; tracked `D/capacity_bfp8_all_lofi.json` and the BF16 placement smokes; `doc/optimized_full_model/logs/trace_positive.log.gz`.
  Affected path: Existing generator trace preparation and later eager prefill helpers.
  Control or comparison: Fresh tracked runs include program-cache allocations; the unchanged environment's prior positive control rejects a deliberately live unsafe buffer and succeeds after its release.
  Likely subsystem: Generic allocator warning when tracking is disabled for timing.
  Investigation performed: Inspected the serial runner's explicit `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, tracked repeated execution, frozen program counts, and the small implementation diff, which adds no broad allocation exemptions or new trace-lifetime scheme.
  Resolution: controlled. The selected policy passes tracked maximum-context/non-aligned prefill followed by terminal decode; timing evidence remains separately untracked.

- Observed anomaly: The initial readiness gate rejects the historical limitations-file schema.
  Evidence: `D/stage_gate_initial_schema_failure.log`, current `doc/eval_readiness/limitations.json`, and `D/stage_gate.log`.
  Affected path: Readiness metadata validation.
  Control or comparison: Required `items` schema with preserved historical records.
  Likely subsystem: Metadata schema compatibility.
  Investigation performed: Inspected final schema, evidence paths, and final actual stage-check output; verified that the final P1 artifact equals the selected runner result.
  Resolution: fixed.

- Observed anomaly: UMD reports an unknown motherboard/tray mapping, and one rejected BFP4-down run reports 1,343 MHz versus requested 1,350 MHz.
  Evidence: Hardware log headers, including `D/bfp4_down_lofi.log` and selected-run logs.
  Affected path: Discovery metadata and a rejected candidate's clock settling.
  Control or comparison: Four expected local Blackhole devices are discovered and closed successfully; selected and baseline serial runs complete. No device-health/reset failure appears.
  Likely subsystem: UMD metadata and clock tolerance, not model fallback.
  Investigation performed: Classified warning origin and affected run; checked command ordering and selected repetition evidence.
  Resolution: controlled. Neither warning invalidates the selected policy's accuracy or same-regime comparison.

## Scope Inspected

- Goal/skill paths: Supplied original stage contract; installed `stage-review`, `datatype-sweep` and `references/precision-reasoning.md`, `tt-device-usage`, `qualitative-check`, `tt-enable-tracing`, and model-bringup startup/validation instructions under `/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/`.
- Artifact paths: `D/README.md`, `D/work_log.md`, all 19 candidate JSON/config pairs, baseline repetitions, CSV/JSON ledger, selected/default/propagation/selection artifacts, main and natural reference hashes, P1 final/gate artifacts and local score caches, four capacity records, context contract, short-prefix controls, placement smokes, resume checks and both negative controls, token-out selected/baseline records, qualitative raw outputs and metadata, limitations, AutoDebug/AutoFix reports and failure logs, final host/hooks/stage logs, command ledger, and both Pareto PNGs.
- Code paths: All stage-modified TT code and host tests; `tt/precision.py`; datatype evaluation, reference, cache, reasoning-text, context-report, queue and plot runners; existing prefill/trace and benchmark consumers as needed; installed P1 cache-fingerprint logic.
- Commands run: Read-only `git status`, `git diff`, `rg`, `cat`, `sed`; Python JSON/CSV/hash/counter/chronology analysis; CPU-only `torch.load(weights_only=True)` of existing P1 score caches; gzip inspection of prior trace-positive evidence; image viewing of both pyplot artifacts. No test or model was executed.
- Independently re-derived results: Every matrix row's policy matches loaded tensor dtypes and compute objects; references still match their recorded hashes; all timed samples have 99 or 511 model replays and zero captures/releases; thresholds agree with raw scores; the selected row is the fastest P1/full-model pass; selected artifact and candidate agree; current selected default matches all source hashes; P1 caches match current source/config/text; the serial command ledger has no overlapping jobs.
- Performance assessment: Baseline median 113.574848 traced teacher-forcing tokens/s/user with 0.475605 spread; selected candidate 114.539849 and default 114.414255. Same original warmed token-out harness measures selected 116.141003 versus baseline 114.809884 resident tokens/s/user. Every resident repetition records 126 model/sampling replays and zero timed token/position/page refreshes, readbacks, captures, or releases. Reported TTFT is the harness's separate warmed first-token measurement, as disclosed.
- Capability assessment: Both KV dtypes and BF16 reference preserve 262,144 context; selected default has its own full-layer capacity run, non-aligned 262,143 prefill, terminal decode, and unchanged program count. Selected resumed-prefill passes unchanged thresholds with both negative controls failing as expected. No vLLM adapter was required or inferred.

## Residual Risk

- Accuracy and speed are supported for the stated references, workload shapes, mesh, and serial regime. The small speed gain is supported by both selected default reproduction and the independent token-out comparison but is not a serving benchmark guarantee.
- Long-context capacity evidence is not a full-model long-context accuracy benchmark, and batch-32 short-prefix evidence does not imply 32 simultaneous maximum-context requests fit.
- Later serving work must consume the common selected policy and preserve cache/context contracts; batching, serving trace reuse, and serving throughput remain future-stage validation.
