# Stage Review

Verdict: more-work-needed

Independent inspection of stage 01, functional decoder for `Qwen/Qwen3.8-27B`, at 2026-10-23 14:04 UTC. This is an interim review of a live worktree, not a claim that the owner had submitted a completed stage. Branch `exp26-qwen38-bringup`; base/HEAD `0611d6e393882ce28e811b023b5f7d23bba1aced`. Only the functional decoder is in scope. No optimization, multichip, full-model, generation or serving work is requested by this review.

## Required Work

- P2: Finish the final correctness and trace-allocation validation on the repaired implementation.
  Evidence: `tests/validate.sh` now includes final B32 resume/replay for both kinds, linear B8/B16 and full B16 replay checks, full B1 FP64 correlation control, mixed-position B8 full-attention replay, and the synthetic pytest matrix. At inspection, `linear_b32_final.json`, `full_b32_final.json`, `full_b1_fp64.json`, `ragged_decode.json`, and `synthetic_pytest.xml` did not exist. The existing final-source watcher runs cover B1; performance runs disable allocation tracking. Earlier B32 resume results predate parts of the recurrence/configuration repair sequence and do not close this final validation matrix.
  Why this matters: the trace skill requires the captured variants to replay safely with program-cache allocations included. Mixed per-request positions and final real-shape synthetic checks are also unfinished acceptance evidence, rather than demonstrated failures.
  Required next step: complete the serial validation script with `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, retain results and logs, and resolve failures. Short sequence checks suffice for the missing batch variants; no repeat of the valid maximum-context or watcher runs is requested unless implementation changes invalidate them.

- P2: Complete and adjudicate the remaining decode performance rows.
  Evidence: complete `whole_layer_timing.json` files existed for linear B1/B8 and full B1/B8/B32 at the final inventory check. Linear B16 was still exporting after a source-location error; full B16 and the linear B32 decode-only replacement were not complete. The initial `profile_linear_b32.json` has correctness results but its capture lost signposts/markers and cannot supply performance. `performance.json` still contained only linear B1 and full B1/B32.
  Why this matters: the configured default decode batch set is 1, 8, 16 and 32. The stage has valid batch-1 timing, but its required supplemental matrix is unfinished. Process exit status does not prove capture completeness.
  Required next step: finish the pending reports, reject incomplete measured windows, and retain copied raw ops CSV, filtered CSV, human-readable table and timing provenance for each accepted decode row. For the linear B16 source-location failure, either prove that every measured-window event and device timing survived using the raw host/device evidence, or use a clean replacement capture. The linear B32 decode-only replacement is reasonable supplemental coverage because warmed prefill is already measured at B1.

- P2: Reconcile the final evidence index, workload record and anomaly classifications.
  Evidence: `doc/agentic_serving.json` still has `tested_batch_sizes: []`; README/work log and telemetry accurately mark work pending but are not final. Full B32 decode is 63195.406 us versus B1 2456.269 us. Of that, matmul kernel time is 60067.072 us versus 1923.443 us. Linear B8 matmul time is 15679.296 us versus B1 1977.859 us. Raw inputs have separate batch dimension B and a physical 32-row tile for each logical one-token row. This explains a material limitation that the final evidence should state explicitly.
  Why this matters: consumers need a consistent account of which batches were proved, which captures were accepted, and what the reported traffic percentage represents. The compulsory-traffic lower bound is not measured bus utilization or evidence of weight reuse across requests.
  Required next step: update README, work log, agentic-serving JSON, performance JSON and the local compact telemetry packet with final results and precise artifact paths. Classify the large-batch scaling using the measured operation times and geometry. Functional-stage disclosure supported by these measurements is sufficient; this finding does not request optimization. Retain the 4096-input/128-decode/B1/concurrency1 headline separately from the estimated agentic planning workload.

## Other Concerns

- Repository-wide ignore rules exclude `ops.csv`, every `*.csv`, `*.log`, and `generated/` watcher evidence. The owner confirmed accepted compact CSVs/correctness logs/watcher logs will be explicitly retained when checkpointing after clean-pass. Keep the multi-GB Tracy intermediates excluded. The files currently exist and were inspectable, so this does not invalidate their results.
- The saved `hf_modeling_reference.py` is a different complete source revision from installed Transformers 5.12.1. `AUTODEBUG.md` identifies both hashes and proves the full-attention class equivalence; tests import the installed package. Preserve that distinction in reproduction instructions. No reference substitution or gate bug was found.
- The custom recurrence kernel's introductory comments still describe normalization/exponentiation and padding as “host-side”; the wrapper actually performs these using TTNN on device. This is misleading commentary, not an observed fallback.

## Hard-Check Gaps

- Several early maximum-context JSON files lack source hashes. This is not a new rerun requirement: the chronological work log and unchanged prefill path support their use, and final-source watcher/profile provenance matches the current implementation. Future evidence should keep the hashes already emitted by the runner.
- Resume checks compare continued outputs with continuous execution, while snapshots are independent device clones by inspection. They do not separately serialize a resume-state PCC metric or implement a serving prefix manager. The stated decoder contract and per-kind ownership/byte accounting are present; neither additional metric nor a serving manager is required here.
- Only the last 128 HF outputs are compared in maximum-context prefill. The reference constructs state from the entire prefix, and TT computes every logical output. This is a declared comparison scope, not a context reduction.

## Anomaly Ledger

- Observed anomaly: original linear recurrence diverged at 16385 tokens.
  Evidence: `long_linear_debug/baseline_states.json/log` and `bf16_l2.json/log`; recurrent-state PCC approximately .9076.
  Affected path: linear attention's persistent recurrent state.
  Control or comparison: CPU chunk-schedule control approximately .999984 state PCC; changing L2 precision did not repair the failure.
  Likely subsystem: repeated state truncation through unpack/FPU arithmetic.
  Investigation performed: read AutoDebug/AutoFix, descriptor, compute kernel and final long-context logs; verified separate FP32 state/SFPU and matrix-consumer copies.
  Resolution: fixed for reviewed B1 long-context evidence; final-source B32 and intermediate batch validation remains required above.

- Observed anomaly: full-attention maximum-context decode PCC .690184/.609275.
  Evidence: `full_262142_decode_max.json` versus `full_262142_decode_k256.json`.
  Affected path: long-context SDPA decode.
  Control or comparison: identical real weights/seeds; repaired configuration gives .999960/.999992 and unchanged .996382 prefill.
  Likely subsystem: long online reduction and SDPA program configuration.
  Investigation performed: read AutoDebug/AutoFix and the explicit k256/core16/exact-exponent configuration; k512's L1 rejection is preserved.
  Resolution: fixed by the combined configuration; individual-knob causality is correctly not claimed.

- Observed anomaly: FP32 correlation rounded/clamped to 1.0 for a large output array.
  Evidence: `profile_full_b32.json` records same-tensor FP64 PCC .999974527 and relative RMSE .007774166.
  Affected path: host-side validation statistic.
  Control or comparison: same output/reference tensors, different accumulation precision.
  Likely subsystem: FP32 correlation accumulation.
  Investigation performed: inspected FP64 runner statistic and retained comparison metrics.
  Resolution: controlled; the queued full B1 FP64 control remains part of final validation. Historical 1.0 values must not be described as bitwise model parity.

- Observed anomaly: profiling process succeeded despite missing instrumentation data.
  Evidence: original linear B1 marker failure, linear B32 source-location/marker failure, and the owner's new linear B16 late source-location failure report.
  Affected path: capture/export, not decoder mathematics.
  Control or comparison: clean replacement linear B1 has all signposts and 3459 prefill/101 decode device operations; invalid B32 is excluded.
  Likely subsystem: profiler storage and Tracy source-location capacity.
  Investigation performed: independently parsed accepted B1/B8/full B32 windows and logs; verified timing formulas and trace IDs. B16 late-error adjudication and B32 replacement were pending.
  Resolution: more-work-needed for the pending rows.

- Observed anomaly: large-batch decode time grows nearly with batch size.
  Evidence: operation totals and input geometry in Required Work above.
  Affected path: dense projection/MLP matmuls with `[B,1,H]` input.
  Control or comparison: batch-1, linear B8 and full B32 raw profiles use the same BF16/HiFi4 policy.
  Likely subsystem: separate tiled batch matrices with a logical one-row M dimension; repeated work/weight reads are consistent with measured scaling.
  Investigation performed: summed raw matmul kernel durations and inspected physical/logical dimensions.
  Resolution: more-work-needed for final evidence classification only; no optimization gate is introduced.

- Observed anomaly: HF “falling back to torch” warning, local plotting/cache and optional Tracy GUI-copy warnings, motherboard/subset-MMIO notices, and small AICLK settling differences.
  Evidence: correctness/profile logs; both watcher logs contain no fatal/error/invalid/assert/sanitize/overflow entries.
  Affected path: CPU HF oracle, optional report tooling, and hardware discovery rather than the decoder forward.
  Control or comparison: source/runtime audit keeps Torch out of TT forward; accepted raw CSVs and human tables exist; local mesh device 0 is selected; reported device timing uses the profiler's 1350 MHz header.
  Likely subsystem: environment/reference/tooling metadata.
  Investigation performed: read complete small correctness logs and filtered profiler warnings, excluding compiler-command false matches; checked both watcher logs and output files.
  Resolution: controlled. Histogram bucket overflow counts are not dropped profiler markers.

## Scope Inspected

- Goal/skill paths: supplied stage-01 contract; installed `stage-review`, `functional-decoder`, `agentic-serving`, `tt-device-usage`, `tt-enable-tracing`, and `model-bringup` skills; local `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: `doc/context_contract.json`, `doc/agentic_serving.json`, README/work log, config/weight/input statistics, both watcher JSON/log trees, short/resume/B32/16385/262142/262144 results, gate and numerical AutoDebug/AutoFix records, `performance.json`, and the local telemetry packet `bf452eab-a14c-4c27-9161-a094297f5789.json`.
- Profiler paths inspected in depth: `tracy/linear_b1_complete`, `tracy/full_b1`, `tracy/full_b32`, and `tracy/linear_b8`, including raw `ops.csv`, human reports, timing JSON and corresponding runner logs. Full B8 completion was inventoried but not yet inspected in depth in this interim review.
- Code paths: all three files under `tt/`; `tests/run_decoder.py`, `ragged_decode.py`, `collect_perf.py`, `roofline.py`, `validate.sh`, `reference_control.py`, `test_functional_decoder.py`; relevant installed HF model source read through `docker exec ... sed` without imports.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, file inventory, `git status/branch/rev-parse/check-ignore`, and small Python standard-library scripts for AST parsing, SHA-256, JSON, CSV, marker/timing extraction and warning checks. No TTNN import, device access, reset, server, model test or long experiment was launched. Only this review report was written.
- Final implementation hashes: `tt/functional_decoder.py` = `e843748bd41908b43565470155d21c784669f28cef42d2976361a4193ee193db`; `tt/fp32_recurrence.py` = `ed602a0203895c94f7747e8d2c5b5ccd989d11bae713796ebd3b9254c676da77`; `tt/kernels/fp32_recurrence.cpp` = `89df91a22f9cd4d3d7d06eb28009d995f905d9a28105f75faf5bad67b7ee8796`. These match final-source watcher/profile provenance.

## Residual Risk

- No new stage-critical implementation defect was established by this inspection. The verdict reflects incomplete required validation/reporting plus the unclassified batch-scaling anomaly.
- Existing B1 evidence supports both layer kinds, arbitrary tested boundary tails, real weights, state resumption, paged-address refresh, deterministic traced replay, and the full advertised context. Full-attention maximum-context prefill has less PCC margin than short context and compares a declared final-output subset.
- This is one device and one layer of each meaningful kind. It does not establish every batch/context Cartesian combination, generator-wide trace reuse, sampling feedback, complete-model quality, multi-user serving capacity, or optimized performance.
- A follow-up independent review must assess the completed final artifacts and any intervening implementation changes before clean-pass or a local checkpoint commit.
