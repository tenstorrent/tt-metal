# Stage Review

Verdict: clean-pass

Independent final rereview of stage 01, functional decoder for `Qwen/Qwen3.8-27B`, on 2026-10-23 at approximately 14:27 UTC. Branch `exp26-qwen38-bringup`; base/HEAD at review `0611d6e393882ce28e811b023b5f7d23bba1aced`. The initial review is preserved in `STAGE_REVIEW_initial.md`. All initial required-work findings are closed. This verdict covers the functional decoder only.

## Required Work

- None.

## Other Concerns

- Large-batch performance is a documented functional-layout limitation. The independent batch planes and padded one-token matmul geometry explain the measured scaling. No optimization is claimed or required for this stage.
- The staged evidence is substantial: 200 stage-owned files, approximately 432 MB, including eight accepted raw ops CSVs. The largest CSV is approximately 123 MB. This review authorizes stage closure under the specified local-checkpoint workflow; no remote publication is part of the task.
- Checkpoint creation and recording its SHA intentionally follow this clean-pass. They are owner handoff steps, not missing implementation or validation gates.

## Hard-Check Gaps

- No remaining hard-check gap prevents this stage from passing.
- Maximum-context prefill compares the final 128 HF outputs while constructing reference state from the entire prefix; TT computes all logical outputs. This scope is explicit in the context contract and README.
- Some early maximum-context JSON files predate provenance hashing. The initial inspection established chronology and unchanged relevant paths. The only implementation changes since that review are proven comment/docstring edits, so no numerical revalidation of those runs is needed.
- Prefix resumption is proved through continued-output PCC against continuous execution and independent device clones. The API, per-kind hit rule, state/payload bytes and release lifetime are documented. A serving prefix manager and generator-wide trace cache are outside this functional-layer contract.

## Anomaly Ledger

- Observed anomaly: long linear recurrence originally failed PCC.
  Evidence: `long_linear_debug/baseline_states.*`, `bf16_l2.*`, `fp32_local_v2.*`, and `linear_262142_decode_max.*`.
  Affected path: FP32 recurrent state.
  Control or comparison: the CPU chunk-schedule control preserved state; L2 changes did not fix the device error; the local FP32 state/SFPU repair passes long context and the final B8/B16/B32 tracked replay cases.
  Likely subsystem: state truncation through unpack/FPU arithmetic.
  Investigation performed: initial code/kernel inspection plus final source-equivalence and batch-artifact verification.
  Resolution: fixed.

- Observed anomaly: maximum-context full-attention decode originally gave PCC .690184/.609275.
  Evidence: `full_262142_decode_max.json`, `full_context_debug/AUTOFIX.md`, and `full_262142_decode_k256.json`.
  Affected path: long-context SDPA decode.
  Control or comparison: the combined k256/core16/exact-exponent configuration produces .999959588/.999991536 with the same real weights/seeds; final B1/B8/B16/B32 checks also pass.
  Likely subsystem: long online reduction and SDPA program configuration.
  Investigation performed: inspected the explicit configuration and preserved failed-capacity/control evidence.
  Resolution: fixed by the combined configuration; individual-knob causality remains appropriately unclaimed.

- Observed anomaly: large-output FP32 correlation rounded/clamped to 1.0.
  Evidence: `profile_full_b32.json` and `full_b1_fp64.json`.
  Affected path: host-side validation statistic.
  Control or comparison: same-tensor full-B32 FP64 PCC .999974527 and relative RMSE .007774166; final full-B1 4096-token FP64 PCC .999962094 and relative RMSE .008713095.
  Likely subsystem: FP32 statistical accumulation.
  Investigation performed: verified the FP64 implementation, final source hashes and exact JSON correspondence to the final validation log.
  Resolution: controlled; historical 1.0 PCC is not presented as bitwise model parity.

- Observed anomaly: incomplete original linear B1/B32 profiler captures despite successful process completion.
  Evidence: rejected capture logs; accepted `tracy/linear_b1_complete` and `tracy/linear_b32_decode`.
  Affected path: profiling/export.
  Control or comparison: the B1 replacement has complete warmed prefill/decode windows. The B32 replacement retains 4096 prefill and 128 traced steps, profiles the final decode, and disables only optional GUI event push. Its decode is 70860.338519 us over 486 device ops; its log has no dropped-marker/source-location failure.
  Likely subsystem: profiler capacity and Tracy source-location ceiling.
  Investigation performed: independently extracted accepted signposts, firmware envelopes, op counts and trace ID/replay session; verified excluded captures are not indexed as performance evidence.
  Resolution: controlled through valid replacement captures. Supplemental B32 warmed prefill is explicitly unmeasured; required B1 warmed prefill exists for both kinds.

- Observed anomaly: linear B16 hit a source-location limit during shutdown after both measured windows.
  Evidence: `tracy/linear_b16/capture_validation.json`, `measured_cpp_device_rows.csv`, raw C++ profiler data, and `profile_linear_b16.log`.
  Affected path: late optional GUI device-event push.
  Control or comparison: all four exported signposts exist; all 11171 prefill and 286 decode rows exactly match raw C++ data by global call count, trace and replay session. The raw C++ time spans contain exactly those row counts, so this check also excludes unexported device rows within the windows.
  Likely subsystem: optional GUI event capacity after numerical measurement.
  Investigation performed: independently verified all 11457 measured records, firmware timestamps, raw-C++ SHA-256 and event ordering.
  Resolution: controlled; the numeric windows are complete and accepted.

- Observed anomaly: large-batch decode time grows nearly with batch size.
  Evidence: raw batch profiles, README, `PERFORMANCE.md`, `performance.json` and `tests/roofline.py`.
  Affected path: separate `[B,1,H]` tiled batch planes in dense projections/MLP.
  Control or comparison: full B32 matmuls consume 60067.072 us versus 1923.443 us at B1; linear B8 consumes 15679.296 us versus 1977.859 us at B1, under the same BF16/HiFi4 policy.
  Likely subsystem: independent padded matmul planes and repeated weight movement.
  Investigation performed: initial raw geometry/time inspection; final report and traffic-formula verification.
  Resolution: controlled as a measured functional-layout limitation. DRAM percentages are explicitly estimated traffic/peak, with separate batch-plane weight reads, not hardware utilization counters.

- Observed anomaly: HF Torch-fallback warning, plotting/cache and optional GUI-copy warnings, motherboard/subset-MMIO notices, small clock-settling differences.
  Evidence: correctness/profile logs and both watcher trees.
  Affected path: CPU oracle, report tooling and discovery metadata.
  Control or comparison: runtime Torch/conversion guards pass; device selection remains one local ASIC; profiler timing uses its recorded 1350 MHz header; both watcher logs are clean.
  Likely subsystem: environment/reference/tooling.
  Investigation performed: warning scans and source/boundary inspection in the initial review; final validation/new accepted profile logs contain no unsafe-allocation, corruption, fatal or assertion failures.
  Resolution: controlled. The saved installed HF source is now clearly distinguished from the upstream inspection snapshot; misleading kernel comments were corrected.

## Scope Inspected

- Goal/skills: the supplied stage-01 contract and installed `stage-review`, `functional-decoder`, `agentic-serving`, `tt-device-usage`, `tt-enable-tracing`, `model-bringup`, plus the checkout's trace-correctness guide. The initial review supplies detailed code/context/watcher inspection.
- Final correctness: `final_validation.log`; linear B8/B16/B32 and full B16/B32 final real-weight JSONs; full B1 FP64 control; B8 ragged positions `[33,31,1,385,128,32,129,127]`; synthetic JUnit, summary and all 14 individual results. All six final non-ragged real-weight JSONs exactly match records in the log. The validation script exports allocation tracking 1 and program-cache skipping 0 before fresh test processes.
- Final acceptance results: all final real-weight prefill/decode/changed-input/resume PCCs exceed .995, all repeat checks are bitwise deterministic, and runtime fallback audits pass. Mixed-position same-trace PCC is .999965301/.999954488. JUnit reports 14 tests, zero failures/errors/skips, 153.336 seconds. Independently recomputed synthetic minima are .999832111/.999822253/.999858970 for linear and .999607100/.999788174/.999788174 for full prefill/decode/changed decode.
- Performance: independently recomputed all eight decode firmware envelopes and all seven measured prefill envelopes from raw `ops.csv`, matched timing JSON and performance JSON, checked trace 0/replay 132, checked human/CSV reports, and rederived roofline fields from the pure host calculation. B1 prefill/decode is 284743.540741/2660.428148 us linear and 108787.729630/2456.268889 us full.
- Final records: README, work log, `PERFORMANCE.md`, context contract, agentic-serving JSON, installed HF source hash, source-cleanup proof, and local telemetry packet `bf452eab-a14c-4c27-9161-a094297f5789.json`. Telemetry retains the exact 4096-input/128-decode/B1/concurrency1 workload and matches both kinds' B1 timing, FLOP/traffic and peak fields.
- Source continuity: independently reconstructed the old source from the documented replacements and matched the initial hashes; compared Python AST excluding the changed module docstring and C++ content excluding comments. Executable implementation is unchanged. All final non-ragged real and synthetic tests record current hashes.
- Current implementation SHA-256: `tt/functional_decoder.py` = `a0aa08c6d5dbd5296b57e602832b6c06e4cb1971f4c9b09eec783165e66768f9`; `tt/fp32_recurrence.py` = `ed602a0203895c94f7747e8d2c5b5ccd989d11bae713796ebd3b9254c676da77`; `tt/kernels/fp32_recurrence.cpp` = `78cf3afd692ed074a047ec402b8e8b49f56183e833d7687f9284ef8af4472c48`.
- Retention: inspected the 200-file Git index. Every file is inside this autoport; accepted raw/filtered CSVs, human tables, correctness logs, final results and watcher logs/kernel maps are staged. Multi-GB `.logs/` and `reports/`, rejected raw capture directories, weights and pytest symlinks are excluded.
- Commands: read-only Git inspection, file reads, and Python standard-library/host-only scripts for AST/hash, JSON/XML/CSV, timing, provenance, log and index checks. No TTNN import, hardware use, reset, server, target test or long experiment was performed. Only this final review report was overwritten.

## Residual Risk

- This proves one device and one real-weight representative of each meaningful layer kind, with the advertised 262144-token context and stated boundary/batch tests. It does not prove every batch/context Cartesian combination.
- Long full-attention prefill has less PCC margin than short context and a declared 128-output comparison subset. The entire valid prefix contributes to reference state.
- This stage does not establish full-model text quality, sampling feedback, generator-wide trace reuse, multi-user serving capacity, or optimized performance. Those are outside the user's authorized scope.
- Local checkpointing should retain the final report and exact stage-owned artifacts. No required implementation, validation or evidence work remains before that handoff.
