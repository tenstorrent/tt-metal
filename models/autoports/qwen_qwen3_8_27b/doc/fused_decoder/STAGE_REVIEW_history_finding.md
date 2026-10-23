# Stage Review

Verdict: more-work-needed

Independent final review of stage 02, Qwen/Qwen3.8-27B, on live branch `exp26-qwen38-bringup`, base `2d41497c9965b277c78dc033b4d0f8bdd49b261e`. Reviewed runtime SHA256: `f38fb8f9b2f193fe8f27e11f4b5908d99a428eb27e3eebc3601c49c60f41e28c`, byte-identical to `row_major_state_candidate.py`. One concrete graph rewrite remains unresolved. No runtime correctness defect was identified in the inspected evidence.

## Required Work

- P2: Remove or earn the rejection of the remaining prefill history tilize/untilize roundtrip.
  Evidence: `tt/fused_decoder.py:165` applies `to_layout(ROW_MAJOR)` after slicing a tiled projection at the nonaligned three-row history offset. In `tracy/final_linear/ops.csv`, every one of the 32 prefill chunks executes `UntilizeDeviceOperation -> SliceDeviceOperation -> TilizeWithValPaddingDeviceOperation -> UntilizeWithUnpaddingDeviceOperation -> CopyDeviceOperation`. The two adjacent conversion operations total 353.240 us of summed device-kernel duration across the measured prefill. This sum is evidence of removable work, not a prediction of whole-layer speedup. `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:244` selects the row-major path for nonaligned begins, line 393 converts the input to row-major, and lines 196–205 restore the original input layout before returning. The outer conversion immediately undoes that restoration. `PATTERNS.md` has no measured rejection of moving the conversion before this slice, while the work log claims history-update roundtrips have been removed.
  Why this matters: The original stage contract requires exhausting relevant graph rewrites and eliminating unnecessary tilize/untilize in both measured phases. Selecting row-major persistent history removed the decode roundtrip but leaves this newly exposed prefill roundtrip. This is an observed redundant conversion pair with an available API-level adaptation, not a request to begin later-stage matmul or precision tuning.
  Required next step: Test converting the projection to row-major before the `length >= 3` history slice so that the slice returns row-major directly, preserving the efficient short-history branch. Adapt if necessary and compare whole-layer timing; retain it only with correctness and performance evidence, or document a concrete correctness/op-contract/slower-measurement rejection. Cover the changed history branch at its boundary and representative nonaligned/tile/chunk tails, continuation/snapshot and a split batch. Reuse unchanged decode/full-attention evidence where source and execution dependencies establish it remains valid. If selected, reproduce affected linear prefill performance and preserve or verify the traced-decode winner; update inventory, final source/provenance, performance and packet before fresh independent review. No full unrelated validation sweep is requested solely for this small change.

## Other Concerns

- The final docs and metadata substantially close the initial reconciliation finding. The exact-ID packet exists and correctly awaits independent review and local commits. Commit absence is intentional at this point; local stage-owned commits remain a post-clean-pass action.
- `doc/agentic_serving.json` points its prefix-contract link at `fused_decoder/README.md#contract`, which is not a current heading. The contract text itself is present under `API, state and context`; correct the anchor during the remaining documentation update.

## Hard-Check Gaps

- No new evidence format or extra runtime counters are required. The wrappers enable `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` before process import; retained successful tracked tests and source establish the allocation gate. No allocation exemptions or corruptible-buffer acknowledgments appear in this runtime or its runners.
- Recomputed hashes match the runtime and all indexed final linear evidence. Candidate15 and final source differ only in the linear allocation branch and history update; the full-attention dependency path is AST-identical. Reusing its retained evidence is supported.
- `synthetic_pytest.xml` has 18 tests with zero errors/failures/skips; the affected rerun has 9 tests with zero errors/failures/skips. All 18 selected final synthetic cases exceed 0.995 and report deterministic replay. Real final evidence covers both watcher runs, requests of lengths 1/2/3/4/31/32/33/127/128/129/385 and short reuse, odd linear B3, ragged full B8, all six B8/B16/B32 4096+128 workloads, and exact/nonaligned maximum context. All inspected scores meet the unchanged acceptance bar.
- Raw profiler windows rederive exactly: linear 83381.831852 us prefill / 2328.098519 us decode; full attention 93214.924444 / 2187.320000 us. Functional controls rederive as 284743.540741 / 2660.428148 and 108787.729630 / 2456.268889 us. The selected full envelopes beat all retained comparable candidate profiles. Decode windows contain trace ID 0, replay session 132. Prefill is warmed eager execution. Windows include gaps; summed kernels are not used as their denominator.
- Selected raw CSV copies match their original report CSVs, both phase CSV/text reports exist, and operation audits match raw rows. Every measured selected matmul is BF16/HiFi4. Whole-layer roofline calculations rederive as 22.867475%/64.867014% linear and 21.066271%/68.024797% full for prefill useful FLOPs/decode estimated DRAM respectively. Supplemental batch latencies remain explicitly host-wall measurements.

## Anomaly Ledger

- Observed anomaly: Redundant row-major-to-tiled-to-row-major conversion during prefill history update.
  Evidence: The exact 32 adjacent conversion pairs in selected `final_linear/ops.csv`, current line 165, and slice implementation cited above.
  Affected path: Linear prefill chunks with nonaligned history-slice starts.
  Control or comparison: Row-major persistent history already removes five operations from decode, reducing it to 56 operations; prefill still has 995 operations.
  Likely subsystem: Composite slice restores its input layout.
  Investigation performed: Raw sequence/count/duration inspection and source-level control-flow analysis.
  Resolution: more-work-needed; the proposed ordering has not been measured.

- Observed anomaly: Minimal split and dedicated SwiGLU fusion can be slower despite reducing operations.
  Evidence: `minimal_split.json` reports 2789.998 us traced host screening versus 2357.664 us selected gate-activation control; snapshot hash matches recorded provenance and direct functional PCC passes. `tracy/c3_full` records 2685.596 us device decode; matching-precision geometry screens retain passing alternatives and L1 failures.
  Affected path: MLP.
  Control or comparison: Selected independent gate matmul with fused SiLU, packed gate/up controls, final profiles.
  Likely subsystem: Specialized matmul geometry and movement.
  Investigation performed: Candidate source, JSON provenance/results and comparative windows inspected.
  Resolution: controlled; the initial minimal-split experiment finding is closed.

- Observed anomaly: Long-context linear recurrent-state PCC differs from functional; full-attention prefill PCC is approximately 0.9964.
  Evidence: Initial retained state comparison is 0.998997698 versus functional 0.999973834. Final linear exact-context prefill is 0.999916484; nonaligned prefill/decode/changed-input scores are 0.999910521/0.999961657/0.999993965. Full exact/nonaligned prefill is 0.996466700/0.996417056.
  Affected path: Long-context prefill/state.
  Control or comparison: Functional full-attention scores 0.996442795/0.996382296 and preserved original threshold/context; final-position decode passes both kinds.
  Likely subsystem: Existing BF16 attention behavior and different chunkwise recurrence/normalization rounding.
  Investigation performed: Runtime/reference paths and final/control outputs compared; the individual rounding contributions were not isolated experimentally.
  Resolution: controlled within the unchanged 0.995 layer contract; no precision or capability reduction.

- Observed anomaly: Candidate API/layout errors, whole-MLP L1 capacity failure, optional profiler/report warnings and the packed-full AICLK warning.
  Evidence: Retained candidate adaptations, geometry logs, `expert_mlp_supported.log`, selected/raw profiler artifacts, and final work-log ledger. Packed full reports 1331 MHz while conversion follows the 1350 MHz header.
  Affected path: Rejected alternatives and evidence tooling/startup.
  Control or comparison: Corrected candidate passes, selected profiles without that clock warning, complete numeric windows, and watcher logs without matched fatal/assert/overflow/corruption signatures.
  Likely subsystem: Op contracts/L1 capacity and optional host tooling/metadata.
  Investigation performed: Final and initial-review evidence checked; raw completeness and conversion arithmetic independently verified. Using the lower rejected-control clock would make that control slower, so it cannot reverse selection.
  Resolution: fixed for selected paths / controlled for rejected alternatives and numeric reporting.

## Scope Inspected

- Goal/skill paths: installed stage02 goal; `model-bringup`, `graph-fusing`, `agentic-serving`, `tt-device-usage`, `tt-enable-tracing`, `stage-review`; repository `TraceCorrectness.md`; enabled AutoDebug configuration.
- Artifact paths: final README/work log/inventory and initial review; validation/provenance summaries; all indexed final real/synthetic results and XML; raw selected/baseline/comparative profiler windows and selected phase reports; watcher output; candidate snapshots/screens; context and agentic documents; exact packet `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/2afbed0c-78d4-425a-abb4-e5dcd018a314.json`.
- Code paths: complete fused decoder, runner, shell validation wrappers, synthetic/ragged tests, reporting/roofline/collection code, selected candidate source and TTNN composite slice.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, `diff`, `git status/branch/rev-parse`; small Python3 JSON/XML/CSV/AST/hash analyses. No TTNN imports, hardware use, tests, servers or resets. Only this review report was written.

## Residual Risk

This is single-layer evidence using real model weights and supplied/synthetic hidden activations, not full-model generation or serving-quality evidence. Long prefill compares the final 128 outputs using the entire prefix while TT computes all outputs. The 128-step timing workload is executed; its oracle checks the initial two decode steps plus restored repeated replay, not every performance-loop output. No hardware was independently rerun by the reviewer. The current results support substantial improvement and preserved tested semantics, but the explicit redundant prefill conversion prevents the stage's graph-exhaustion claim from passing.
