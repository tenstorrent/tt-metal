# Stage Review

Verdict: clean-pass

Independent closure review of stage 02, `Qwen/Qwen3.8-27B`, on live branch `exp26-qwen38-bringup`, based at `2d41497c9965b277c78dc033b4d0f8bdd49b261e`. Reviewed runtime SHA256: `41ba2e2fec65d6c5bb7503a6ff1ca7a905a8164f349eb8ef9d8a399dccdefc6e`, byte-identical to `history_slice_candidate.py`. The final evidence was reviewed after the stage owner declared it ready. No required implementation, validation, or measurement work remains in this review. Local checkpoint commits and completion metadata are the stage owner's required post-review actions; they were intentionally absent during review.

## Required Work

- None. The remaining history-slice finding in `STAGE_REVIEW_history_finding.md` is closed by the final source, affected validation, and reproduced complete-device profile.

## Other Concerns

- No additional blocking concern was identified. The README's accidentally renamed packed-control directory was corrected to `tracy/packed_final_linear` during reconciliation and verified afterward.
- The final history change optimizes prefill. Decode's small change from 2328.099 to 2325.657 us is correctly described as run variation on the unchanged decode graph. The selected decode still beats the retained tiled-state and packed controls; this is not evidence for another decode optimization.
- The inventory supports closure of the relevant graph-fusion patterns on this hardware and precision. It does not establish that later sharding, precision, matmul tuning, full-model, or serving stages have no remaining work.

## Hard-Check Gaps

- No missing mandatory check or unsupported evidence-format requirement was identified. Tests import `FusedDecoder` directly. The optional functional control runs separately from the runtime and measured windows. Audited correctness runs forbid Torch operations and `ttnn.from_torch`/`to_torch` inside the forward path.
- Final linear provenance matches the reviewed source in all seven indexed real checks and nine synthetic cases. Their runner hashes also match the current runner. All indexed scores meet PCC >= 0.995; the lowest indexed linear score is 0.999806192. Synthetic XML records nine tests, zero failures/errors/skips, 87.757 seconds. The original 18-case suite has zero failures/errors/skips; its nine full-attention cases remain applicable.
- Re-derived AST comparison against `candidate15.py` finds changes only in `allocate_state` and `_delta`. Removing the linear-only allocation branch leaves the full-attention allocation AST unchanged. Constructor, shared normalization/MLP, snapshots, RoPE, attention, and prefill/decode orchestration are identical. This supports the retained full-attention correctness and `final_full` profile, whose source hash matches candidate15. The lowest indexed full-attention PCC is 0.996417056 at nonaligned maximum context.
- Final affected coverage includes both history branches and lengths 1/2/3/4/31/32/33/127/128/129/385 plus short reuse, snapshot/continuation, B3 split-batch execution, B8/B16/B32 4096+128 workloads, exact 262144 prefill, and 262142 prefill followed by the final two supported decodes. Full-attention coverage additionally includes remapped pages and B8 ragged positions `[33,31,1,385,128,32,129,127]`. Repeated replay is bitwise deterministic and changed-input/position replay passes.
- Correctness wrappers set `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0` before TTNN import. The runner prepares stable buffers and host refresh tensors before capture, warms decode, restores state, and exercises repeated replay. No corruptible-buffer acknowledgment or broad allocation exemption appears. Watcher and profiler runs are separate. Both retained watcher files contain no matched fatal/assert/sanitize/overflow/corruption/error signature.
- Raw selected `ops.csv` copies are byte-identical to their original report CSVs. Both phases have nonempty tt-perf-report CSV and text artifacts. Re-derived firmware envelopes match `whole_layer_timing.json`, `performance.json`, and the exact-ID telemetry packet. Prefill is warmed eager execution; decode is trace ID 0, replay session 132. The denominator includes all operations and intervening gaps.

| Layer kind | Functional prefill us | Selected prefill us | Functional decode us | Selected decode us | Prefill / decode speedup |
|---|---:|---:|---:|---:|---:|
| linear_attention | 284743.540741 | 83008.777778 | 2660.428148 | 2325.657037 | 3.430282x / 1.143947x |
| full_attention | 108787.729630 | 93214.924444 | 2456.268889 | 2187.320000 | 1.167063x / 1.122958x |

- The selected linear profile has 931 prefill operations versus 995 before the final fix. Its history sequence is now `UntilizeDeviceOperation -> SliceDeviceOperation -> CopyDeviceOperation`; all 32 intervening tilize/untilize pairs are absent. Decode retains 56 operations, no tilize, and the one untilize required to update row-major history. Full decode retains 35 operations, no tilize/untilize, and the two cache-update sharding conversions. No additional redundant layout roundtrip was identified in the selected windows.
- Every selected measured matmul has BF16 inputs/weights and HiFi4 fidelity. Independent arithmetic reproduces whole-layer prefill useful-FLOP percentages of 22.970245% linear and 21.066271% full, and decode compulsory-DRAM estimates of 64.935112% and 68.024797%. The basis is one p300c ASIC, 165.888 TFLOP/s at HiFi4 and 512 GB/s, using the captured 1350 MHz header. These are explicitly estimated algorithmic rooflines, not hardware bus counters or averages of per-op utilization. Supplemental batch results remain labeled host-wall measurements.
- `context_contract.json` preserves 262144 and points to final affected evidence. Row-major history saves 593920 physical bytes/request; total linear state is 3207168 bytes/request. `agentic_serving.json` retains the 32-conversation planning workload, batches 1/8/16/32, final evidence links, and the corrected prefix-contract anchor. The exact-ID packet's headline numbers and source references match the final reports; review/commit status correctly remains pending until this verdict is recorded.

## Anomaly Ledger

- Observed anomaly: Prefill history previously restored tiled layout immediately before converting back to row-major.
  Evidence: Prior `final_linear` has 32 tilize/untilize pairs and 995 prefill operations; current line 165 converts before slicing, and `history_slice_final` has 931 operations with neither redundant operation.
  Affected path: Linear prefill history updates with length >= 3, including nonaligned tails.
  Control or comparison: 83381.831852 us before versus 83008.777778 us after; unchanged decode control flow.
  Likely subsystem: Composite slice's restoration of the input layout.
  Investigation performed: Source and slice contract inspection, raw operation sequences/counts, provenance, and all affected correctness artifacts.
  Resolution: fixed. Both state branches, continuation, split batches, maximum context, and synthetic tails pass on the final source.

- Observed anomaly: Fewer operations did not guarantee faster MLP alternatives.
  Evidence: Dedicated SwiGLU full decode is 2685.596 us; packed controls are 2377.982 us linear and 2227.382 us full. Direct minimal-split screening is 2789.998 us versus 2357.664 us gate-activation control.
  Affected path: Shared MLP.
  Control or comparison: Selected gate matmul with fused SiLU and final complete-device measurements.
  Likely subsystem: Dedicated matmul geometry and required movement.
  Investigation performed: Candidate source/configuration, matching BF16/HiFi4 geometry screens, passing equivalence, and L1 diagnostics inspected. Whole-MLP supported-config retry requires 7497408 bytes versus a 1461120-byte CB budget.
  Resolution: controlled. Adapted slower alternatives and concrete capacity failures justify rejection; the previous minimal-split experiment gap is closed.

- Observed anomaly: Long-context linear recurrent-state correlation is lower than the functional control; full-attention long-prefill PCC is approximately 0.9964.
  Evidence: Retained state comparison is 0.998997698 versus 0.999973834. Final linear exact/nonaligned prefill is 0.999916484/0.999910521, with nonaligned decode/changed-input scores 0.999961657/0.999993965. Full exact/nonaligned prefill is 0.996466700/0.996417056.
  Affected path: Long-context recurrence and attention prefill.
  Control or comparison: Functional full scores 0.996442795/0.996382296, unchanged acceptance threshold/context, and passing final-position decode for both kinds.
  Likely subsystem: Different intermediate rounding in chunk recurrence/normalization and existing BF16 long-context attention behavior.
  Investigation performed: Final/control scores and arithmetic paths compared. Individual rounding contributions were not experimentally isolated.
  Resolution: controlled within the unchanged layer contract; no precision or capability reduction and no material output regression established.

- Observed anomaly: Candidate API/layout errors, profiler/report warnings, and a rejected packed-full AICLK warning.
  Evidence: Retained supported-config/cropping retries and L1 failures; optional Tracy GUI-copy, Matplotlib, and pandas warnings; packed-full 1331 MHz startup warning against the 1350 MHz profiler header.
  Affected path: Adapted/rejected candidates and evidence tooling/startup.
  Control or comparison: Passing selected paths, complete numeric timestamp windows and reports, clean watcher files, and selected profiles without the AICLK warning. Histogram overflow wording denotes out-of-range histogram buckets, not lost profile data.
  Likely subsystem: Op contracts/capacity, optional host tooling, and clock metadata.
  Investigation performed: Actual diagnostics, report completeness, source CSV identity, and timestamp arithmetic inspected. Using 1331 MHz would make the already slower rejected packed-full control slower still.
  Resolution: fixed for selected runtime paths / controlled for rejected alternatives and reporting.

## Scope Inspected

- Goal/skill paths: installed stage02 goal; `model-bringup`, `graph-fusing`, `agentic-serving`, `tt-device-usage`, `tt-enable-tracing`, and `stage-review`; repository `TraceCorrectness.md`; enabled AutoDebug inventory.
- Artifact paths: final README/inventory/work log and both historical reviews; validation summary, all indexed final real/synthetic results and XML; source/provenance proof; final/baseline/comparative raw profiles and phase reports; watcher files and validation logs; candidate results and diagnostics; context and agentic documents; exact packet `/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/2afbed0c-78d4-425a-abb4-e5dcd018a314.json`.
- Code paths: full `tt/fused_decoder.py`, final and prior candidate snapshots, runner and runtime audit, shell wrappers, synthetic/ragged tests, profile extraction and roofline/report scripts, relevant TTNN slice/operation contracts.
- Commands run: read-only `cat`, `sed`, `rg`, `git status/diff`; small Python3 JSON/XML/CSV/AST/SHA256 analyses. No TTNN imports, hardware commands, tests, servers, or resets. Only this review report was written by the reviewer.

## Residual Risk

This pass applies to the staged single-layer contract with real target weights and supplied/synthetic hidden activations. It is not full-model generation or serving-quality evidence. Long prefill compares the final 128 outputs using the entire prefix while TT computes all outputs. The 128-step performance workload executes all steps; its explicit oracle covers the initial two decodes and restored repeated replay, not every performance-loop output. Maximum context is tested at B1 and larger batches at 4096, not every batch/context combination. The reviewer independently inspected and recomputed existing evidence without rerunning hardware. Future runtime changes require affected revalidation.
