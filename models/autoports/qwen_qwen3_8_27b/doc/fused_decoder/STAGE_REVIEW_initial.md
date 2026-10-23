# Stage Review

Verdict: more-work-needed

Independent inspection of stage 02, Qwen/Qwen3.8-27B, on the live `exp26-qwen38-bringup` worktree based at `2d41497c996`. This is an initial review, not a closure review. The inspected selected implementation is `candidate15.py`, SHA256 `a55fbb2aa851882004a4be715e28b6ca49697690be07c6b61a0c4d97074e5e46`. It was byte-identical to `tt/fused_decoder.py` during the source/provenance check. The validation suite completed while this review was running.

## Required Work

- P2: Finish the two remaining graph experiments and review the resulting selection.
  Evidence: `PATTERNS.md` explicitly leaves row-major convolution history unresolved; `row_major_state_candidate.py` is prepared. The stage owner also prepared `minimal_split_candidate.py`. The selected source converts convolution history to row-major at line 155 and back to tiled state at line 166. The inventory's rejection of `minimal_matmul_split` based only on a slower, stronger SwiGLU variant does not itself measure this distinct equal-width gate/up candidate.
  Why this matters: The original goal requires an earned rejection or acceptance of relevant movement/fusion candidates and selection by correctness plus complete-layer latency.
  Required next step: Complete the prepared experiments, adapt recoverable API/layout failures, and record exact source/config/results. A slower correct candidate can be rejected without profiling the entire suite. If an alternative becomes selected, reproduce the affected headline phase measurements and validate affected paths on that final implementation. For row-major state, include both history-update branches (`length < 3` and `>= 3`), nearby padded tails, continuation/snapshot behavior, and split-batch operation. Reuse unaffected full-attention evidence.

- P2: Reconcile final documentation, stage metadata, and artifacts with the accepted implementation.
  Evidence: `README.md` still says final boundary/watcher/ragged/synthetic evidence and final performance are pending. `work_log.md` still calls candidate13 the final candidate. `performance.json` has empty supplemental batch maps although all six `workload_layer{0,3}_b{8,16,32}.json` files now exist. `doc/agentic_serving.json` still identifies stage01 and points exclusively to functional-stage evidence. The exact-ID telemetry packet correctly remains marked in progress, with candidate15 headline values.
  Why this matters: The current write-up is intentionally unfinished, but cannot serve as the final reproducible stage result. Existing good evidence should be summarized accurately rather than rerun solely for presentation.
  Required next step: After candidate selection, update README/work log/inventory/performance and stage02 agentic metadata, preserving accepted earlier-stage records. Include final source identity, commands/configs, direct-equivalence and HF scores, final supplemental batch rows with their host-wall measurement basis, rejected candidates, limitations, and exact artifacts. Classify the warnings listed below. Obtain final independent rereview, then create and record stage-owned local checkpoint commits without pushing.

## Other Concerns

- `packed_full.log:23` reports AICLK settling at 1331 MHz while `whole_layer_timing.json` uses the profiler header's 1350 MHz. State that timing normalization follows the captured header and classify the startup warning; do not describe every run as independently verified exact 1350 MHz. The selected full-attention candidate is faster even in the currently recorded cycle-normalized comparison, so this warning alone does not establish a wrong winner or require a new sweep.
- `rotary_embedding_llama_fused_qk` has a real direct-contract mismatch: the implementation requires sharded decode inputs and a tile-local 32x32 transform, while the selected partial rotate-half operation exchanges 32-column halves. The inventory appropriately distinguishes this from the already measured rotate-half/decode-head alternatives. Any future attempt to claim an adapted fused-QK path has been rejected should account for projection/norm/cosine channel permutation and the cache convention, rather than treating “not a drop-in” as a measured rejection. No additional experiment is required by this initial review solely for that speculative alternative.

## Hard-Check Gaps

- No completed gate is being failed solely for lacking stronger counters or a new artifact format. Script configuration plus successful tracked runs establishes `TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; no exemption scope or corruptible-tensor acknowledgment appears in the reviewed code.
- The suite has now completed: `synthetic_pytest.xml` reports 18 tests, zero failures/errors/skips, 172.945 seconds. All 18 retained JSON results are deterministic and their minimum prefill/decode/changed-input PCCs are above 0.99960/0.99977/0.99977. `odd_batch.json`, `ragged_decode.json`, all six workload batch JSONs, watcher cases, and both exact/nonaligned maximum-context cases exist and pass. These results validate candidate15; selection of a later runtime change requires only the affected checks.
- Commits are properly absent at this initial-review point. The required local checkpoint is a post-clean-pass action, not grounds for premature stage completion.

## Anomaly Ledger

- Observed anomaly: Reduced op count made the dedicated SwiGLU candidate slower.
  Evidence: `tracy/c3_full/whole_layer_timing.json` is 114860.008 us prefill / 2685.596 us decode; matching-precision geometry screens and L1 failures are retained in `swiglu_k*n*.json/.log`.
  Affected path: Dense MLP.
  Control or comparison: Candidate15 full attention is 93214.924 / 2187.320 us; packed control is 95321.468 / 2227.382 us.
  Likely subsystem: Dedicated matmul kernel geometry and movement.
  Investigation performed: Source, profile rows, geometry results, and explicit L1 diagnostics inspected.
  Resolution: controlled; the slower implementation is rejected. The separate minimal-split experiment remains required above.

- Observed anomaly: Long-context linear recurrent-state PCC is lower than the functional implementation.
  Evidence: `linear_max.log` records state PCC 0.998997698; functional README records 0.999973834. Final candidate15 nonaligned output PCCs are 0.999910521 prefill, 0.999961657 decode, 0.999993965 changed-input decode.
  Affected path: Chunked prefill recurrence and L2 normalization.
  Control or comparison: Functional long-context state/output evidence; final exact262144 and nonaligned262142 checks.
  Likely subsystem: Different intermediate rounding in chunkwise BF16 matrix algebra and fused normalization.
  Investigation performed: Arithmetic paths and retained state/output comparisons inspected.
  Resolution: controlled within the unchanged 0.995 contract; the explanation is plausible, not a claim that individual rounding contributions were experimentally isolated.

- Observed anomaly: Full-attention prefill PCC at maximum context is about 0.9964.
  Evidence: `final_layer3_262142.json` 0.996417056 and `final_layer3_262144.json` 0.996466700.
  Affected path: Long-context prefill.
  Control or comparison: Accepted functional results 0.996382296 and 0.996442795.
  Likely subsystem: Existing long-context BF16 attention accumulation behavior.
  Investigation performed: Final and baseline scores and preserved SDPA configuration compared.
  Resolution: controlled; no material regression or capability reduction established.

- Observed anomaly: Candidate API/shape/configuration errors and whole-MLP capacity failure.
  Evidence: Work log plus corrected concat/rotary cropping, causal-convolution supported config, `swiglu_k32n10.log`, `swiglu_k16n16.log`, and `expert_mlp_supported.log` (7497408-byte CB requirement versus 1461120-byte budget).
  Affected path: Rejected or adapted fusion candidates.
  Control or comparison: Passing corrected candidate JSONs and selected BF16/HiFi4 runtime.
  Likely subsystem: Op layout/configuration constraints and L1 capacity.
  Investigation performed: Source and diagnostics inspected; failures were followed by supported adaptations.
  Resolution: fixed for selected paths / controlled for rejected paths; no device-health failure is established.

- Observed anomaly: Profiler/report and startup warnings.
  Evidence: Selected profile logs report a missing optional Tracy GUI-copy destination and pandas mixed-column warnings; report text contains Matplotlib cache and unclassified-operation warnings. Hardware startup also reports motherboard-identification fallback/subset-device advisory; packed full control has the AICLK warning above.
  Affected path: Evidence presentation, startup metadata, and nominal timing basis.
  Control or comparison: Raw numeric CSVs exist, copied `ops.csv` is byte-identical to each source CSV, all required signposted device timestamps are present, and whole windows rederive exactly. Watcher logs have no matched fatal/assert/sanitize/overflow/corruption signature.
  Likely subsystem: Optional GUI export/report tooling and startup metadata.
  Investigation performed: Logs and raw numeric measurement artifacts checked.
  Resolution: controlled for numeric completeness; final documentation classification still required.

## Scope Inspected

- Goal/skill paths: installed `prompts/model_bringup_multigoal/02*txt`; `model-bringup`, `graph-fusing`, `agentic-serving`, `tt-device-usage`, `tt-enable-tracing`, and `stage-review` skills in tt-model-bringup0.1.21; repo `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Artifact paths: fused README/work log/inventory/performance; candidate and correctness JSONs; selected/packed/baseline whole-layer reports; selected raw `ops.csv` and prefill/decode CSV/text reports; watcher logs; final validation, synthetic XML/results; context/agentic documents; functional README/performance; exact attempt packet `2afbed0c-78d4-425a-abb4-e5dcd018a314.json` under the user-specified telemetry path.
- Code paths: entire `tt/fused_decoder.py`, runner, `fused_run.sh`, `validate_fused.sh`, synthetic and ragged tests, reporting/roofline/collection scripts; candidate snippets and relevant TTNN op validation/contracts.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, `ls`, `git status`; small Python3 scripts reading JSON/XML/CSV, computing SHA256, and rederiving device-window timing. Initial `python` lookup failed because the host executable is named `python3`. No TTNN imports, hardware commands, tests, servers, or resets were run by this reviewer. Only this report was written.

## Residual Risk

No concrete runtime correctness defect was identified in candidate15. Both layer kinds preserve the functional public shape/state API, 262144 logical context, nonaligned tails, paging, and repeated replay within the tested layer scope. Selected profile provenance matches candidate15, dominant measured matmuls are BF16/HiFi4, and the raw complete-window results support the reported speedups: linear 3.409x prefill / 1.131x decode, full 1.167x / 1.123x. Host conversions occur in setup/reference work outside the audited runtime.

This remains a single-layer evaluation with synthetic hidden activations and real target weights. Long prefill compares the final128 outputs using the complete prefix; all TT outputs are computed. The 128-step timing workload is executed, while the correctness oracle explicitly checks the initial two decode steps and repeated restores; it is not a full-model generation or serving-quality claim. The reviewer did not independently rerun hardware. A later selected candidate can invalidate affected candidate15 evidence, and the unresolved experiments/documentation prevent clean-pass now.
