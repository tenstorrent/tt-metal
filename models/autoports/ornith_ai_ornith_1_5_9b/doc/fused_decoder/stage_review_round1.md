# Stage Review

Verdict: clean-pass

Independent review, round 1 of 3, stage 2 fused-decoder for `ornith-ai/Ornith-1.5-9B`. Reviewed the live staged worktree on `main`, starting HEAD `d55afc0918f5862d121467f8927c58d4a3b93b70`. The delivered implementation SHA256 is `8f094843786f8f388b241449c0322a684fc398244500274f34ae3862fb07cc10`. No devices, servers, resets, implementation edits, or hardware tests were used in this review.

## Required Work

- None. The inspected implementation and completed evidence satisfy the stage contract. This verdict covers the delivered layer implementation and evidence; the stage owner must still create the authorized local checkpoint and record its SHA. The supplied run plan prohibits pushing, and no push is authorized by this review.

## Other Concerns

- Performance handoff: the linear single-token history update still has an opportunity to reuse its row-major history. `tt/fused_decoder.py:243` converts `state.conv` for causal convolution, while the `n < 3` branch at lines 262–263 concatenates/slices tiled history. The final decode profile shows the associated additional untilize/tilize operations. An adapted row-major history-update path is a reasonable next-stage experiment; its correctness and net latency benefit have not been measured. This is not evidence of wrong state, and the passing stop rule bounds further candidate work.
- The small-gain convergence evidence is for the shared projection-policy search on the linear layer: the last two kept refinements improve 1.211509621 -> 1.210115300 -> 1.207822040 ms, or 0.1151% and 0.1895%. The selected shared policy was then validated on full attention, but a separate full-attention geometry sweep was not performed. Preserve that distinction in later performance claims; do not describe the full layer as independently geometry-exhausted.
- The final profiler's displayed matmul core counts and derived FLOP percentages should not be interpreted literally. The raw final CSV records `CORE COUNT=80` for the sharded projection programs, while the rendered tables display eight cores and approximately 143–152% FLOP utilization. The measured kernel times, runtime BF16/HiFi4 attributes, and independently calculated bandwidth-based stage roofline remain usable. This is a reporting/accounting limitation, not evidence that the decoder exceeds a physical compute peak.

## Hard-Check Gaps

- `tests/validate_fused.py:99–110` records ordinary residual-subtracted `contribution_pcc` without asserting its threshold, except in the per-row continuation branch. The current final artifacts nevertheless pass: their minimum positive contribution PCC values are 0.999396861 linear and 0.999623895 full. The paired and long-context runners explicitly assert contribution PCC. Add the ordinary assertion when maintaining this runner so a future contribution regression cannot silently pass the HF battery; no new device run is needed to establish the current values.
- The 128-token accumulating trajectory is eager functional-versus-fused equivalence. Traced evidence separately covers changed requests, state restoration, and repeated replay. This satisfies the layer-stage repeated/stress contract, but does not establish a generator's sustained traced token-feedback loop or interleaved multi-signature lifetime. Those remain later-stage work, as the documentation states.

## Anomaly Ledger

- Observed anomaly: allocator warnings in untracked watcher/profiler runs say allocations made with an active trace could be corrupted.
  Evidence: `linear_watcher.log`, `full_watcher.log`, and both `*_profile_final.log`; tracked `*_final.json` and `*_long.json`; `trace_alloc_tracking/positive_control.log`.
  Affected path: eager prefill and output allocations around retained decode traces.
  Control or comparison: the same final layer path and correctness battery pass with `TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`; the deliberate unrelated live allocation raises before replay.
  Likely subsystem: generic allocator warning when tracking is disabled.
  Investigation performed: inspected preparation/capture/replay ordering, exact output acknowledgment, deletion of prefill temporaries before replay, and snapshot ownership. Only the owning trace is live; its output is consumed after replay and before the next overwrite. No broad exemption scope or program-cache exclusion exists.
  Resolution: controlled.

- Observed anomaly: final instrumented single-replay walls imply less than 70% roofline, whereas the headline warmed trace reaches more than 70%.
  Evidence: `*_profile_final.json`, final profiler CSVs, `default_{linear,full}_s1024.json`, `performance_summary.json`.
  Affected path: timing regime and stop-rule accounting.
  Control or comparison: final matched benchmarks take seven groups of 50 nonblocking replays with one synchronization per group; profiler walls time one guarded, synchronized replay.
  Likely subsystem: per-call synchronization and instrumentation overhead.
  Investigation performed: recomputed medians, kernel sums, op gaps, and same-run wall residuals. Linear reconciles as 1.144266 + 0.041831 + 0.086464 = 1.272561 ms; full as 1.093674 + 0.046840 + 0.084196 = 1.224710 ms. The separately labeled unprofiled medians are 1.207541 and 1.129713 ms.
  Resolution: controlled; the README makes the regime distinction explicitly.

- Observed anomaly: rendered matmul utilization exceeds 100%, and experimental GDN/convolution names are unclassified.
  Evidence: `tracy_final/{linear,full}/decode_perf_report.txt` and corresponding raw `decode_ops.csv`.
  Affected path: profiler classification and derived utilization, not recorded device duration.
  Control or comparison: raw program metadata and an independent sum of actual kernel-duration rows.
  Likely subsystem: report core-count/peak assumptions and missing operation categories.
  Investigation performed: compared raw and rendered core fields, inspected runtime compute attributes, and recomputed stage bandwidth accounting without rendered FLOP utilization. The stage does not use the implausible compute percentages for its speedup or stop-rule arithmetic.
  Resolution: controlled, with the reporting limitation retained under Other Concerns.

- Observed anomaly: initial fusion/configuration attempts fail op layout or L1 validation.
  Evidence: concat-heads failure and adapted success logs; `dram_r1_c32_b0_linear.log`, `dram_r1_c64_b0_linear.log`, `dram_r2_c32_b0_linear.log`; subsequent successful candidate JSONs.
  Affected path: rejected historical candidates.
  Control or comparison: adapted head concatenation/splitting, smaller K blocks, and two/four-reader sharded variants complete correctness and timing.
  Likely subsystem: operator contracts and static L1 capacity.
  Investigation performed: checked failures against source/configuration and measured retries; these failures were not treated as a blanket rejection of the optimization family.
  Resolution: fixed or controlled by adapted candidates.

## Scope Inspected

- Goal/skill paths: installed stage-2 goal template `prompts/model_bringup_multigoal/02-fused-decoder.txt`; supplied original goal contract; installed `stage-review`, `graph-fusing`, `optimize`, `tt-device-usage`, and `tt-enable-tracing` skills; model-bringup startup/stop-rule instructions; optimize performance-accounting reference; repository `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`.
- Code paths: all of `tt/fused_decoder.py`; comparison with `tt/functional_decoder.py`; `tests/validate_fused.py`, `compare_fused.py`, `long_context_fused.py`, `test_fused_decoder.py`, `dram_fusion_candidate.py`, `fusion_candidate.py`; evidence/profile/candidate shell drivers; `fused_perf_reports.py`, `summarize_fused.py`; reference weight/PCC helpers in `run_decoder.py`.
- Artifact paths: stage README, work log, pattern audit, context contract, candidate measurements and relevant failed/adapted logs; both final, watcher, paired, trajectory128, long and final-profile reports; all 20 matched length-sweep benchmark JSONs; allocation positive control; final raw/profile CSVs and provenance; watcher logs/audit; driver completion logs; pre-review hook output; archive manifest and all 159 gzip archives.
- Commands run: read-only `rg`, `sed`, `cat`, `nl`, `git status`, `git diff --cached`, `git rev-parse`; small Python scripts using only filesystem/JSON/CSV/hash/gzip/statistics operations to verify source/provenance/archive hashes, enumerate positive/negative metrics, recompute medians and speedups, sum profiler rows, inspect raw trace metadata, scan bounded warning excerpts, and calculate physical cache/weight bytes. The only write was this report.
- Source identity: final correctness, watcher, context, paired/stress, benchmark, and profiler evidence match the delivered source hash. The only production addition is `tt/fused_decoder.py`; the functional implementation is unchanged. Every compressed archive checked against its manifest hash.
- Accuracy and branches: both layer kinds execute the standalone fused graph. The final battery covers 1,2,3,4,31,32,33,1023,1024,1025,2051,17 tokens, continuation starts on both sides of tile/chunk boundaries, independently permuted pages, reconstructed physical KV, and ragged batch 32. Positive metrics pass 0.995; intentionally broken-history controls fail. The 128-step paired trajectory minimum contribution PCC is 0.999467492 linear and 0.999960542 full. Long prefill executes 262139,262143,262144 logical tokens; sampled full-history reference minima are 0.998479128 and 0.998122931 for contribution PCC. Final-slot decode and deterministic replay pass.
- Capacity: the advertised 262144 context is retained. Full-attention K/V allocation is 1,073,741,824 bytes, the added packed RoPE lookup is 67,108,864 bytes, and additional sharded projection allocations match 438,304,768 linear / 419,430,400 full bytes. Complete long runs establish layer-level feasibility with this overhead.
- Performance: independent median/sum arithmetic reproduces linear prefill 13.615562 -> 5.867048 ms and decode 1.698006 -> 1.207541 ms; full prefill 8.682155 -> 6.533450 ms and decode 1.897858 -> 1.129713 ms. Other matched lengths also improve. Final defaults reproduce the fastest measured correct candidates. Final signposted decode windows contain one retained trace ID, replay session 4, and runtime BF16/HiFi4/FP32-accumulation attributes. They contain 48/54 device operations, with no host operations in the measured graph.
- Stop rule: independently calculated active padded projection/state traffic gives 0.864256/0.827400 ms lower bounds at the explicitly stated nominal 512 GB/s. Final unprofiled warmed traces achieve 71.5715%/73.2399%; the final two kept shared-policy refinements gain less than 3%. Remaining candidates are permitted handoffs at this measured stopping point. No later pipeline stage is implemented.

## Residual Risk

- This is layer-level evidence using real layer weights with synthetic hidden inputs. It does not certify accumulated full-stack accuracy, natural-language generation, dense batch throughput, serving, sampling, or token feedback.
- Long-context numerical agreement is sampled at first/middle/terminal windows with complete prior history, not every output row.
- Timing conclusions apply to serialized chip-0 Blackhole runs and the documented warmed replay regime. Sub-percent candidate ordering is near timing noise; the material gains against functional and interleaved fused controls are much larger.
- Integration must preserve caller-owned state, bounds checking, complete warmup, stable trace buffers, and output lifetime. Additional live trace variants and generator request interleaving require their own later-stage evidence.
