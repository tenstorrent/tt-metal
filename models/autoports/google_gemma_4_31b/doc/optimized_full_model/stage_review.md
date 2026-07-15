# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- P2: The repository still has unrelated dirty state outside Stage 07: deletion of `tt_metal/python_env/requirements-dev.txt`, untracked `.exp_run/` and `fusion_tests/`, and older `doc/full_model/` profiler/triage outputs. The Stage 07 tracked files and `doc/optimized_full_model/` tree are path-separable, so this does not block the stage, but the parent must use a path-scoped checkpoint and inspect the staged diff before committing.
- P2: `doc/context_contract.json` deliberately retains top-level `stage: "full-model"` and the prior full-model `decode_status`, while the stage-specific `optimized_full_model_status` and nested `optimized_full_model_plan.status` are `evidence_complete_rereview_pending`. The optimized-full-model scope is now explicit and still preserves context 262,144; the parent should advance only the stage-specific status after accepting this review.

## Hard-Check Gaps

- No TT hardware, watcher, profiler, model, server, or vLLM command was run during this rereview. Existing target-hardware logs and JUnit/JSON evidence were inspected directly.
- Full Ethernet watcher instrumentation remains physically unavailable because the instrumented active-fabric binary exceeds its 25,600-byte configuration buffer before model execution. The retained worker-watcher run passes with `TT_METAL_WATCHER=10` and `TT_METAL_WATCHER_DISABLE_ETH=1`; this is a scoped instrumentation limit, not a skipped worker assertion.
- The block-3 four-shard timing is a short reduced trace, but it is independently ineligible because the same-hidden full-model aligned diagnostic changes greedy token 669 to 108 and only 16.07% of pre-softcap BF16 logits match exactly. Additional timing repetitions cannot change that correctness rejection.

## Anomaly Ledger

- Observed anomaly: The selected four-input-shard LM-head geometry previously lacked a compatible larger-`in0_block_w` frontier.
  Evidence: Current source/default is split 8,192, four input shards, block 2. `candidate_results.csv`, candidate JSON/JUnit, and raw logs now cover every larger legal divisor of 42: block 3 passes at 337.545868 t/s/u but is slower and changes aligned greedy 669 -> 108; block 6 clashes at L1 allocation 1,351,040 versus static-CB end 1,381,120; blocks 7/14/21/42 require 1,581,824/2,986,752/4,391,680/8,606,464 bytes against 1,572,864-byte L1.
  Affected path: TP-local BF16/HiFi2 terminal LM-head projection.
  Control or comparison: Block 2 measures 339.823107 t/s/u, repeats as the final default at 339.164348 t/s/u, is bit-identical to the legacy LM head across all 262,144 aligned pre/post-softcap logits, and preserves device/host greedy token 669.
  Likely subsystem: DRAM-sharded matmul K-block geometry and L1 circular-buffer capacity.
  Investigation performed: Re-derived 168 hidden K tiles / four shards = 42 tiles/shard; inspected source validation/program config, candidate ledger, block-3 perf/aligned JSON, and block-6/7/14/21/42 XML plus raw failures.
  Resolution: fixed. Block 2 is the only passing correct winner in the compatible frontier.

- Observed anomaly: The previous top-level compact profiler summary belonged to the rejected block-3 profile.
  Evidence: `profiler_sha256.txt` now exactly matches SHA-256 for selected `profiler_raw_ops.csv.gz`, `tt_perf_report.csv`, `tt_perf_report.txt`, and `tt_perf_summary.csv`. `gzip -t` passes; the 472,199-byte repository artifact expands losslessly to the original 5,995,589-byte CSV with SHA-256 `7997ff98469f6dad6f3dbb4cfb7b7f058910791671105a363f94f8db14f2e133`. Compression keeps the byte-exact raw evidence below the 500 KB pre-commit limit while compact CSVs remain directly inspectable. The selected summary hash is `b588f2...`, distinct from rejected block 3's `43fb8f...`; selected raw and rejected raw are also distinct. The console names the suffix-correct `tt_perf_summary.csv` output.
  Affected path: Selected-profile provenance and topology percentages.
  Control or comparison: The selected detailed CSV contains 155 operations totaling 2,833.08975 us before display rounding; the compact summary reports 2,833.08 us, including width-sharded matmuls 1,935.15 us/68.31%, exact greedy 298.93 us/10.55%, and async all-reduce 50.28 us/1.77%.
  Likely subsystem: Prior report output naming and stale artifact propagation.
  Investigation performed: Recomputed hashes, file identity, detailed/stacked CSV sums, op counts, and selected LM-head dtype/fidelity/geometry rows.
  Resolution: fixed.

- Observed anomaly: The former 0.904983 ms subtraction was incorrectly labeled terminal/sampling/orchestration cost.
  Evidence: Sorting selected detailed rows by global call count reproduces non-overlapping spans: input 0.025503 ms, sliding layer 0.44878175 ms, full-attention layer 0.479633 ms, terminal 1.565191 ms, and sampler/feedback 0.313981 ms. Scaling the representative layers gives `50*0.44878175 + 10*0.479633 + 0.025503 + 1.879172 = 29.1400925 ms`.
  Affected path: Lower-bound/gap performance accounting.
  Control or comparison: Matched observed median is 29.254761467 ms/token, so observed minus modeled is 0.114668967 ms or 0.3935%. The Stage 05 value 28.356925 ms is separately and correctly labeled a sum of standalone layer medians, not an additive captured-trace lower bound or terminal measurement.
  Likely subsystem: Reporting arithmetic across unlike measurement regimes.
  Investigation performed: Independently summed detailed CSV spans and recomputed every scaled and observed quantity in `perf_summary.json`, README, and work log.
  Resolution: fixed.

- Observed anomaly: The earlier single-sample full-path comparison showed +8.42% TTFT and -1.74% overall throughput.
  Evidence: `full_token_out_matched_baseline.json` and `full_token_out_matched_selected.json` each contain one discarded warmup and five recorded full-60-layer 149/100 samples from the source-current focused harness. Baseline/selected medians are 444.008/452.409 ms TTFT, 24.88885/24.98654 overall t/s/u, and 33.87453/34.18247 steady t/s/u; all ranges and raw samples are retained.
  Affected path: Complete generator request latency and token-out throughput.
  Control or comparison: Every one of the ten samples has 99 model-trace replays, zero token/page/full-logit refresh/readback traffic, two position and two RoPE setup refreshes, three synchronizations, and one seed-token readback. Source resets state per call and places construction outside TTFT.
  Likely subsystem: Historical setup/cache-sensitive measurement provenance, plus a controlled 8.40 ms selected TTFT cost.
  Investigation performed: Recomputed medians/min/max/percent changes, inspected every counter, and inspected the benchmark-only source path and configuration overrides.
  Resolution: fixed. The matched result is +1.89% TTFT, +0.39% complete-request throughput, and +0.91% steady throughput; the old samples are no longer headline evidence.

- Observed anomaly: The first 249-row full-prefill readiness run failed the fixed-M sharded LM-head program contract.
  Evidence: The pre-fix log records the shard-height failure. Current source normalizes once, partitions arbitrary logical M into contiguous 1--32-row ranges, projects each range, concatenates along M, and softcaps once. The focused 33-row hardware JUnit passes, and the final exact 249-row full-stack log reports top-1 91/100, top-5 100/100, top-100 100/100.
  Affected path: Arbitrary non-aligned full-logit prefill.
  Control or comparison: Static range tests cover 1, 31, 32, 33, 63, 64, 65, 149, and 249 rows; public prompt length remains unrestricted up to context 262,144.
  Likely subsystem: Fixed-M terminal program/layout contract.
  Investigation performed: Inspected pre/post-fix logs, autofix reports, current source, focused JUnit, and current static test.
  Resolution: fixed.

- Observed anomaly: The initially selected block-3 accumulation geometry changed the Fibonacci trajectory, while some base-checkpoint prompt controls repeat corpus phrases.
  Evidence: Block-3 aligned JSON changes the same-hidden greedy winner; selected block 2 is exactly equal to legacy. The six-prompt selected outputs match Stage 06 TT token-for-token, two prompts match HF for all 64 tokens, and the separate 100-token story is coherent English with zero adjacent repetition and zero repeated trigrams.
  Affected path: Greedy generation quality and terminal numerical stability.
  Control or comparison: `GemmaTokenizer.chat_template` is absent, so HF and TT both use exact plain-tokenizer completion prompts. Repetition in the two exact-HF corpus controls is shared control behavior, not TT-only degeneration.
  Likely subsystem: Block-3 LM-head accumulation geometry; base-checkpoint completion distribution for shared repetition.
  Investigation performed: Applied the qualitative-check rules; inspected prompt-format metadata, rendered prompt/token artifacts, HF/TT outputs, mechanical degeneration JSON, aligned logits, and autoregressive text directly.
  Resolution: fixed/controlled. Block 3 is rejected and the selected path passes qualitative signoff.

- Observed anomaly: Runtime logs emit the conservative active-trace allocation warning during cooperating sampler-trace setup.
  Evidence: Warning is present in teacher-forcing, autoregressive, and qualitative logs.
  Affected path: Second trace registration, not steady replay.
  Control or comparison: Sampler resources are prewarmed before model capture; both traces are released together before mutation; focused normal and worker-watcher tests pass; all repeated full-path counters show no steady host fallback or stale refresh.
  Likely subsystem: Conservative trace-region registration warning.
  Investigation performed: Inspected trace capture/release/prewarm source, warning logs, counters, reset/recreate/static ownership tests, and the classified anomaly ledger.
  Resolution: controlled.

- Observed anomaly: The merged four-device profiler chronology contains a 667.9 ms inter-stream gap.
  Evidence: Selected detailed report/raw CSV.
  Affected path: Merged timestamps only.
  Control or comparison: Kernel durations sum to 2.833 ms in the reduced window; unprofiled full-path timing is the latency source; trace counters show a single final steady synchronization rather than per-token synchronization.
  Likely subsystem: Asynchronous device-stream timestamp merge.
  Investigation performed: Compared chronological gap, device-time sums, signpost boundaries, and unprofiled matched timing.
  Resolution: controlled; chronology is not used as token latency.

## Scope Inspected

- Goal/skill paths: supplied Stage 07 optimized-full-model contract; `.agents/skills/stage-review/SKILL.md`; `.agents/skills/optimize/SKILL.md`; `.agents/skills/qualitative-check/SKILL.md`.
- Artifact paths: `doc/optimized_full_model/{README.md,work_log.md,AUTOFIX_STAGE_REVIEW.md,optimization_checklist.md,provenance.md,runtime_fallback_audit.md,anomaly_ledger.md,perf_summary.json,candidate_results.csv,full_token_out_matched_*.json,profiler_sha256.txt,profiler_raw_ops.csv.gz,tt_perf_report.{csv,txt},tt_perf_summary.csv,*.xml,*.log}`; candidate and selected/rejected qualitative/autoregressive trees; `doc/context_contract.json`; previous-stage controls cited by those artifacts.
- Code paths: `tt/model.py`, `tt/generator.py`, `tests/test_full_model.py`, `tests/test_full_model_contract.py`, `tests/run_full_model_qualitative.py`; tracked diff from HEAD and dirty-worktree scope.
- Commands run: read-only `git status/diff/rev-parse`, `find`, `rg`, `sed`, `jq`, `sha256sum`, `cmp`, CSV/XML/JSON parsing and arithmetic, and the static contract suite with `LD_LIBRARY_PATH=$PWD/build/lib`. No TT hardware was opened. Static result: 22 passed.

## Residual Risk

- The profile-derived full-path operation model scales one real sliding and one real full-attention layer across the 60-layer stack. Its 0.39% agreement with the independently timed full path is strong validation, but it remains a modeled decomposition rather than an all-layer profile, as required by the optimization skill's profiler-safety rule.
- Full Ethernet watcher coverage is unavailable for the documented instrumentation-capacity reason; worker-kernel watcher coverage and normal fabric execution are clean.
- The selected block-2 default is well supported: fixed decoder dtype/fidelity/KV/CCL/residual policy remains visible in runtime rows; context remains 262,144; non-aligned and mixed/fixed-slot state passes; canonical device sampling and feedback avoid host argmax, generic TopK, full gather, and per-token readback; full prefill/teacher forcing meet 100% top-5/top-100; qualitative and autoregressive controls are inspectable; matched full-path performance improves complete-request and steady throughput. No unresolved stage-critical contradiction remains.
