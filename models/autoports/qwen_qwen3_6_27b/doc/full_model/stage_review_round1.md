# Stage Review

Verdict: clean-pass

Independent reviewer, round 1 of 3, 2026-11-14. Stage 6 full-model, Qwen/Qwen3.6-27B, pinned HF revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`. Reviewed the live dirty `main` worktree based on `7be8c4348666622686a6bdf5efa72e73ecf4c9b0`. This is an inspection verdict, not a new hardware run or a vLLM acceptance result.

## Required Work

None. The inspected implementation and completed evidence satisfy the stage's functional, numerical, trace, context, batch, sampling, and performance contracts. The owner must still record this review and create the authorized local stage-owned checkpoint commits before declaring the overall goal complete. The run plan prohibits pushing; this review does not authorize a push.

## Other Concerns

- Keep the capacity scope explicit at the next stage. Physical B1/P8193 has the full-context runtime acceptance; physical B32/P8224 has complete B1/B32 preparation and short fixed-slot lifecycle acceptance. B32 supports 1–32 live requests through inactive slots. These results do not establish 32 simultaneous full-context requests, a maximum physical pool, allocator high-water marks, or a full-context capacity budget for every intermediate physical constructor batch. The current README, context contract, and capacity accounting correctly distinguish these claims.
- The shared qualitative suite supplies 12 outputs per side across six prompts and two sampling modes, with one seed per prompt/mode. I read every output. They are coherent within the inspected windows, but mostly terminate during thinking/planning. This supports the stated observational smoke, not completed-answer quality or statistical HF/TT equivalence. A later content failure still needs the prescribed matched, blinded HF control before attribution.
- Greedy selection is semantically correct but does not promise HF's lowest-index tie identity when more than 32 maxima tie within one vocabulary shard. The explicit 65-way case selects another true maximum. Preserve this distinction in later deterministic comparisons.
- Serving support is an evidence-backed mapping to existing primitives, not an implemented adapter. In particular, the pinned plugin disables resident reuse when `supports_async_decode=False`, so generator token-out throughput is not achieved serving throughput. Host penalties/logprobs/filtering, scheduling, cancellation, and HTTP behavior remain integration tests. Prefix caching is legitimately excluded for `recurrent_state`; existing-state continuation is separately implemented and tested.
- Nonblocking documentation housekeeping: the README points to `commands.json` for HF reference-generation provenance, but that manifest has no reference-generation entry. The actual pinned reference, its exact-match cache key, `reference_pinned.log`, and the original command plus local-snapshot substitution in `work_log.md` establish the required provenance. Adding that command to the manifest would improve navigation; no reference regeneration or review round is needed.

## Hard-Check Gaps

- `stage_gate.log` checks output degeneracy, the context declaration, and serving-row inventory. It does not itself enforce the numerical threshold. The separate current-source `collect_accuracy.py` receipt closes this gap: I independently parsed the raw aggregate counts and confirmed prefill 100/100/100 and decode 99/100/100 for top-1/top-5/top-100, each over exactly 100 targets.
- `performance_final.json` deliberately has `acceptance_ready=false`: this uninstrumented timing run is not the allocation-tracking gate. The distinct current-source `trace_contract_native.json`, all64 mixed/B32 acceptance, and maximum-context run supply that gate. Treating the performance file alone as trace-allocation acceptance would be incorrect.
- General trace probing is reduced to real layers 0 and 3; all64 runtime evidence complements it. The direct probe observes exact capture bindings, persistent token/position contents, changed/unchanged tables, and newly written physical pages. All64 performance and maximum-context evidence retain 24456 programs and three traces; all64 mixed lifecycle retains 834 programs. The evidence is complementary rather than an assertion that every test independently covers every dimension.
- Some focused sampler/host-compatibility receipts predate the final native metadata repair and lack embedded source identities. The Python generator is unchanged across that repair, the native change has exact independent mapping/device evidence, and final-source trace, numerical, autoregressive, mixed, B32, and max-context receipts pass. I found no affected functional path requiring repetition of the entire focused suite.

## Anomaly Ledger

- Observed anomaly: B32 preparation originally exhausted contiguous DRAM despite remaining aggregate free space.
  Evidence: `AUTODEBUG_b32_capacity.md`, `AUTOFIX.md`, the baseline/compressed memory summaries, and `reshape_mapping/independent_review.md`.
  Affected path: cached native tiled-reshape mappings during complete prefill preparation.
  Control or comparison: diagnostic pre-capture cache clear isolated retained allocations; mapping size/multiplicity matched the dominant native buffers.
  Likely subsystem: native program-owned mapping tensors, rather than changed model policy or physical context capacity.
  Investigation performed: inspected the two-file native diff; independently reran the CPU proof (2322 exact comparisons); inspected the 36-case device test and current-source Watcher receipt; checked complete repaired all64 B32 acceptance.
  Resolution: fixed. Equal tile-aligned widths retain one map per tile row and reconstruct the column offset. Unequal/partial widths retain the original path. No production cache eviction or context reduction was introduced.

- Observed anomaly: common sampling deallocated a borrowed persistent CCL output.
  Evidence: `AUTOFIX.md`, ownership-negative/fixed receipts, `SamplingCCL.line_all_gather`, and current retained-trace tests.
  Affected path: split sampling candidate gathers.
  Control or comparison: the borrowed-output failure is preserved; the adapter now returns a sampler-owned clone.
  Likely subsystem: tensor ownership at the CCL/sampler boundary.
  Investigation performed: inspected ownership, capture ordering, and the direct `tt_out_tok` connection to the next decode input.
  Resolution: fixed. No broad corruptible-allocation exemption was added.

- Observed anomaly: final prefill/resume numbers are not universally bit-exact with other execution partitions.
  Evidence: `mixed_prefill_native_all64.json` and `readiness/prefill_resume_check.json`.
  Affected path: uniform bulk prefill versus solo, and split recurrent-state publication versus uninterrupted prefill.
  Control or comparison: ragged/reordered all64 rows and preemption rebuild are exactly equal to solo; uniform bulk has minimum PCC 0.999833 and complete top-1/5/100 agreement. Resume case means are at least 0.9994319; the worst individual PCC is 0.9857546 against the runner's explicit 0.98 floor, with minimum top-1 agreement 0.9922179 and every last token matching. Both missing-history and fresh-suffix controls fail in both eligible windows.
  Likely subsystem: partition-dependent numerical execution and BF16 published recurrent state.
  Investigation performed: inspected the private FP32 prefill workset, once-per-public-call BF16 publication, unaligned head handling, and raw positive/negative control metrics.
  Resolution: controlled within the required gates. This is not a claim of universal bit equality or automatic prefix-cache support.

- Observed anomaly: adversarial greedy ties can select token 70001 instead of HF's 70000.
  Evidence: `sampler_ties.json`, common `TTSampling`, and `sampling_choice.md`.
  Affected path: more than 32 equal maxima in one local shard.
  Control or comparison: all returned IDs are members of the true maximum set; cross-shard ties and changed unique winners also pass.
  Likely subsystem: local top-k tie retention.
  Investigation performed: inspected semantic k=1 versus physical k=32 and the true-maximum assertions.
  Resolution: controlled; the goal requires semantic greedy, which passes.

- Observed anomaly: profiler op-to-op gaps exceed the measured window, while generic advice suggests implausibly large tracing savings.
  Evidence: `perf_terminal/profiling_analysis.md`, `kernel_accounting.json`, signposted workload, and raw CSV location in the report.
  Affected path: performance interpretation across profiler-drain boundaries.
  Control or comparison: per-device kernel sums and measured window wall times; separate uninstrumented all64 timing.
  Likely subsystem: profiler window accounting.
  Investigation performed: checked signpost/drain placement and inspected the per-device kernel accounting. The stage excludes cross-window gaps rather than reporting them as removable model stalls.
  Resolution: controlled. Same-policy adapted LM-head candidates are correct but slower in uninstrumented complete-terminal comparisons. Sampling/readback does not dominate the full-model path.

- Observed anomaly: a closing-think marker and a single doubled “thinking” phrase appear in generated text; the suite often reaches its token cap while planning.
  Evidence: both actual readiness completion files and every output in `qualitative/{hf_suite,tt_suite}.json`.
  Affected path: prompt formatting and limited generation windows.
  Control or comparison: the readiness HF/TT outputs match in all 128 token IDs; HF also emits the doubled phrase on the story prompt. The unstripped canonical chat template is covered separately by the shared suite.
  Likely subsystem: matched model/template behavior and the 128-token cap.
  Investigation performed: read the actual outputs, rendered prompts, tokenizer/revision metadata, and the recorded 24-versus-25-token whitespace caveat.
  Resolution: controlled observations, with no new qualitative failure attributed and no statistical equivalence claim.

- Observed anomaly: primary synthetic timing output repeats one token extensively; setup-inclusive readiness TTFT also differs greatly from warmed TTFT.
  Evidence: `tests/profile_full_terminal.py`, `performance_final.json`, teacher-forcing log, and `performance_summary.json`.
  Affected path: benchmark workload and measurement boundaries.
  Control or comparison: the timing prompt is explicitly arbitrary deterministic token IDs and carries no quality claim. Separate natural/chat generation and the changed-token trace probe exercise functional feedback. Warmed timing excludes 93.06 seconds of preparation, while readiness TTFT includes it.
  Likely subsystem: workload choice and timing scope, not evidence of a frozen token-feedback buffer.
  Investigation performed: traced the measured call through `generate`, independently recomputed all run rates and medians, checked equal output sequences, counters, trace IDs, and persistent addresses.
  Resolution: controlled for performance-only use. Confirmed warmed TTFT 86.9243 ms and token-out 30.9676 tokens/s/user over 128 input/128 output tokens; the separate 31.05 teacher-forcing and 31.5035 logits-only rates remain correctly labeled.

- Observed anomaly: logs include motherboard-discovery fallback, L1 semaphore-fragmentation advice, and a B32 clock settling at 1337 MHz rather than 1350 MHz.
  Evidence: current trace/B32/mixed/max-context logs and native Watcher capture.
  Affected path: platform discovery and allocation diagnostics.
  Control or comparison: four-chip FABRIC_1D setup completes, strict allocation tracking includes programs, measured complete preparations succeed, and native Watcher has no error matches.
  Likely subsystem: platform metadata and resource advisories.
  Investigation performed: distinguished the discovery fallback from model/host inference fallback and checked completed current acceptance. B32 is not the primary latency measurement.
  Resolution: controlled; no unclassified device-health or inference-fallback failure found.

## Scope Inspected

- Goal/skill paths: `doc/full_model/goal_contract.txt`; installed stage-review, full-model, qualitative-check, tt-enable-tracing, optimize, tt-device-usage skills and model-bringup startup; repository `TraceCorrectness.md`.
- Artifact paths: model and full-model READMEs; work log; commands and hooks; stage gate; current numerical/performance/context/B32/mixed/trace/resume receipts and corresponding log excerpts; actual `.refpt` payload and cache-key JSON; both readiness completion files and token metadata; both shared qualitative suites/verdicts; serving contract/mapping/audit; context and capacity accounting; runtime audit; sampling choice/ties/ownership/seed/host receipts; AutoFix/AutoDebug and reshape proof/device/Watcher/memory evidence; terminal profiling and LM-head comparison/layer-bound evidence. Paths are relative to `models/autoports/qwen_qwen3_6_27b` unless stated otherwise.
- Code paths: complete `tt/model.py`, `tt/generator.py`, `tt/bounded_prefill.py`; accepted decoder source identity/policy references; common sampling state/trace/tie paths; full-model trace, slot, context, terminal, host-compatibility, and reshape probe sources; both changed native reshape files and the unchanged writer's mapping contract. The accepted multichip decoder hash remains `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b`.
- Pinned serving sources: checkout HEAD verified as `042a94863ad27d88207bcf71abdf67122c6f4c27`, clean worktree; decode-reload and capability docs plus worker/model-runner references. The 79-row contract has no missing rows and every supported evidence path exists. Excluded capabilities are named explicitly.
- Commands run: read-only `git status`, `git diff --cached`, `git rev-parse`, `rg`, `sed`, `cat`, and small Python artifact analyses; CPU-only mapping proof piped to a compact stdout summary; host-only Torch load of the existing reference. Independently checked seven current source hashes, Watcher raw-log digest, numerical counts, autoregressive token equality, timing arithmetic, layer-bound raw values, and evidence-path existence. No TTNN import, device use, server, reset, target hardware test, or implementation edit occurred. Only this report was written.

## Residual Risk

- Inspection and finite acceptance cannot prove every scheduler interleaving or long-generation accuracy case. The existing evidence is sufficient for stage 6; later integration must preserve the exclusive cache lease, preparation-before-capture rule, caller-owned resident-position/page admission, and recurrent-state exclusion for automatic prefix caching.
- Native reshape evidence covers the model's standard tile geometry, both packing directions, BF16/FP32/UINT32, cache reuse, and unequal/partial-width fallbacks. It is not an exhaustive test of every unrelated reshape consumer or architecture.
- Maximum-context acceptance proves execution/capacity through the final legal position, not long-context answer quality. The fixed B32 evidence likewise establishes capability rather than concurrent-serving throughput.
- Preserve the accepted stage-5 carried confidence-bound note and rejection ledger; this full-model pass does not reinterpret that earlier performance uncertainty.
