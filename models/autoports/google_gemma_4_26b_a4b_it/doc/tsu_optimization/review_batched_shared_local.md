# Stage Review

Verdict: clean-pass

This is an independent review of the incremental shared-decode batching checkpoint for `google/gemma-4-26B-A4B-it` on Blackhole TP4/QB2. It is not completion of the full optimization/release goal. CI run 36564611976 addresses the earlier C1 checkpoint and predates this batch change; a new exact image and the full 23-row matrix remain separate required qualification. This review does not waive them. SWE run 36530661132 is outside this review and was not touched.

The inspected TT-Metal worktree is `mvasiljevic/gemma4-ttft-opt` at `eb1d2af61c7630c7424e9593b646093ef7be45af`, with live incremental changes. TTI remains clean at `ba2aea35fded98a19821c34ac61aa05bd63bf502`.

`E` below means `models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/tsu_optimization`.

## Required Work

None for this bounded local checkpoint. The P2 comparison-harness finding was fixed and validated during review. The fresh default server completed all six matched benchmark cohorts and reproduced the selected gain.

## Other Concerns

- No production defect was found in the guarded implementation. It selects dense active batches 8 through 32 only, with Linear topology, replicated residuals, paired MoE reduction, fused tail, and the exact selected `_SharedMLP` implementation. Smaller/larger/incompatible configurations use the inherited decoder. `tt/model.py` additionally routes batches containing inactive interior slots through per-slot decoder calls, so those calls remain on the original B1 path.
- The candidate preserves the decode weight/fidelity selection by calling `_SharedMLP.decode_batch`; ordinary multirow `__call__` continues to select prefill weights. Attention, cache updates, router, and indexed expert execution remain per slot. Production batching adds no host read, host tensor construction, or sampling path. The generator retains nonblocking model replay, the existing sampling trace, and device token feedback.
- Selected-default serving results independently recomputed from raw JSON are below. Every normalized command matches its control; all 104 generated texts, input lengths, output lengths, model IDs and tokenizer IDs match exactly, with zero failed requests. The earlier opt-in run independently passed the same 104 comparisons. There are two repeats per row. TSU is `1000 / mean TPOT`.

| ISL/OSL/C/N | Control TPOT ms | Final default TPOT ms | Control TSU | Final default TSU | TSU change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096/128/1/4 | 19.583877 | 19.585127 | 51.06241 | 51.05915 | -0.00638% |
| 128/128/32/32 | 680.429811 | 643.759471 | 1.46966 | 1.55338 | +5.69628% |
| 128/128/8/16 | 180.455993 | 175.440160 | 5.54152 | 5.69995 | +2.85900% |

- The final C32 median ITL reduction, 677.3433 to 640.9244 ms, agrees with the complete 30-layer generator's paired 677.4438 to 641.0122 ms result. Thus the observed gain includes steady decode rather than only moving capture/prefill work between metric phases. Final mean TTFT/E2EL are 2123.78/4611.09 ms for C1, 13444.34/95201.79 ms for C32, and 2992.37/25273.27 ms for C8, all below their paired controls. The C1 TPOT difference is small relative to repeat variation and its implementation path is unchanged; no C1 speedup is claimed.
- Opt-in candidate TPOT was 19.6144/643.5137/175.1956 ms for C1/C32/C8. The final default reproduces that performance. `E/batch_default_async/launch.json` has no batching override and matches every current runtime hash; server logs attest all 30 layers, context 262144, 32 slots, async scheduling, device sampling, and the unchanged trace reservation. `E/batch_default_comparison.json` agrees with the independently recomputed raw results. Additional guards still running at review close are not included in this verdict.
- The candidate's 18 concurrent shared-suite replies exactly match the control's text, finish reason, and token usage, as well as the pinned six-prompt suite. I read all six unique outputs. Prompt metadata contains the pinned revision, chat template, rendered messages and token IDs; the endpoint is `/v1/chat/completions`. Nonaligned prompt lengths and early completion of short replies exercise changing active-slot layouts. No new output-quality defect was observed.
- Current default runtime SHA256 is `1635882a40eff01005d57de94725c5b068d05c5f1a8a289705e27079f736b7f9`. Reversing only the constructor's environment default from `1` to `0` reconstructs the candidate/safety hash `c6c55a8bb55be7253e32e6ab6a9d6d4515ccdd95842c4721e6c53f8c021e3e4f` exactly. Other runtime file hashes match their candidate manifests.

## Hard-Check Gaps

- Watcher evidence disables Ethernet checks, retaining worker checks. The exact allocation-tracking command records `TT_METAL_TRACE_ALLOC_TRACKING=1`, tracebacks enabled, and program-cache exclusions disabled. This is scoped Watcher evidence, not full Ethernet instrumentation.
- The B32 final contract and current-source B8 repaired-control contract each contain six exact layer-output and full per-rank K/V comparisons, spanning layers 0/5, changed/reordered activations and advancing positions across 32-token page boundaries. These are representative-layer checks. All-layer coverage comes from the complete-model probe and actual serving, not an all-layer instrumented run.
- Maximum-context correctness is inherited from the unchanged attention/cache path. This increment keeps the 262144 context setting and aggregate capacity, full-width page tables, 32 scheduler slots and 1 GB trace reservation; it does not newly execute a 262144-token prompt or claim 32 simultaneous maximum-length requests.
- The updated context contract distinguishes zero new persistent tensor payload from nonzero retained row temporaries, concat buffers, ordinary batched MoE RS/AG outputs and trace scratch. It makes no unmeasured transient-peak claim.
- Reduced B8 profiling describes the serialized control with real layers 0/5 and the terminal/sampler path. It is not a serving profile or a full-stack device breakdown. I verified all four compressed per-device window hashes/counts and independently reconstructed the 14.0960–14.0997 ms windows. The advice table and those windows cannot establish exhaustion of all remaining optimization candidates.
- The vectorized expert-union experiment remains probe-only despite an approximately 4 ms complete-generator improvement beyond shared-only batching. The later SDPA geometry investigation is also outside this checkpoint's selected production change. Neither is accepted or rejected as a final serving optimization by this review.

## Anomaly Ledger

- Observed anomaly: selecting default-on batching could make the comparison harness test the enabled path against itself.
  Evidence: the initial `check_tsu_batch.py` and `probe_tsu_batch.py` inherited the constructor default before capturing their reference.
  Affected path: future reference/candidate correctness and timing evidence.
  Control or comparison: earlier accepted artifacts ran while the default was off, so their serialized references remain valid.
  Likely subsystem: test-harness control selection.
  Investigation performed: raised a P2 finding; the owner explicitly sets every decoder's baseline flag to false before capture and records it in JSON. I inspected both fixes. `E/batch8_default_contract.json` matches the current runtime and the repaired harness revision used for that run, records `baseline_batched_shared_decode=false`, and passes all six exact output/KV cases. A later probe-only `experts_index_union` choice changes harness/helper hashes; I inspected its dispatch and verified it leaves the forced baseline and `candidate == "runtime"` branch unchanged. `E/batch_policy_default.log` reports 48 passing tests.
  Resolution: fixed; no outstanding required work from this finding.

- Observed anomaly: batching removes the intervening per-slot MoE collective before persistent attention-buffer reuse.
  Evidence: `_decode_shared_batch`, `CollectiveBufferPool`, and `doc/tsu_optimization/autodebug_batch_collectives.md`.
  Affected path: persistent Linear attention RS/AG output ownership and captured semaphore reuse.
  Control or comparison: inherited per-slot path and exact B8/B32 checks.
  Likely subsystem: cross-rank collective ordering.
  Investigation performed: inspected the independent source report, actual separate RS/AG calls, Linear final-reduction setup and payload-ready waits, dispatch prior-worker completion wait, and unconditional program-config writes. The next RS needs every rank's contribution before its subsequent AG can overwrite the previous gathered output. Fresh residual and routed/shared-input owners remain live until concatenation. The production guard explicitly excludes Ring; the comment now states the actual proof.
  Resolution: controlled for Linear TP4, one CQ/subdevice, matching rank order, separate RS/AG workloads. No claim for concurrent callers or future fused/streaming collectives.

- Observed anomaly: fully batched attention changes output logits/top-1; initial batched norms retained aliases of persistent attention output.
  Evidence: `E/batch32_attention.json`, `batch32_attention_contract.json`, `batch32_rowsdpa.json`, `batch32_norms_owned.json`, and probe source.
  Affected path: rejected attention/norm experiments, not selected runtime.
  Control or comparison: serialized reduced control and shared-only candidate.
  Likely subsystem: SDPA batch geometry and persistent-output ownership, respectively.
  Investigation performed: the attention experiment keeps exact caches but changes a reduced full-model top-1; adapting it to per-row SDPA restores exact logits and takes 49.5784 ms, slower than the roughly 47.39 ms control. Cloning attention results before norm batching restores exact logits but takes 51.1091 ms. Neither experiment is installed in production. B2 shared batching is likewise exact but slower, 4.3437 versus 4.0997 ms, and the production guard excludes it.
  Resolution: controlled; adapted candidates were measured and rejected rather than dismissed at their first failure.

- Observed anomaly: generic allocation warnings appear after capture, and shutdown reports nanobind reference leaks.
  Evidence: candidate/control server logs, final Watcher/allocation logs, and the prior selected-local review.
  Affected path: trace scratch lifetime and process teardown.
  Control or comparison: the same warning families occur in the inherited control; no new failure appears during accepted requests.
  Likely subsystem: allocator warning emitted for any allocation while a trace is live; binding teardown diagnostics.
  Investigation performed: inspected allocator warning semantics and trace-managed temporary ownership, current-source reduced allocation checks, completed serving requests, and normal shutdown ordering. The new batched path does not add post-capture host allocations; the existing generator warm/capture/restore and per-request trace invalidation remain unchanged.
  Resolution: controlled within the finite tested lifecycle; not proof of arbitrary future allocations or a long-running memory soak.

- Observed anomaly: benchmark random-token prompts produce refusals/repeated control-like text under forced `ignore_eos`; four official qualitative answers hit their 256-token cap.
  Evidence: raw benchmark `generated_texts` and both concurrent qualitative JSON files.
  Affected path: interpretation of throughput stress text and quality evidence.
  Control or comparison: all 104 benchmark texts match exactly; all 18 prompt-correct qualitative texts, finish reasons and token counts match their control and pinned suite.
  Likely subsystem: synthetic benchmark inputs/forced length and configured qualitative output cap.
  Investigation performed: inspected benchmark commands and actual generated text separately from the official suite. For example the haiku, translation and explanatory/code/story replies remain the same; long replies stop at the same partial sentence/code position as the control.
  Resolution: controlled. Synthetic benchmark completions are not used as a standalone quality verdict.

- Observed anomaly: B32 profiler readout and an initial reduced allocation setup did not produce usable evidence.
  Evidence: preserved failed profiler/probe logs and `E/batch8_profile_retry`.
  Affected path: diagnostic evidence collection.
  Control or comparison: the successful reduced B8 retry with full aggregate cache allocation and full-width page tables.
  Likely subsystem: profiler volume/host memory and harness cache-capacity setup.
  Investigation performed: only the successful reduced retry is used for attribution; compressed per-rank fields/hashes and timings were checked. Failed artifacts are not counted as passes. No live-serving profiler result is used.
  Resolution: controlled for this checkpoint's profiling claims.

## Scope Inspected

- Goal: optimize actual vLLM/TTI TSU while preserving correctness, 262144 context and 32 slots; compare identical workloads, prove trace/page/cache safety, retain measured wins, and profile only reduced non-serving paths.
- Skills: complete installed `stage-review`, model-bringup startup, `optimize`, `tt-enable-tracing`, `vllm-integration`, `qualitative-check`, `tt-device-usage`, and `tt-profiler`; applicable LLM optimization and profiler-report references. Verified enabled AutoDebug dependency and its package environment. No child reviewer was spawned.
- Code: incremental `tt/multichip_decoder.py`; surrounding model/generator/adapter/CCL source; `tests/{test_tsu_batch_policy,check_tsu_batch,probe_tsu_batch,tsu_batch_candidate,filter_tsu_profile,summarize_full_profile}.py`; `tools/{compare_tsu_benchmarks,tsu_qualitative_batch,tsu_benchmark,ttft_server}.py`; relevant native dispatch/Linear RS source. New unrelated probe tooling is excluded from the selection verdict.
- Evidence: work log and context contract, final B32/current B8 contracts and logs, prior B8 runtime contract, full-model paired batch probe, rejection artifacts, CPU logs (176 passed plus 3 subtests; final 48 passed), reduced B8 raw windows/manifests/advice, control/candidate/default launch records, raw serving commands/results/logs, concurrent qualitative outputs, and source inspection report/AUTOFIX follow-up.
- Commands: read-only `git`, `rg`, `sed`, `cat`, file inventory and small Python JSON/hash/CSV/text analyses. I did not import TTNN, open hardware, start a server, reset devices, run model tests, modify implementation, or send external messages. This report is the only reviewer-authored file.

## Residual Risk

Finite representative-layer and serving evidence supports this local change, not exhaustive model quality, maximum-context requalification, every sampling mode, or a long concurrency soak. The source proof depends on the current Linear collective/dispatch contract. Default reproduction is complete; remote qualification of the new batch-enabled image and the full matrix remain separate. Further production changes require their own focused evidence and review.
