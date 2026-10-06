# Stage Review

Verdict: clean-pass

Independent read-only review, 2026-09-29. This verdict covers the **selected-local TSU checkpoint only**: default-enabled exact-signature eager-prefill decode retention, async throughput configuration, and BFP4/LoFi LM-head K1. It does not qualify a new immutable image, complete the remote matrix, or declare optimization exhausted.

Paths below are relative to `models/autoports/google_gemma_4_26b_a4b_it`; `E` denotes `readiness_vllm/tsu_optimization`.

## Required Work

None for this bounded local checkpoint. No actionable source correctness defect or unsupported local acceptance claim remained after inspection and the final selected-default controls.

## Other Concerns

- Independently recomputed primary serving results: eight measured 4096-input/128-output/C1 requests give pooled TPOT **19.650091 ms**, TSU **50.89035**, TTFT **2123.388 ms**, and E2EL **4618.949 ms**. The matched local baseline gives 23.260252 ms, 42.99179 TSU, 2094.308 ms, and 5048.360 ms. Thus TSU improves 18.37% and E2EL improves about 8.5%; pooled TTFT worsens by 29.08 ms. This is not a TTFT improvement. All 20 final primary/short/8K texts and lengths match the original baseline exactly.
- Full-generator K4→K2→K1→K4 controls, five steady repeats each, give mean TPOT 19.641300/19.629814/19.628722/19.639584 ms. All 128-token outputs match. K1's 11–13 microsecond saving exceeds the approximately 1.7 microsecond K4 return-control drift, but is only about 0.06% of full decode and is not separately resolved in serving noise. Precision is unchanged.
- Final long-output, 32K, and C32 guards complete all 38 requests with exact K4-async control texts and lengths. C32 mean TPOT is 680.538670 ms and median ITL is 677.347571 ms, essentially unchanged from the earlier async K4 control. The inherited batch-row loop in `tt/optimized_decoder.py` remains a distinct performance opportunity, not a solved problem.
- I read all six actual qualitative outputs, not just their pass flags. The final 18 greedy responses exactly match the accepted shared suite; all six sampled texts, finish reasons, and usage match the prior synchronous controls. Prompt rendering, token IDs, pinned HF revision, canonical chat template, and chat endpoint are recorded. This supports absence of a new local quality regression, not a fresh broad accuracy qualification.

## Hard-Check Gaps

- **Still required for the overall task:** build and test the new exact immutable image; run focused CI and the full 23-row qualification matrix; complete the requested commit/push workflow. These gates are pending, not waived or satisfied by this pass. The local run used the recorded compiled image with candidate Python bind-mounted.
- No full-30-layer device-only profile or serving profile was collected. This follows the serving-stage profiler exclusion. `doc/tsu_optimization/perf_summary.json` correctly leaves full-model device duration and roofline null. The 2548.310/2474.757 microsecond windows are reduced real layers 0/5 plus terminal/sampler/recorder, not full-model timings.
- Final trace-allocation/Watcher stress uses reduced layers with full 262144 cache allocation. Final serving exercises 32K context and 32 scheduler slots; maximum-context and broader sampling guarantees also rely on inspected inherited capacity/sampling evidence and unchanged configuration, rather than a new final-image 262144-token request or complete sampling matrix.
- Ethernet Watcher instrumentation was disabled for its documented capacity limitation; worker checks remain enabled. CPU results (125 model contracts, 111 TTI tests) were reviewed from logs, not rerun by this reviewer.

## Anomaly Ledger

- Observed anomaly: eager prefill allocates while a decode trace remains live, raising stale-input/allocator concerns.
  Evidence: `tt/generator_vllm.py`, `tt/generator.py`, `E/final_watcher.json`, `E/final_tracking.json`, associated raw Watcher logs.
  Affected path: retained single-slot greedy decode after eager prefill.
  Control or comparison: changed prompt tokens, reversed live physical pages, tile-boundary lengths, repeated signatures, and 4096→4097→4096 transitions against eager controls.
  Likely subsystem: trace ownership and input refresh.
  Investigation performed: inspected cache-object/tensor identity, address/spec signature, exact shape/stride/dtype checks, fallback invalidation, blocking prefill completion, persistent token/position/page updates, and pending-key promotion. Checked 31 Watcher rows and 11 explicitly allocation-tracked rows; matching warmed signatures have zero recaptures/program-cache growth and unchanged model/sampler trace identities.
  Resolution: covered for the selected narrow path; no unsafe-survivor exclusion is used. Unsupported signatures continue through invalidation/fallback.

- Observed anomaly: asynchronous page growth or slot changes could let queued decode consume stale tables.
  Evidence: installed plugin `async_decode.py`, `model_runner.py`, `input_batch.py`; inherited `doc/optimized_vllm/adapter_queued_reads.json` and `batch_cache_watcher.json`; final serving guards.
  Affected path: queued async decode and allocator-driven growth.
  Control or comparison: installed/host core-plugin file hashes match; inherited simultaneous-request/queued-buffer controls; final 1024-output, 32K, and C32 exact comparisons.
  Likely subsystem: scheduling and async input ownership.
  Investigation performed: read actual image plugin source via read-only container commands. Real new block IDs disable overlap; pending work drains before non-steady/layout-changing input preparation. Same-layout page changes refresh persistent tables. The direct queued-read test changes unused columns, so it was not mistaken for allocator-growth proof.
  Resolution: static drain contracts and real serving growth controls agree; no new plugin change or stale-page defect found.

- Observed anomaly: large TPOT gains can reflect phase attribution rather than faster device decode.
  Evidence: raw baseline, retained-sync, retained-async and selected-async chat/guard JSON; `E/direct_paths.json`; server capture logs.
  Affected path: performance claims, especially C32.
  Control or comparison: sync retention isolates recapture removal; async changes host overlap; direct full-model buffered/per-token/queued paths separate readback behavior.
  Likely subsystem: scheduler timing and trace setup.
  Investigation performed: recomputed pooled metrics and compared TTFT, E2EL and steady ITL. Direct queued full-model decode is about 19.626 ms; final C1 serving is 19.650 ms, but prompts differ, so this is not a precise host-overhead subtraction. C32 async raises TTFT while steady ITL barely changes.
  Resolution: primary TSU claim is supported; no high-concurrency device-speedup or TTFT-win claim is supported. Historical token-readback counters undercount a synchronous token read path; zero does not mean no host token delivery.

- Observed anomaly: generic profiler advice suggests tracing savings even though replay is already traced.
  Evidence: `E/profile_retry/{raw_ops.csv.gz,summary.json,decode_report_advice.txt}` and `E/profile_short/{raw_ops.csv.gz,summary.json,decode_perf_report.txt}`.
  Affected path: bottleneck attribution.
  Control or comparison: raw per-rank timestamps versus merged advice rows.
  Likely subsystem: report heuristics and cross-op/rank timing.
  Investigation performed: verified raw CSV hashes and independently reconstructed max-rank first-FW-start to last-FW-end windows. The large 4K advice gap is dominated by the first Slice interval outside the measured replay window. Short-profile advice still reports a theoretical 35 microseconds of gap savings despite tracing; it is not an established avoidable host gap. TopK is the large-indices implementation, not a presumed one-core generic TopK.
  Resolution: reduced profiles are usable for local attribution only; no full-stack duration or automatic tracing-win inference is made. Short versus 4K whole-window differences also include K1 versus K4, while the SDPA extrapolation remains explicitly an estimate.

- Observed anomaly: initial profile and launch attempts were invalid; an 8K prefill-trace candidate exceeded capacity.
  Evidence: retained failed logs, corrected `profile_retry` and `profile_short`, `doc/tsu_optimization/AUTOFIX.md` and work log.
  Affected path: experiment provenance and prefill-trace feasibility.
  Control or comparison: corrected recorder index and source precedence; final source-hashed runs; full-model versus reduced trace capture.
  Likely subsystem: harness setup, Python import precedence, and trace-buffer capacity.
  Investigation performed: checked the interrupted first profile was not used despite its wrapper status, the old-source 1024-cap launch supplied no accepted candidate timings, and the full-model long-prefill trace requested 1,032,519,680 bytes beyond the configured trace region.
  Resolution: rejected artifacts/candidates remain disclosed. Reduced success is not substituted for full-model feasibility; the selected eager-prefill path does not lower advertised context or cache capacity.

- Observed anomaly: head geometry rejections could be based on the wrong precision or only one allocation failure.
  Evidence: `E/head_bfp4_*.json`, `E/head_full_paths.json`, real recorded terminal-input fixture and `tests/probe_full_lm_head.py`.
  Affected path: selected BFP4/LoFi terminal projection.
  Control or comparison: native DRAM, conversion-inclusive L1, alternate grids/subblocks, multiple DRAM readers and chunk widths, and adapted small chunks supporting larger K blocks.
  Likely subsystem: geometry, layout conversion and L1 capacity.
  Investigation performed: checked precision-locked legal cases and exact allocation rejections, real-model local top-1/PCC controls, and full-generator paired confirmation. Adapted small-chunk DRAM best is about 483.230 microseconds versus its 390.019 control; conversion-inclusive L1 best is about 383.041, slower than selected native K1 at about 375.951 in its sweep.
  Resolution: material head families have meaningful measured rejection evidence. Prior residual/fusion/CCL rejections remain inherited evidence with their original scope, not newly repeated exhaustive proof.

- Observed anomaly: benchmark refusals, qualitative truncation and awkward wording could conceal output regression.
  Evidence: raw generated texts in all serving JSON and `E/{eager_sync,selected_async}/qualitative.json`.
  Affected path: output quality.
  Control or comparison: original benchmark texts and accepted canonical-chat greedy/sampled controls.
  Likely subsystem: artificial benchmark prompts, generation limit and baseline model behavior.
  Investigation performed: read every unique qualitative output and compared actual response fields. Four long prompt cases hit the configured 256-token cap in both greedy and sampled controls; inherited thermodynamics wording and the sampled Fibonacci description mismatch are also identical in control. Random-chat refusals are baseline-identical and are not treated as the qualitative gate.
  Resolution: no new regression; truncated prose/code is not claimed to be a complete answer, and exact control matching is not a claim that every baseline statement is factually perfect.

- Observed anomaly: nanobind reports leaked binding objects during interpreter shutdown.
  Evidence: `E/selected_async_server.log` after “Application shutdown complete”; original `E/baseline_sync/chat/s4096-o128-c1-n4-r0.log:53` onward reports the same warning family.
  Affected path: process teardown.
  Control or comparison: original baseline also reports 973 leaked types and 4455 functions; all selected requests finish successfully.
  Likely subsystem: existing binding teardown reference ownership.
  Investigation performed: compared warning timing/type with the baseline and checked normal shutdown log ordering.
  Resolution: classified as inherited teardown diagnostics, not silently ignored or presented as proof of runtime memory safety. No evidence here attributes a new serving failure to the selected change.

## Scope Inspected

- Goal/skills: supplied selected-local contract; complete installed `stage-review`, model-bringup startup, `optimize`, `tt-enable-tracing`, `vllm-integration`, `qualitative-check`, and `tt-profiler`; review router/core with trace and vLLM-serving domains. The router's two-domain limit selected the lifecycle/serving changes; unchanged precision/CCL implementations were assessed through stage evidence rather than additional domain checklists.
- Repositories: tt-metal branch `mvasiljevic/gemma4-ttft-opt`, base `557b527e096cba271ea270d922cf6f22eb598b54`; inference-server branch `mvasiljevic/gemma4-ttft-monorepo-compat`, base `318c40a5d6f5636d58e11ebd9bbe258c02cf72ce`; live selected uncommitted changes. No unrelated files or workflow runs were altered.
- Code: `tt/{generator,generator_vllm,model}.py`; changed and added TSU/slot/profiling tests and benchmark tool; installed async plugin core; TTI `workflows/model_specs/dev/llm.yaml` and its launch-argument test. Source hashes in final safety/launch manifests were checked against current implementation.
- Artifacts: TSU work log, topology audit, AUTOFIX, context contract, reproduction commands and performance summary; raw serving/profile/head/safety evidence above; inherited optimized-full-model, optimized-multichip and optimized-vLLM controls. Final server startup records 262144 maximum model length, 32 sequence slots, async scheduling, device sampling and unchanged trace capacity.
- Commands: read-only git diff/status/check, source/log searches, small standard-library JSON/CSV/hash analyses, and read-only inspection of installed container files. No test, server, hardware open/reset, or implementation mutation was performed by this reviewer. Only this report was written.

## Residual Risk

The selected local behavior is supported, not exhaustively proven: finite prompt coverage, inherited maximum-context/sampling evidence, reduced-layer allocation instrumentation, unresolved batch-row serialization, and unmeasured full-stack device breakdown remain limitations. Revalidate exact source/image provenance and the full remote matrix before overall release closure. Any subsequent implementation change, particularly batched decode or plugin scheduling changes, needs fresh focused correctness, performance and qualitative controls rather than inheriting this verdict.

## Publication Addendum — 2026-09-29

Verdict and selected-local scope remain unchanged: **clean-pass**, no additional required work.

The repository's 500 KB per-file limit changed profile packaging, not measurement. Full `raw_ops.csv.gz` captures cited above remain preserved locally but are ignored and are **not durable committed artifacts**. The durable replacements are:

- `E/profile_retry/raw_decode_ops.csv.gz`, `raw_decode_ops.manifest.json`, and `summary_window.json`: 1,146 rows including both signposts, all 310 original columns; compressed size 225,805 bytes. Decompressed window SHA256: `9c17600f43de41631f0a7db1c581e046d2a7a599f0b4ca300f7378577c9564e9`.
- `E/profile_short/raw_decode_ops.csv.gz`, `raw_decode_ops.manifest.json`, and `summary_window.json`: 1,146 rows including both signposts, all 128 original columns; compressed size 217,412 bytes. Decompressed window SHA256: `7505425eea952bfdc9c1c37b81bbad754dd9d26286e758f6aad3e696e8f8ebe8`.

I inspected `tests/filter_tsu_profile.py` and independently decompressed both full and filtered CSVs. Each filtered header and every parsed row/field exactly equals the corresponding contiguous `PERF_DECODE` through `PERF_DECODE_END` slice of the original; no in-window rows, columns or values were dropped. Original/window hashes and manifest counts match, and each original summary's `windows` object exactly equals `summary_window.json`. Both compressed windows are staged for publication; both full captures are ignored. Outside-window warmup/prefill reconstruction still requires the local originals, so the committed extracts must not be presented as complete captures. The retained advice tables and scoped summaries retain their original limitations.

The publication-only test adjustment uses the repository `expect_error` fixture, which still wraps `pytest.raises` with exception-type and message matching. `E/publication_cpu_contracts.log` reports 21 passed and two deprecation warnings. This does not replace the earlier broader CPU coverage. All 12 runtime implementation SHA256 values still match `E/selected_async/launch.json`; no runtime source change invalidates the reviewed measurements. This followup used only read-only source/artifact checks and updated this report; no hardware or test execution was performed by the reviewer.
