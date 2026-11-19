# Stage Review

Verdict: more-work-needed

Independent round 1, stage 7 optimized-full-model, ornith-ai/Ornith-1.5-9B.
Reviewed the live staged worktree on main over
ebd521f744bb61f787a164f6b207c6defb8dd306. No hardware, tests, servers, resets,
or implementation edits were performed by the reviewer.

## Required Work

- P1: Reject resident replay windows that cross an unmapped causal page before submitting any trace.
  Evidence: `tt/generator.py:973` adds `decode_resident`, but its only position/page admission check is the configured-context comparison at lines 993–997; it never checks `state.host_tables`. The existing reload preflight explicitly permits `-1` for unused pages (lines 801–802) and checks mappings only through the position being submitted (lines 814–820). Consequently, with context 64 and table `[[0, -1]]`, an ordinary valid decode at position 31 succeeds and leaves `host_positions=(31,)`, `position_advances=1`. `decode_resident(1)` then passes its context check and submits position 32 against page `-1`. This is a source-derived counterexample; the unsafe device call was not run. The actual full-attention path calls `paged_fused_update_cache` (`tt/multichip_decoder.py:1272`). Its writer reads the page as an unsigned physical block and computes an address without an unmapped-page guard (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/kernels/dataflow/writer_paged_fused_update_cache_interleaved_start_id.cpp:108`); only a negative **position** skips the update. The sibling `decode_forward(..., reload_inputs=False, reload_page_table=False)` also omits resident context/page validation in `_decode_preflight`.
  Why this matters: Partially allocated scheduler-owned page tables are admitted by the explicit cache contract. A valid initial decode can therefore be followed by an accepted public resident call that produces invalid cache addressing, rather than a host rejection or a request for the next mapped page. Fully mapped benchmark tables do not cover this case.
  Required next step: Validate every active slot's entire requested decode window against the authoritative resident host position and mapped causal-page prefix before queuing work. Preserve inactive rows and zero device transfers. Apply the same admission invariant to ordinary one-step resident decode so the sibling path cannot bypass it. Add focused host checks for the exact last mapped position, first unmapped position, a multi-step window crossing a hole, inactive rows with unmapped tables, and unchanged counters on rejection. Then exercise a valid mapped page-boundary transition and changed-table continuation against explicit-feedback control with allocation tracking, retaining trace IDs/program counts. Rerun affected resident performance/contract evidence after the guard change; the invalid-address call itself must remain host-only negative coverage. No decoder arithmetic or broad accuracy/geometry sweep is requested for this admission fix.

## Other Concerns

- Primary warmed TTFT is honestly reported as 90.017 to 92.049 ms. The three samples overlap (before 89.992/90.017/92.559, after 92.049/95.309/89.602 ms), and the separate capacity-matched head control does not establish a head-induced regression. Retain the reported primary number and its limitation; do not claim a TTFT improvement.
- The shared-suite output remains cap-limited. I read all six current texts and both current HF/TT autoregressive pairs. The reasoning is coherent, prompt05 completes its translation, and the other cap-limited outputs are not evidence of completed answers or runnable code. Exact current/prior token, prompt, seed, and sampling equality holds for all six shared prompts. The existing ten-seed blind prompt04 control records HF 4/10 versus TT 1/10 content failures and 10/10 cap-limited outputs on each side; the corresponding limitations carry `original_model: "same"`. These are controlled model limitations, not Required Work.
- Local commit creation and recording the resulting SHA remain the explicitly planned post-review bookkeeping. The user prohibited pushing and vLLM work. Neither action was performed by this review.

## Hard-Check Gaps

- The 45-test host suite does not cover sparse resident page tables: `tests/full_model_resident_test.py` supplies no `host_tables` on its synthetic state, so it cannot discover the required-work counterexample. Existing resident device/performance cases allocate every page in their timing window.
- Some earlier stage-7 artifacts (`resident_tracked`, shared qualitative, raw autoregressive, resume) predate the final selective-restore integration. They are not among the 21 commands claimed to match final production hashes. Current tracked full-stack B32, isolated copy, sampling-transition and Watcher evidence covers the changed selective branch; the unchanged B1/prefill paths and exact-token comparison support the retained earlier evidence. This does not itself require a wholesale rerun, but future source-change attribution should continue to distinguish reused from final-source evidence.

## Anomaly Ledger

- Observed anomaly: Resident replay can cross from an admitted mapped prefix into an unused `-1` page.
  Evidence: Source-derived position-31/table-`[0,-1]` counterexample above.
  Affected path: New fixed-step resident API and ordinary one-step resident admission.
  Control or comparison: Reload admission rejects an unmapped current causal page; resident admission does not.
  Likely subsystem: Host request/window validation before traced paged KV update.
  Investigation performed: Followed host position/page metadata through `_decode_preflight`, `_execute`, the new API, and the actual paged-cache writer; inspected the host and device coverage.
  Resolution: more-work-needed.

- Observed anomaly: One inactive B32 row previously added a large full-model decode cost.
  Evidence: Matched final control/selected resident latency 18.017388/11.471057 ms; identical final token vectors; bitwise copy cases and full-stack inactive-state checks.
  Affected path: Per-linear-layer recurrent-state protection.
  Control or comparison: Full-pool copy/merge versus selective FP32 NoC copy; all-active final 10.741069 ms.
  Likely subsystem: Avoidable DRAM traffic.
  Investigation performed: Read both copy branches/kernel, full snapshots for scheduler permutations, isolated B1/B3/B32 tests, and guard profile. Recomputed 24 times the per-layer profile savings as 6.656112 ms versus measured 6.546332 ms.
  Resolution: fixed for the measured contract.

- Observed anomaly: Reduced extrapolated device time exceeds full-model quiet-host resident latency.
  Evidence: Recomputed report row sums yield 7.947128 ms stack plus 1.231759 ms terminal/entry, versus 8.708730 ms measured full resident.
  Affected path: Performance attribution, not execution correctness.
  Control or comparison: Exact same-run reduced raw firmware spans are 1.726575/1.729251/1.728515/1.720941 ms, versus 1.808981 ms host time.
  Likely subsystem: Reduced-layer extrapolation, independent rank maxima/gaps and instrumentation.
  Investigation performed: Recomputed all component sums and raw firmware spans; inspected explicit scope labels.
  Resolution: controlled; no negative host-time claim or unsupported 70% roofline claim.

- Observed anomaly: Profiler source-ID collision and subsequent broad B32 report-enrichment failure.
  Evidence: Preserved failed logs and `profiler_debug/AUTODEBUG.md`/`AUTOFIX.md`; accepted narrow final profiles.
  Affected path: Instrumentation/reporting.
  Control or comparison: Byte-identical source-path overlay and narrowed reduced B1/full-boundary plus B32 guard fixtures.
  Likely subsystem: Profiler source-ID/metadata collection.
  Investigation performed: Read the failure disposition and verified all five accepted raw CSV and window hashes. Decode contains 104 operations on each of four ranks.
  Resolution: controlled; rejected broad captures are not used as passing reports.

- Observed anomaly: Allocation warning appears in tracking-disabled teacher-forcing output.
  Evidence: `logs/aime_teacher_forcing_final.log.gz`; tracked final B32/capacity/sampling runs contain no unsafe-allocation failure; positive control explicitly rejects a live unsafe buffer and succeeds after release.
  Affected path: Trace allocation diagnostics.
  Control or comparison: Program-cache-inclusive tracked replays and exact-output acknowledgments.
  Likely subsystem: Generic allocator warning when another trace is already live.
  Investigation performed: Inspected captured logits' producer-before-consumer lifetime, persistent masks/CCL/state, tracker environment and positive-control log.
  Resolution: controlled for exercised traces; the new copy adds no allocation acknowledgment or exempt scope.

## Scope Inspected

- Goal/skill paths: Supplied stage contract; installed 0.1.23 model-bringup, stage-review, optimize, full-model, multichip, tt-enable-tracing, qualitative-check and tt-device-usage skills; relevant LM-head/sampling and prefill optimization references.
- Artifact paths: `doc/optimized_full_model` README, work log, checklist, runtime audit, mesh plan, provenance/dependency/archive/source manifests, baseline and final performance, head geometry and sampler controls, tracking/Watcher/transition/fallback/capacity/resume evidence, accuracy and raw logs, qualitative outputs/controls, five accepted profile windows/reports, accounting tables; context contract and prior serving/quality artifacts.
- Code paths: Complete staged production diff (`tt/model.py`, `generator.py`, `batch_state.py`, `inactive_state_copy.py`, copy kernel); relevant unchanged cache/decode/prefill/sampler contracts and decoder call sites; changed benchmark/probe/admission-test sources; pinned tt-transformers LMHead1D; paged-cache writer.
- Commands run: Read-only `git status`, `git diff --cached`, `git show`, `git rev-parse`, `rg`, `cat`, `sed`, `nl`; small standard-library Python analyses of JSON/CSV/gzip/source hashes and arithmetic. No TTNN import, test execution, device access, server, reset or implementation mutation. Only this report was written.
- Independently verified: 21 final command production-source hash sets match current files; eight claimed preserved source files equal HEAD byte-for-byte; all 94 archive hashes/lengths match; all five raw profile/window hashes match; profile component sums and same-run firmware spans reproduce; measured matmul attributes show BF8 weights and selected HiFi2/LoFi policy; final AIME logs report prefill 99/100/100 and teacher-forced 98/100/100 percent top1/top5/top100; head/sampler controls and final default throughput are consistent with the reported values. The pinned tt-transformers checkout is clean at the documented SHA.

## Residual Risk

- This is an inspection review, not a new hardware run. Required-work consequences are inferred from the actual address calculation, not an intentionally executed invalid address.
- The 262144-token gate demonstrates capacity and finite output, not max-context HF accuracy. B32 at context512 does not establish B32 at maximum context; the report makes neither claim.
- Six one-seed qualitative samples establish preservation/smoke coverage, not broad comparative quality. The prior matched ten-seed judgement applies to its named prompt and limits.
- Performance uses short warmed windows and reduced-layer profiling; reported component extrapolations are estimates, with the same-run reconciliation kept separate.
