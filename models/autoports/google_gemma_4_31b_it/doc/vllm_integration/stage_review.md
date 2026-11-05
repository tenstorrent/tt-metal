# Stage Review

Verdict: clean-pass

Independent final inspection of stage 09, google/gemma-4-31B-it, live worktree based on 9df04b7df6909d3c5e5e8e9b286b539855df7af5. The earlier interim findings are closed by final evidence below. No hardware, API requests, servers or tests were started by the reviewer. Only this report was written.

## Required Work

None. Stage-owned local checkpoint commits and SHA logging follow this clean review as required by the stage-review workflow; they are not claimed already complete. Never push from this stage.

## Final Evidence and Closed Findings

- The canonical full sampling log is byte-identical to final_full/sampling_tests.log and finishes with 76 passed, 75 skipped, two warnings in 512.16 seconds. The final plugin host suite finishes with 2303 passed, 16 skipped in 109.37 seconds. Hooks passed. Plugin dirty paths are exclusively the five existing tests/helpers plus four new host regression test files; runtime src/ is unchanged.
- PC-02/03 correct token-versus-visible-text assumptions while retaining original request recipes and thresholds. PC-04 preserves all ten mixed-penalty requests on an explicitly stable host route using zero bias and additionally tests the six original penalty-free requests on the native route. Raw log ranges confirm route changes in the original native controls and host-only routing in both stable controls: 40 within-host and 40 cross-host comparisons match exactly.
- PC-05 fixes cohort sizes before native samples using the independent reference distribution. For maximum probability m, the two constant-cohort and fewer-than-two-paired-differences union bound is 2*m**(N-1)+m**N+N*(1-m)*m**(N-1). The bound is valid under the stated stable independent-draw assumption. Cutoff ties conservatively enlarge support; top10-only normalization upper-bounds m when ties enlarge support. Unresolved support or excessive required samples fails closed. Negative tests reject constant, out-of-support and insufficient-difference outputs. Original full-text and greedy checks remain. Final top-k cases pass with predeclared 80-row cohorts; no output-adaptive retry is used. This establishes stochasticity/support coverage, not a distribution-fit proof.
- B1 final raw summary reports 4096 input/128 output, one completed request, TTFT 754.870287ms and mean TPOT 28.311535ms, yielding 35.321292 tokens/s/user. B8 and B16 final runs complete all 8 and 16 requests with 128 input/output each. B32 audited CI burst evidence remains separate. README metrics match these artifacts and honestly state one-request percentile limitations.
- Final mixed async check passes after full sampling on the same server, including all fourteen check rows, page-only refresh, full reload, unchanged steady state, nonaligned continuation and synchronous tracked control comparisons. Canonical mixed/prefix/qualitative/sampling artifacts are byte-identical to final_full copies.
- Recomputed every TT source hash against final_source_manifest.json: no mismatch. Recounted final server sampling-mode lines: 402 host-compatible prefill, 4198 host-compatible decode, 444 device prefill, 6337 device decode, matching the lifetime audit. These lifetime counts are correctly separated from benchmark counts.
- Final cleanup artifact records no device owners. Independently checked that runner108112/API108144/EngineCore108190 no longer exist in /proc. B1/8/16 audits record corresponding serialized cleanup.
- Read all twelve final served chat outputs directly and compared the existing pinned-HF and same-code standalone controls. Responses are coherent and topical; no unrequested language or request contamination in final outputs. Length truncation and historical selected-policy language limitation remain explicitly recorded.

## Other Concerns

- Prefill still loops over rows in generator.py. The specific Stage split in vllm-integration/references/prefill-features.md explicitly permits stage 09 to measure this serial path and defers batching, exact batched-equals-solo tests and ratio gates to stage 10. Therefore this is not additional stage-09 required work despite broader wording in stage-review. Label vllm_prefill_burst_benchmark.json as the serial reference for the next stage. Its 32 x 128 / 32-output measurement has no serial_reference field because no batched arm exists yet.
- Each server advertises only [max_num_seqs] as its prepared decode bucket. This is honest capability reporting but small active loads on the B32 server pad to 32; preserve the separate B1 headline rather than implying bucket switching.
- The primary earlier result contains one request. Its P50/P99 do not establish a latency distribution.

## Hard-Check Gaps

- Intermediate status JSON and README wording lag raw artifacts. The final gate should verify paths and statuses rather than interpreting stale pending prose as an authorization blocker.
- Performance-mode audits count sampling routes and token bounds, but endpoint allocator readings are not a transient high-water measurement; existing documentation correctly limits that claim.

## Anomaly Ledger

- Observed anomaly: Three stock sampling tests fail.
  Evidence: sampling_tests.log, sampling_failure_probe.json, allowed_token_ids_live.json, raw_prompt_hf_controls.json, PC-02/03 patches.
  Affected path: Shared test assertions.
  Control or comparison: Original allowlists return valid hidden special tokens; token-level variety exists despite identical first characters; 32 mixed penalty rows match solo/homogeneous/host controls, and pinned HF also defeats the character-count assertion.
  Likely subsystem: Test contract assumptions.
  Investigation performed: Read failure report, proposals and controls; inspected test-only plugin dirty paths.
  Resolution: fixed by authorized PC-02/03 test corrections; final full sampling and host regressions pass. Local commit follows clean review.

- Observed anomaly: Historical sampled thermodynamics heading contains Japanese text.
  Evidence: b32_pool24608/vllm_qualitative_outputs.json and qualitative_language_comparison.json.
  Affected path: Selected numerical policy, not uniquely serving.
  Control or comparison: Same-prefix standalone selected-policy logits reproduce the Japanese candidate preference; all 20 relative candidate gaps match serving to about 3e-8. HF prefers English. Latest 12 served chat outputs are coherent, topical and contain no unrequested language switch.
  Likely subsystem: Selected model precision/output distribution.
  Investigation performed: Read actual latest served outputs and standalone/HF controls plus conditional comparison.
  Resolution: controlled for stage09; retain as selected-policy quality limitation, never attribute it to HF.

- Observed anomaly: Untracked server log warns that allocations with active traces may be unsafe.
  Evidence: readiness_vllm/server.log; b32_pool36896_tracked/mixed_sync.json and runtime_audit.json.
  Affected path: Eager prefill while decode traces live.
  Control or comparison: Representative tracked served-cache replay keeps program entries at 2354, three prepared captures and zero runtime captures/retirements; no broad corruptible allocation exemption appears in autoport code.
  Likely subsystem: Generic untracked-allocation warning.
  Investigation performed: Inspected generator-owned warmup, tracked counterpart, trace signatures and allocator snapshots.
  Resolution: controlled for inspected workload; do not disable tracking to accept a new failing path.

## Scope Inspected

- Goal/skill paths: bringup/run_contract.json, bringup/agentic_evals/inputs.json, stage09 original prompt; stage-review, vllm-integration, agentic-serving, qualitative-check, and prefill-features reference; operator decisions in supplied AGENTS.md.
- Artifact paths: doc/vllm_integration/{README.md,work_log.md,stack.json,plugin_changes.md,PC-02*,PC-03*}; doc/full_model/serving_contract.json; doc/agentic_serving.json; readiness/prefill_resume_check.json; readiness_vllm mixed async/tracked sync, runtime audits, B32 capacity audited benchmark/mode summary, growing traffic, prefix cache, prefill burst, qualitative served/standalone/HF outputs and language probe comparison, original sampling log and controls.
- Code paths: tt/generator_vllm.py; tt/generator.py warmup, prefill, trace and sampling boundaries; tt/prefill.py path search; PC-02/03 exact test patches.
- Commands run: read-only cat/sed/rg, Python JSON/text inspection, git status of plugin checkout. Only this report was written.

## Residual Risk

No unresolved correctness or stage-09 evidence finding remains in the inspected scope. Single-request headline percentiles are not a distribution study. The synthetic growing-prefix workload is cache/capacity evidence rather than a scored agentic evaluation. Historical isolated language drift is controlled as a selected-policy behavior, not attributed to HF. Serial dense prefill and its equality/performance gates are explicitly stage10 work under the specific skill stage split.

The stage-check log returns an advisory for the absent batched/serial ratio; its context and 18-row serving-contract checks pass. The separately executed agentic checker has only the documented tied-weight representation and fixed pre-load KV-budget advisories. These do not contradict the stage09 contract or measured allocator evidence.

Scope additionally includes PC-04/05 records and exact local test implementations; final_full raw sampling/server/process/audit artifacts; final B1/B8/B16 summaries/audits; final_stage_check.log and final_agentic_check.log; plugin_pc02_pc05_host.log. Commands remained read-only file/JSON/hash/process-existence inspection and git status/diff. No pushes were performed or inferred. The stage owner should now make local stage-owned checkpoint commits and record SHAs as the workflow requires.
