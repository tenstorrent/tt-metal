# Stage Review

Verdict: clean-pass

Independent review of Stage 7, optimized-full-model, for Qwen/Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Reviewed the live `mvasiljevic/qwen38-full-bringup` worktree based on `b70863460380bdf57cc0a9c0d5769fe654ac161d`, including the final host-only metadata correction. The reviewer opened no devices, ran no hardware tests or servers, and changed only this report.

## Required Work

- None. The inspected implementation and evidence satisfy the stage's technical gates. The stage owner must still perform the prescribed post-review local checkpoint and SHA logging before declaring the whole goal complete; that ordering is part of the stage-review workflow.

## Other Concerns

- The performance result is specific to warmed S128/G128/B1 and the recorded persistent allocation. History retains a request-sized high-water capacity, copies that allocation on each append, and transfers the whole allocation at delivery. The implementation and README state this correctly. Larger retained allocations and cold captures can cost more; the reported result does not claim otherwise.
- Maximum context and batch 32 are separate coverage points. The all-layer context checks execute S262143/G2 and S262144/G1; they are capacity and position checks, not a new 262144-token HF accuracy oracle. This limitation is disclosed in the context contract.
- The inherited Stage 5 decoder policy, residual layout and rejection ledger remain authoritative. No decoder source change or newly selected rejected precision policy was found. The selected head geometry is an additional terminal optimization, with real-activation candidate controls.

## Hard-Check Gaps

- No missing hard gate was found. The reduced profile is the required real representative-layer method, not an all-64-layer device measurement. Its expanded kernel total is an accounting estimate. The README and summary correctly distinguish that estimate, profiled gaps, complete-delivery host timing and the nominal read-bandwidth bound.
- The benchmark records warmed comparisons and final reproduction, rather than confidence intervals over many independent launches. The evidence supports the reported observed improvements; it does not establish a statistical performance distribution.

## Evidence Re-derived

- **Accuracy and output:** `readiness_final.json` and its successful log/exit record give prefill top1 99/100 and decode top1 98/100; both top5 and top100 are 100/100. The runner invokes the standard prefill and traced teacher-forcing entrypoints with the pinned AIME24 chat reference. Actual autoregressive HF/TT output was inspected, as were all shared-suite outputs and controls. Independent host checks verified matching rendered prompts/token IDs, six EOS-complete final outputs of 418/148/1700/380/145/337 tokens, the story's exact 1024-token prefix, clean packaged degeneracy results and the generated Fibonacci function for inputs -1/0/1/2/8.
- **Complete delivery:** `after_full_final.json` records 59.2953516 ms TTFT and 40.3850053 tokens/s/user with deferred complete delivery. The 127-step steady counter set is exactly 127 model replays, 127 sampling replays and 127 history appends. One history read follows the loop. Inspection of `generate`, `_append_history`, `_read_history` and the executable guards confirms that the final transfer and output-list construction belong to decode timing. All 128 tokens equal the immediate and eager-prefill controls. Immediate delivery is separately 40.3284339 tokens/s/user; queued final-token checking and teacher forcing retain their distinct boundaries.
- **Before/after:** `before_full.json` measures 86.3122316 ms TTFT and 39.4331123 tokens/s/user with immediate delivery. Final immediate delivery is directly comparable; deferred delivery is explicitly marked unavailable before the change. The 8192-byte fabric candidate's 40.3809707 tokens/s/user is reproduced by the final default's 40.3850053. Decoder dtype/layout and host-pool settings remain fixed.
- **Raw profiler:** Independently parsed all four `device*_ops.csv.gz` files using the `PERF_TOKEN_OUT` signposts. Each contains 144 device operations in that interval, all bearing model/sampling trace and replay IDs. Hidden-norm boundaries independently locate the representative linear/full layers and terminal work. Decoder matmuls use BFLOAT4_B/LoFi; the four terminal matmuls use BFLOAT8_B/HiFi2. LargeIndices TopK uses 110 cores, with 81/2-core route preparation/finish, and the only sampling gathers expand 32 local candidates to 128. No selected generic TopK, force-argmax or full-vocabulary gather was found.
- **Accounting:** Recomputed expanded kernel totals are 23.817045, 23.883777, 23.865967 and 23.756443 ms across devices, using 48 linear and 16 full-attention layers plus entry/terminal/sampling/history once. Complete delivery is 24.7616657 ms/token. This gives approximately 3.7–4.2% excess relative to the expanded kernel estimate, below the goal's investigation threshold. The much larger gap-inclusive expansion is explicitly not interpreted as host overhead. The final profile and benchmark both use cache256/page-table `[1,8]`/history127.
- **State and trace safety:** `contract_final_full_b32.json` and the implementing tests cover mixed prompts31/33 in fixed slots31/0, inactive recurrent/convolution state, exact physical-page remapping, all32 history lanes, nonzero-cursor recapture, overflow rejection and continuation PCC0.999214470. `prefill_contract_full.json` covers logical lengths1/31/32/33/4095/4096/4097, changed prompts/modes, public output lifetimes and changed-table physical KV writes. Fifty captures run with program-cache misses forbidden. The separately instrumented watcher/tracker run passes on real layers0/3 across all4 devices, with Ethernet checks enabled.
- **Provenance:** Current model/generator/decoder hashes match the final benchmark, readiness, full-batch, full-prefill, watcher and profile manifests. Final-validation-index hashes and saved profiler-source hashes were independently checked. The accepted source ASTs parse; recorded final pre-commit, context, stage-runner and qualitative checks exit0. No stage C++ change requires a new build.

## Anomaly Ledger

- Observed anomaly: Short qualitative budgets cut off planning, stories or code; the story extension also inherited an incorrect `executed_steps=1024` field.
  Evidence: `tt_qualitative*.json`, their exact HF controls, and `qualitative_metadata_correction.json`.
  Affected path: Qualitative evaluation and artifact metadata.
  Control or comparison: HF also truncates at the short budgets; the TT story reaches coherent EOS at1700 tokens and exactly retains the earlier1024-token prefix.
  Likely subsystem: Generation budget and copied control metadata.
  Investigation performed: Read actual outputs; compared prompt IDs and controls; checked each recorded `perf.decode_tokens+1` against the corrected TT execution budget; independently reran host degeneracy and generated-code checks. Text, tokens and timings were unchanged by the metadata repair.
  Resolution: controlled / fixed. The longer TT story is completion evidence, not an equal-budget HF comparison.

- Observed anomaly: Generic live-trace allocation warnings in uninstrumented runs.
  Evidence: Final full-model logs, `check_prefill_tracing.py`, `prefill_contract_full.json`, `watcher_prefill_final.json`.
  Affected path: Owned prefill, decode/sample trace scratch and public output lifetime.
  Control or comparison: Separate allocation-tracked watcher run and all-layer retained-output tests.
  Likely subsystem: Trace allocator lifetime rules.
  Investigation performed: Inspected persistent prefill inputs/output allocated before capture, model/sample-before-prefill capture order, trace release on cache/signature changes, all-device address preservation and public outputs surviving subsequent traced generation.
  Resolution: controlled. No observed corruption or stale-output failure remains in the tested path.

- Observed anomaly: Reader1 numerical failure, reader3 padding failures, larger-block L1 failure and a LoFi head PCC miss.
  Evidence: `AUTODEBUG_head_geometry.md`, compatible reader1/3 JSONs, `head_16384_block10_reader2.log`, `head_16384_block5_lofi_report.json`.
  Affected path: Rejected LM-head candidates.
  Control or comparison: Corrected reader-dependent padding restores PCC; adapted reader1/3 are slower. Selected reader2/block5 passes. LoFi rank3 PCC0.998926342 misses the existing0.999 real-activation gate.
  Likely subsystem: Bank stride/padding, L1 capacity and multiply fidelity, respectively.
  Investigation performed: Compared candidate geometry, actual local PCC/greedy outputs and warmed component timings. Rejections include adapted controls rather than only first API errors.
  Resolution: controlled. Selected head retains BFP8/HiFi2 and valid reader2 geometry.

- Observed anomaly: Startup ownership/sysmem failures and one later initialization transfer stall.
  Evidence: `AUTOFIX_startup.md`, `AUTOTRIAGE_head_startup.md`, recovery/list/mesh logs and exit records.
  Affected path: Device initialization before the candidate executes.
  Control or comparison: Subsequent exact TP4 mesh open/close and identical candidate retry pass; final model gates complete.
  Likely subsystem: Infrastructure ownership/initialization; the isolated transfer-stall root cause remains unproven.
  Investigation performed: Reviewed bounded recovery evidence, captured triage, successful device discovery and mesh teardown. No failed initialization run contributes a candidate timing.
  Resolution: controlled historical incidents, with recurrence risk retained.

- Observed anomaly: Watcher Ethernet instrumentation initially exceeds its code buffer; generic motherboard and L1-semaphore advice also appears.
  Evidence: `watcher_contract_b3.log`, final watcher environment/log, successful mesh logs and `anomaly_ledger.md`.
  Affected path: Instrumented setup and allocation advice.
  Control or comparison: NOINLINE1/fabricO3 watcher configuration passes without disabling Ethernet checks; physical four-device Ring discovery succeeds and selected full-model allocations fit.
  Likely subsystem: Instrumentation size, board identification and semaphore placement.
  Investigation performed: Checked final environment and successful accepted runs; no runtime fallback was inferred from these warnings.
  Resolution: fixed / controlled.

- Observed anomaly: Profiler worker-count denominator is misleading, expanded gaps exceed full wall time, and an interim profile used cache160/history1.
  Evidence: Raw operation attributes/advice tables, `perf_summary.json`, final matched-buffer profile.
  Affected path: Performance interpretation.
  Control or comparison: Native reader counts imply16/24 compute workers, and final raw geometry is cache256/history127. Kernel-only and gap-inclusive accounting are reported separately.
  Likely subsystem: Reporting assumptions and representative-profile geometry.
  Investigation performed: Independently re-derived raw operation partitions, dtype/fidelity, sampling route and kernel totals; verified current summary input hashes.
  Resolution: controlled / fixed. Invalid utilization percentages and mixed-run differences are not claimed as measured full-model utilization or host overhead.

- Observed anomaly: Final Tracy export warns about a missing optional wasm/UI trace-copy source and pandas mixed-type inference.
  Evidence: `profile_final_buffers.log` near its completed CSV export; `tracy/profile_final_buffers/device*_ops.csv.gz` and rendered phase tables.
  Affected path: Optional trace viewer copy and CSV parsing.
  Control or comparison: Raw signposts, complete all-device rows, replay identities, source hashes and table-aligned numeric accounting remain available and parse successfully.
  Likely subsystem: Export convenience path and dtype inference.
  Investigation performed: Independently parsed the compressed raw CSVs and reproduced accepted numeric accounting.
  Resolution: controlled. No missing required performance artifact or corrupted measurement was found.

## Scope Inspected

- Goal/skill paths: `/home/mvasiljevic/qwen38-full-rerun/live-logs-stage7/07-07-optimized-full-model.prompt.txt`; `.agents/skills/{stage-review,multichip,optimize,tt-device-usage,full-model,tt-enable-tracing,qualitative-check}/SKILL.md`.
- Artifact paths: This directory's README, work log, checklist, fallback/qualitative/anomaly reports, candidate JSONs, accepted benchmark/readiness/contract/watcher logs and manifests, all-device final raw compressed CSVs and phase tables, `final_validation_index.json`, qualitative correction/metrics, actual TT/HF output and controls; `../context_contract.json`; referenced inherited Stage5 contract/rejection evidence.
- Code paths: `../../tt/{generator,model,multichip_decoder,optimized_decoder}.py`; `../../tests/{benchmark_full_model,check_deferred_generation,check_prefill_tracing,full_model_contract,run_full_model,run_readiness,tt_qualitative,check_qualitative_artifacts,summarize_full_model_perf}.py`; stage launch/profile scripts.
- Commands run: Read-only `git status`, `git diff`, `rg`, `cat`, `sed`/`tail`; small Python JSON/hash/AST/CSV analyses and in-memory qualitative/functional-code checks. One initial reviewer CSV query confused display row IDs with global call counts and failed; it was corrected to select native signposted windows and then passed on all four devices. No implementation or hardware test was run by the reviewer.

## Residual Risk

- Representative-layer profiling and focused finite tests cannot prove every joint batch/context/history allocation or every future external scheduler lifecycle. The reviewed evidence proves the stated tested contracts and preserves advertised capability.
- Cold trace setup, large history high-water allocations and infrastructure recurrence remain practical costs/risks. They are disclosed and do not contradict the measured warmed path.
- The report applies to the reviewed runtime hashes. Subsequent runtime edits require validation appropriate to those edits. Post-review metadata/checkpoint bookkeeping may proceed without rerunning device tests when runtime behavior is unchanged.
