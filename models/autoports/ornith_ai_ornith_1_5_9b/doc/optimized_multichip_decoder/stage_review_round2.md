# Stage Review

Verdict: clean-pass

Round 2 of 3. Independent follow-up review of stage 5, optimized-multichip-decoder, for `ornith-ai/Ornith-1.5-9B`. Reviewed the live `main` worktree at HEAD `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`. The production decoder remains SHA256 `b1d5016dd769ab65fce82e27d483f73a33fa2e746e29e21952d521052948d2d8`.

R1 is closed. The replacement measurements meet the separate diagnostics-off timing requirement, and their summaries reproduce the raw results. No new correctness, measurement-contract, or implementation finding arose from the remediation. This was a follow-up review of the changed evidence and host acceptance code, not a second review of the whole optimization search.

Paths below are relative to this evidence directory unless prefixed by `tests/` or `tt/`, which are relative to the model autoport directory.

## Required Work

- None.
- **Closed — R1: Separate diagnostics-off performance evidence with matched before/after settings.**
  **Evidence:** `review_timing_jobs.json`, `review_timing_additional_jobs.json`, `review_timing_map.json`, all 24 mapped result JSONs and command/log sidecars, `device_jobs.jsonl`, `final_timing_audit.json`, and `tests/stage5_timing_evidence.py`.
  Independent checks found exactly one successful completed ledger job for each replacement artifact. Every command explicitly sets allocation tracking and tracebacks to `0`, unsets watcher, device profiler and slow dispatch, invokes plain Python without Tracy or `--profile`, and records serial execution on devices 0–3. The actual job intervals are disjoint from one another and from other recorded hardware jobs. They run from 07:43:58 to 08:04:08 UTC, totaling 1,190.457633 seconds of recorded job elapsed time.
  All 26 production hashes match current files and every new command sidecar. Both measured harnesses, the frozen stage-four decoder, and its wrapper match their current hashes and retained source snapshots. Final rows select the default decoder; before rows select the frozen stage-four decoder and workspace. Normalized paired command differences are implementation selection, the old explicit prefill chunk size versus the new default, and an explicit versus computed 64K capacity that resolves to the same value. Workload shapes, capacities, starts, answer lengths, reference paths, I/O contract, timing loops and warmups match.
  I independently checked 145,540 finite PCC values in the replacement runs: all meet 0.995, with minimum 0.9957183599472046. I recomputed 166 reported timing medians and all 45 table rows across 11 unique before/after pairs, including CSV/JSON agreement and speedup/reduction arithmetic. The preserved 82 correctness jobs remain successful and bound to the same 26 production hashes; they were not replaced by the untracked timing runs.
  The host acceptance functions pass on all actual pairs. Twenty additional in-memory negative controls reject enabled tracking/profiling, missing diagnostic disables, missing watcher/profiler/slow-dispatch unsets, a concurrent regime or wrong devices, and mismatched steps, warmups, repeats, layer, layout, trace setting, lengths, reference or harness hashes. No files were created for these controls.
  **Summary propagation:** `final_performance.{csv,json}`, `final_selection.json`, `final_functional_bar.json`, `final_roofline.json`, `final_context128k_roofline.json`, README, topology decisions and AutoFix follow-up use the replacement results. Historical tracked values remain labeled as historical evidence.

The independently reproduced packed-stack results are:

| Workload | Before, ms | After, ms | Speedup |
|---|---:|---:|---:|
| B32, 4K prefix, 1,024 answer steps | 19.855421339 | 0.828630803 | 23.9617× |
| B32, 64K prefix, 128 answer steps | 24.221872608 | 3.635507589 | 6.6626× |
| B32, 128K prefix, 1,024 answer steps | 28.889897815 | 6.731745554 | 4.2916× |
| B1, short prefix | 0.578599097 | 0.483845011 | 1.1958× |
| B32, short prefix | 18.990079849 | 0.587435695 | 32.3271× |

Recomputing the recorded bandwidth-floor formulas gives FUNCTIONAL fractions 0.4041900314 and 0.4151627821, above the unchanged 0.10 gate. The primary roofline fraction is 0.5609639398; the supplementary 128K fraction is 0.7103679070. The primary result is not presented as reaching 70%.

## Other Concerns

- **Previous SDPA concern: open as an explicitly retained residual risk, with no new required correction.** The independent new 128K timing run retains the same minimum PCC, 0.9957183599472046, only approximately 0.00071836 above the gate. README and `context128k_autofix/AUTOFIX.md` retain the empirical nature of the capacity-aware policy, the BF16 running-statistic limitation, and the K128 failure/K256 and K512 controls. The remediation does not alter this policy or establish broader full-model numerical coverage.
- **Previous stop-rule concern: closed.** Direct interval merging, excluding the explicitly identified CPU calibration commands, reproduces the frozen 8.001707033316295-hour cutoff and the updated 9.715293385452695-hour total. The latter includes mandatory repair and validation. The meaning of these intervals remains explicit; they are not kernel execution hours. Untried optimization candidates remain handoff notes and are not reopened by this review.
- **Previous small-prefill disclosure concern: closed; measured tradeoffs remain visible.** The clean table now has three negative prefill rows: linear S513, 1.367500052 to 1.397887245 ms (+2.2221%); full-attention S512, 1.364132855 to 1.419158187 ms (+4.0337%); and full-attention S513, 1.435679849 to 1.439174637 ms (+0.2434%). README retains all three. These are measurements from five warmed samples per side, not evidence of universal prefill improvement or an assertion that every difference is noise. B1 packed decode, private S1024 decode, and both FUNCTIONAL bars pass their comparisons/gates. No production change or new quality failure accompanies these timing replacements.
- **Previous checkpoint concern: open only as the authorized post-review workflow.** Local checkpoint commits and recording their SHAs remain to be performed after this verdict. The supplied plan says never push; no push state was inferred from Git history. This review does not authorize full-model or vLLM work.
- **Checkpoint-hook qualification:** `checkpoint_hooks.log` explicitly skips the mutating formatting hooks and the 500 KiB size hook; it is not evidence that every repository hook passes. Current staging includes 86 paths above that size, including compressed profile evidence and the JSONL ledger. The work log discloses the local-checkpoint exceptions, while separate production and changed-host-file style checks are retained. Preserve this distinction when recording the checkpoint. This is a packaging/workflow qualification, not a reason to repeat numerical evidence or another stage review.

## Hard-Check Gaps

- **Closed — timing-regime acceptance.** The summarizer now calls the host timing audit and validates every before/after pair before emitting the final timing audit/table. The current validator's hash matches `timing_acceptance_host_checks.json` and `review_remediation_style.json`. The independent positive and negative controls above confirm its relevant rejection behavior.
- **Closed — retained short-prefill evidence index.** `final_manifest.json` points to `retained_short_prefill_evidence.json`. All six mapped artifacts exist and match the recorded content/source hashes. The mapping distinguishes retained correctness coverage from final timing evidence and links the source-equivalence decision already accepted in round 1; no new boundary sweep is needed.
- **Closed — top-level context index.** `doc/context_contract.json` now points to the two `final_v2_max_context_*` artifacts at top level. Both bind the unchanged production hash, and the supported context remains 262144.
- **Archive check scope:** `archive_integrity.json` matches the current manifest hash, count of 2,904 gzip archives and 4,549,881,981 expanded bytes. I additionally verified the raw and decompressed hashes of the 28 affected timing/hook archive entries. I did not independently decompress the entire historical archive collection in this follow-up.

## Anomaly Ledger

- **Observed anomaly:** Tracking-enabled or mismatched diagnostic timing had been used in final performance summaries.
  **Evidence:** Round-1 R1; replacement `review_timing_*` results, sidecars, ledger and final summary artifacts.
  **Affected path:** Warmed prefill, traced warmed decode, FUNCTIONAL and roofline reporting.
  **Control or comparison:** Eleven matched frozen-stage-four/final-default pairs with explicit diagnostics-off settings; original tracked correctness runs retained separately.
  **Likely subsystem:** Measurement configuration and summary acceptance.
  **Investigation performed:** Independent command/interval/source/PCC/median/table checks and twenty host negative controls.
  **Resolution:** fixed — R1 closed.

- **Observed anomaly:** Three clean short-prefill medians regress by 0.24–4.03%.
  **Evidence:** `final_performance.json` and `review_timing_{stage4,final_v2}_short_prefill_perf_{0,3}.json`.
  **Affected path:** Linear S513 and full-attention S512/S513 warmed prefill.
  **Control or comparison:** Same reference, length list, warmup, repeat count, harness and diagnostic settings against frozen stage four; raw samples retained.
  **Likely subsystem:** Per-shape performance tradeoff and host timing variability; this review does not assign an unproven single cause.
  **Investigation performed:** Recomputed medians and percentage changes and inspected individual samples; checked README disclosure and the separate B1 decode/FUNCTIONAL results.
  **Resolution:** controlled — reported measured tradeoffs, without an across-the-board improvement claim; no additional optimization search required after the accepted stop rule.

- **Observed anomaly:** Replacement logs contain generic untracked trace-allocation warnings and frozen-baseline computed-memory-config warnings.
  **Evidence:** `review_timing_*` logs and their original corresponding logs; retained tracked `final_v2_*` controls.
  **Affected path:** Trace harness allocations and frozen stage-four matmul configuration.
  **Control or comparison:** The production/harness bytes are unchanged; retained tracked correctness controls remain current. Computed-memory-config warnings occur in the frozen-before runs, and their counts match the corresponding original artifacts.
  **Likely subsystem:** Known allocator advisory with tracking off and the frozen baseline's output-memory-config selection.
  **Investigation performed:** Scanned all new logs for errors/fatal signatures, compared warning provenance, and checked successful source-bound result/PCC evidence. No new exception or failed job was found.
  **Resolution:** controlled — no new trace-lifetime or runtime-fallback defect is established by these unchanged warnings.

## Scope Inspected

- **Goal/skill paths:** `goal_contract.txt`; installed `stage-review/SKILL.md`, `optimize/SKILL.md`, `tt-device-usage/SKILL.md`, `tt-enable-tracing/SKILL.md` and model-bringup startup instructions under `/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/`.
- **Artifact paths:** Round-1 review; 24 new timing results and all associated sidecars/logs; timing job lists/map/audit; final manifest/audit/performance/selection/FUNCTIONAL/roofline artifacts; README, relevant work-log/topology/AutoFix updates; retained-short-prefill mapping; context contract; stop and updated time accounting; host check artifacts; checkpoint hook log; archive manifest/integrity and affected archive entries.
- **Code paths:** Changed `tests/stage5_timing_evidence.py`, `tests/summarize_stage5_final.py` and `tests/prepare_stage5_manifest.py`; relevant unchanged runner, packed-stack/probe timing loops, frozen-decoder wrapper, archive and time-accounting code. Production files and harness snapshots were hash-checked, not re-reviewed in full.
- **Commands run:** Read-only `git rev-parse`, targeted `git diff --check --cached`, `git diff --cached --name-only`, `rg`, `cat`, `sed`, and Python standard-library analyses over existing artifacts. Host acceptance functions were loaded directly with bytecode writes disabled; negative-control paths were in-memory objects. The first independent time-accounting attempt included early CPU calibration rows whose sidecars still name devices; excluding those commands as explicitly documented reproduces both ledger totals exactly.
- **Restrictions observed:** No TTNN import, device access, reset, server, vLLM execution, long test, implementation edit or child reviewer. The only review-created file is this report.

## Residual Risk

- Finite layer/stack tests and source inspection do not prove accuracy for all activation distributions. The unchanged capacity-aware SDPA margin and the prior maximum-context/all-layer limitations remain explicit.
- Timing samples establish these workloads and this serial regime, not a broad confidence interval over host scheduling or all future shapes. Small prefill tradeoffs remain visible.
- This decoder-stage pass does not establish full-model generation quality, serving trace reuse, token feedback, prefix caching or all-layer runtime memory fit. Those remain later-stage obligations.
- Local checkpoint creation and SHA logging follow this passing review. No further device rerun or review round is required for R1.
