# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None. The stage-owned changes are still uncommitted at review time; the main agent must create the required local checkpoint commit after this clean pass, exclude unrelated `third_party/tt-metal/`, never push, and append the branch/SHA to `work_log.md`.

## Hard-Check Gaps

- None material to the stage contract.

## Anomaly Ledger

- Observed anomaly: the prior review found mixed selected-run and stale resident-memory values in `doc/context_contract.json`, plus no direct selected BF16 capacity artifact in its provenance list.
  Evidence: `results/all_bfp4_lofi_bf16_kv/full_context_coverage.json` reports 2,198,460,928 bytes/device after weights, 3,137,985,024 after cache, and 3,174,425,088 after execution. The current context contract reproduces all three values. Its 536,870,912-byte transient reserve gives exactly 3,674,855,936 bytes/device resident-plus-reserve, also reproduced in the contract. The direct BF16 full-context artifact is now present in `capacity_evidence.artifacts`.
  Affected path: selected BF16 KV capacity and context-contract provenance.
  Control or comparison: direct arithmetic over the selected full-context allocator snapshots and the contract's reserve field.
  Likely subsystem: context-contract regeneration.
  Investigation performed: independently re-derived all requested values from the raw selected-run JSON, checked cache delta (939,524,096 bytes), free-space values, BF16 physical-tile accounting, explanatory prose, and artifact existence.
  Resolution: fixed.

- Observed anomaly: equal-top-5 dominated points had previously been eligible for the plotted Pareto frontier.
  Evidence: `generate_artifacts.py` now uses pairwise non-domination. Independent calculation over the nine trace-verified runnable rows yields only `all_bfp4_lofi_bf16_kv` on the top-5 frontier; the top-1 frontier contains the selected policy, attention-HiFi2, and BFP8-LoFi. Both PNGs exist at 1800x1080, mark the selected point red, and the generator draws the required dotted gate line.
  Affected path: `top1_perf_pareto.png` and `top5_perf_pareto.png`.
  Control or comparison: direct non-domination calculation from `sweep_results.json`.
  Likely subsystem: plot generation.
  Investigation performed: inspected the plot generator and recomputed the frontiers from the raw rows.
  Resolution: fixed.

- Observed anomaly: the selected BF16 KV policy is marginally faster than the BFP8 KV baseline and therefore needed unambiguous default-path reproduction.
  Evidence: selected traced teacher-forcing performance is 110.971641 t/s/u versus 110.814953 for the baseline. `results/selected/post_selection_token_out.json` identifies `selected_precision_config.json`, reports the complete BFP4/LoFi/BFP8-activation/BFP8-CCL/BF16-KV runtime summary, and separately measures 110.991436 device-only and 110.402775 caller-visible token-out t/s/u.
  Affected path: selection and later serving-comparison baseline.
  Control or comparison: selected candidate evidence, baseline row, and post-selection normal-construction artifact.
  Likely subsystem: precision propagation and benchmark provenance.
  Investigation performed: compared config paths, runtime summaries, trace statistics, workload shape, and all performance fields.
  Resolution: controlled.

- Observed anomaly: the BF16-all-weights canonical candidate cannot construct its LM head.
  Evidence: `AUTODEBUG.md`, `AUTOFIX.md`, and `results/autofix_lm_head/` show the dtype-aware minimum legal K-block adaptation and real-weight controls; BF16 still requires 2,003,712 bytes/core for each tested 32K/16K/8K vocabulary split against 1,572,864 bytes/core available.
  Affected path: canonical BF16/HiFi4 candidate.
  Control or comparison: BFP8 width-1 succeeds after the same focused fix, while BF16 fails at the physical minimum across three split sizes.
  Likely subsystem: LM-head circular-buffer L1 capacity.
  Investigation performed: rechecked the AutoFix hypothesis ledger, focused controls, blocker row, and exclusion from Pareto ranking.
  Resolution: controlled exact runtime blocker.

## Scope Inspected

- Goal/skill paths: original Falcon3 stage-8 datatype-sweep goal; `.agents/skills/datatype-sweep/SKILL.md`; `.agents/skills/stage-review/SKILL.md`; supplied `$autofix` and `$tt-device-usage` requirements.
- Artifact paths: datatype-sweep README/work log; selected config; ten candidate configs; JSON/CSV results; refreshed 100-token AIME24 reference and logs; all candidate evidence; AutoDebug/AutoFix reports and controls; selected BF16 full-context and non-aligned evidence; post-selection token-out evidence; qualitative suite/verdict; context contract; plot generator and PNGs.
- Code paths: current diffs in `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py`, and `tests/full_model_evidence.py`, including required default artifact loading and runtime precision summaries.
- Commands run: read-only `sed`, `rg`, `jq`, `git status`, `git log`, and local Python JSON/CSV/path/arithmetic/Pareto/image-metadata checks. No TT device, reset, server, vLLM, profiler, or hardware command was run. Only this report was modified.

## Residual Risk

- All nine runnable full-model candidates pass the 90% top-1 and 98% top-5 gates on the refreshed 100-token reference; their ranking uses trace-verified teacher-forcing only. The fastest passing row is the selected BFP4+LoFi/BF16-KV policy.
- The matrix includes BFP4+LoFi for every material attention, MLP, and LM-head group, isolated fidelity controls, direct BFP8 LoFi/HiFi2 controls, activation/CCL/KV switches, and the AutoFix-supported BF16 blocker.
- The normal generator requires and consumes the selected artifact by default; measured runtime summaries prove weights, fidelities, activation, residual, CCL, KV, logits, sampling, and LM-head geometry propagation. No vLLM integration was started.
- The selected BF16 KV policy preserves the advertised 32,768-token batch-1 context, complete 28-layer final-page coverage, and non-aligned 33/47 and 2049/2079 prompt paths. Prompt-correct base-model qualitative controls pass.
- The remaining administrative action is the post-review local checkpoint commit and SHA logging required by the stage contract; it does not invalidate the reviewed technical evidence.
