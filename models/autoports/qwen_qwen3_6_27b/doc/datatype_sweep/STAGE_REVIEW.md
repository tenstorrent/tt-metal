# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- `selected_bfp8_activation_ccl` is derived from the safe BF16/full-attention baseline, not from `full_attention_bfp4_lofi`; its README label `selected + BFP8 residual/CCL` is therefore imprecise. The row does construct and exercise BFP8 activation/residual and both BFP8 CCL fields, passes the accuracy gates, and is materially slower at 5.97 t/s/u, so this naming/provenance issue does not undermine selection.
- The top-5 chart is mathematically correct, but all twelve candidates have 100% top-5 and several labels overlap. This is cosmetic; every point is present and the sole non-dominated point is the selected 7.00 t/s/u row.

## Hard-Check Gaps

- There is no independent profiler/perf-report artifact enumerating every dominant op's runtime dtype and fidelity. The exact constructed 64-layer `PRECISION_CONFIG` summaries, strict policy loader, source propagation, and completed full-model candidate logs are sufficient evidence for this stage, including the corrected controls.
- The stage checkpoint commit is intentionally pending this clean review. The stage owner must still isolate and checkpoint stage-owned changes as required by the stage-review workflow.

## Anomaly Ledger

- Observed anomaly: The first MLP and linear HiFi2 controls were confounded by restoring BF16/HiFi full attention.
  Evidence: The discarded logs remain as `logs/selected_bfp4_{mlp,linear}_hifi2_teacher_forcing_confounded.log`; they are excluded from the 12-row result matrix. Direct comparison of each current candidate against `full_attention_bfp4_lofi.json` shows only `config_id` plus the named fidelity delta: MLP changes only `compute_fidelities.mlp`; linear changes only `linear_attention_internal`, `linear_input_projection`, and `linear_output_projection`.
  Affected path: Same-dtype fidelity controls and fastest-passing-policy selection.
  Control or comparison: Selected all-BFP4/LoFi candidate and its 7.00 t/s/u traced row.
  Likely subsystem: Candidate-generation provenance.
  Investigation performed: Field-by-field normalized JSON comparison; sweep-row-to-candidate comparison; inspection of constructed runtime summaries and final aggregates in both corrected logs.
  Resolution: fixed. Corrected MLP HiFi2 is 93/100 top-1, 100/100 top-5/top-100, 6.72 t/s/u; corrected linear HiFi2 is 93/100, 100/100, 100/100, 6.90 t/s/u. Both preserve BFP4/LoFi full attention and are slower than selected LoFi.

- Observed anomaly: BFP8 MLP-down initially failed with a static-CB/L1 overlap.
  Evidence: Preserved `*_pre_autofix.log` failures and `AUTODEBUG.md` isolate width 17 at the TP4 MLP-down geometry; both MLP-down call sites now consume `mlp_down_in0_block_w`; BFP8 width-1 HiFi2 and LoFi full-model reruns complete at 6.26 and 6.64 t/s/u.
  Affected path: All-projection BFP8 decode.
  Control or comparison: Same BFP8 dtype with width 17 failure versus width 1 completion across both fidelities.
  Likely subsystem: Precision-dependent matmul program geometry and previously ignored policy plumbing.
  Investigation performed: Inspected AutoDebug/AutoFix evidence, source, candidate overrides, pre-fix logs, constructed policies, and final aggregates.
  Resolution: fixed.

- Observed anomaly: The first BFP8 activation/residual+CCL run failed at packed-QKV head splitting.
  Evidence: The pre-AutoFix log records the exact `nlp_create_qkv_heads_decode` BF16/FP32 input assertion. The implementation now adapts only packed QKV to BF16 immediately before that op; the rerun summary constructs BFP8 activation and both BFP8 CCL fields on all 64 layers and completes at 97/100 top-1, 100/100 top-5/top-100, and 5.97 t/s/u.
  Affected path: Reduced-precision activation/residual and CCL candidate.
  Control or comparison: Exact pre-fix assertion versus successful original-command rerun.
  Likely subsystem: Fixed input dtype contract of the head-splitting op.
  Investigation performed: Inspected pre/post logs, local adapter source, runtime policy summary, and aggregate.
  Resolution: fixed.

- Observed anomaly: The original Pareto implementation admitted dominated points.
  Evidence: `build_results.py` now uses pairwise maximize-accuracy/maximize-throughput dominance with strict improvement. Independent recomputation over all 12 JSON rows gives top-1 frontier `{full_attention_bfp4_lofi, baseline_bfp8_ccl, full_attention_bfp8_hifi2}` and top-5 frontier `{full_attention_bfp4_lofi}`; both regenerated PNGs match.
  Affected path: Required Pareto charts and policy interpretation.
  Control or comparison: Independent frontier computation from `sweep_results.json` and direct PNG inspection.
  Likely subsystem: Prior order/equality handling in plot construction.
  Investigation performed: Read the plot builder, recomputed dominance, checked artifact timestamps, and visually inspected both charts.
  Resolution: fixed.

- Observed anomaly: Platform-discovery, firmware-version, nanobind teardown-leak, and capacity shutdown warnings appear in preserved logs.
  Evidence: Candidate logs reach final aggregates and close devices; `host_tests.log` records 10 passed before teardown warnings; capacity artifacts contain structured terminal results.
  Affected path: Environment metadata and process shutdown, after stage-critical results.
  Control or comparison: Successful aggregate/test/capacity outputs and completed device close sequences.
  Likely subsystem: Platform metadata and Python binding teardown rather than model execution.
  Investigation performed: Searched logs for errors/warnings and checked their ordering relative to final results.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: original datatype-sweep contract in `.agents/skills/datatype-sweep/SKILL.md`; `.agents/skills/{stage-review,qualitative-check,tt-device-usage}/SKILL.md`; prior `doc/datatype_sweep/STAGE_REVIEW.md`.
- Artifact paths: `doc/datatype_sweep/{README.md,work_log.md,AUTODEBUG.md,AUTOFIX.md,sweep_results.json,sweep_results.csv,selected_precision_config.json,build_results.py,make_candidate_configs.py}`; all 12 candidate JSONs; corrected and confounded HiFi2 logs; pre/post-AutoFix logs; host tests; token-out, mixed-slot, qualitative, and capacity artifacts; both Pareto PNGs; `doc/context_contract.json`.
- Code paths: `tt/{precision_config.py,model.py,multichip_decoder.py,optimized_decoder.py,generator.py}` and `tests/test_full_model_public_contract.py`; stage-scoped git status/diff.
- Commands run: read-only `sed`, `find`, `git status`, `git diff`, `jq`, `rg`, `wc`, `stat`, `sha256sum`, normalized JSON diffs, a small read-only Python consistency/frontier check, and direct PNG inspection. No hardware, server, vLLM, or implementation command was run.

## Residual Risk

- The selected artifact exactly matches `full_attention_bfp4_lofi.json` and is the fastest of 12 passing, trace-verified full-model rows at 93% top-1, 100% top-5/top-100, and 7.00 teacher-forcing t/s/u. The three material selected BFP4 families now have coherent same-dtype HiFi2 controls, all slower than LoFi.
- The normal default construction path consumes the selected artifact and propagates weight, activation/residual, CCL, KV/state, LM-head, sampler, and compute-fidelity fields. Post-selection evidence covers warmed token-out (17.8968 t/s/u), non-aligned S65/S63 mixed slots with cache reset/reuse, B1 advertised-context capacity, bracketed B32 capacity, and a prompt-correct six-case HF-controlled qualitative suite.
- Remaining risk is limited to the absence of an op-level profiler cross-check and minor documentation/plot presentation issues noted above; neither contradicts the constructed runtime evidence or the selected-policy result.
