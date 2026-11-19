# Stage 8 independent review brief

Stage: datatype-sweep; target Qwen/Qwen3.8-Flash-Next.
Checkout: /home/ttuser/qwen38-flash/tt-metal, branch main; live worktree based on ccb44bb0cb36efb5eb26718a5520dbdf1ec41dc5. No push is authorized. No vLLM integration belongs to this stage.

## Original goal

Use $datatype-sweep and $tt-device-usage for the Qwen/Qwen3.8-Flash-Next optimized full model in the current tt-metal checkout.

Subagents are authorized when a selected skill requests or recommends them.

Start from the completed optimized full model and choose the fastest weight, activation, CCL, and KV-cache datatype configuration that still satisfies the required full-model accuracy. Do not begin vLLM integration work in this goal.

Goal completion requirements:
- The baseline optimized full model accuracy and performance are refreshed using the main AIME24 chat-template readiness reference with 100 generated tokens.
- Full-model candidate results are captured in models/autoports/<model>/doc/datatype_sweep/sweep_results.json and sweep_results.csv, including config id, dtype policy, compute-fidelity policy, top-1/top-5/top-100, TTFT, trace-verified teacher-forcing decode t/s/u, measurement regime, command, hardware, mesh, and pass/fail status.
- The selected precision config is written to models/autoports/<model>/doc/datatype_sweep/selected_precision_config.json and is either the model default or a required config artifact that later full-model/vLLM construction paths actually consume by default. It must include weight groups, layer exceptions, compute fidelities, activation/residual dtype, CCL dtype, KV-cache dtype, and logits/sampling dtype assumptions. Evidence must prove the measured runtime path consumes these fields; a JSON field ignored by hard-coded model code does not count.
- Recompute doc/context_contract.json for every KV-cache dtype candidate that could change memory capacity; run max context only for the selected config if its KV dtype/layout changed, else a 32K–64K smoke. Do not complete after reducing advertised capability unless a hard physical device limit, e.g. DRAM, requires it and evidence records the largest feasible value.
- Datatype changes preserve valid non-aligned prompt-length support; if a KV-cache or trace-buffer dtype/layout changes chunking, rerun the non-aligned prompt check.
- Two pyplot charts exist: top1_perf_pareto.png and top5_perf_pareto.png. Each chart plots evaluated full-model configs, draws the Pareto frontier, marks the selected point in red, and shows a vertical dotted line at the minimum allowed accuracy.
- doc/datatype_sweep/README.md and work_log.md record thresholds, selected config, rejected configs, Pareto interpretation, commands, plots, performance, accuracy, limitations, and exact artifacts.
- Precision starts from BFP8; a weight group moves to BFP4 only if it passes P1 (readiness_check.run_precision_reasoning, see $datatype-sweep). doc/eval_readiness/precision_reasoning.json for the selected config passes the runner gate, or the loss is a recorded limitation (BFP4 required to fit).
- The final selected config is the fastest evaluated config that satisfies the full-model top-1/top-5 gate and P1 using traced teacher-forcing decode performance. Eager/untraced decode numbers are invalid for ranking and selection. The sweep includes likely lower-precision wins and canonical policies, with evidence for rejected configs. For every material BFP4 matmul group considered or selected, the sweep includes a BFP4+LoFi candidate for that same group, or an exact TTNN/runtime blocker plus `$autofix` evidence.
- After selecting the winning config, run the same warmed token-out no-readback benchmark used by optimized full model through the normal selected-config construction path. Record it separately from teacher-forcing as post-selection token-out TTFT/decode t/s/u. Later reports and vLLM comparisons must use this post-selection token-out number when available.
- $stage-review returns `clean-pass` (or `pass-with-carried` after 3 rounds); findings fixed/rereviewed. Local commit stage-owned touched-repo changes; never push; log SHAs.

Unmet requirements, review findings, failed gates: work. Stop only after `$autofix` fails.

## Latest controlling user instruction

> Stage time/token cap reached. Keep the best version that passes the required gates, run only the remaining required gates once, write untried ideas as hand-off notes in the stage work log, and complete the stage (pass-with-carried is acceptable for non-blocking items).

Apply this instruction over the skill's usual three-round prerequisite for nonblocking carry. Missing/failed correctness or required evidence is still work. Do not request additional optional search, geometry tuning or timing repetitions; inspect the recorded handoff instead.

## Read first

- Installed skills: /home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/{stage-review,datatype-sweep,tt-device-usage,qualitative-check}/SKILL.md
- Model root: models/autoports/qwen_qwen3_8_flash_next
- doc/datatype_sweep/README.md, work_log.md, evidence_manifest.json, sweep_results.json/csv, selected_precision_config.json, top1_perf_pareto.png, top5_perf_pareto.png.
- doc/datatype_sweep/final_validation_v1.json and its qualitative evidence; post_selection_token_out_v1.json and provenance; p1_final_v1.json and runtime; selected_precision_propagation_audit.md/json; logs/stage_gate_v2.log.
- doc/eval_readiness/precision_reasoning.json, limitations.json; doc/context_contract.json.
- Runtime changes in tt/precision.py, tt/model.py, tt/multichip_decoder.py, tt/generator.py, tt/cache_pool.py, tt/cache_budget.py and relevant test/report harnesses.
- AUTOFIX.md, initial_precision_capacity_audit.md, kv_drift_audit.md, runtime_warning_audit.md and weight_memory_residual_audit.md for rejected policies/anomaly controls.

Review existing evidence and generated text directly; do not open devices, reset hardware, run experiments or mutate implementation. A review report file is the sole authorized write. Required Work is for correctness, contract or missing mandatory evidence; optional unrun performance ideas belong in Other Concerns under the controlling cap. Use the skill verdict format and name exact evidence.


## Additional owner resume instruction

> Owner decision: resume stage 8 once more. Finish the remaining 32K-context hardware smoke test for the selected precision configuration, record the result, complete the stage review and close stage 8. No further precision search.

Inspect `final_validation_v1.completion_recovery.json` for the interrupted wrapper metadata and preserved actual completed child evidence.
