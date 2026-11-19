# Datatype sweep independent review brief

Review round1 of at most3. All acceptance jobs and required checks have finished.
Use the installed stage-review skill independently; do not use TT hardware,
run model/evaluation/benchmark jobs, or edit implementation code. CPU artifact
analysis is allowed. Save the verdict in stage_review_round1.md.

Original goal contract:
/home/ttuser/qwen36-35b-a3b/multigoal_logs/08-08-datatype-sweep.prompt.txt

Latest user scope: [owner_decision.md](owner_decision.md) and [owner_acceptance.json](owner_acceptance.json), plus [stage_cap_instruction.txt](stage_cap_instruction.txt),
verbatim. The broad precision search and further repair experiments were
stopped at the cap; only the active P1 run and remaining acceptance ran once.
No vLLM work or push. The owner reports retained baseline precision, not a
broadly established precision optimum. Determine the stage verdict independently.

Starting completed optimized-full-model checkpoint: e03236d2.
Stage checkpoints: a8fdbbc11b9bf58e597e5d4003dec559e3a50a99 and
c873e26136c8140fe419e523525265ee7d3939dc, plus current stage-owned working tree.
Production source is unchanged since c873; report-only plotting and documents
changed during final acceptance. Review relevant stage diffs and actual files.

Primary artifacts, relative to this directory:

- README.md, work_log.md, selected_precision_config.json.
- sweep_results.json/csv, top1_perf_pareto.png, top5_perf_pareto.png;
  baseline_bfp8_v5.json and v6.json with provenance sidecars.
- p1/baseline_bfp8_v1.json, p1/baseline_bfp8_v1.process.json, score bindings/manifest;
  ../eval_readiness/precision_reasoning.json and its runtime identity binding;
  stage08_final_gate.json. The advisory HF slice was skipped by runner memory
  preflight; the required P1 selected check passes.
- post_selection_performance.json and post_selection_token_out.json.
  Queued no-readback token-out is the requested later comparison metric;
  caller-observed readback timing is separate.
- final_acceptance/nonaligned_1_33_2049.json;
  final_acceptance/resume_default.failed.json;
  final_acceptance/resume_baseline_equivalence.json;
  final_acceptance/resume_diagnostics/ and hf_resume_control/;
  resume_wrapup_assessment.md, AUTOFIX_resume.md.
- final_acceptance/qualitative/verdict.json, tt_greedy_outputs.json,
  paired_review.md, all six HF/TT text pairs, degenerate_check.json.
- final_acceptance/context_32769_alloc65536.json (complete and passed);
  ../context_contract.json, context_candidates.json, p1_memory_fit.json,
  memory_fit/accuracy_bf16_experts20_ep.json.
- ../eval_readiness/limitations.json preserves historical legacy records and
  adds the installed reader's items schema and the explicit stage8 owner acceptance.
- run_final_acceptance.sh, final_acceptance/*.attempt.json, logs/final_*.log.

Runtime paths: ../../tt/precision.py, model.py, generator.py,
multichip_decoder.py, prefill.py, lm_head.py. The default factory resolves the
selected JSON and explicit baseline override remains available. Native grouped
MoE internals have fixed mixed precision; see native_precision_contract.json.
Runtime summaries must demonstrate actual consumption, not only JSON intent.
Relevant tests/reporting: ../../tests/datatype_candidate.py,
datatype_report.py, datatype_p1_runner.py, datatype_runtime_identity.py,
datatype_resume_check.py, run_full_model_performance.py,
run_optimized_full_model_prefill.py. No serving adapter was added.

All device jobs are serialized with normal closure. Required gates must be
assessed individually; selection eligibility alone is not a stage-pass claim.
All numerical repair experiments and untried policies are separately labeled.

Live tt-metal checkout branch `main`, HEAD c873e26136c8140fe419e523525265ee7d3939dc. No pushes or other repository edits. The context wrapper lost its exit status across interruption; passing artifact and normal device closure are on disk and the attempt records recovery transparently. Read the latest owner instruction as part of the governing contract: no further resumed-prefill investigation, retain precision and complete after review.
