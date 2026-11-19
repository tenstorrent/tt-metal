# Stage 7 validation and evidence plan

CPU-only plan dated2026-11-16. It records requirements and reusable evidence, not completed stage7 gates. The parent stage owns serialized device jobs and final outcomes. Scope is optimized full model/generator on the existing1x4 Blackhole mesh. No server/vLLM implementation, broad dtype frontier or push.

## Required new evidence and what can be reused

| Requirement | Exact prior evidence and scope | Stage7 action |
|---|---|---|
| Before/after128-in/128-out B1 TTFT, token-out, true teacher timing | `../full_model/performance/lifetime_fixed.json`; all40, tracking1,127 decode steps, observed and queued metrics separate | New same-harness baseline and final with diagnostics disabled; historical tracked result is not the untracked baseline. Label teacher-forcing sampled readiness vs true logits-only99 HF inputs vs fixed-token control |
| AIME24 prefill and teacher accuracy | `../full_model/{prefill,teacher}_accuracy_lifetime_fixed.json`, each top1.98/top5=top100=1.0/100 positions | Refresh both through `run_full_model_readiness.py`, keep top5>=.98 and top100=1.0. Preserve reference cache/provenance, regenerate only if model/revision/template metadata mismatches |
| Autoregressive/shared qualitative suite | `../full_model/qualitative/tt_greedy/`, six paired128-token completions; HF controls and rendered prompts | Rerun selected TT path on six shared prompts plus autoregressive mode; reuse matched pinned HF control when settings identical; run degeneracy checker. Content failure adjudication needs10 blind matched samples per side, not a small anecdotal sample |
| Mixed/nonaligned/seed/page/trace contract | `../full_model/generator_replay_probe/lifetime_fixed_watcher.json`, reduced layers0/3, capacity3,155 model+sampling replays, entries8489 unchanged | Fresh reduced affected-path test with tracking1/skip-program-cache0, final replay after any eager prefill; unchanged/changing pages, changed tokens/positions, inactive middle row, reordered mixed lengths, sampled→greedy reuse and seed isolation |
| Trace allocation tracker active | `../full_model/reduced_positive.json`; earlier positive control | Positive control in final environment; reject deliberately live unsafe allocation. Retain exact acknowledgments and aliases; no broad capture/forward exemption |
| Larger batch | `../full_model/full_batch32_trace_budget.json`, all40,512MiB reservation, entries8551 unchanged; short state smoke, chip0 inactive assertions. `../full_model/batched_teacher/result.json` all40 B3 top5/top1001.0 | Preserve B32 state/capacity; refresh changed terminal/feedback path at B>1 and fixed inactive slots. Add all-rank checks where change touches distribution; B32 performance remains separate from B3 accuracy |
| External ownership/API | `../full_model/external_cache_probe/result.json`, `serving_commands_probe/result.json`, `serving_contract.json` | Reuse untouched contract source/fixtures with explicit revision scope; rerun affected reload/reset/remap/async output consumers if buffer identities or lifecycle change. No adapter implementation |
| Resume PCC limitation | `../full_model/owner_decision.md`, `owner_acceptance.json`, `qualitative/hf_resume_control/verdict.json`; raw failed resume retained | Preserve accepted HF-reproduced limitation and top1/negative-control thresholds. Do not reopen unchanged numerical gate or relabel raw failures as numerical PASS; changed prefill/cache arithmetic requires affected requalification |
| Context | `../context_contract.json`, `../full_model/context_acceptance/final.json`: all40 C262144,262143 logical prefill and final-position decode | New32K–64K smoke on final public generator, preferably32769 nonaligned with a few traced steps, all40 cache/state checks. Keep supported262144; max rerun only if KV dtype/layout changes. Update final persistent census when buffers change |
| Watcher | Stage6 reduced replay90 clean dumps/all4 chips | Separate fresh affected reduced run at `TT_METAL_WATCHER=10`, profiler unset. Do not call it all40/max-context Watcher coverage |
| Runtime fallback | `../full_model/runtime_fallback_audit.{md,json}` plus source guards and final-source performance counters | Refresh measured-path source review, runtime guard and counter evidence. `throw_exception_on_fallback` alone is not enforcement. Explicit host compatibility remains separately labeled |
| Native kernel profile | `../full_model/performance/profile_reduced_v4_drains/`: validated32896 execution UIDs,748 metadata entries, ops CSV and six perf reports | Use prior report to prioritize; profile final changed reduced real layers0/3 including terminal+sampler. Do not profile all40. Preserve CSV/provenance/command logs and per-rank timings, advice dispositions |
| Stage review/commit | Stage6 round2 clean-pass; stage5 round3 pass-with-carried | New fresh xhigh stage7 review only after required gates finish; fix/review findings. Local stage-owned commits only, log SHAs; never push |

## Smallest useful command surfaces

Run from repo root after sourcing the established lane and installed bringup environment. Every device command needs a bounded `timeout`, one process on the mesh at a time; fuser check and no other owner's process termination. These are command templates, **not executed by this auditor**. Choose output paths beneath `doc/optimized_full_model/`; never overwrite prior stage artifacts.

- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_performance --output <stage7 JSON>`: existing complete128/128 performance harness, default context1024. Diagnostics off for ranking; separate tracked validation. `--reduced` is diagnostic only.
- `python models/autoports/qwen_qwen3_6_35b_a3b/tests/run_full_model_readiness.py --mode prefill --output <JSON>` and `--mode teacher`: complete accuracy runners.
- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_generator_replay_probe --context 1024 --section all --output <JSON>`: reduced mixed/nonaligned/seed trace exercise; tracking environment set before process import. Allow at least1800s for tracked/Watcher runs.
- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_probe --generate --positive-control --output <JSON>`: tracker control. `--all-layers --generate --batch-size 32 --batch-probe` is the full B32 capacity/state surface; retain sufficient512MiB mesh trace reservation from accepted runner.
- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_sampler_contract_probe --iterations 100 --output <JSON>`: semantically greedy candidate comparison; ensure final model construction selected by probe matches final head policy.
- `python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/run_tt_controls.py --mode suite --output-dir <stage7 directory> --max-new-tokens 128`; run `--mode autoregressive` separately and keep prompt-format metadata/HF controls.
- Native reduced profiling: adapt `tests/collect_full_model_components.sh`/`profile_full_model_components.py` and prior `profile_reduced_v4_drains/commands.log`. Keep mid-run drain/source tracking settings that fixed earlier source-overflow/marker-loss failures. Six report outputs and advice must refer to completed collection, not intermediate files.

The existing performance runner has a fixed128-token workload, and maximum-context acceptance runner is fixed to the max. Neither alone implements short/long/chunk-tail TTFT or a bounded32K smoke. Add a narrow stage7 harness/parameter to exercise those requests without inadvertently repeating the262K capacity run.

## Boundary-driven cases

`tt/prefill.py::attention_chunks` selects linear chunks64; full chunks32..256 only at absolute256 boundaries, otherwise singleton decode. MoE physically rounds nonsingletons to32; singleton follows a distinct branch. The stage6 public API supports these cases and changing them is not an acceptable optimization shortcut.

For changed prefill/embedding/transport, use logical lengths1,31,32,33,63,64,65,127,128,129,255,256,257 as inexpensive planner/host cases, then select device cases on both sides of the actual changed branch (do not automatically run a full matrix). Include one continued prefix with nonzero unaligned start, one mixed/reordered multirow call equal to solo, and long nonaligned32769. If only terminal head geometry changes, check physical M32 with logical1/3/32 and prefill final-row/all-logits32-row consumers rather than an unrelated complete chunk sweep.

For TTFT record short128, a longer representative length (for example1024), and a relevant chunk tail257 alongside physical chunks/counts; add before/after measurements on the same logical requests. Per-chunk kernel table should distinguish supported64/128/256/1024/largest MoE chunks and terminal final-row calls. An unsupported generic chunk size is a documented contract row, not fabricated measurement. Stage7 preserves mixed-row prefill equality; parallel dense batched-prefill throughput is stage10 work.

For each changed trace-boundary candidate, exercise device token feedback across at least two different steps, inspect exact stable token/position inputs outside timed loops, test both changed/unchanged page tables, and finish after the last eager helper with trace replay. All live traces retain stable IDs with no request-time release/recapture. Resident measured loop must show model+sampling replays=N, token/position/RoPE/page refreshes=0, readbacks=0, no blocking replay/sync per token, no host mask/cache reconstruction. Caller-observed output delivery is a separate metric and never feedback.

## Checklist disposition map and budget

| Optimize family | Stage7 evidence destination/action |
|---|---|
| Whole graph topology, mesh shards, collective families | `topology_audit.md`; candidate rows then final selected construction evidence |
| Layer QKV/shared packing, residual layout, active MoE, cache dtype, per-role decoder geometry | Reuse exact stage5 decisions/provenance if unchanged; no broad precision frontier or rejected-policy promotion |
| New terminal matmul program config, Blackhole readers, phase layout | Measure real head BFP8/HiFi2 with one-reader control and legal reader/block/core/output alternatives; preserve complete producer/head/sampler layout costs. Record exact shape/L1 blockers after adaptation, not first API errors |
| Terminal norm, embedding, CCL persistence/links | Keep/reject measurements or exact blocker; final memory/trace lifetime evidence |
| Split sampling | Correct greedy alternatives and non-greedy capability; real negative-logit/padded-vocab/tie cases; no full-vocab gather in selected path |
| Prefill/TTFT | Short/long/tail before/after, real chunk/kernel inventory and dispatch attribution |
| Profile advice/dtype propagation | Final reduced native tables/CSV/provenance; each actionable advice row kept/rejected/blocker; prove actual dominant runtime dtype/fidelity |
| Roofline/device/e2e | `perf_summary.json`; useful-byte arithmetic from topology audit, per-kind profile times×30/10 plus single terminal path, same-run e2e regime, explicit derivation/limitations |
| Acceptance | Fresh accuracy/qualitative, trace/control, context, batch/state, Watcher, fallback and stage gate, then independent review |

Stop optional optimization when every checklist line is measured/decided/blocked, the last two kept gains are each<3%, or four-hour stage wall cap. Put remaining ideas under `Not tried (stop rule)` with expected gain where known. Correctness and the10% FUNCTIONAL gate are not waived by that stop. Stage5's accepted linear roofline shortfall is not a stage7 exemption: topology audit derives full-model ceiling21.6314ms/token, while the old model is25.0819ms. This must be resolved or reach an explicit AutoFix failure; report honest failing status rather than manufacturing a pass.

Write README headlines from the final default run, including baseline and final warmed TTFT, queued token-out, caller-observed token-out when measured, and trace-verified teacher-forcing as distinct numbers. Keep raw failures, exact commands, source identity, mesh/device environment, benchmark regime, carried owner decisions, and artifact paths in work_log. Reviews are inspection-only and cannot substitute for missing required runs.
