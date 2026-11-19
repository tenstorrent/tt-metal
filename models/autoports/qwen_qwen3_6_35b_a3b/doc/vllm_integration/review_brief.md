# Stage 9 review evidence brief — ready for independent review

All required runtime collection, cleanup and final checker have finished. This file is an evidence map, not a verdict or waiver.

Original goal: `/home/ttuser/qwen36-35b-a3b/multigoal_logs/09-09-vllm.prompt.txt`. Target: Qwen/Qwen3.6-35B-A3B. Worktree is live under `/home/ttuser/qwen36-35b-a3b/tt-metal`; stage baseline47c0bbd606a691e6111b46c5ee1eb9e01fd423d8. Plugin base042a94863ad27d88207bcf71abdf67122c6f4c27 plus local committed repair5d1785b76d17a8989c9495f6939abea6cef62f09; upstreamvLLM0.31.0. No push is authorized.

Owner loosened the wall cap to finish the remaining gates. Preserve the completed maximum-context and P5 runs; do not request repeats for documentation changes. The inherited full-model resumed-prefill limitation was explicitly accepted at the earlier stage; that acceptance does not waive new serving defects.

## Read first

- `README.md`, `work_log.md`, `manifest.json`, `acceptance_inventory.json`.
- `../../tt/generator_vllm.py`, `generator.py`, `serving.py`, `serving_state.py`, `serving_capabilities.py`; common/native sampling diff and native MoE reader completion drain.
- `FINAL_RUNTIME_FALLBACK_AUDIT.md`, `sampling_debug/FINAL64_ADAPTER_AUDIT.md`, `plugin_changes.md`.

## Final-source evidence

- `../../readiness_vllm/adapter_sampling64_dormant/result.json`: actual reduced adapter, stale opaque inputs, pending results, remaps, changed/unchanged pages, sampling transitions and dormant top-k40/64 preservation; four captures/releases, stable programs.
- `../../readiness_vllm/sampling64_probe/`: default32 exact baseline, native64 fixtures, TP4 canonical pipeline and narrow/wide timing. Preserve initial install/harness failures; isolated results are not full-model quality claims.
- `../../readiness_vllm/final64_gate/lifecycle_summary.json`: same-server18checks, nine requests, actual chunk starts0/1792/3840/4101 with concurrent decode, model and sampling signature/replay accounting. `execution.json` records exact commands, log offsets, raw exits, routes and errors for every final gate. Targeted7, P2all15, fullsampling68pass/74skip/0fail and qualitative12 reviewed/degeneracyPASS.
- `../../readiness_vllm/qualitative_controls/tt_outputs_final64.json` and `tt_final64_quality_review.json`: all12 complete stored outputs read, exact previous32 tokens, six answers capped256. Compare final served output collection against these and prompt-correct HF controls; do not infer quality from the PASS label alone.
- Final performance is under `../../readiness_vllm/final64_perf/`, measured on a separate audit0/tracking0 server. Final artifacts are promoted; historical baseline is preserved in final64_perf/previous_canonical/. Matched B2 native/adapter before/after controls: `serving_overhead/{baseline,after,comparison}.json`, all18 token trajectories exact. Historical B1 teacher-forcing is only a lower-bound reference; B1 uses different MoE arithmetic from grouped EP (see `../full_model/AUTODEBUG_MIXED.md`).

## Required carried-forward evidence and anomalies

- P3 `../eval_readiness/chat_conformance.json`:15/15. P5 `../eval_readiness/long_context_accuracy.json`:nine8K/32K/128K recalls, all pass. `p5_execution.json` correctly records recovered result and unknown original process exit.
- Context `../context_contract.json`, `../../readiness_vllm/max_context_acceptance.json`:262143 nonaligned input+1output=262144, HTTP200, no reduction, no repeat.
- Full-vocabulary standalone logit and exposed top20 HTTP equality: `../../readiness_vllm/logit_determinism/`. Explicitly scoped historical source evidence; new native sampler does not change model logits. Async/nonasync real page-growth controls: `async_control_comparison.json` and growth runtime audit.
- Prefix caching uses the documented recurrent_state exclusion; `prefix_cache_check.json` records it. Prefill remains serial; final burst artifact is required, no batched improvement claim.
- Historical sampling67pass/74skip/1mixedfail remains raw. `sampling_debug/SAMPLING_DISPOSITION.md`, `MIXED_ROUTING_RESUME_AUDIT.md`, and `../../readiness_vllm/mixed_route_analysis_legacy_parsed.json` distinguish demonstrated admission routing from the individually unattributed older TopP failure. Final rerun passes68/74skip/0fail; read it directly; do not silently rewrite a failure as PASS.
- Separate constrained thinking comparison remains HF2/10 versusTT6/10, FAIL_COMPARISON. `sampling_controls/native_card_all10_attribution.json` proves all10 trajectories/4620tokens exactly match standalone TT, including all six secondary failures; `native_card_all10_quality_review.json` reads every output. `sampling_debug/QUALITY_CONTROL_CONTRACT_SUPPLEMENT.md` gives contract scope. Judge the inherited full-model finding independently; neither attribution nor prose grants an accuracy waiver.
- Completion-drain repair: `AUTOFIX.md`, `AUTOTRIAGE_RECURRENT_ZERO.md`, `NOC_COMPLETION_AUDIT.md` and native controls. Original hang causality remains provisional. No profiler, watcher or optional MMIO diagnostic is required or repeated.

Final shutdown records: final64_tracked_server/shutdown_record.json and final64_performance_server/shutdown_record.json, both4/4releases/noholders. Canonicalpromotion: final64_perf/promotion.json. Checker: stage_check_final64.json and logs/stage_check_final64.log, exit1 advisory missingserial_reference for serialprefill; no critical failure. Independentreview andlocaltt-metalcommit pending. No acceptance claimed.
