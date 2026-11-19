# Fused decoder review contract

Stage2 fused-decoder, Qwen/Qwen3.8-Flash-Next. Runtime scope is only tt/fused_decoder.py plus tests/docs. No optimized decoder, multichip, full model or serving work.

Deliver actual fused default exercised by tests; preserve functional prefill/decode, paged/recurrent request state, determinism, layer0 GDN/layer1 PLE/layer3 indexed QSA, arbitrary nonaligned lengths, context262144 and prepared PLE4768/4767 capacity. Measured warmed prefill and traced warmed decode must beat the best correct pre-fusion candidates; official tt-perf-report tables/CSV for both phases, no unnecessary measured host/layout fallback, repeated replay and Watcher correctness, all-pattern assessment, exact evidence/docs and independent review. Local stage-owned commits only; never push.

## Explicit owner wrap-up update

At2026-11-13 15:12 UTC the owner said the stage exceeded its wall budget and must wrap within about one hour: stop tuning, keep best passing version, run remaining required gates once, no exhaustive capacity or repeated max-context sweeps, use32K–64K smokes, logs plus one manifest, no per-artifact archiving. Record untried ideas as handoff. This supersedes the old70%-roofline/8h-device-time stop criterion. Optimize skill0.1.24 Stop Rule applies the owner wall cap. Context dtype/layout are unchanged.

PCC>=0.995 is whole-output/per-layer-kind aggregate. Per-row/user/step is diagnostic floor>=0.99 with top-1 agreement. A per-row miss reproduced in a matched HF control is a known limitation, recorded and carried. Isolated decoder returns hidden streams, not token logits; verify metric domain rather than asserting generated-token agreement.

## Review inputs

Live checkout branch main, base70dc6f5dd24298741e2f108a281160677e986155. No push. Start with README.md, manifest.json, final_performance.json, wrap_accuracy.json, wrap_final_plan_progress.json, work_log.md, per_op_findings.md, runtime_boundary_audit.md, pattern_assessment.json, ../context_contract.json and ../eval_readiness/limitations.json. Raw logs/results support these claims. Files larger than repository500KiB cap remain local, no archival pass required. Historical failed logs stay failed and must be classified, not erased.

Skills: /home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills/{graph-fusing,optimize,tt-device-usage,stage-review,tt-enable-tracing}/SKILL.md and tt-autodebug/0.1.8/skills/autofix/SKILL.md.

Reviewer is fresh xhigh, read-only except report file, no device/TTNN/server/reset or heavy experiments. Re-derive claims from code/artifacts. Return clean-pass / more-work-needed (pass-with-carried only round3) using stage-review required format. Required gates must be finished before review starts.
