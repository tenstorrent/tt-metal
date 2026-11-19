# Independent review contract

Stage3 optimized-decoder, Qwen/Qwen3.8-Flash-Next, live checkout `/home/ttuser/qwen38-flash/tt-metal`, branch `main`, starting commit3cd9937dfdbb2917990443e2e937e4dd3b9e60c7. No push authorized. Only `tt/optimized_decoder.py`, stage tests and docs may change; predecessor fused code and unrelated raw artifacts must remain untouched. Local checkpoint follows passing review.

Original user requirements:

- Deliver an actual optimized runtime exercised by tests, preserving fused prefill/decode semantics, paged state, determinism, meaningful layer kinds and arbitrary non-aligned logical lengths.
- Preserve context/capacity.32768–65536 smoke; max rerun only if KV dtype/layout changes. Capability reduction requires physical-limit/largest-feasible evidence.
- Whole-output PCC at functional0.995 bar; per-row diagnostic; matched original-HF misses may be classified known limitations. Explain material deltas.
- Report warmed prefill and traced warmed decode before/after. Final code must beat best correct eligible baseline/candidates, not merely reduce ops.
- BFP8 default. Measure BFP4/LoFi expert candidates with real weights and actual profiler rows; stage8 makes final precision choice.
- Operation-topology audit before knobs; address model-applicable precision/fidelity, sharding, fused/packed same-input projections, DRAM-sharded matmuls, prefill configs, SDPA/composites, memory/program/compute configs, movement and active MoE. Kept packed projections must beat tuned legal separate controls or demonstrate invalidity.
- Official advice-enabled prefill/decode tables/CSV. First API/layout error does not reject a material optimization; adapt and retry or AutoFix.
- No unnecessary host roundtrip/fallback/layout movement in measured runtime. Stress/repeated runs and watcher-clean correctness.
- README/work log include commands, exact artifacts, before/after PCC/perf, configs, advice conclusions, rejections, limitations. Assess optimize checklist until its stop rule; record untried handoff items; quality gates unchanged.
- Fresh stage-review clean-pass, or pass-with-carried only after3rounds; fix/rereview findings. Local isolated commits only, log SHAs, never push. Failed gates remain work until AutoFix fails.

Skills to read from installed package root `/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24/skills`: `stage-review/SKILL.md`, `optimize/SKILL.md`, `tt-device-usage/SKILL.md`, applicable tracing/evidence references. AutoDebug/AutoFix evidence is in this folder's `autofix_accuracy/`.

Read raw artifacts as evidence; README/work-log claims are not proof. Inspect source, final factory metadata, actual profiler rows, accepted versus failed runs, context scopes, trace tracking, watcher and accuracy controls. Compact copies preserve the full content of oversized HF JSONs; local originals remain at the recorded paths. No text generation occurs in this decoder-only stage.

Reviewer is read-only, no TT device use, reset, servers, long tests, or source edits. Small CPU artifact inspection is permitted. Write the requested review report only. Do not spawn a nested reviewer. Return stage-review's full verdict format, including required work, other concerns, gaps, anomaly ledger, inspected scope and residual risk.
