Use $optimize and $tt-device-usage for the Qwen/Qwen3.8-Flash-Next multichip decoder in the current tt-metal checkout.

Subagents are authorized when a selected skill requests or recommends them.

This creates optimized-multichip-decoder state for the repo-local TTNN autoport pipeline. Start from the completed multichip decoder and optimize it in place; do not begin full-model or vLLM work.

Goal completion requirements:
- The measured path is the multichip decoder path on the target mesh. Single-chip or replicated fallback does not satisfy completion.
- Multi-chip prefill and decode PCC remain at the accepted baseline for every meaningful layer kind (whole-output; per-row PCC is diagnostic; misses HF reproduces are known limitations).
- Warmed prefill and traced warmed decode latency are reported before and after this multichip optimization pass.
- tt-perf-report tables and CSV/provenance logs exist. Actionable advice is tried. Rejections have before/after evidence. A first TTNN/API error is not enough to reject an optimization that would remove a material op, collective, reshard, or layout conversion.
- The pass starts with an operation-topology audit: repeated same-input matmuls, material collectives, reshard/layout conversions, fused/packed projections, fused matmul-CCL paths, lower-movement residual layouts, action taken, and evidence.
- Dominant multi-device costs are compared as coherent families: residual layout, collective placement, fused CCL+matmul, packed vs separate projections, activation/CCL dtype, and persistent buffers. A lower-movement family is not rejected by measuring it only with an immediate restore to the old replicated residual contract.
- Model-applicable async CCLs, fused matmul-CCL paths, collective placement, preallocated buffers, residual layout, activation sharding, DRAM-sharded decode matmuls, and precision/fidelity reductions have evidence. For material rejections, adapt shape/layout/padding/weights and retry, or use `$autofix`.
- Preserve or improve the multichip context contract. If KV-cache dtype/layout, activation sharding, or per-device memory changes max context, update doc/context_contract.json. Reductions require hard physical evidence and the largest feasible value. Context: 32K–64K smoke; max only if KV dtype/layout changes.
- Preserve support for valid non-aligned logical sequence lengths; optimization may add internal alignment, but it must own padding/masking/slicing rather than requiring aligned public inputs.
- The final path avoids gather/reshard/all-reduce between decoder layers. Any remaining inter-layer collective has measured evidence showing it is faster overall. Write down the inter-layer residual layout contract so full-model bringup can preserve it instead of rediscovering it.
- Runtime fallback audit is clean, stress coverage matches risk, and watcher-clean evidence exists.
- models/autoports/<model>/doc/optimized_multichip_decoder/README.md and work_log.md record commands, before/after PCC/perf, CCL findings, rejected options, limitations, and artifacts.
- Applicable items in this prompt and `$optimize` are tried with evidence until the $optimize Stop Rule (checklist done, last 2 kept <3%, or wall cap); untried items are work-log hand-off notes; quality gates are unchanged.
- MoE: keep gate-selected active-expert execution; dense all-expert execution needs evidence it is faster in decode. Follow the $multichip MoE rule; report decode at batch 1 and the largest batch with diverse routing, plus the MoE share of the step.
- $stage-review returns `clean-pass` (or `pass-with-carried` after 3 rounds); findings fixed/rereviewed. Local commit stage-owned touched-repo changes; never push; log SHAs.
- Final optimized numbers come from the final default path, not from an earlier candidate that the default no longer reproduces.

Unmet requirements, review findings, failed gates: work. Stop only after `$autofix` fails.
