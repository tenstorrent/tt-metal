# Stage03 optimized decoder contract

Target: Qwen/Qwen3.8-27B. Scope: `tt/optimized_decoder.py` plus its tests/docs. Preserve fused prefill/decode, paged cache, determinism, both real layer kinds,262144 context, valid unaligned lengths and functional PCC>=.995. No multichip, full-model or vLLM implementation.

Required: real optimized path in delivered tests; topology audit first; canonical precision policy; sharded layouts; packed versus legal tuned separate same-input projections; DRAM-sharded matmuls; large-prefill configs; SDPA/composite ops; runtime movement reduction; MoE applicable/not-applicable determination. Mandatory material BFP4/LoFi geometry trials. Synthetic-only PCC cannot reject a real-weight win. Final default must beat strongest correct traced-decode candidate, not merely functional baseline. Warmed before/after prefill and traced decode; advice-enabled tt-perf-report tables/CSVs; try actionable advice and record measured rejections/adaptations. No unnecessary torch/conversion/host fallback in measured runtime. Repeated runs/stress and separate watcher-clean checks. Update context contract when memory/KV policy changes, without capability reduction unless physical limit proven.

README/work_log: commands, PCC, latency, configs, advice/rejections, limitations, exact artifacts, completed optimize checklist. Fresh xhigh stage-review clean-pass required; findings are work. Local stage-owned commits only; never push; log SHAs. Stop only after AutoFix fails, not because a gate remains unmet.

Headline performance workload:4096 input tokens,128 supplied-hidden-input decode steps for this single-layer stage,B1,concurrency1. Report linear_attention and full_attention separately. Device-time roofline denominator is the entire layer firmware execution window including all operations and gaps. Useful FLOPs/estimated traffic, physical device theoretical peaks and assumptions must be explicit. No substitution of host wall time into device fields.

Compact JSON evidence packet must use the supplied template identifiers and be written locally even offline:
`/home/mvasiljev/agentic-research-data/experiment-26/bringup-logs/qwen38-27b/20261023-astra-medium-exp26-clean/telemetry/packets/ee9997b0-4c77-4500-b9e6-9af11e547881.json`.
Unavailable values stay null/omitted with missing_reasons; no raw logs/tensors in packet.

Selected skills: installed tt-model-bringup0.1.21 `optimize`, `agentic-serving`, `tt-device-usage`, `stage-review`, `tt-enable-tracing`; tt-autodebug0.1.8 `autodebug`/`autofix` when required. Startup telemetry plugin0.1.7 is active. User explicitly authorizes subagents requested/recommended by a selected skill.
