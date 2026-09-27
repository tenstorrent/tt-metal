# Full-path profiling and checklist

The measured end-to-end all-30-layer result is in `performance.json`: 4096 input,
128 generated tokens, batch 1, one request, median of three warmed requests.
Device instrumentation is deliberately restricted to actual layers 0 and 5 plus
embedding, final norm, vocabulary-sharded LM head, split sampler and token recorder.
These reduced measurements are diagnostics, never full-model device latency.

`profile/summary.json` records the raw CSV path, SHA256 and whole device windows,
including gaps (maximum rank first firmware start to last firmware end). Prefill
is 667191.590 us; decode is 3082.769 us. Prefill is eager and profiler/dispatch
heavy; its 667.644 ms host signpost bounds the device window. Decode has three
traces: model up to 2553.024 us, sampler up to 514.067 us and recorder up to
9.821 us. Maxima occur on different ranks and must not be summed as an exact
critical path. The decode host signpost is 3.281 ms. Neither phase is a substitute
for the all-layer TTFT or autoregressive throughput.

The advice-backed human tables and compact CSVs are
`profile/{prefill,decode}_perf_report.{txt,csv}`; console logs preserve invocation
output. Large prefill CSV and advice text tables are also preserved byte-for-byte as
`.gz` files for the repository size limit; uncompressed files remain local.
Raw captures stay local under `profile/reports/` and `profile/.logs/`.
The table merges rank operations and can reorder the three tiny recorder ops;
raw per-rank trace windows establish recorder ordering and total cost.

| Area / report advice | Decision and evidence |
| --- | --- |
| LM head DRAM sharding | Current row is BF16 x BF16, HiFi4, 930 us decode / 931 us prefill, about 78.4% per-op modeled bandwidth. Precision-locked adapted chunks/readers/K-block sweeps include movement and correctness; best DRAM path 1245.46 us loses to selected interleaved K4 936.85 us. `terminal_lm_head_plan.md`, 44-row candidates CSV and native JSONs. |
| Matmul grid / output subblocks / large K blocks | 11x10 K4 selected versus K8, K11 and legal smaller grids; K22/44/88 retried with smaller DRAM chunks after full-width L1 overflow. Same real normalized input and BF16/HiFi4 policy throughout. Native reports confirm the selected policy. |
| Greedy sampler | Correct physical tile-padded split path 508.518 us versus force-argmax 3300.512 us; known winners pass on every rank. Profile uses TopkRoutePrep/TopkLargeIndices and candidate collectives, not full-vocabulary gather or generic TopKDeviceOperation. `sampler_comparison.json`. |
| Trace boundary gap advice | All three decode traces already replay nonblocking on one queue. Report recommends tracing for about 34 us of boundary gaps even though trace IDs prove tracing is enabled. The recorder trace costs under 10 us, <0.05% of full token time. Combining sampler with model would violate the canonical split contract. |
| Token output / host orchestration | Fixed-step generation retains history on device, reads once at completion, and uses device token/position/RoPE advance. 127 model/sampler/recorder replays, no host input updates/syncs. `performance_comparison.json`, `buffered_extended_watcher.json`, source runtime audit. |
| Embedding / terminal norm / logits | Hidden-sharded embedding with entry gather; accepted replicated residual flows directly to final norm and vocab-sharded head. Only local padded vocabulary shards reach sampling. No logits host transfer on measured generation. |
| Decoder CCL / residual sharding / persistent buffers | Unchanged accepted Stage05 family: TP4 attention, EP4 prefill/indexed TP4 decode experts, async collectives with persistent L1 output and layer-private semaphores. Carried-shard/fused/lower-movement families were adapted through consumers and rejected by measurements; `../optimized_multichip_decoder/{candidate_comparisons,residual_contract,final_perf_findings}.md`. No policy or rejection ledger changed. |
| Decoder SLOW/subblock/DRAM advice | Stage05 final-policy native rows and geometry controls cover QKV/output/router/shared/sparse matmuls, DRAM readers and sparse N2/K alternatives; retained configurations win whole-layer comparisons. This stage changes only terminal head K blocking. |
| Prefill input-L1 advice | Stage05 `prefill_producer_summary.json` tests all real producer roles for both kinds with repeated and combined controls; no reproducible material whole-layer gain. Avoid repeating an already measured losing family. |
| Fidelity advice | Router weights are BF16, so generic BFP8 advice is inapplicable there. Accepted per-group decoder policy is frozen by this goal, including accuracy rejections; no broad datatype frontier search here. |
| Prefill expert utilization | Dynamic 32-token expert unions differ from eight active experts per token. Report's `--active-experts 8` annotation is diagnostic; no prefill FLOP percentage is claimed from it. |

The stage's cross-run layer-stack bound is 19.842905 ms from 25 sliding and five
full-attention warmed medians. Full token-out is 20.262 ms including final output
transfer, 2.11% above the stack alone. The >10–15% gap trigger is not reached even
before terminal allowance. Approximate DRAM traffic lower-bound assumptions and
sources are in `layer_stack_comparison.json`; this is not measured utilization.
Full-model device times and roofline fields remain null in telemetry. No sum of
matmul times or average utilization is represented as whole-model performance.

Recorder storage is persistent, but indexed-fill plus copy moves the history each
step. At the required 128-token workload its measured cost is negligible. Very
long fixed-length output may spend more time copying history; this is an explicit
performance limitation, not a context or functional cap. Memory accounting
retains maximum context and the existing reserve in `../context_contract.json`.
