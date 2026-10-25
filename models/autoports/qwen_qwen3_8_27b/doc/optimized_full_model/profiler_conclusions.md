# Profiler conclusions and remaining gap

Only real layers0/3 plus the complete embedding/norm/head/sampling/collector path
were profiled. CSVs and advice-enabled human tables are in `tracy/b1` and `tracy/b32`.
`scope.json` identifies synthetic4096+127 token history and decode position4223;
this is attribution evidence, not the4096/128 full-model autoregressive benchmark.
No reduced device latency or roofline is promoted into full-model telemetry.

|Reduced complete device envelope,µs|B1 before|B1 after|B32 before|B32 after|
|---|---:|---:|---:|---:|
|Prefill|16331.56|16073.86|489961.31|440997.34|
|Decode|3104.42|3120.09|5544.81|5551.25|

B32 prefill removes about48.96ms from this reduced path by packing terminal user
rows. The final head is physical32x5120x62080 at every supported batch, rather than
one padded matmul tile per prefill user. The same BF16/HiFi4 head reaches about75.7%
of report-estimated DRAM bandwidth at1.65ms on the slowest reported chip. Nineteen
real-hidden geometry controls, including adapted large-K and DRAM-sharded families,
are in head_candidates.csv; those families do not improve the whole terminal path.
No numerical policy was changed for a small timing gain.

The decode collector adds indexed-fill/copy/increment operations (about5–6µs on a
representative chip), while removing per-token host readback from unprofiled full
requests. This explains why full token-out throughput improves slightly while the
reduced device envelope is nearly unchanged/slightly larger. No device-kernel-only
speedup is claimed for buffering. B1 prefill variation is not attributed to packing,
which changes the multi-user case.

Current whole-layer windows, including every operation and internal gap, are mapped
in `reduced_decode_attribution.json`. The parser verifies the full preceding opcode
sequence against the existing source-aligned layer boundaries before reusing them;
cycles are subtracted only within each ASIC's clock domain and the maximum complete
per-device window is reported.

|Current profiled decode component,µs|B1|B32|
|---|---:|---:|
|Embedding and shared RoPE|49.69|43.18|
|One linear_attention layer|464.72|2036.62|
|One full_attention layer|372.65|1003.61|
|Terminal, sampler, collector|2239.66|2477.19|

The derived48×linear+16×full stacks are28.269ms and113.815ms; adding terminal/input
work gives30.558ms and116.336ms. These profiled estimates include instrumentation
and a fixed4223-position history. They are not directly subtractable from the
unprofiled whole-model wall time or a mathematical lower bound.

The requested lower-bound guide from optimized multichip layer host medians is
computed separately in `layer_stack_comparison.json`:

|Batch|48×linear+16×full,ms|Full token-out,ms|Excess before terminal allowance|
|---|---:|---:|---:|
|1|27.149|28.367|4.49%|
|8|47.410|47.585|0.37%|
|16|72.086|72.860|1.07%|
|32|111.872|113.066|1.07%|

Those inherited layer medians include per-layer replay overhead and predate the
stage06 rotary core-group correction; they are an explicitly approximate comparison,
not new full-model device measurements. Combined traces amortize dispatch. Even
before allowing terminal work, no batch has a10–15% unexplained excess. Current
profiles independently show no dominant sampler/collector gap. The measured
avoidable host boundary was removed; EOS-aware windows bound short-answer overrun.

The selected sampler uses TopkRoutePrep/TopkLargeIndices/TopkRouteFinish and the
canonical Sampling operation. There is no generic TopKDeviceOperation, force-argmax
or full-vocabulary gather in measured decode. Gather payloads are the32 local
candidates. B32's explicit async candidate gathers win the matched sampler controls;
legacy gathers remain faster for B1/B8/B16. Persistent sampler output buffers with
an ownership-safe clone lose; decoder persistent buffers and pooled CCL remain on.

`kernel_policy_audit.json` verifies ordinary BFP4 matmuls at LoFi, fused MM/RS inner
attributes at LoFi, B1 prefillK4/K17, B32 decodeK6/K17 andK24/K17, and the BF16/HiFi4
head. The perf-report's generic MATH FIDELITY column mislabels fused inner matmuls
as HiFi4; embedded compute_kernel_config is authoritative. The two B32 small
1536x5120 fused rows labeled SLOW are inherited, non-dominant42–45µs rows, not the
head/sampler. The accepted same-policy8/10/11-core,K24/K48,downK34 controls and
selected fractured residual contract are in
`../optimized_multichip_decoder/decode_rs_grid_selection.json` and its raw result
references. Stage07 preserves that earned selection rather than reverting to a
replicated stream or mixing precision policies.

Report-generation matplotlib cache and unclassified-op notices do not remove raw
rows. The unreplayed masked-trace notice is scoped in runtime_audit.md; selected
measurement windows have complete device-cycle data. Raw logs/CSV/provenance remain
local with artifact_manifest.json receipts; compact tables and summaries are committed.
