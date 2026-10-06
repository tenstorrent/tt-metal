# Serving runtime audit

The final runtime hash manifest is candidate_source_manifest.json. The serving
plugin revision is unchanged at 7f72b1c6e905f5137fe3377f2e7b42738d3f271d.
Its model registry resolves TTAutoportGemma4ForCausalLM to this checkout's
AutoportGemma4ForCausalLM adapter. Server logs confirm trace_mode=decode_only,
sample_on_device_mode=all and async scheduling.

The adapter passes scheduler-owned hybrid attention caches and per-layer page
tables unchanged to the existing generator. `_bind` creates persistent token,
position, RoPE-position and page-table tensors and releases prior traces before
rebinding. Equal table contents skip copies. Real allocator growth disables
steady overlap and drains pending decode before host state reload in the plugin.
The optimization does not change these branches. requests_final.json verifies
actual page growth and concurrency with exact isolated and prior-stage controls.

`decode_forward(read_from_device=False)` returns the existing device public-token
view. The canonical sampler trace now copies sampled feedback tokens into that
public tensor before completion. `_replay` and the common sampler execute trace
with blocking=False. The adapter's deferred `.cpu(blocking=False)` reads one
local token tensor and records a CQ0 event. Host tensor conversion occurs only
after completion (or synchronous plugin contract). No extra replay, device
allocation, host argmax, full-logits read, token/position upload, layout conversion,
reshard or page copy was added to steady device-sampled decode.

Optional GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1 supports shared constrained and
logprob compatibility tests. It is not selected by the primary or burst greedy
benchmarks. The standalone `host_sampling` oracle remains explicit and keeps
its original eager output formatting. The optimized device sampler delegates
to the full-model canonical greedy localTopK32/candidate-gather path and retains
force_argmax=False. It does not introduce an adapter sampler or host feedback.

The new host tests verify actual canonical capture ordering, unchanged result
identity, no post-replay eager slice and preservation of host compatibility.
Reduced device allocation tracking validates the exercised sampler/model traces;
Watcher validates multirow cache use with the documented Ethernet instrumentation
limit. adapter_queued_reads.json also passes two queued decode/read submissions before
any wait, then validates old host outputs across batch rebind.

Device allocation warnings in server logs are inherited warnings from request
shape changes behind a live trace. The focused allocation tracker is enabled
with program-cache checking retained and no unsafe-survivor failure. Tensor
storage ownership and capacity are unchanged. Successful requests/cache usage
returning to zero do not alone prove all-shape allocator stability; no measured
allocator peak or general long-soak claim is made.
