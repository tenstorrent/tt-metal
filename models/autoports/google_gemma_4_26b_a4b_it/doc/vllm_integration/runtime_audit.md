# Serving runtime audit

The final launch recipe is `full_server_command.json`. `readiness_vllm/server.log`
records `sample_on_device_mode=all`, `trace_mode=decode_only`, max_model_len262144,
max_num_seqs32 and async scheduling. The adapter constructs Gemma4Generator
without its standalone `host_sampling` option, so that option remains False.
Canonical device decode executes the model trace nonblocking and invokes the
full-model split sampler; its persistent token output feeds the next device
step. The adapter returns token IDs only in this mode. `adapter_trimmed_batch.json`
and `requests_reduced_trimmed.json` exercise removal of trailing plugin wire
padding without dropping interior inactive rows or consulting stale host
positions during steady async replay. Per-layer page-table changes refresh only
changed tables; unchanged tables do not copy, as recorded in
`adapter_changed_pages.json`.

The explicit environment opt-in GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1 permits
vLLM-owned CPU sampling for shared tests. The plugin's
`model_runner.py::check_perform_device_sampling` chooses this mode for min_p,
bad words, logit bias, allowlists, minimum tokens, structured output and TP4
logprobs. The adapter returns logits from the generator's existing low-level
path; it does not implement a CPU sampler or argmax. Mode changes release the
previous trace before binding the next mode. `requests_reduced_compat.json`,
`reduced_host_minp.json` and numeric host-sampler unit checks cover this path.
Neither benchmark profile requests these constraints or logprobs. Their explicit
temperature0 overrides the checkpoint generation defaults, so benchmark metrics
use the canonical greedy device sampler.

The startup warning about disabling chunked prefill is upstream's general
hybrid-model warning. This adapter intentionally accepts full-prompt prefill and
rejects nonzero continuation positions; the server disables chunked prefill and
allows a token budget matching the context contract. Hybrid page capacity
accounts for all six cache groups before sliding retirement. Non-aligned prompt,
page-growth and concurrent allocator checks exercise this contract. The
scheduler warning about AsyncScheduler is upstream's generic warning for a
custom scheduler; the TT plugin owns the proven deferred-read/overlapped decode
pipeline tested against synchronous outputs.

Skipping worker warmup is explicit. The generator compiles and captures the
actual input signature on first use, including the exact sampling mode. The
benchmark CLI performs a startup request before measuring. Inductor is disabled
because execution uses TTNN. The file-descriptor limit warning did not prevent
48 queued requests from completing in logprob checks. Unknown VLLM_RPC_TIMEOUT
and unauthenticated cached-HF notices do not change model execution.

The allocation-after-capture warning was investigated in
`AUTODEBUG_trace_allocations.md`. Reduced layers0/5 canonical split replay passed
with TT_METAL_TRACE_ALLOC_TRACKING=1, TT_METAL_TRACE_ALLOC_TRACEBACKS=1 and
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0: `adapter_trace_allocations.json` and
`.log` preserve the result. The tracker found no surviving unsafe allocation at
model or sampler replay. This resolves the exercised lifecycle warning; it is
not an address-overlap proof for untested configurations. Full30-layer model
logits traces also passed the same tracker settings in
logit_determinism_full.json/.log; canonical full-model split tracking also passed
four256-token controls in qualitative_256_controls.json/.log. The full initial server
was stopped via runner SIGINT, with no remaining runner, vLLM entrypoint or
EngineCore processes. The final full-model runner exited0 after SIGINT55104; API55123 and EngineCore55165
exited, and the final ps audit found no serving processes (`final_cleanup.json`).
The final check runner exited0 with72 sampling passes/one documented logprob-cap
skip, plus both completed benchmark profiles. Final log inspection found no new
ERROR/Traceback/FATAL or fallback indication; warnings match the categories above.

The selected full-model teacher-forcing control in
`../datatype_sweep/selected/readiness.json` records52.1824 tokens/s/user for
161 input and100 output tokens at batch1, including99 caller-token injections.
Its approximately19.1635 ms decode/token is only a lower-bound latency reference
for decoder execution; the workload and feedback contract differ from serving.
It is not the vLLM headline or a same-workload speedup claim. The current serving
measurement is19.7128 ms TPOT for4096 input/128 output/B1/C1. A same-harness
serving baseline retaining32 padded wire rows took58.6442 ms TPOT for the same
4096/128/B1/C1 shape; `baseline_padded_decode/` preserves that result. Removing
only unused trailing wire rows resolves that measured serving-specific overhead.

The inherited common device sampler limits effective stochastic top-k to32;
`format_sampling_params` maps top_k>32 or top_k<1 to32. This applies to ordinary
device requests even when checkpoint defaults say64 or a shared test asks100.
The greedy path remains the canonical split-sampling greedy implementation;
this is not a new greedy top-k fallback. Explicit host-only compatibility
requests use vLLM CPU semantics. These limitations must accompany stochastic
sampling results; primary greedy performance does not depend on the top-k cap.


Shutdown audit detail: the installed API server uses abort/timeout=0 for this
runner stop. It reports force-stopping one remaining EngineCore, then records
engine-manager completion and FastAPI application shutdown. The process audit
confirms all three owned PIDs are gone; this is not a claim of graceful device
teardown. Nanobind reports105 instances/973 types/4455 functions at interpreter
exit, exactly the same counts as the prior `server_decode_seed_fixed.log`
shutdown after a much smaller workload. These are binding-reference teardown
warnings preserved in the logs, not leftover serving processes or a measured
per-request growth claim. No warning was suppressed and no additional reset or
profiling was performed.
