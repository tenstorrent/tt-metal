# Runtime and ownership audit

The default runtime is the30-layer accepted TP4 stack. Embeddings, both rotary
representations, final norm, vocabulary-sharded head, softcap, cache operations,
collectives and sampling execute on TT devices. Decoder dtypes, fidelities,
expert strategies, residual layout and rejection ledger are unchanged.

`tests/check_full_sampling.py` runs prefill, model decode, sampler precompile
and trace replay inside `tests/runtime_audit.py:device_only`. Torch dispatch
and TTNN host-tensor construction/readback APIs raise inside that scope. These
checks pass (`sampling_contract.json`). Host checkpoint conversion, tokenization,
cache allocation, request sampling setup and scheduler state binding remain
explicit setup boundaries.

Default token-out replay submits model and sampler traces nonblocking. The
sampler writes the persistent decode input; model trace increments RoPE/cache
positions and sampled seed state. The public caller may read output tokens;
that read synchronizes their availability. Counter `synchronizations` counts
explicit synchronization calls, not the unavoidable wait inside token readback.
`full_logits_readbacks` is zero on measured default token-out generation.

Full-logit host boundaries are deliberate: readiness `prefill_logits`, optional
all-position prefill logits, test diagnostics, and `host_sampling=True` greedy
compatibility. Host mode rejects unsupported sampled/penalty policies. It is
excluded from optimized token-out measurements. Teacher forcing explicitly
refreshes token input via its callback; it does not refresh positions per token.

The standalone generator owns its cache/table. `reset` clears owned KV and
request state in place and retains tensor/trace identities; caller-owned cache
contents are not cleared. Warmed requests with the same greedy policy and prompt shape reuse existing
traces and cache allocations; changed prompt contents are refreshed at request
setup. A tracked changed-prompt replay matches a fresh capture. New signatures
release old traces before potentially different prefill programs. The final TTFT metric
includes request trace preparation. Low-level callers supply cache, page table,
prompt lengths, slot IDs and positions. Same-shape unchanged tables skip copies;
changed content copies into the existing allocation. Shape/cache/batch/active
mask changes bind a new trace and require explicit input tokens.

Public batch token outputs use preallocated storage and a pre-warmed slice with
`output_tensor`, rather than keeping an allocation alive across trace replay.
MixedB3 inactive-slot and B32 independent-cache checks pass trace allocation
tracking. Full30-layerB32 feedback and independent endpoint-slot controls also
pass (`trace_full_batch32.json`). B32 uses a1GB trace reservation; B1 uses100MB.

The2-layer diagnostic may saturate tens of thousands of logits at softcap30.
A same-prefix oracle proves host and common-device greedy choices are both
maxima. This tests the common sampler's documented tied-candidate behavior; it
is not a full-model text quality assessment. All-layer AIME and qualitative
results are recorded separately.

Watcher and profiler run separately. Watcher status and exact command are
recorded in the work log; initial inlined Watcher fabric firmware exceeded
its code buffer before model execution, requiring the prior-stage NOINLINE
setting. That retry exposed a native multicast all-gather unused-scatter
initializer assertion for4096-byte pages. Evidence was preserved and a bounded
reset, four-device listing and TP4 mesh smoke restored healthy devices.
The guarded native initialization passes both2048B/4096B page controls,
eager and traced, on linear/ring routes. The original model Watcher regression
then passes with allocation tracking (`trace_watcher_fixed.log`). The CI build
wrapper could not run because Docker is unavailable (`native_build.log`);
affected device kernels were JIT-compiled by the passing hardware tests.

Final audit artifact: `sampling_contract.json`, written by the successful
`sampling_contract_pass.log` run. Penalty counts cover the prefill token and
traced decode tokens. Optional TP4 logprob requests are rejected before
request-state mutation; there is no silent None result for such requests.
