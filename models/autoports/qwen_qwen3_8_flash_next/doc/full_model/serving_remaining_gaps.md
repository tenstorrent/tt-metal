# Serving evidence still required

This table groups the remaining `missing` ledger rows into concrete work. It
does not declare pending hardware checks passed. No server or adapter was added.
Accepted paged K/V is BF16/BF16; BFP8 was rejected in optimized stage5.

| Gap | Required implementation or invocation | Acceptance evidence |
| --- | --- | --- |
| Allocator/global block IDs/cache ownership | Verify implemented deferred setup, allocator `(shape,dtype,tensor_idx)`, opaque pool ownership and per-layer table routing; finish adapter-facing mapping of all48 layer kinds without allocating recurrent layers as KV | `tests/probe_shared_kv_pool.py` then reduced generator with real writes, global disjoint IDs, remap and slot release; transient allocation peak accounted |
| Full context and resource estimate | Implemented `get_max_tokens_all_users` includes pooled-state budget, non-KV scratch and pinned output/null reservation; replace unmeasured planning reserve with acceptance measurements | One advertised-max prefill/decode acceptance; physical limit only with the required bracket pair |
| Prefill/chunk/resume | Current `prefill_forward`, `supports_chunked_prefill`, mixed3/33 prompt lengths, sampled first-token, retained-state continuation | Passing `readiness/prefill_resume_check.json` plus `tests/probe_full_model_serving.py`; previous full_accuracy_v1 resume failed |
| Decode/reload/inactive/remap/release | Current fixed-slot low-level path, unchanged versus changed tables, dormant feedback, state/seed migration, ended-request reuse | Current serving probe passes batched versus solo outputs/state, no per-token position/table refresh and survivor unchanged |
| Seeded/common sampling | `tests/probe_seeded_sampling.py` uses actual common sampler, including actual sampled-prefill wrapper | PASSED `seeded_sampling_device_v2.json`: actual common sampler across reset, row order, pause, migration and synthetic-logit sampled-prefill wrapper; decoder-output integration remains in serving probe |
| Host logits and request features | `decode_forward(sample_on_device=False,...)`, `read_decode_output`, `process_decode_output_host` expose raw full-vocabulary logits with correct order and independent host snapshots | Serving probe delayed reads and full logits match; host-mode seeded generator control passes. Existing10 CPU host-helper tests cover utility math only |
| Logprobs/penalties/min-p/bad words/bias/allowed IDs/min tokens/structured outputs/custom processors | Preserve raw logits boundary for the pinned plugin's built-in processors; do not run model HostSampler again when plugin owns sampling | Boundary device evidence plus source-mapped processor ownership, not a `plugin_owned` exemption or invented launch option |
| Warmup/lifetime | Bind actual caches before prepare; warm every migrated/upload/sampling shape; idempotent `close` releases captures before caller closes mesh | All current keys under allocation tracking, retained request-time trace IDs, no freed pool/feedback addresses |
| Fabric/setup | Existing `model_capabilities.fabric_config` now declares accepted ring; verify caller provides documented trace/L1 resources (no model resource capability exists) | Model-side setup mapping and an allocation/trace proof using it; probe-only configuration does not satisfy this API |

The readiness `build_generator` factory is implemented and exercised by
`full_accuracy_v1.json` initialization/preparation, but that artifact is an
overall failure due to resumed prefill. It cannot establish full-stage acceptance.
Device penalties and asynchronous resident-output reuse are explicitly disabled
using existing pinned capability keys. Prefix caching is disabled for the allowed
`recurrent_state` exclusion; retained-state resumed prefill remains mandatory.

CPU documentation audit updated `serving_contract.json` to distinguish supported
APIs, explicit documented defaults/declarations, and missing device evidence.
Shared-pool allocation is now integrated, including deferred state construction,
per-layer routing and dormant-slot validation; 49 CPU tests pass. Allocator rows
remain missing until the prepared real reduced shared-pool test passes.
