# Pinned serving mapping to full-model primitives

Source audit of plugin `042a94863ad27d88207bcf71abdf67122c6f4c27`, vLLM 0.31.0,
2026-11-14. This is the full-model interface design and evidence map. There is
no vLLM adapter, model registration, server execution, or plugin modification
in this stage. A supported design row means the existing generator supplies
the underlying operation; it does not mean the future adapter hook exists.
`plugin_owned` is reserved for an actual plugin gap under the skill's rules;
normal use of the already implemented plugin host sampler is not such a gap.

## Construction, physical storage, and lifetime

Pinned `docs/MODEL_SETUP.md` and `src/vllm_tt_plugin/model_runner.py`
(`_build_per_layer_specs`, `_kv_cache_shape`, `_allocate_kv_caches`,
`_block_tables_per_layer`) define the translation:

| Plugin entry | Existing primitive and translation |
| --- | --- |
| `initialize_vllm_model(hf_config, device, max_batch_size, max_seq_len, tt_data_parallel, optimizations)` | Construct `Model(device, checkpoint=...)` on the existing worker-owned 1x4 TP mesh, retain the requested B/context, then construct `QwenGenerator` after cache allocation. The model rejects other meshes. `build_generator(model_dir, mesh_device)` is the readiness entry; its model_dir is the autoport directory, not the HF checkpoint. Checkpoint override is explicit. |
| `get_kv_cache_spec` | Describe the 16 full-attention layers at global indices 3,7,...,63 as one homogeneous AttentionSpec group. Four global KV heads become one local KV head, head dimension 256, block size 32. Keep the 48 GDN conv/recurrent states in the opaque model-owned cache object, not artificial Mamba/attention groups. |
| `allocate_kv_cache_per_layer` / `allocate_kv_cache` | Use `Model.create_cache` / each decoder's `create_paged_state` and `create_recurrent_state` to construct the ordered 64-entry cache list, then bind it through `QwenGenerator(..., kv_cache=cache, page_table=table)`. Native allocation uses `max(table)+1` physical pages, not virtual table length; a bootstrap table can name the last physical page before installing the scheduler table. Honor per-layer tensor-index aliasing if supplied, reject incompatible shapes/grouping, and preserve accepted BF8 KV and BF16 persisted GDN. The plugin's torch-facing dtype is not permission to change this physical policy. |
| `release_persistent_capture()` | Map to `QwenGenerator.close()`, which releases retained trace IDs. The worker closes its mesh later. Dispose the generator with its exclusive cache lease; do not use `close()` as request reset and then reuse its `prepared=True` object. |

The external pool is **exclusively bound to one generator for its lifetime**.
Caller allocation is supported; concurrent independent generators sharing the
same pool are not. `prepare()` and `reset()` zero the entire bound physical
pool. Ordinary prefill clears only its destination GDN slots and fills mapped
pages; `release_request(slot)` clears that slot's GDN state, leaving KV page
reclamation to the scheduler. Cache/table object identities stay stable after
capture. Initial external device tables are explicitly installed at binding.

B32 with virtual per-request context 262144 does not imply 32 simultaneous
262144-token allocations. The default dense constructor would allocate that
impossible pool. Serving must bind an explicitly budgeted physical pool and
virtual page tables, with unused table tails allowed to alias valid pages.
Writes to active live ranges must not alias another live request. The active
fill-range certification retains native batched fill when only unused tails
alias. `mixed_prefill_native_all64.json` proves all 64 real layers at B3 with 96 virtual
table entries per row and only 64 physical pages. Uniform33 native fill remains
enabled; its top-1/top-5/top-100 comparisons all pass, with minimum PCC
0.9998330967 and maximum absolute logit error 0.25 versus solo execution. The
ragged33/128/513 prefill/decode comparisons are bitwise equal to solo execution.
This focused run is not a B32/max-context capacity measurement.

`capacity_accounting.md` records the validated budget Q262144 before plugin headroom for physical B1/P8193 and B32/P8224, with full virtual context262144. The pool includes the null block and obeys `ceil((Q + 32*B)/32) + P_extra <= P`; this generator has no out-of-pool KV scratch. Complete preparation and stated runtime workloads pass with all64 weights and retained resources. Resident measurements are not allocator high-water marks, and no extrapolated maximum pool is claimed. `get_max_tokens_all_users` is mapped to this conservative operating budget; the absent-hook131072 default is not used as evidence.

## Warmup and execution

Pinned `docs/MODEL_LIFECYCLE.md` invokes eager prefill/decode warmup first, then
trace warmup according to trace mode. Both are setup-only and precede requests.
The mapping uses the existing bounded helpers' `prepare()` for all physical
lengths 1..512 at B1 and the configured B; this is independent of logical context
length. For the eager decode warmup phase, the existing `_decode_graph`,
`_sample_graph`, and `_reset_graph` are the exact operations used by generator
preparation. The final traced warmup invokes `QwenGenerator.prepare()` once;
it completes every prerequisite and restores state before retaining decode,
sample, and reset traces. Repeated warmup hooks must respect `prepared` and
must not compile additional signatures or reset live requests after admission.
No new model math or sampler is needed in the adapter. These are design
translations, not existing named vLLM hook implementations.

Current full-repertoire lifecycle evidence is `trace_contract_native.json`:
real reduced layers [0,3], B1/context2048, lengths1..512 prepared, 23705
program-cache entries before/after 13 public request cases, three unchanged
trace IDs, strict allocation tracking including program-cache allocations,
and `acceptance_ready=true`. Cases include nonaligned lengths2..1025, resumed
starts31/512, greedy/stochastic sampling, and changed/unchanged page mappings.
The artifact's model-path and native reshape source hashes match the current checkout. The separate
tracker positive control, `trace_alloc_tracking/positive_control.log`, records
an expected rejection of a live allocation introduced after capture.

`mixed_prefill_native_all64.json` separately covers all64 real layers with focused
preparation, B3 mixed/uniform requests and release/rebuild lifecycle, retaining
834 programs and three traces. This closes the stale claim that current-source
warmup/lifecycle evidence is absent. It does not certify complete configured-B32
preparation or its physical pool capacity; `slots32_all64_pool_fixed.json` separately supplies that completed acceptance.
These are generator primitive proofs, not existing named vLLM warmup hooks.

Runtime prefill accepts eager execution for either `enable_trace` value.
Runtime decode requires `enable_trace=True`; a future launch must select the
existing `decode_only` (or `all`) trace mode and warmup. It must reject a runtime
untraced decode launch, not silently switch execution strategy.

`prefill_forward` consumes request-order token histories, absolute start/end
positions, request-order page-table rows, and explicit destination `empty_slots`.
`prompt_lens` denotes absolute ends, not chunk lengths. It owns physical chunk
boundaries, padding/slicing, cache fill, and final-row selection. Same-length
full configured batches use the accepted native batched path. Ragged/subset
requests use its B1 helper in their fixed slots. A continuation requires the
same valid GDN/KV state from the previous chunk. Only the final scheduled chunk
receives `sampling_params`; intermediate chunks do not sample or advance RNG.
The low-level resumed checks pass for starts 31 and 512. These do not replace
the separately required all64 readiness resumed-prefill gate.

`decode_forward` requires vectors covering configured fixed slots, positions
-1 for inactive rows, stable cache identity, and slot-indexed page tables.
Translate plugin presence of `sampling_params` to `sample_on_device=True`;
absence means False. The plugin does not send the generator's custom boolean.
Output is synchronous torch `[B,1]` tokens or `[B,1,V]` logits, even if the
optional `read_from_device` argument is false. With `supports_async_decode=False`
this is an allowed synchronous return, not a deferred readback contract.

For resident decode, **the caller must ensure every active device position is
below the configured context and has a valid allocated physical page before
every replay**. The resident path deliberately ignores stale host token and
position arguments; it does not read device positions to perform a host limit
check. The caller tracks its scheduled token budget and issues an authoritative
inactive-row transition before exhausting capacity. `generate` performs its
own prompt-plus-generation bound check. A virtual table with aliased unused
tails does not authorize writing those tails without scheduler allocation.

Version-1 reload commands map directly to generator parameters. Full reload
copies token, position, RoPE, active mask, and tables even when equal. Table-only
reload copies the actual stable table tensor and preserves token/position/RoPE.
No-reload preserves all resident inputs. `slot_remap` is a full capacity
permutation and is applied once before sampler reload/reset. Inactive rows
preserve GDN, token, RoPE and RNG state. The plugin's
`TTAsyncDecodeController.plan_decode_reload` sends sampler reload/reset only
on device-sampling transitions. Host mode sends neither and omits sampling
parameters; wider host top-k cannot reach `configure_sampling` through that
path. A direct caller explicitly requesting a dormant device-sampler upload
must still provide supported device parameters. Ignoring a commanded reload
would violate version 1.

The pinned plugin uses resident reuse only for `supports_async_decode=True`.
This model declares False; therefore a future plugin run will authoritatively
reload each token. The measured generator token-out path separately exercises
resident feedback. Do not describe that generator measurement as achieved
vLLM serving throughput or claim async support to remove those reloads.

## Existing host sampler and request features

Pinned `TTModelRunner.check_perform_device_sampling` (model_runner.py lines
3983–4083) selects one sampler for a submitted forward. The existing generator
boundary is `prefill_forward(..., sampling_params=None)` or
`decode_forward(..., sample_on_device=False)` returning full logits; the runner
selects scheduled rows and performs its already implemented processing in
`_get_output_tokens` (sampling metadata and `host_sampler` invocation around lines
4400–4444). `host_compatibility.json` verifies the explicit boundary and token
feedback with a top-k50-compatible host callback. It does not execute each
vLLM processor or certify served API output. These rows close source-design
mapping, with end-to-end server tests reserved for integration.

| Request feature | Pinned host operation |
| --- | --- |
| presence/frequency/repetition penalties | `supports_device_penalties=False` routes active requests to host. SamplingMetadata carries prompt/output token history and all three penalty tensors. |
| logprobs, including sampled-token-only 0 | Four devices are outside the plugin's device-logprob set {8,32}; all generated-token logprobs use host logits and `sampler_output.logprobs_tensors`. API all-vocabulary logprobs remain subject to the server max_logprobs policy; prompt_logprobs are rejected by the pinned plugin. |
| min_p, logit_bias, min_tokens | `has_active_logitsprocs()` forces host and its configured LogitsProcessors apply them. |
| allowed_token_ids | Host SamplingMetadata gets the selected `allowed_token_ids_mask`. |
| bad_words | Host SamplingMetadata gets `bad_words_token_ids`. |
| custom logits processors | `model_config.logits_processors` forces host; the runner constructs the registered processors and passes its selected LogitsProcessors. |
| structured_outputs | `supports_device_grammar=False` keeps both prefill and decode on host; `apply_grammar_bitmask` masks logits before host sampling. No device grammar methods are needed. |
| stochastic top-k outside 1..32, including unbounded | `max_device_top_k=32` routes to host; temperature-zero greedy remains device eligible. |
| seed | Device streams are keyed by signed64 seed plus absolute token position; host uses runner-owned torch generators. Existing `seeded_sampling_policy="host"` may pin seeded requests to the host route. No host/device RNG stream parity is claimed. |

The plugin tests `tests/tt/test_host_only_params.py`, `test_logprobs.py`, and
`test_request_isolation.py` exercise these expectations. They were read, not
run here. Device sampler proof includes `prefill_sampling_tracking.json` and
`sampler_ties.json`. Greedy always selects a maximum; with more than 32 equal
maxima in one shard, the selected token can differ from HF's lowest-index tie
choice. That is semantic greedy, not first-index parity.

## Request lifecycle and evidence limits

`release_request(slot)` plus a new `prefill_forward(start_pos=0, empty_slots=...)`
reinitializes recurrent history. `slots32_resident_tracking.json` proves release
is isolated from another live slot, inactive state preservation and remapping.
`mixed_prefill_native_all64.json` proves all64 repeated mixed-batch versus solo
prefill/decode outputs, slot permutations, bounded external allocation, and
release/rebuild isolation. `prefill_sampling_tracking.json` uses real reduced
[0,3] layers at B4 to prove omitted sampling rows stay unchanged and final
resumed chunks sample exactly once. Its stochastic top-k32/17 cases repeat
identically after request-order and physical-slot permutations with seeds0/-1;
explicit reload/reset also accepts both signed64 extremes without token history.
These are real model kernels, with artifact scope stated separately.

Pinned `MODEL_LIFECYCLE.md` / `DECODE_RELOAD_CONTRACT.md` already assign request
IDs, cancellation, stop strings, EOS, max_tokens, KV block release, and result
suppression to plugin/vLLM scheduling. The generator supplies one output token
and fixed-slot release, not an HTTP cancellation implementation. Preemption
releases GDN state, then fresh prefill reconstructs it from committed history;
KV-only prefix reuse cannot restore GDN and is explicitly disabled for
`recurrent_state`. Existing-state continuation is a distinct supported operation.
The combined all64 `mixed_prefill_native_all64.json.preemption_rebuild` check now
passes: fresh-history rebuild prefill and the following decode are bitwise equal
to the solo baseline, other live slots remain unchanged, and trace/program
identities remain stable. This supports the low-level preemption mapping;
server scheduling/cancellation timing remains future integration validation.

Changed/unchanged page-table proof is current in `trace_contract_native.json`:
at position32 the new physical page63 is written, former page1 stays unchanged,
and every bound device table matches. The only steady-phase host write is the
explicit page-change upload; ordinary resident steps perform no host token,
position, RoPE or table refresh. Capture bindings prove the sample output token
buffer is the next decode input buffer, and device positions remain coherent.
`prefill_sampling_tracking.json` separately covers authoritative seed resets and
ignored stale resident host inputs. Neither artifact establishes all64 maximum
context, final throughput, or end-to-end served sampling. The separate final capacity artifacts close `get_max_tokens_all_users` as described below.


## Final operating-budget evidence

The missing aggregate-budget row is closed by `max_context_final.json` (B1/P8193, full context) and `slots32_all64_pool_fixed.json` (physical B32/P8224, full preparation and slot lifecycle). The future adapter can return Q262144 before the pinned plugin adds32 tokens per physical slot; the existing generator binds that bounded pool through its full-width page table. This mapping requires no new model path. Capacity coverage is explicitly physical B1/B32; B32 admits1..32 live/inactive requests. Other constructor batch sizes do not inherit an unmeasured full-context budget. Null reservation, page rounding and scheduler admission/preemption remain as documented in `capacity_accounting.md`. No largest-physical-pool claim or context reduction is made.

`serving_contract.json` now has79 rows and zero missing rows. Existing-state resumed prefill passes on the final native build (`prefill_resume_native_receipt.json`); KV-only prefix caching remains excluded for recurrent state. These are full-model primitives and interface mappings, not a tested vLLM adapter or server.
