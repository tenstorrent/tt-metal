# Full-model serving gap audit

Source-only AutoFix investigation, 2026-11-14. This report does not implement a
vLLM adapter, edit `serving_contract.json`, open devices, or certify hardware
results. The pinned plugin is
`042a94863ad27d88207bcf71abdf67122c6f4c27`, targeting vLLM 0.31.0. Generator
source changed concurrently during this audit; the parent owns fixes and
acceptance evidence. The earlier slot report's findings are not repeated as
unfixed claims; the parent reports that the repaired B32 tracked probe passed.

The full-model stage can close the contract without implementing optional
speculation, device grammar, device penalties, or asynchronous output DMA.
It cannot close real generator gaps by assigning them to the future adapter.
The important remaining source findings were ordinary disabled-logprob
sentinels, seeded reload without history, and low-level device-sampled prefill.

## Required generator fixes or explicit proof

1. **Disabled logprobs use a negative sentinel.** The inspected decode
   validation rejected `num_logprobs != 0`. The plugin's
   `TTModelRunner._sampling_params_for_padded_decode` sets
   `enable_log_probs = num_logprobs >= 0`; ordinary disabled values are negative.
   Consequently a normal device request could fail validation even with
   `enable_log_probs=False`. Permit the disabled sentinel and reject actual
   requested device logprobs. Do not confuse API `top_logprobs=-1` (all vocabulary)
   with the internal disabled sentinel: the plugin's accompanying enable flag
   distinguishes request handling.

2. **Seed resets cannot assume `output_tokens` exists.** In
   `TTAsyncDecodeController.submit_decode`, history is supplied only when
   `model_input.prompt_tokens` is present, for penalty handling. Ordinary seeded
   decode therefore reaches `_reset_rng` without output history. The inspected
   implementation then selected offset zero on every sampling-state transition,
   restarting continuing streams when another request arrived or prefill
   interleaved. The command requires reconstructing the same stream position
   from available authoritative state. An explicitly documented
   absolute-position seed stream is viable: align from `start_pos` only when
   `reload_inputs=True`, advance once on device per sample, and make initial
   prefill sampling use the corresponding convention. Do not derive anything
   from stale host positions when `reload_inputs=False`. Signed 64-bit API seeds
   must survive until conversion at the device boundary. Include seed 0, -1,
   large positive/negative seeds, remaps, inactive rows, and a reset transition
   without token history in the probe.

3. **Low-level prefill must not swallow device sampling.** The inspected
   `prefill_forward(..., **kwargs)` ignored `sampling_params` and returned
   logits. `TTModelRunner.submit_prefill` sends that keyword for
   `sample_on_device_mode="all"`, and then interprets output as token IDs.
   High-level `generate` already samples its first token, but that does not
   establish multi-row low-level prefill sampling. Either supply an explicit
   reusable generator primitive that consumes device last-row logits and
   per-request sampling parameters, or implement the keyword in
   `prefill_forward`. Preserve omitted slots and scatter submitted sampling
   rows by `empty_slots`; intermediate chunks receive no sampling parameters
   and must not advance the RNG. A future adapter may select the existing
   `decode_only` mode and host-sample prefill, but that narrow launch mapping
   is not evidence that an unrestricted `all` mode is supported. Do not mark
   this row supported merely because a wrapper can reimplement sampling later.

4. **Prove actual external-cache binding.** The low-level generator accepts
   caller-supplied cache and table allocations at construction and requires
   stable identity thereafter. This is compatible with serving if adapter
   allocation happens before generator preparation and capture. Evidence
   should use a caller-created bounded physical page pool rather than only
   default per-slot dense pages; include shuffled physical pages, fixed slots,
   and unchanged/changed page contents. The shared KV-token pool need not hold
   `batch_capacity * max_context` tokens simultaneously. Its physical capacity
   and the maximum individual context are separate quantities.

5. **Warmup is generator-owned.** The full-model skill requires all supported
   eager preparation before first capture, restoration of mutated state, and
   retained traces across public workloads. The adapter hook names themselves
   are stage-9 work; preparation of all underlying paths is stage-6 work.
   Document how both plugin warmup phases map onto the generator without
   capturing first and compiling a later variant. If untraced runtime decode
   is deliberately rejected, state that restriction; do not silently change
   to host or eager decode when the plugin sends `enable_trace=False`.

## Cache grouping: no artificial GDN KV group

The model has 16 identical full-attention KV layers, four global KV heads,
head dimension 256, 32-token pages, and 48 GDN layers whose conv/recurrent
state is owned by the generator. The selected physical KV dtype remains the
optimized decoder's BF8 policy; recurrent persistence remains BF16. No dtype
change is authorized by the plugin's Torch-facing cache specification.

`MODEL_SETUP.md` and `TTModelRunner._block_tables_per_layer` state that one KV
group omits `page_tables_per_layer`, even if that one group has heterogeneous
per-layer shapes. Therefore **per-layer page-table routing is not required
for this one-group full-attention design**. Record the conditional feature as
not applicable to the declared layout, not as an unimplemented mandatory
generator capability. An unsupported multi-group table must not be silently
accepted and ignored. No documented model capability key is needed to invent
a second group for GDN state.

`TTModelRunner._validate_kv_cache_groups` accepts `AttentionSpec` and
`UniformTypeKVCacheSpecs` only. Treating GDN as an upstream `MambaSpec` group
would add an unsupported plugin contract. Keep its state in the adapter's
opaque cache object alongside the 16 physical attention-cache entries.

The future adapter must map the plugin's ordered attention-layer specs to
full model layer indices 3,7,...,63. The exact count/order produced by pinned
vLLM's `get_num_layers_by_block_type(...,"attention")` must be checked at
integration; do not confuse full model indices with compact attention indices.
This is adapter translation, not a reason to add 48 fake paged caches now.

The optional `get_max_tokens_all_users` must use the measured weight/state/
trace/CCL budget from `context_contract.json`; the plugin's absent-hook
131072-token default is not a physical memory measurement. Cache pool size,
page width/padding, and any scratch pages must be included. The constructor
must continue requiring the optimized 1x4 TP mesh.

## Legitimate capability choices

These are documented declarations/defaults, not requests for new plugin gates.
Each `not_supported` row should name the exact declaration and request effect.
Each supported row still needs an existing model-local evidence path.

| Row | Defensible declaration and effect |
| --- | --- |
| `supports_chunked_prefill` | `True` once resumed/mixed-length low-level evidence passes. Scheduler processes absolute `[start,end)` chunks; no intermediate sample. Internal generator chunking alone does not prove this. |
| `supports_prefix_caching` / `prefix_caching` | `False`, reason explicitly `recurrent_state`: vLLM KV blocks do not reconstruct GDN conv/recurrent state on a new prefix hit. Existing-state continuation remains supported. User explicitly still requires the resumed-prefill check, despite the skill's generic exclusion allowance. |
| `output_tokens_per_step` | `1`: ordinary autoregressive output, no block-output semantics. |
| `supports_sample_on_device` | `True` for the proven greedy and stochastic low-level paths. Record prefill mode coverage explicitly. |
| `max_device_top_k` | `32`: stochastic k outside 1..32, including unbounded k, routes to host. Greedy temperature zero must remain greedy regardless of request k. |
| `supports_device_penalties` | Explicit `False` (default is **True**). Presence/frequency/repetition requests remain supported via the host-logits boundary. |
| `supports_device_grammar` | `False`: structured prefill and decode remain host-sampled; ordinary structured requests are not rejected. No deferred payload/mask methods are required. |
| `supports_async_decode` / `async_decode` | `False`: plugin disables async scheduling and sends authoritative forward inputs each step. This is permitted and does not remove generator device token feedback or separate decode/sample traces. |
| `supports_compact_host_logits` | `False`: return full fixed-slot host logits; plugin selects scheduled rows. |
| `supports_selective_host_readback` | `False`: no selected-row DMA contract. Full output is valid. |
| `supports_spec_decode` | `False`: speculative launches are rejected at admission. No verify, candidate state, proposer, or acceptance implementation is needed. |
| `spec_ids_are_sampled`, `supports_async_spec_decode` | `False`, conditional speculative extensions unused. |
| `supports_narrow_decode` | This is a **SpecPlan field**, not a model capability. Not applicable when no SpecPlan is exposed; ordinary width-one decode remains supported. |
| `spec_requirements`, `spec_hidden_handoff` | Empty documented defaults; speculative entrypoints excluded. |
| `supports_stage_inputs`, `supports_stage_outputs` | `False`: no Omni intermediate stage payload/frame contract for this text model. |
| `folds_dp_into_lanes` | `False`: no in-process lane-DP mesh folding. Standard DP would be independent 1x4 replicas with enough hardware; it is not a fallback to four single-chip models. |
| `tt_adaptive_block_output` | `False`; `tt_adaptive_block_max_prompt_tokens=0`, `tt_block_kv_extent_tokens=0`: all block-output extensions unused. |
| `fabric_config` | Declare the accepted FABRIC_1D configuration using actual TTNN enum objects in the future adapter. JSON records names; no topology or CCL strategy change. |

The current JSON inventory omits the six capability rows
`tt_adaptive_block_output`, `tt_adaptive_block_max_prompt_tokens`,
`tt_block_kv_extent_tokens`, `folds_dp_into_lanes`, `spec_requirements`, and
`spec_hidden_handoff`. Add explicit rows so the inventory covers every
documented model capability, even when excluded by defaults.

## Call mapping without implementing stage 9

| Plugin call | Generator behavior now / adapter-only translation later |
| --- | --- |
| `initialize_vllm_model` | Construction mapping to `Model(open_mesh)` plus `QwenGenerator` with validated capacity/cache. Registration, architecture aliases and loader class are stage 9. |
| `get_max_tokens_all_users` | Export measured total KV-token budget; classmethod spelling is adapter work, accurate budget is full-model work. |
| `get_kv_cache_spec` | Optional one homogeneous attention-group design described above. The optional hook may be omitted if pinned HF/vLLM metadata gives the correct one-group geometry. |
| `allocate_kv_cache_per_layer` / alternative `allocate_kv_cache` | Translate specs to physical selected-dtype caches and owned recurrent state, preserving sharing/identity. Underlying explicit-cache support must already exist and be tested. |
| `warmup_model_prefill`, `warmup_model_decode` | Map to generator-wide two-phase preparation, never independently prepare model-owned buffers after capture. |
| `prefill_forward`, `batched_prefill` | Existing explicit API accepts mixed absolute starts/ends and slots; require per-row-equals-solo evidence and completion of low-level sampling semantics. Native uniform batching is separate from the correctness of one multi-row call. |
| `decode_forward` | Map absent `sampling_params` to explicit `sample_on_device=False`; present params to device path. Plugin does **not** send the generator's custom `sample_on_device` keyword. This flag translation is adapter work, not a model rewrite. |
| `read_decode_output` | May be absent with synchronous host Torch outputs. `MODEL_EXECUTION.md` permits that even when front-packed submission requested `read_from_device=False`. Do not claim asynchronous DMA support. |
| `process_decode_output_host` | May be absent when decode already returns `[B,1,V]` or `[B,1]` host Torch output. Otherwise a thin layout/readback conversion is adapter mapping. |
| `release_request`, `slot_remap` | Actual recurrent and sampler ownership are generator requirements; fixed buffer/permutation evidence is required now. |
| `note_state_slots_moved` | Optional identity-metadata callback. Can be omitted when all ownership is already moved by the generator and there is no separate session map. Never gather twice. |
| `release_persistent_capture` | Map to generator `close()` while mesh remains open. The mesh belongs to the worker; close must not reset or close it. |
| `decode_input_update_contract` | Version 1; all four reload commands, validation-before-mutation, explicit legacy reset rejection, and seeded state semantics are generator work now. |
| `spec_plan`, `propose_draft_tokens`, speculative wide `decode_forward` | Not supported through `supports_spec_decode=False`; no implementation required. |
| `enable_device_grammar`, `device_grammar_enabled`, `sample_deferred_decode` | Not supported through `supports_device_grammar=False`; host grammar uses ordinary full logits. |
| `pop_stage_outputs` | Not supported through `supports_stage_outputs=False`. |

Add rows for the alternative allocator and conditional extension hooks instead
of leaving their absence implicit. `read_decode_output` being absent is not a
missing row when the documented host-output alternative is explicitly mapped.
The stage's `supported` evidence may include a design artifact for the adapter
mapping, but that artifact must point to existing tested generator behavior;
it must not describe absent model logic as future wrapper work.

## Host compatibility is real support, not a device claim

On this 4-device TP mesh, `TTModelRunner.check_perform_device_sampling` routes
**every logprob request, including 0**, to host because device logprobs require
devices-per-rank in `{8,32}`. No new capability is necessary. It also routes
all active penalties to host when `supports_device_penalties=False`; min_p,
bad_words, logit_bias, allowed_token_ids, min_tokens, and active/custom logits
processors are always host-only. Structured requests use host grammar when
device grammar is false. The generator must return unfiltered full-vocabulary
last-position logits without advancing its sampler on this path, then consume
the externally sampled token on the next authoritative decode.

Thus request-feature rows for logprobs, the three penalties, token filters,
and structured outputs should identify the explicit compatibility boundary
and its evidence. They are not globally unsupported requests and do not need
custom device implementations. Their exact transformation, HTTP formatting,
EOS/stop/max-token enforcement and request scheduler belong to existing vLLM
code; implementing another host sampler in this full-model stage is unnecessary.
This is ordinary documented division of responsibility, not a `plugin_owned`
gap requiring a plugin patch or a justification for changing it.

`logprobs=0`, top-N, all-vocabulary logprobs and prompt logprobs are distinct.
The first three can use full logits subject to the server's configured limit;
the pinned plugin README explicitly rejects **prompt logprobs** at request
validation. Add the latter exclusion rather than falsely claiming it.

Host and device RNG streams need not match. The plugin documents
`seeded_sampling_policy="host"` for API-seeded requests requiring consistent
host RNG ownership when companions change batch-wide routing. This existing
option does not excuse broken device seeded reloads. The mixed isolation test
assumes companions cannot change sampler route, but with device penalties
disabled its penalized companions do change it; interpret such tests under an
explicit route policy and separately prove seeded device behavior at k<=32.

## Test inventory and implications

The ordinary `tests/tt` suite was read by source, not executed. It uses an
already-running server and requires no new stage-6 server. Its direct model
requirements and later served evidence are:

| Source | Model contract exercised |
| --- | --- |
| `test_seeding_and_variety.py` | Seeded repeats, shuffled slots, seed0/-1, unseeded variety, mixed temperatures, k=1 semantic greedy, k=10 stochastic. Most seed tests use k=50 and therefore test **host** sampling under max_device_top_k32; add model-local k<=32 proof. |
| `test_request_isolation.py` | Per-request mixed sampling parameters and shuffled request order; account for batch-wide route changes described above. |
| `test_tt_penalties.py` | Per-request repetition/presence/frequency penalties and unchanged unpenalized neighbors. With explicit false device capability, full-logit host compatibility covers these. Some output-change assertions depend on HF confidence and need the user's matched HF qualitative control before investigation. |
| `test_logprobs.py` | Sampled logprob0, top-N 1/3/5/10, all-vocabulary chat mode, requests beyond one hardware batch. Raw full-vocabulary logits are the required model boundary. |
| `test_host_only_params.py` | min_p, bad words, bias, allowlist, minimum tokens. Existing host processing; do not claim the high-level diagnostic host-argmax mode implements all of these. |
| `test_structured_output_dp1.py` | Choice/regex/JSON mixed with plain requests and two waves of slot reuse. Host grammar is legitimate; claiming device grammar additionally requires device-grammar log evidence. |
| `test_chunked_prefill.py` | Solo and long recall plus four concurrent ragged chunk continuations. Tests skip without declared serving budget and cannot themselves prove chunking was enabled. Generator resume/mixed-row proof is required now; server logs are stage 9. |
| `test_config.py`, `test_tile_size.py` | Plugin config parsing and 32-token tile constant; no additional model operator required. |
| `tests/tt/spec/*` | Enumerated all test functions and read suite README plus the full speculative contract. These are opt-in speculative/dummy-model regression tests, skipped with spec-k0. They do not require Qwen to implement speculation when its documented capability is false. Ordinary release/remap/preemption responsibilities remain relevant independently. |

Add explicit request rows for custom logits processors, prompt_logprobs,
preemption/resume, release/reuse/cancellation, stop/EOS/max_tokens, request
seed width, optional multimodal/mRoPE exclusion, and trace mode if the final
contract intends complete request coverage. For stop/EOS/length, the existing
one-token low-level API leaves policy to the scheduler; show that it does not
internally generate extra committed tokens or reset neighboring requests.
Do not invent `supports_original_prompt_lens`: that declaration is not in
this pinned plugin's documented capabilities.

## Sources read for this audit

- Full `docs/MODEL_INTERFACE.md`, `MODEL_SETUP.md`, `MODEL_CAPABILITIES.md`,
  `DEVICE_GRAMMAR_CONTRACT.md`, and `SPEC_DECODE_CONTRACT.md`.
- `MODEL_EXECUTION.md` call/output/sampling sections, `MODEL_LIFECYCLE.md`
  from the preceding slot investigation, `DECODE_RELOAD_CONTRACT.md` commands,
  transition/input-authority and requirements sections, and all relevant
  `SCHEDULING.md` ordinary, chunked, async, lane, DP, block/spec sections.
- README sampling routing/request seeds/host readback and prompt-logprob
  restrictions; `diffusion-gemma.md` is a linked block-output example and is
  outside this ordinary one-token text model's selected extension contract.
- `src/vllm_tt_plugin/model_runner.py`: cache grouping/allocation/table
  expansion, parameter padding, sampling route and prefill submission;
  `async_decode.py`: decode keyword/reload/history delivery; `worker.py`:
  default homogeneous spec and sizing interface.
- Ordinary test sources listed above, `tests/tt/conftest.py`, request utility
  definitions, speculative suite README and speculative test function inventory.
- Current generator/model/cache ownership sources and serving JSON; selected
  full-model skill and its prefill-features reference.

The final JSON still needs evidence-bearing statuses in place of `missing`.
This audit is a map of defensible closures and concrete remaining work; it is
not itself a clean-pass verdict or a substitute for the required stage review.
