# AutoDebug: minimum Qwen/Qwen3.8-27B vLLM integration

Date: 2026-10-25. Inspection only; no model execution, hardware access, server launch, source repair, or AutoDebug relaunch.

**Two concrete defects in the inspected baseline deserve attention: the generator uploads temperature instead of inverse temperature, and the pinned plugin can silently relocate a continuing prefill's state ownership without moving its recurrent state.** Most remaining findings are missing adapter contracts, not failures of the standalone generator. At investigation start, no vLLM adapter existed in the model directory; this report does not claim an observed serving failure. Concurrent repair files appeared during final review and are outside this baseline diagnosis.

The minimum implementation should keep the existing Model, Generator, Sampling1D, external-state binding, and split model/sampling traces. Add a thin serving adapter and narrow generator helpers for request-state ownership and explicit decode updates. A correct synchronous path can precede resident async serving. Automatic prefix caching and scheduler-driven chunked prefill require additional work; merely enabling their existing capability flags is incorrect.

## Evidence and revision boundary

- tt-metal baseline/HEAD: `fe82645a041cf665a621dc6211c0e9354edb995b`. All model-source line references below refer to that committed version, which matched the initial inspection. Baseline generator SHA-256: `2ff93d3cd0774c683c0c01ca67073f13324ea88687f2c7d3aee9aa8a745c130c`.
- Local plugin HEAD: `1badbd02bb6911dfdb59c206bdc00c6c1e8dc0bf`, clean nested worktree at initial inspection, at `bringup/dependencies/vllm-tt-plugin`. Its installation script pins vLLM 0.30.0 (`docs/install-vllm-tt.sh:42`). Plugin references also identify that baseline; concurrent test edits appeared later. This report uses that plugin's documentation and implementation, not the deprecated Tenstorrent vLLM fork.
- Existing container `qwen38-functional-build`: read-only source/metadata inspection through `docker exec ... sed/cat` found `/opt/venv/lib/python3.10/site-packages/vllm-0.30.0+empty.dist-info/METADATA` reporting `0.30.0+empty`. No Python package import or execution of vLLM was needed. An installed vLLM commit was not established.
- Local checkpoint `.cache/qwen38-full/config.json`: architecture `Qwen3_5ForConditionalGeneration`; 64 text layers, comprising 16 full-attention and 48 linear-attention layers; four KV heads, head dimension 256; maximum context 262144. These are local checkpoint facts, not a claim about a newly downloaded revision.
- References below abbreviate the model directory as **M** = `models/autoports/qwen_qwen3_8_27b`, and the plugin directory as **P** = `bringup/dependencies/vllm-tt-plugin`. Use `git show <baseline>:<path>` for model references if the working files have changed.
- During report review, another writer modified `M/tt/generator.py` and created `M/tt/generator_vllm.py` and `M/tests/probe_serving_state.py`. One observed change already reciprocates the public temperature. Those evolving changes were neither made nor validated by this investigation; findings below describe the original intervention requirements, not outstanding defects asserted against that newer work.

Direct observations are source contracts and control flow. Predicted bad tokens, stale-state effects, and distribution changes follow from those contracts; hardware reproduction and serving validation remain outstanding. Existing probe results were not rerun.

## 1. Confirmed generator bug: public temperature has the wrong polarity

`M/tt/generator.py:153-175` accepts `temperature`, validates positive finite values, and copies that value directly into `self.temp`. `_sample` passes it unchanged to the common Sampling1D module (`:140-145`). Sampling1D passes the tensor to `ttnn.sampling` (`models/common/modules/sampling/sampling_1d.py:438-454`).

The lower contract is unambiguous:

- `ttnn/cpp/ttnn/operations/reduction/sampling/sampling_nanobind.cpp:45` documents the tensor as **1/T**.
- `.../device/kernels/dataflow/writer_interleaved.cpp:94-106` broadcasts the supplied BF16 value without reciprocating it.
- `.../device/kernels/compute/sampling.cpp:480-493` multiplies selected logits by that value before softmax.
- Both the native common formatter (`models/common/modules/sampling/params.py:197`) and the older serving formatter (`models/common/sampling/generator.py:832-839`) implement the reciprocal conversion.

For logits `[0, 1]`, `k=2`, `p=1`, and public `T=2`, the intended larger-token probability is `sigmoid(0.5) = 0.622459...`; the current path computes `sigmoid(2) = 0.880797...`. At `T=0.5`, the outcomes reverse. This is a wrong distribution, not a low-precision tolerance question. `T=1` and greedy `k=1` conceal the error.

Counterevidence considered: perhaps the setter intentionally takes inverse temperature. `M/doc/full_model/sampling_contract.md:19-23` calls its input temperature and cites temperature 2 without such a convention. `M/tests/probe_trace_contract.py:45-49` checks reset determinism at that value, which both distributions satisfy. `probe_sampling.py` checks greedy selection, also unaffected.

**Smallest repair:** normalize public temperature once in the shared generator parameter path, preserving direct Sampling1D reuse. For serving, also translate vLLM `T=0` into `(k=1,p=0,inverse_temperature=1)`; the current setter rejects zero. Avoid double reciprocation if a common formatter has already prepared device parameters. Test `T=0.5`, `1`, `2`, mixed greedy/stochastic rows, and the zero-temperature convention with captured upload values; later use synthetic logits to validate the distribution direction on device.

## 2. Public decode is host-authoritative; the resident mechanism already exists

The implementation has two different interfaces that must not be conflated:

| Path | Input authority and behavior |
|---|---|
| `Generator._decode`, `_sample` (`M/tt/generator.py:127-145`) | Consume persistent device tokens/positions/RoPE; model trace advances position once; sampler writes directly into the same token input. The selected top-k branch advances seeds; the optional argmax branch returns before that increment. |
| `Generator.generate` (`:584-596,603-613,661-667`) | Uses those traces directly. Ordinary device sampling does not reconstruct token feedback from host readback. |
| Public `Generator.decode_forward` (`:503-550`) | Always reads host positions, chooses the active trace, and uploads active mask, token, position, and RoPE before replay. Extra reload keywords are swallowed by unused `**kwargs`. |
| `Model.decode_forward` (`M/tt/model.py:124-131`) | Device computation only; no host refresh. There is no reason to rewrite its mathematical layer loop for resident feedback. |

The plugin's version-1 reload planner deliberately permits stale host tokens/positions on steady device decode (`P/src/vllm_tt_plugin/async_decode.py:415-463`). After one step consumes `(t0,L)`, the resident state contains `(t1,L+1)`. Calling today's public method with stale `(t0,L)` and `reload_inputs=False` still reloads and consumes `(t0,L)` again. Attention revisits position L; the linear recurrence advances again from its already-updated state. This is a direct future adapter mismatch, not evidence that standalone `generate()` is broken.

**Minimum extension:** make the serving-facing update commands explicit and delegate them into generator-owned execution:

1. Apply any full-capacity slot permutation before using request state.
2. `reload_inputs=True`: copy authoritative tokens, positions, RoPE inputs, masks, and required page mapping; establish retained active-slot metadata.
3. `reload_inputs=False`: derive no forward input, active mask, or RNG counter from the host token/position arguments. Use resident values and retained metadata.
4. `reload_page_table=True`: update only the persistent page map, independently of forward-input reload. Existing `refresh_page_table` is reusable (`M/tt/generator.py:118-125`).
5. Honor `reload_sampling_params` and `reset_sampling_state` independently. The current combined setter resets seed buffers whenever parameters are changed and is not this four-command protocol.
6. Return logits when the plugin chooses host sampling; otherwise replay the canonical sampler once. Do not sample during readback.

Advertise `decode_input_update_contract=1` only when the implementation handles those commands. The plugin supplies them at `async_decode.py:1315-1323`; no new plugin configuration is needed. Reject legacy `reset_batch` even if the signature accepts `**kwargs`. For direct callers, safe defaults are `reload_inputs=True` and the other three commands False; full reload already includes page tables, and sampling-state reset requires authoritative inputs (`P/docs/DECODE_RELOAD_CONTRACT.md:21-39`). A synchronous version-1 adapter may keep `supports_async_decode=False`, require full reload, and reject unsupported resident combinations explicitly (same document, Partial adapters). It must still implement slot movement and sampling-state commands; disabling async does not solve request ownership.

### Submission and completion boundaries

`execute_trace(...blocking=False)` at `M/tt/generator.py:541,545` enqueues work. `return_device=True` returns a reusable persistent device tensor, not a completed independent result. `host_logits` and the token `to_torch` path (`:385-387,549-550`) perform host readback. Preparation explicitly synchronizes at `:294,300,357`.

The serving adapter must translate `read_from_device`, not silently ignore it. A minimal synchronous adapter may materialize and return host tensors, which the plugin accepts even when it requested deferred output. A genuine async adapter needs `read_decode_output(output, async_read=True)`, with event-backed host storage for each submission, plus `process_decode_output_host` when retaining TTNN host handles; the plugin skips that conversion if output is already torch. Enqueue the step-N read before any step-N+1 overwrite of its persistent output; retain transfer storage until consumption. The plugin waits for read events before conversion (`P/src/vllm_tt_plugin/async_decode.py:1403-1427,1531-1559`; reference implementation `models/tt_transformers/tt/generator.py:3421`). On another command queue, establish an explicit producer/read/reuse dependency.

Tokens are replicated `[1,1,1,B]`; read one replica and return `[B,1]`. Logits are vocabulary-sharded `[1,1,B,V/4]`; concatenate the four vocabulary shards, trim real vocabulary, and return `[B,1,V]`. Preserve submitted-row order. A bare persistent `self.tokens` handle retained until after another replay is not a safe output snapshot.

## 3. Preserve the chosen canonical sampler; add serving state deliberately

The existing generator already uses common Sampling1D and its `tt_out_tok` mechanism. `M/doc/full_model/sampling_contract.md:3-18` explicitly chooses it over the older SamplingGenerator because the generator owns TP4 buffers and traces. Migration to the older wrapper is not a prerequisite for correctness. Existing performance comparisons are historical evidence only, not validation performed here.

The missing serving layer includes:

- Per-row vLLM controls, including greedy temperature zero, stochastic top-k bounds, unspecified seed `None`, and request-owned RNG lifetime. The baseline setter supports only numeric seeds in a reserved uint32 range, defaults to 1234, and `_sample` increments all seed entries on the selected top-k path (`M/tt/generator.py:153-175,140-145`). Reapplying the original seed whenever any batch member changes would restart existing streams.
- Remapping sampler state even while host sampling has temporarily made the device sampler dormant. Initialize/reset state only as commanded, including unseeded state; do not consult stale positions during resident decode.
- Generator-owned prefill sampling: reuse the existing terminal-logits copy followed by the sample trace (`:584-594`). The adapter should not create an unrelated token-selection implementation.
- A real host-logits path. The plugin omits `sampling_params` when it chooses host sampling. Current public decode defaults to sampled token output, so merely forwarding plugin kwargs returns the wrong output kind. Standalone `generate(host_sampling=True)` is greedy-only and is not the vLLM host sampling implementation.

Existing capabilities already express an initial device boundary: `supports_sample_on_device=True` only after the path works, `max_device_top_k=32`, `supports_device_penalties=False` until penalties are implemented, and no device grammar declaration. The penalty default is **True**, so omitting it would misrepresent this sampler. The plugin routes unbounded/larger stochastic top-k, active unsupported penalties, host-only processors, and TP4 logprob requests to its host sampler (`P/src/vllm_tt_plugin/model_runner.py:3617-3716`). Do not clamp unrestricted vLLM sampling to 32 candidates and claim equivalent behavior.

Native `SamplingState1D` and `SeedManager1D` offer reusable request-state mechanisms if implementing broader device sampling (`models/common/modules/sampling/sampling_state_1d.py:63,529-558,731`; `seed_manager_1d.py:308-371`). They are not a constructor-only replacement: prepared metadata capacity is 32 here, while kernel k/p/temp tensors must match actual logits rows B (`sampling_device_operation.cpp:155-159`). Capture uses the device-only body; dynamic host seed preparation stays outside capture. Reuse that normalization/state code selectively or integrate it with explicit B1/B32 tests and preparation-state restoration. A seed-buffer update is distinct from a host token-feedback loop.

In particular, `SeedManager1D.synchronize(reset_batch=True)` preserves unchanged streams (`seed_manager_1d.py:265-306`); its flag is not a direct implementation of every plugin reset command. Implement commanded initialization and position realignment explicitly, including None seeds, and omit stale host positions during resident steps. Use the existing `salt_duplicate_seeds=False` setting for independent vLLM requests with equal seeds (`sampling_state_1d.py:71-73`).

## 4. External attention KV and recurrent state need different ownership

External binding is already implemented (`M/tt/generator.py:32-54,435-450,505-514`). Generator preparation even warms external pools through temporary states and restores request buffers before capture (`:177-364`). Replacing this with an independent private serving KV pool is unnecessary and would break scheduler ownership.

The serving allocator should return the complete opaque **64-entry model state list**, in model-layer order:

| State per TP4 device | Required allocation and ownership |
|---|---|
| 16 attention layers: key/value each `[num_blocks,1,32,256]` | Use the scheduler's exact physical block count and unchanged IDs, with selected model KV precision. |
| 48 linear layers: conv `[B,3,2560]`, BF16 row major | Fixed request-state slots, independent of attention page IDs. |
| 48 linear layers: recurrent `[B,12,128,128]`, FP32 tiled | Same slot ownership as conv; preserve precision and trace-bound addresses. |

Evidence: `M/tt/multichip_decoder.py:507-522`, `M/tt/model.py:71-72,103-107,127-129`. The ordinary attention allocator derives `B*ceil(context/256)*8` pages, so it cannot be reused unchanged when the scheduler allocates a different physical pool. Extend allocation narrowly to accept external attention geometry, or allocate PagedState in the adapter and bind it before Generator preparation. The plugin passes allocator results back as opaque storage (`P/src/vllm_tt_plugin/model_runner.py:764-794`).

Use the existing hybrid model identity `is_hybrid=True` so installed vLLM counts the 16 full-attention layers; otherwise its transformer path counts all 64. This is an existing upstream model-interface attribute, not a proposed plugin capability. Installed vLLM `config/model.py:1594-1609,1638-1647` establishes the distinction. Validate the plugin's 16 attention allocation specifications and map compact indices 0..15 to original model layers 3,7,...,63. If later emitting per-layer cache specifications, their parsed indices must be compact: the plugin rejects indices outside the attention count (`model_runner.py:872-889`). A homogeneous attention group plus separately owned linear state is sufficient initially. The plugin's generic hybrid KV path accepts AttentionSpec/UniformTypeKVCacheSpecs; it does not implement a general recurrent-state/Mamba allocator (`:730-747,875-876`).

Keep 32-token page geometry for this model. The persistent page-map width must cover configured context and be a multiple of eight int32 entries (`M/tt/generator.py:49-54`), e.g. `8*ceil(C/256)`. Scatter scheduler rows into fixed destination slots and pad unused columns with reserved null block **0**, preserving all live IDs. Allocate that physical K/V page and initialize it to finite zeros. Do not manufacture private block IDs or use arbitrary `-1` tails. The plugin already clears unused table entries to zero (`P/src/vllm_tt_plugin/input_batch.py:727-756`). SDPA reads a whole K chunk before masking and its page lookup casts IDs to uint32 without an invalid-page check (`ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp:60-69`; `sdpa/device/kernels/dataflow/dataflow_common.hpp:64-68`). Inactive `current_pos=-1` is a separate whole-row skip. Validate that live positions have real mapped blocks; null padding is not additional request capacity.

Provide `get_max_tokens_all_users` with a justified pool budget. The plugin otherwise defaults to a base budget of 131072 tokens, adds per-user output reservation, and reports synthetic memory rather than profiling free DRAM (`P/src/vllm_tt_plugin/worker.py:438-464,730-770`). Include weights, attention pools, fixed linear state, persistent prefill buckets, traces, sampling buffers, and preparation scratch. Linear state alone is `48*(3*2560*2 + 12*128*128*4) = 38,486,016` bytes per slot per device, before allocator overhead. A 262144 advertised context does not imply that many tokens can be allocated for every concurrent request. Preserve `M/tt/precision.py`'s selected-policy constructor and explicitly reconcile scheduler dtype metadata with the physical TT cache dtype.

### Required slot operations

Prefill row i must write linear state into `empty_slots[i]`, not necessarily i. New request admission must zero only the destination linear slot; continuation must retain it. `Generator.reset()` clears every tensor in every state, including external attention pools (`M/tt/generator.py:366-383`); it is unsuitable for per-request admission/release.

Decode `slot_remap` is a simultaneous **destination-to-old-source** permutation across the full state capacity. Example: old slots `[A,B,unused,unused]`, submitted decode `[B]`, remap `[1,0,2,3]`. B moves to slot 0 and off-batch A must survive in slot 1. Gather conv/recurrent and any slot-owned sampler/session state through temporary storage, then copy back in place. Do not gather physical attention pages by slot. Preserve old sources until all destinations are formed, and prewarm any device operations before retaining traces.

The plugin builds and commits the full permutation correctly (`P/src/vllm_tt_plugin/model_runner.py:1311-1383`; `async_decode.py:1310-1311,1392-1395`), but today's generator ignores it through `**kwargs`. `note_state_slots_moved` may update host bookkeeping after acceptance; it must not apply the device permutation a second time. `release_request` receives the current owner slot. Keep request-scoped clearing separate from model-lifetime resource release (`P/docs/MODEL_LIFECYCLE.md:42-75`).

## 5. Confirmed pinned-plugin defect: continuing prefill can be reassigned without state transport

`P/src/vllm_tt_plugin/model_runner.py:1268-1309` excludes all currently prefilling requests from `held`, prefers each packed row as its slot, and overwrites `_req_state_slot`. It does not preserve the old slot of a continuing request. Prefill receives only destination `empty_slots`; the source permutation is computed only on decode (`:2450-2456,3761-3773`).

Concrete sequence:

1. A owns slot 0 after finishing prefill; long request B owns slot 1 after a partial prefill.
2. Decode interleaving lets A run while B is still partial. A finishes; its slot claim is released.
3. B's next prompt chunk is packed alone at row 0. `_alloc_prefill_state_slots([B])` produces `[0]` and changes B's recorded owner from 1 to 0.
4. B's convolution/recurrent tensors still reside at slot 1. An otherwise correct incremental adapter trusts destination slot 0, continues from wrong state, and no later decode remap recovers the lost ownership.

The sequence is supported by decode interleaving/partial-prefill hiding (`P/src/vllm_tt_plugin/scheduler.py:806-867,920-944`) and finish processing (`model_runner.py:970-1006`). It does not depend on an imagined order of unrelated logs. A small inspection-only transcription of the allocator produced `{B:1} -> empty_slots=[0] -> {B:0}`; no plugin execution was used. A model that reconstructs every prefix from zero might conceal the bug, but the target generator's continuation consumes prior linear state.

**Smallest repair before enabling incremental chunked serving:** preserve every continuing request's valid existing slot. Reserve all existing owners before assigning slots to new requests, including continuing requests encountered later in the submitted list. Then allocate only genuinely unowned requests. No new capability or argument is needed. Add host regression tests for the sequence above, reordered continuing rows, and a new row preceding a continuing row. Existing `P/tests/test_state_slots.py` covers substantial decode movement and fresh-prefill isolation, but its re-prefill example already owns slot 0 and misses this transition.

Runtime manifestation is unverified because no target adapter or serving run was executed. The ownership contradiction itself is established by source.

## 6. Chunked prefill needs semantic translation and a real offset policy

Plugin runtime prepares **full history**, one absolute start per submitted row, and exclusive chunk ends (`P/src/vllm_tt_plugin/model_runner.py:2103-2127,3728-3773`). Core Generator expects chunk-local tokens, one shared scalar `start_position`, and local lengths for every fixed slot (`M/tt/generator.py:432-445`). It silently ignores plugin `start_pos` in `**kwargs`.

For plugin `(start_pos=128,prompt_lens=192)`, the correct computation is the 64 tokens `tokens[128:192]` at absolute positions 128..191. A direct forward uses 192 tokens from zero. Merely renaming start_pos without subtracting it from prompt_lens is also wrong. The adapter must group compatible starts, slice actual chunks, scatter them into `empty_slots`, preserve other live slots, and gather terminal outputs back into submitted order. Group outputs must be retained before another call overwrites `self.prefill_output` (`M/tt/generator.py:446,487-493`).

The baseline core requires a nonnegative multiple-of-128 start. Its internal 4096-token splitting and physical rounding do not make arbitrary scheduler chunks supported. The plugin's existing `tt.prefill_chunk_alignment` defaults to 128 (`P/src/vllm_tt_plugin/config.py:658-681`), but its scheduler explicitly permits progress off that grid when a running request has insufficient residual budget (`scheduler.py:763-785`). Example: start 128, residual budget 64 after earlier scheduled work, prompt end 512 returns a 64-token partial chunk; next start is 192 and the core rejects it. A final one-token chunk is still prefill, not decode. The helper also contains a permanent-small-cap branch, but constructor validation rejects such caps on normal aligned launches (`scheduler.py:265-281,356-364`); that branch is not evidence of another live failure case.

Safe choices are:

- Initial path: explicitly leave scheduler chunked prefill unimplemented/disabled with the existing false declaration; full prompt calls can still use the generator's internal chunk walk. This is a staged integration limitation, not evidence that chunking is impossible or a complete implementation of it.
- Full incremental support: fix continuation slot ownership, normalize per-row start/end/slot semantics, and implement arbitrary reachable offsets. Options include a generator-owned teacher-forced logits-only decode bridge for a short unaligned span before resuming aligned prefill, or a properly validated prefill kernel path. Do not round the start backward and replay tokens into already-advanced recurrent state. An alternative stricter scheduler policy must address progress and leftover budgets explicitly, not assume the existing alignment option guarantees it.

Intermediate prefill results must not advance device RNG. The plugin forces host sampling for a submitted prefill containing intermediate chunks and suppresses those intermediate outputs (`P/docs/MODEL_EXECUTION.md:57-61`); respect that route. Preemption/resume with prefix caching disabled rebuilds state from token history, including generated tokens, after clearing only the replacement request's linear slot.

## 7. Attention page reuse does not implement automatic prefix caching

A prefix hit at position 128 requires both attention K/V and each linear layer's conv/recurrent state **after exactly 128 tokens**. Zero/new/stale state plus valid K/V is insufficient: the next convolution consumes history, and recurrence consumes its initial state (`M/tt/multichip_decoder.py:605-622,665-667,695-697`; `functional_decoder.py:240-245`). Changing the page table cannot reconstruct either field.

There is no prefix-keyed boundary checkpoint, restore, immutable sharing, or eviction lifecycle in this baseline generator. The full-state snapshot helper in `M/tt/optimized_decoder.py:218-222` is not such a cache. The plugin's existing attention-only cache interface does not supply these missing semantics.

**Minimum truthful declaration is `supports_prefix_caching=False`.** If prefix reuse becomes a requirement, implement exact-boundary snapshots of all linear fields, restoration into independent request slots, identity/invalidation tied to actual prefixes, and eviction/reuse tests. Account for checkpoint memory and supported hit boundaries. Disabling automatic prefix caching does not remove the need for ordinary paged KV, remapping, slot reuse, or preemption/resume.

## 8. Adapter/setup work required even for the initial synchronous path

| Boundary | Minimum concrete work |
|---|---|
| Registration | Add the model's serving class using existing registration. The local checkpoint names `Qwen3_5ForConditionalGeneration`, not a guessed Qwen3.8 Python class. The plugin's built-in target is `models.demos.blackhole.qwen36.tt.qwen36_vllm:Qwen36ForCausalLM`, whose module is absent in this checkout (`P/src/vllm_tt_plugin/platform.py:1443-1456`). A bare-name `TT_MODEL_CLASS_OVERRIDES` entry registers both bare and TT-prefixed aliases (`:1347-1362`) and makes early model inspection select the intended class. |
| Registry inspection | Satisfy installed vLLM's structural text-generation interface: constructor accepting `vllm_config`, callable `embed_input_ids`, `forward` accepting `input_ids` and `positions`, and `compute_logits`. The plugin uses its custom factory/forward methods instead; compatibility methods may raise clearly if invoked unexpectedly. `nn.Module` inheritance is not itself required by these checks. Evidence: installed `vllm/model_executor/models/interfaces_base.py:59-109,135-144`, `registry.py:885`. |
| Text-only configuration | Implement the text path without advertising SupportsMultiModal. Do not mutate checkpoint architecture or RoPE data just to bypass configuration. Validate the actual checkpoint, TP4 1x4 mesh, configured batch 1..32, and unsupported lane/topology requests. |
| Construction | Implement `initialize_vllm_model(hf_config,mesh_device,max_batch_size,max_seq_len,tt_data_parallel,optimizations)` using the existing Model/precision path. Delay Generator cache binding/capture until the plugin invokes allocation; `build_generator()` otherwise allocates/prepares immediately. Keep model modules importable in all resolving processes, including the existing readiness-contract dependency. |
| Allocation | Bind the exact external attention pool plus complete linear state list before preparation; expose a justified KV budget, 32-token page geometry, and fixed padded map. |
| Batch shapes | Use one prepared fixed B capacity initially and pad inactive decode rows with position -1. If publishing `tt_supported_decode_batch_sizes`, publish only prepared capacities. Prefill still needs packed-to-slot scattering. |
| Warmup | Implement both plugin warmup methods. Plugin calls eager prefill/decode first and later trace warmup; prepare every bucket/slot operation actually supported before capture. Respect requested execution modes; explicitly reject an unimplemented mode rather than ignore `enable_trace`. Initial `trace_mode=decode_only` matches existing eager prefill plus traced decode. |
| Sampling/output | Explicitly distinguish host logits and device tokens, normalize vLLM parameters, and provide correct output shapes. Initially synchronous host sampling is a correctness baseline; it is not evidence of resident-device serving performance. |
| Lifetime | Implement idempotent `release_persistent_capture` while the mesh is open. Release all unique decode/sample/buffered traces after queued work/readback completes; do not close the worker-owned mesh. Existing exception cleanup is not a normal serving shutdown hook. |

Sources for call order: `P/docs/MODEL_INTERFACE.md`, `MODEL_SETUP.md:34-69,119-146`, `MODEL_LIFECYCLE.md:9-37,98-109`; core default builder `M/tt/generator.py:695-704`.

**Text-only MRoPE is still part of this checkpoint's serving interface.** Installed vLLM `config/model.py:1852-1853` and `transformers_utils/config.py:595-631` detect nested `mrope_section` regardless of multimodal class support. The plugin then unpacks prefill as `(output,rope_deltas)` (`P/src/vllm_tt_plugin/model_runner.py:332,3775-3780`) and forwards `rope_deltas_all_users` on decode (`async_decode.py:1343-1357`). The model already constructs identical three-axis text positions (`M/tt/model.py:59-64`). Return correctly ordered zero text deltas and accept/validate the corresponding decode metadata; returning bare logits under this config is incompatible. This does not establish image/video support.

Use existing capability keys and configuration. No new capability, environment read, model-name allowlist, or additional-config option is justified by these findings. The plugin's AGENTS.md also requires specific authorization and necessity analysis before adding any such interface; this inspection added none.

## 9. Minimum discriminating tests

These are proposed validation, **not tests run in this investigation**.

| Test | What must distinguish passing from failing |
|---|---|
| Host sampling normalization | Capture device parameters for T=.5/1/2/0, per-row k/p, None/seeded inputs, unrestricted k routing, and penalties. Reciprocal bug must fail the old implementation. |
| Host plugin continuation slots | A finishes while B is partial at another slot; B continuation preserves its prior slot. Add new-before-existing and reordered continuations. |
| Host adapter input translation | Start/end pairs `(0,33)`, `(128,192)`, `(256,257)`, heterogeneous starts, nonidentity empty_slots, shorter submitted batch than capacity. Verify slices, absolute positions, output ordering, and zero deltas. |
| Host allocation/metadata | Verify 16 attention specs become the correct entries in a 64-layer state list; exact pool count, compact-to-original mapping, map width, null padding, and unsupported geometry rejection. |
| Host registry/config | Resolve the actual bare and TT-prefixed architecture, classify as a text-generation hybrid model, count 16 attention layers, retain text MRoPE metadata, and avoid accidental multimodal registration. |
| Device steady decode | Compare serial authoritative calls with repeated public no-reload calls whose host inputs are deliberately poisoned. Tokens, positions, KV writes, and recurrence must agree; token/position refresh counters must stay unchanged on resident steps. |
| Device page-only refresh | Allocate a newly consumed page, change only that mapping, then repeat unchanged maps. Verify correct attention and no token/position/RoPE refresh; use nontrivial physical IDs. |
| Device slot lifecycle | Nonzero independent requests, new admission while peers are live, cyclic full-capacity remaps with off-batch owners, completion/reuse, preemption/resume, and host/device sampling transitions. Compare against independent execution, not just inactive-row equality. |
| Device sampling/readback | Seeded/unseeded streams, unchanged requests across batch changes, prefill-to-decode, dormant sampler remap; delayed step-N read consumed after N+1 submit must still contain N's output. Include B1 and B32. |
| Device chunking | Whole prompt versus split continuation at 128, odd-128 offsets, 4096 boundaries, ragged final 1/31/33/127-token pieces, and an off-grid scheduler budget. Compare conv/recurrent state, logically initialized attention KV, next decode logits, and untouched live peers. Unused/padded KV bytes need not match; intermediate chunks must not sample. |
| Prefix caching, if implemented | Cold versus hit at each supported boundary, simultaneous sharers with independent recurrent slots, eviction, block reuse, and preemption. Matching attention pages alone cannot pass. |
| Preparation/shutdown | External live-state preservation, prepare failure cleanup, remap/clear operator warmup, address stability, and release after completion before mesh close. |

Relevant existing tests to extend/read: `M/tests/probe_trace_contract.py`, `probe_mixed_slots.py`, `probe_external_prepare.py`, `probe_sampling.py`; `P/tests/test_state_slots.py`, `test_decode_reload_contract.py`, `test_prefill_chunk_alignment.py`, `test_chunked_prefill_policy.py`, `test_seeded_sampling_policy.py`. Plugin host tests have import-only TTNN stubs. Hardware tests must be separate and explicitly synchronized at the observation boundary.

## 10. Report review and other potential issues

The first-draft claims were checked against producers and consumers, rather than accepting similar-looking names or existing test labels:

- **Temperature stays a headline bug:** traced public input through Sampling1D, binding, dataflow, and compute; the documented setter is not an inverse-scale API. Reset determinism cannot falsify it.
- **Continuation slot loss stays a conditional headline bug:** the plugin changes ownership without transporting state on an allowed schedule. The missing adapter prevents a current serving reproduction, not the source contradiction.
- **Host refresh is an integration gap:** correct for the existing host-authoritative low-level API; wrong only if presented as version-1 resident decode. The internal standalone loop already supplies resident feedback.
- **External cache binding is present:** preparation uses scratch and restores external bindings. Do not headline a missing external-cache mechanism or assert warmup necessarily destroys live external prefixes.
- **Inactive masking is present:** conv/recurrent preservation and attention write suppression exist (`M/tt/multichip_decoder.py:631-638,677-682,698-714,775-777,809-815`). Tests still need serving slot ownership; a mask does not implement remapping.
- **128 alignment is insufficient:** inspected the exception branches in the pinned scheduler, not just its default config name. Review removed the permanent-small-cap case from live failure claims because constructor validation rejects it; residual-budget continuation still demonstrates the mismatch. A final ragged chunk is not inherently invalid when its start is aligned.
- **No kernel arithmetic rewrite is justified for serving integration:** the demonstrated failures originate at temperature normalization, allocation/ownership, and call translation. Numerical accuracy and full-model throughput were not re-evaluated.
- **Further test gaps:** full-context inactive RoPE index overflow after very long resident runs; selected precision/pool capacity under actual serving concurrency; added request-state operations and scratch allocations under trace allocation tracking. These are not promoted as observed bugs.

Inspection performed: source/test/doc reads; revision and installed-package metadata checks; independent xhigh investigations followed by adversarial first-draft reviews of sampling, hybrid state, and plugin prefill; pure host arithmetic/transcription checks for temperature, state memory, slot reassignment, and the off-grid chunk example. Review corrections are incorporated above. This investigation changed no production source and wrote only this report. Concurrent working-tree changes are explicitly excluded above; their presence is not test evidence.
