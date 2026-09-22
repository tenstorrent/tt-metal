# Gemma dFlash adaptive contract validation

`Gemma4DFlashContractForCausalLM` gives the plugin real draft tokens and the
corresponding target posterior. The plugin owns acceptance. Ordinary decode
returns one token per request and can use deferred device readback. The model
retains each request's committed prefix so the original request can resume
drafting after ordinary batched decoding.

## Implementation boundary

`ContractRequest` represents one request generation. `release_request` marks
`ContractRequest.live=False`; a later prefill creates a different
`ContractRequest` even when the state slot and first physical block are reused.
`ContractStep` keeps the submitted request objects and cloned page tables.
Ordinary proposal callbacks consume `ContractStep` objects in submission order.
Speculative completions return their own `ContractStep` as `VerifyOutput.hidden`.
`VerifyOutput.hidden` contains host ownership metadata, not target hidden tensors.

1. `prefill_forward` saves the complete submitted token prefix for each request.
2. `decode_forward` records the first prefill-sampled input token once and submits
   ordinary target decoding when `spec_mode` is absent.
3. `propose_draft_tokens` records the authoritative committed output tokens at
   the supplied output positions. Later ordinary submissions cannot overwrite
   the saved page tables or request generations.
4. `propose_draft_tokens` declines drafting while a later ordinary completion
   awaits its proposal callback. The plugin drains initial and changed-layout
   submissions before submitting another decode, so initial solo execution and
   a batch transition to solo have a completion at which reconstruction can run.
5. `_contract_rebuild` eagerly recomputes the surviving request's committed
   prefix in separate target KV, excluding the newest anchor, to capture drafter
   residual taps. `_contract_rebuild` preserves committed serving KV and restores
   serving page tables and tail state before `_spec_bootstrap` seeds the drafter
   against the original serving KV. The replay does not publish or append another
   copy of any committed token.
6. `_contract_refresh` installs the completion's per-layer page tables and
   refreshes the fused decoder before `contract_commit` and `contract_replay`.
7. `decode_forward` checks request identity, anchor, position, and every valid
   draft before returning a retained posterior. When a peer joins,
   `decode_forward` runs ordinary target work for the other rows and masks the
   speculative owner's row out of that ordinary work.

Reconstruction adds eager target-prefill work when a request starts or resumes
drafting. Transition latency and long-context device memory use are unverified.
Steady batched ordinary decoding does not reconstruct drafter state.
`GEMMA4_DFLASH_MAX_SPEC_ISL` also limits reconstruction at the current committed
prefix length. Prefix caching remains unsupported.

Both CT allocation entry points prepare reconstruction storage before trace
capture. `_contract_prepare_rebuild_storage` uses the configured serving context,
rounded to the existing prefill bucket, and each target layer's effective block
size. Scratch physical block zero receives padded writes. The remaining scratch
blocks hold the reconstructed prefix. Dedicated page-table tensors cover batch
keys 1 and `max_batch_size`; reconstruction rejects unprepared page-table shapes
before allocation. `_contract_covers_position` declines drafts if the padded
prefix exceeds prepared capacity, then ordinary decoding continues.

The 31B P150x8 configuration with a 4096-token context and 64-token blocks needs
65 scratch blocks per layer. BF16 scratch KV occupies 487.5 MiB per device;
dedicated page tables for batch keys 1 and 32 add about 0.483 MiB per device.
`spec_plan` adds this fixed storage to `extra_bytes_per_seq`, counts shared layers
conservatively, and rounds page-table rows for device alignment. The plugin does
not enforce these byte budgets. Available device memory and allocation overhead
need separate validation. Larger configured contexts increase scratch storage;
the scratch allocator does not copy the serving pool's full block count.

`_contract_rebuild_context` isolates persistent page-table bookkeeping, sliding
tail pools and request keys, bounded-fill validity buffers, prefill input stashes,
slot markers, and decode metrics. Scratch prefill temporarily uses unbounded
storage while retaining sliding attention semantics. `_ct_eager_prefill` also
disables generator-level traced chunks, so scratch execution cannot reuse a
prefill trace that binds serving KV. The context restores original references
after success or failure. Failed reconstruction releases captured temporary taps.
`release_persistent_capture` releases scratch tensors after serving traces.

For runtime continuation with positive `start_pos`, CT `prefill_forward`
recomputes the complete supplied prefix from zero through eager prefill. The
paired plugin supplies that prefix through `prompt_lens`. This fallback avoids
depending on an unavailable or incomplete sliding tail from a previous chunk.
The fallback preserves row ownership and page-table sanitation but adds repeated
prefill work. The recorded P150x8 pair below validates external continuation in
the tested configuration. Continuation cost and behavior outside that configuration remain
unverified.

`supports_narrow_decode=True` is backed by initial ordinary completion handling.
`supports_async_decode` and `supports_async_spec_decode` both default to false.
`GEMMA4_CONTRACT_ASYNC=1`, set before adapter import, enables the explicitly
tested asynchronous path in the configuration documented below. A default change
requires a separate rollout decision; completion of Step 3 does not enable
either declaration automatically.

The contract resolves `GEMMA4_DFLASH_VERIFY` once, defaulting to five drafts.
`spec_plan` and both decoder construction paths use that count, so physical
verification writes occupy exactly `effective_k + 1` positions. Admission rejects
counts outside the actual drafter block size and rejects disabled packed
verification, width-set preparation, or eager decode warmup. These diagnostic
settings cannot satisfy the adaptive contract's physical allocation and capture
requirements.

`warmup_model_decode(enable_trace=False)` prepares drafter weights, persistent
verify inputs for every configured width, fixed-capacity context projection,
and fused programs before ordinary trace capture. Trace-enabled warmup captures
the same decoder. `prepare_widths` rejects new widths after any width is captured.
`_contract_covers_position` checks the actual physical verification extent and
captured coverage before `contract_commit` or reconstruction. An uncovered position returns
zero drafts and continues ordinary decoding.

`_contract_covers_position` also checks every target layer with
`cache_position_modulo` set. Packed verification writes `P_v` positions starting at
the committed anchor before attention reads target KV. The bounded ring needs
at least `P_v - 1` positions beyond `sliding_window`, or every candidate position
must precede the first ring wrap. An unsafe position returns zero drafts before
`contract_commit`, reconstruction, or replay, and ordinary decoding continues. The contract
does not enlarge the bounded KV pool. The current hardware evidence below
records bounded fallback and ring-headroom coverage, including the separately identified exact-ring supplemental witness.
Behavior outside the tested extents remains unverified.
Bounded eager-only execution also declines drafts while prepared verification
widths remain uncaptured. Lazy capture writes through zero page tables at
position zero, but bounded slot zero owns those physical blocks. Bounded
speculation therefore requires capture before serving requests. Unbounded lazy
capture and bounded startup trace capture retain their existing behavior.

`_contract_prefill_tables` clones the submitted page tables and masks unused
columns before traced prefill or prefix reconstruction. Traced padding must not
write stale columns that alias live prompt pages. The masking uses each layer's
physical KV block size and preserves bounded sliding-ring columns. Decode and
speculative refresh retain the allocator's full lookahead tables.

`release_persistent_capture` releases fused and ordinary traces while the mesh
is open. Request-level release retains the width set for later requests; final
shutdown bypasses that retention and releases each trace once.

## Contract target attention policy

`Gemma4DFlashContractForCausalLM.__init__` selects target attention arithmetic
after inherited model construction and before eager program preparation or
trace capture. Each target layer uses
`Gemma4AttentionConfig.decode_rope_fast_and_approximate_mode=True` and
`Gemma4AttentionConfig.decode_sdpa_max_cores_per_head_batch=1`.
`apply_rope_decode_peruser` supplies `fast_and_approximate_mode=True` to both
multiplications. Ordinary and packed decode supply the same SDPA core limit.
The fused batch-1 RoPE path and natural batch-1/batch-32 selection are unchanged.

`Gemma4ForCausalLM` and `Gemma4DFlashForCausalLM` retain the existing arithmetic
by default. Their ordinary multiplication and SDPA calls omit these optional
keywords; their packed SDPA retains its existing conditional limit of 16 or 8.
Normal adapter initialization constructs separate targets, so selecting the
contract policy leaves independently constructed adapters unchanged.

Controlled P150x8 experiments with fixed ordinary batch 32 and these arithmetic
settings match all 704 completed output tokens across four sync/async schedules.
The controlled experiments do not prove parity with natural batch selection or
all request histories. A cancelled request ends at different token counts, and
some proposal histories differ despite equal completed outputs. Synthetic
operator checks establish consistency on their tested operands, not improved
FP64-reference accuracy. The current hardware evidence below records
natural-batch validation. Performance remains unverified. Both async capability
defaults remain false.

## Contract packed ring read order

`Gemma4DFlashContractForCausalLM` selects `rotate_ring_reads=True` for its fused
`DFlashFusedDecoder`. Ordinary attention and the legacy dFlash adapter retain
their existing page-table behavior. Packed bounded sliding attention reads a
separate persistent page table; all KV writes retain the natural ring table.

1. `_pv_width_install` records the natural host page row with `clone()` and
   allocates a separate sliding read table for every prepared width before
   ordinary or fused trace capture. Full-attention read overrides remain `None`.
2. `refresh_page_tables` copies the current request's natural page rows into
   the write buffers and replaces the active width's cloned host rows.
3. `_pv_upload(start)` rotates the sliding read table and sliding mask together
   by the first query's oldest live 64-token chunk. `_pv_upload` copies both
   inputs into their persistent buffers before `contract_replay` submits the
   trace. Each packed query retains exactly the same physical K/V membership.
4. `ttnn_packed_verify_forward` passes the persistent read tables through each
   layer's packed inputs. `packed_decode_forward` uses `read_page_table` only
   for `_packed_verify_sdpa`; fallback writes still use `page_table`, and staging
   writes still use `hot_pt`.
5. `_spec_release_decoder` retains read buffers across request sessions. Final
   teardown releases traces first and then each owned read buffer once.

The common six-row read table starts at the first query's oldest live chunk.
When later queries cross a 64-token window boundary, later queries can mask an
entire leading chunk. Host tests establish exact physical membership across
these boundaries. The current hardware evidence below records full-model output
comparisons for the tested request corpus. Synthetic
P150x8 checks show that the paired table/mask rotation can match ordinary SDPA
on the tested post-wrap operands. The latency cost of the additional
host-to-device page-row copy
and full-model performance remain unverified. Both async defaults remain false.

## Current P150x8 evidence

The tested model source is `c285a6688ef1c4fe67e88563fbb04599abc2123b`,
paired with vllm-tt-plugin `d1b6a5dadb261856b507123a7cc5e91ab4cf71b8`.
The target is `google/gemma-4-31B-it` at checkpoint
`842da3794eaa0b77d5f08bae87a17459d91ff475`; the drafter is
`z-lab/gemma-4-31B-it-DFlash` at checkpoint
`eabd648301ce28583cc14757912e5e0f84e152e1`. The reserved P150x8 runs use
native traces, greedy K=5, six physical verification positions, 64-token pages,
a 4096-token model context, and natural batch-1/batch-32 target selection.
Prefix caching and hybrid KV groups are disabled. Synchronous scheduling uses
`GEMMA4_CONTRACT_ASYNC=0`; asynchronous scheduling uses
`GEMMA4_CONTRACT_ASYNC=1`. The latter opt-in enables both async declarations.
Neither declaration is enabled by default. These schedules use one or two live
requests. The physical batch-32 path does not establish support for 32 concurrent
requests. Other checkpoints, devices, draft counts, and sampling modes remain
unverified.

All 316 production host tests pass with lower device operations stubbed.
Independent source review and CI pre-commit also pass. Host checks do not
substitute for the following device evidence.

`sync-ring-policy-50` and `async-ring-policy-51` each complete the same 29
schedules with a 2048-position bounded ring and 1024-token sliding window.
The runs complete 1,288 and 1,290 speculative replays. All 49 completed request
streams match token for token, totaling 4,908 tokens per mode. Both runs drain
request and ring state, exit zero, and close TT devices. Independent audits
verify the frozen harness, source and runtime hashes, checkpoint manifests,
5,260/5,264 ownership snapshots, 52 released generations per mode, and 30
fresh-wave clear/trace-recapture sequences per mode.

The observer records initial narrow bootstrap, initial batched ordinary decode,
real proposals at peer arrival, the original request resuming speculation after
batched decode, page crossings, cancellation, row reuse, and physical page reuse.
Async execution records a new ordinary submission while an earlier native
readback remains unresolved. `release_request` calls `_contract_synchronize`
before releasing rings for two cancelled generations. Those `ContractRequest`
objects remain invalid when their delayed completions arrive. These observations
establish native submission/readback ordering;
physical DMA overlap and equality of every device KV element remain unverified.

The original strict synchronous gate passes. The strict paired comparison
retains different action prefixes in dynamic schedules 004/005/006 and a
cancelled request with 13 versus 14 output tokens. That cancelled request's
common prefix is exact. A comparison policy frozen before these runs permits
those three dynamic schedules to use cancellation-prefix and lifecycle checks,
while requiring every completed surviving output to match. That separate
29-case comparison passes. The strict failure is preserved; complete output
parity does not assert identical cancellation timing or proposal histories.

`sync-ring-exact-52b` and `async-ring-exact-53` complete five schedules
with a 1024-position ring and a 1024-token sliding window. All ten completed
request streams, totaling 664 tokens per mode, match exactly; action definitions
and committed action prefixes also match. Both runs exit zero and close TT
devices. The synchronous strict gate passes. The asynchronous strict gate retains
one coverage failure: the original 1016-token schedule does not witness a peer
prefill while a real proposal remains outstanding. The unchanged 1008-token
supplement observes that transition and passes its own gate. Aggregate coverage
passes, while the original strict failure remains recorded.

`sync-ring-chunked-54` and `async-ring-chunked-55` use unbounded KV with
external 128-token prefill grants. The runs complete 39 and 40 speculative
replays, respectively. A returns 96 tokens and B returns 16; all token IDs,
action definitions, and committed/admission prefixes match exactly. Both strict
chunked gates pass. The observer records positive-start peer continuation,
ordinary decode between peer chunks, outstanding proposals, and survivor
resumption. Both runs drain state, exit zero, and close TT devices.

Synthetic P150x8 checks separately compare ordinary B1/B32 and packed SDPA
using the same saved device Q/K/V operands. Paired read-table/mask rotation
matches ordinary output for the first query at all seven tested starts. The
checks also verify all six packed rows' physical membership and stable repeats.
Shared queries 2111 and 2124 match across their packed row placements under
both traversal orders. These observations establish operation sensitivity to
traversal and masked-chunk placement; they do not prove arbitrary-input parity
or improved numerical accuracy.

The installed upstream vLLM finalizer calls `model.modules()` on the TT wrapper
and emits an ignored exception. TT trace release and device closure complete,
but warning-free upstream teardown remains unresolved. Both chunked-prefill logs
also retain active-trace allocation and `decode_input_update_contract` legacy
reload warnings. The tests do not establish that every warning is harmless.
During shutdown, the validation harness skips
`torch.accelerator.empty_cache` when PyTorch reports no accelerator allocator.
TT device closure still runs. This host-runtime accommodation is not a model
change. Performance, larger-context memory capacity, forced
preemption, higher actual request concurrency, and ordinary-reference numerical
equivalence remain unverified by this corpus. The generic registry default
switch and throughput comparison remain separate rollout work.

## Host regression command

From the tt-metal checkout, use Python 3.10 or newer with PyTorch, pytest,
pytest-timeout, and NumPy installed. Point `PYTHONPATH` at the matching plugin
checkout so the tests use the production `SpecPlan`, `DraftOutput`, and
`VerifyOutput` types:

```sh
PYTHONPATH=/path/to/vllm-tt-plugin/src python -m pytest -o addopts='' \
  --confcutdir=models/demos/gemma4/tests/unit \
  models/demos/gemma4/tests/unit/test_async_ahead_decode_tokens.py \
  models/demos/gemma4/tests/unit/test_async_prefill_row_authority.py \
  models/demos/gemma4/tests/unit/test_bounded_ring_page_tables_host.py \
  models/demos/gemma4/tests/unit/test_dflash_attention_policy.py \
  models/demos/gemma4/tests/unit/test_dflash_bounded_ring_ownership.py \
  models/demos/gemma4/tests/unit/test_dflash_capture_cleanup.py \
  models/demos/gemma4/tests/unit/test_dflash_contract_adapter.py \
  models/demos/gemma4/tests/unit/test_dflash_contract_bounded.py \
  models/demos/gemma4/tests/unit/test_dflash_contract_prefill_continuation.py \
  models/demos/gemma4/tests/unit/test_dflash_contract_reconstruction.py \
  models/demos/gemma4/tests/unit/test_dflash_contract_width_config.py \
  models/demos/gemma4/tests/unit/test_dflash_ring_read_order.py \
  models/demos/gemma4/tests/unit/test_dflash_width_prepare.py -q
```

The host test module supplies scoped stubs for lower TT imports. `--confcutdir`
keeps the device-runtime fixtures out of this host invocation. The production
adapter and the production inherited width-set bootstrap execute inside the
tests; the tests replace TT device operations with recording stubs. The width
lifecycle tests execute the real decoder preparation and capture code, including
fixed-capacity context seeding, failure cleanup, and preparation of all widths
before the first capture. Coverage tests check fixed-capture exhaustion,
width-set exhaustion, and ordinary fallback without repeated reconstruction.
Width configuration tests execute both production decoder constructors and
compare physical tensor extents with the admitted draft count, including an
unset `GEMMA4_DFLASH_VERIFY` and overridden drafter block sizes.
Bounded coverage tests check the first wrapped candidate, exact headroom
equality, every configured layer, actual decoder width, and ordinary progress
after a verified completion reaches the ring limit.
Reconstruction tests execute both allocation entry points, actual page-table
conversion and updates, first and later reconstruction, nonzero owner slots,
scratch growth rejection, serving KV and tail preservation, input restoration,
partial allocation cleanup, and final release. Continuation tests check eager
dispatch and restoration after errors. These host checks do not establish TT
numerical equivalence or memory sufficiency.

## Step 3 coverage

The checked items below are established for the pinned 31B/P150x8 greedy K=5
configuration in `sync-ring-policy-50` / `async-ring-policy-51`,
`sync-ring-exact-52b` / `async-ring-exact-53`, and
`sync-ring-chunked-54` / `async-ring-chunked-55`, supported by
actual-production host regressions. These checks describe the tested corpus;
they do not establish universal numerical equivalence or default rollout.

- [x] `propose_draft_tokens` initializes real speculative state after the first
      ordinary narrow completion. `_contract_rebuild` also restores drafting for
      a surviving request that initially prefilled in a batch.
- [x] `ContractStep` preserves submitted page-table snapshots.
      `_contract_refresh` refreshes the surviving owner's tables before
      speculative commit and replay. Host tests check table mutation isolation;
      hardware observers check native call order.
- [x] A peer prefills and joins while real drafts remain outstanding.
      `decode_forward` validates the retained proposal and returns its posterior
      for plugin-owned acceptance. `Gemma4DFlashContractForCausalLM` handles row
      changes and subsequent ordinary work while retaining the original
      request's ownership.
- [x] The original request survives `1 -> 2 -> 1` and resumes real drafting after
      batched ordinary decoding. Page-boundary cases cover input lengths
      63, 64, 65, 127, 128 and 129; bounded headroom cases cross the ring boundary.
- [x] Pending ordinary completions retain their request generations. Native
      ordinary submissions overlap unresolved readback; older completions decline
      drafts while later completions remain queued. Cancellation, row reuse and
      physical page reuse reject released generations, with synchronization before
      bounded-ring storage is released.
- [x] All 49 completed request streams match synchronous versus asynchronous
      execution of the same contract adapter. The exact original-survivor case
      passes. The separate predeclared categorized comparison passes all 29 cases;
      the strict paired result remains false for dynamic frontiers in 004/005/006.
      Cancelled A has 13 versus 14 tokens with an identical common prefix.
- [x] Repeated solo output, per-case cleanup, fresh-wave trace recapture, final
      ownership drain and TT device closure pass. These observations do not
      directly compare all device buffers or every KV value.
- [x] `sync-ring-exact-52b` and `async-ring-exact-53` validate safe speculative
      execution, unsafe-write fallback, and subsequent ordinary progress. All ten
      completed streams match. The original async strict missing-witness failure
      remains separate from the passing 1008-token supplement and aggregate gate.
- [x] `sync-ring-chunked-54` and `async-ring-chunked-55` validate positive-start
      continuation, decode between peer chunks, outstanding proposals, survivor
      resumption, lifecycle cleanup, and strict same-adapter token equality.

## Separate numerical and rollout limits

Host tests execute `Gemma4DFlashContractForCausalLM` with lower device operations
stubbed.
Host tests establish ordering, ownership, snapshots, bootstrap, cancellation and
state transitions. Hardware observations establish the recorded native events
and completed outputs, but direct equality of all target KV values, captured
buffer contents, rotated device read tables and physical DMA overlap remains
unverified. Equal completed outputs do not require equal proposal histories.

Same-adapter sync/async equality does not establish ordinary-reference numerical
correctness. The wider plan's ordinary-only baseline, full-vocabulary host
readback and host-sampling validation remain separate Step 6 work. Forced
preemption and ordinary-reference comparison in the wider Step 5 acceptance bar
also remain unverified; cancellation and reuse are not substituted for preemption.

Throughput, transition latency, larger-context memory capacity, higher live
concurrency, broader devices/checkpoints/sampling modes and the registry/default
switch remain separate Step 7 rollout work. Both async defaults remain false.
The upstream `model.modules()` finalizer exception remains visible despite
successful TT closure. The implementation and hardware evidence establish
Step 3 within the explicit support boundary above. This conclusion does not establish universal production
readiness or performance parity.
