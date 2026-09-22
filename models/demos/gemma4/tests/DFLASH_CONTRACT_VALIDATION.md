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
prefill work. Device correctness and the continuation cost remain unverified.

`supports_narrow_decode=True` is backed by initial ordinary completion handling.
`supports_async_decode` and `supports_async_spec_decode` both default to false.
`GEMMA4_CONTRACT_ASYNC=1` enables the experimental validation path. The async
default must remain false until the device checklist passes.

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
does not enlarge the bounded KV pool. Device correctness of this fallback and
of speculative execution with sufficient ring headroom remains unverified.
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
FP64-reference accuracy. Current-source natural-batch validation and performance
remain unverified. Both async capability defaults remain false.

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
these boundaries; full-model numerical parity remains unverified. Synthetic
P150x8 checks show that the paired table/mask rotation can match ordinary SDPA
on the tested post-wrap operands. The additional host-to-device page-row copy
and full-model performance remain unverified. Both async defaults remain false.

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

## Required device evidence before readiness

Use a reserved P150x8 endpoint. Record the exact tt-metal and plugin commits,
checkpoint revisions, environment, trace mode, and all local changes. The plugin
revision must include the generic model-owned contract and K+1 KV lookahead.
Do not use an unrelated plugin PR as an implicit dependency.

- [ ] Initial solo ordinary decode produces nonzero drafts after its completion.
- [ ] Initial batched ordinary decode completes, and an originally batched
      survivor later produces nonzero drafts.
- [ ] A real solo proposal remains outstanding when a peer prefills and joins.
      The original request survives `1 -> 2 -> 1` and resumes real drafting.
- [ ] The ordinary batch submits a second decode before the first readback
      completes. Record controller submission counters and actual scheduled rows.
- [ ] No speculative device work runs while a later ordinary completion remains
      unapplied. Record proposal, completion, and device submission ordering.
- [ ] Cancellation before readback, cancellation after verification, owner
      cancellation, non-owner cancellation, row movement, and physical page reuse
      preserve the surviving request's output and reject stale generations.
- [ ] Greedy K=5 cases cross page boundaries from input lengths 63, 64, 65, 127,
      128, and 129 during initial decode and both batch transitions.
- [ ] Async output matches synchronous output from the same contract adapter for
      identical request schedules. Compare token IDs, not decoded text prefixes.
- [ ] First reconstruction at anchor 158 and transition reconstruction at anchors
      192 or 193 preserve committed serving KV. Compare synchronous and async
      candidate/posterior rows at anchor 201 without forced reconstruction delay.
- [ ] Initial solo and initial batch common-prefix outputs agree after scratch
      reconstruction, including the observed output-index-4 discrepancy.
- [ ] A partial peer prefill resumes from a positive `start_pos`, uses complete
      prefix eager execution, preserves live peer KV, and finishes cleanly.
- [ ] Repeated runs confirm captured buffers, target KV, drafter context, and
      readback events remain valid. Include bounded sliding KV coverage separately.
- [ ] Exact-window bounded rings decline proposals before unsafe candidate writes
      and continue ordinary decoding across the ring boundary. A separate bounded
      run with sufficient ring headroom produces real proposals across that boundary.
- [ ] Record ordinary-reference differences at identical prefixes. Numerical
      differences are not automatically evidence of correctness.

Host tests use the actual adapter with stub device operations. Host tests check
ordering, ownership, context snapshots, bootstrap, cancellation, and state
transitions. Host tests do not verify device buffers, captured traces, KV values,
or readback event behavior. Until the device checklist passes, this implementation
is a draft and does not establish hardware readiness or performance parity.
