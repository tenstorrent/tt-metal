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
   prefix, excluding the newest anchor, to capture drafter residual taps.
   `_spec_bootstrap` seeds the drafter at the newest anchor. The replay does not
   publish or append another copy of any committed token.
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

`supports_narrow_decode=True` is backed by initial ordinary completion handling.
`supports_async_decode` and `supports_async_spec_decode` both default to false.
`GEMMA4_CONTRACT_ASYNC=1` enables the experimental validation path. The async
default must remain false until the device checklist passes.

`warmup_model_decode(enable_trace=False)` prepares drafter weights, persistent
verify inputs for every configured width, fixed-capacity context projection,
and fused programs before ordinary trace capture. Trace-enabled warmup captures
the same decoder. `prepare_widths` rejects new widths after any width is captured.
`_contract_covers_position` checks the actual physical verification extent and
captured coverage before commit or reconstruction. An uncovered position returns
zero drafts and continues ordinary decoding.

`_contract_prefill_tables` clones the submitted page tables and masks unused
columns before traced prefill or prefix reconstruction. Traced padding must not
write stale columns that alias live prompt pages. The masking uses each layer's
physical KV block size and preserves bounded sliding-ring columns. Decode and
speculative refresh retain the allocator's full lookahead tables.

`release_persistent_capture` releases fused and ordinary traces while the mesh
is open. Request-level release retains the width set for later requests; final
shutdown bypasses that retention and releases each trace once.

## Host regression command

From the tt-metal checkout, use Python 3.10 or newer with PyTorch, pytest,
pytest-timeout, and NumPy installed. Point `PYTHONPATH` at the matching plugin
checkout so the tests use the production `SpecPlan`, `DraftOutput`, and
`VerifyOutput` types:

```sh
PYTHONPATH=/path/to/vllm-tt-plugin/src python -m pytest -o addopts='' \
  --confcutdir=models/demos/gemma4/tests/unit \
  models/demos/gemma4/tests/unit/test_dflash_contract_adapter.py \
  models/demos/gemma4/tests/unit/test_dflash_width_prepare.py \
  models/demos/gemma4/tests/unit/test_dflash_capture_cleanup.py -q
```

The host test module supplies scoped stubs for lower TT imports. `--confcutdir`
keeps the device-runtime fixtures out of this host invocation. The production
adapter and the production inherited width-set bootstrap execute inside the
tests; the tests replace TT device operations with recording stubs. The width
lifecycle tests execute the real decoder preparation and capture code, including
fixed-capacity context seeding, failure cleanup, and preparation of all widths
before the first capture. Coverage tests check fixed-capture exhaustion,
width-set exhaustion, and ordinary fallback without repeated reconstruction.

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
- [ ] Repeated runs confirm captured buffers, target KV, drafter context, and
      readback events remain valid. Include bounded sliding KV coverage separately.
- [ ] Record ordinary-reference differences at identical prefixes. Numerical
      differences are not automatically evidence of correctness.

Host tests use the actual adapter with stub device operations. Host tests check
ordering, ownership, context snapshots, bootstrap, cancellation, and state
transitions. Host tests do not verify device buffers, captured traces, KV values,
or readback event behavior. Until the device checklist passes, this implementation
is a draft and does not establish hardware readiness or performance parity.
