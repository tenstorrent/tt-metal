# AutoFix Report: Stage-Review Correctness Gaps

## Starting Evidence

- Source: stage-review findings requesting stronger B>1 correctness, near-limit
  non-aligned coverage, complete runtime fallback closure, and direct DeltaNet
  repeated-input determinism.
- Initial B=2 test only established output shape/finiteness and populated the
  cache through separate B=1 calls. The maximum-context harness ended on aligned
  128/64/2048 boundaries. The source audit omitted autoport-local overrides.

## Hypothesis Experiments

### Batched paged prefill/decode

- Hypothesis: the shared paged prefill is batch-one-only because it calls
  `paged_fill_cache(..., batch_idx=0)`.
- Experiment: replace the smoke with one public B=2 prefill call using disjoint,
  permuted page rows, then decode at distinct positions 64 and 96; compare every
  row to an independent real-weight HF oracle.
- Initial result: verified. The kernel rejected input batch 2 with
  `When no batch_idx_tensor is provided, input_batch must be 1`.
- Fix: the autoport-local attention override device-slices each user row and its
  page tables for the shared B=1 fill/SDPA primitive, then device-concatenates
  outputs. No host conversion was introduced.
- Verification:
  `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k 'runtime_fallback_source_audit or full_attention_batched_paged_prefill_decode_pcc' -s`
- Result: PASS. Prefill PCC: user 0 length 64 = 0.9972510519; user 1
  length 96 = 0.9970104443. Decode PCC: user 0 position 64 =
  0.9957569950; user 1 position 96 = 0.9971083663. Maximum tested batch is 2.

### Non-aligned maximum context

- Hypothesis: the previous aligned 262,144-token traversal did not prove partial
  final chunk/page handling near the advertised limit.
- Experiment/fix: prefill exactly 262,143 tokens for both layer kinds, using a
  final 127-token DeltaNet chunk and final 2,047-token full-attention chunk (63
  tokens in its final page), then decode at position 262,143.
- Verification:
  `pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k advertised_context_layer_harness -s`
- Result: PASS in 40.42 s. Advertised 262,144-token support remains unchanged;
  this supplements the previously passing aligned 262,144-token prefill evidence.

### Runtime fallback closure

- Hypothesis: target-local overrides were absent from the static measured-call
  audit even though the underlying shared helpers were checked.
- Fix: add `_full_attention_forward_with_batched_paged_decode` and
  `_gdn_forward_with_dram_state` to the audited closure, alongside public forward,
  decoder-layer forward, shared attention prefill/decode, and GDN recurrent decode.
- Result: PASS; no `torch`, `ttnn.from_torch`, `ttnn.to_torch`, CPU, or NumPy
  fallback appears in the audited measured runtime sources.

### DeltaNet repeated-input determinism

- Hypothesis: trace-state address stability did not directly prove reproducible
  output after state reset.
- Experiment: run identical real-weight prefill and decode inputs, reset recurrent
  state, and repeat on the same decoder.
- Verification: included in the first focused three-test command.
- Result: PASS; both prefill and decode outputs are bitwise equal after reset.

## Final Status

- Fixed. All four stage-review correctness findings now have focused passing
  evidence.
- Device commands were serialized after the profiler owner released the lock;
  each pytest process closed its device before the next command.
- Remaining note: the public B>1 prefill contract is correct but internally
  row-serial because the shared paged-fill helper does not expose a batch-index
  tensor. This is a performance characteristic, not a correctness fallback.
