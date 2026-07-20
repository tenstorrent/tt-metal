# Runtime fallback audit

## Verdict

No runtime fallback exists in the optimized full-model path. All validation ran with `TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`; the device runs completed without a fallback exception.

## Model and generator path

- Embedding, 40 decoder layers, final norm, untied LM head, and sampling execute on the TP4 mesh.
- Decoder weights and execution retain the selected BFP4_B/LoFi, BFP8_B KV, BF16 activation/residual/CCL, two-link Ring, persistent all-reduce, and native sharded-residual policies.
- There is no single-chip branch, model replication branch, host decoder layer, host LM head, alternate low-precision policy, or silent context reduction.
- Small scalar/control tensors may be replicated across the four ranks as required by TP collectives; this does not replicate the model. Embeddings, decoder weights, KV heads, LM-head vocabulary, and execution remain tensor-parallel.

## Cache ownership and page tables

- The generator owns its default paged KV cache and accepts explicit caller-owned cache/page-table state through the low-level APIs.
- Public prefill owns padding, masks, chunking, cache fill, positions, page mapping, and output slicing.
- Fixed slots and inactive rows remain explicit. Inactive positions use -1 for cache update while rotary positions are safely clamped nonnegative; inactive cache rows remain unmapped and unchanged.
- Page tables are copied only at request setup or when their value changes. Stable replay performs no per-token host rebuild.

## Logit and sampling boundaries

- The default token-out path never performs host argmax or full-logits readback.
- Greedy feedback is direct device `tt_out_tok` to persistent decode token input.
- One sampled token is read for caller-visible output; measured device-only and caller-visible rates are reported separately.
- Full-logits gathering is confined to explicit `return_all_logits`, comparison/evidence helpers, and `sampling_mode="host"` compatibility behavior.
- `force_argmax` and `TTSampling` are comparison paths, not runtime fallbacks.

## Position and token feedback

- Persistent device cache and rotary positions advance with device `plus_one` operations inside the model trace.
- The steady 128-replay measurement recorded zero host token, position, rotary-position, page-table, and sampling-parameter copy deltas.
- Token feedback, position 256, changed page-table propagation, stable page-table reuse, and restored-table exactness are all asserted in `full_model_evidence.json`.

## Reset behavior

- Reset releases both traces before zeroing or reallocating cache state.
- It clears KV cache, device positions, page-table state, sampler state, and request metadata.
- Persistent input and CCL pools are preserved for safe recapture, and CCL capture epochs are reset explicitly.
- Seeded stochastic state is checkpointed/restored across capture; the same seed reproduced the same tokens after safe recapture.

The explicit host compatibility mode is not used in reported optimized performance. No fallback was observed by the runtime or Watcher gates.
