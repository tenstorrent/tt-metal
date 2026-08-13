# AutoDebug Report: reduced split-sampling shape mismatch

## Starting evidence

- Failing log: `doc/full_model/logs/reduced_split_trace.log`.
- Reproducer: construct `build_generator(num_layers=4, max_context=128, batch=1)` on TP4 and call `generate([1,2,3,4,5], 3)`.
- Current failure: `ttnn.sampling` rejects unequal logical shapes for input values and indices.
- The earlier `current_positions` dtype failure in the same concatenated log was already corrected; the latest run reaches the sampler.

## Source diagnosis (inspection only)

### H1: the common sampler's padded index templates are not narrowed to the runtime logits batch

Evidence:

- `TTSampling.__init__` rounds every configured batch up to at least 32 and creates both `tt_indices_tensor` and `tt_indices_device_offsets` with logical shape `[1, 1, 32, ...]`.
- The reduced Qwen model emits logits with logical batch 1.
- `TTSampling.forward` passes the full 32-row `tt_indices_tensor` to local `ttnn.topk`, while top-k values originate from the batch-1 logits.
- It then adds the full 32-row `tt_indices_device_offsets` to gathered indices.
- The terminal validation requires values and indices logical shapes to be identical and reports exactly this mismatch.

Prediction:

- Narrowing both persistent index tensors to `x.shape[-2]` before their first batch-sensitive uses will make top-k values and global indices agree at logical batch 1 while retaining tile-padded physical storage.
- The same change should be a no-op for batch 32.

Focused experiment:

- Add temporary shape instrumentation or a narrow assertion/probe immediately before `ttnn.sampling` to record the gathered values and untilized global-index logical shapes.
- Then apply only runtime-batch slicing to the two index templates and rerun the original reduced TP4 command.

### H2: the Qwen LM-head logits layout itself is incompatible with the common sampler

Evidence against:

- Local top-k and both TP4 gathers complete; failure occurs only after global indices are constructed.
- The vocabulary shard is tile aligned and the tracing contract explicitly recommends keeping 32 local candidates per shard for semantic greedy sampling.

Prediction:

- If H1 is fixed, no LM-head reshape or custom sampler should be necessary.

## Initial conclusion

H1 is the leading causal hypothesis. An equivalent contract-preserving repair is to pad decode logits to the common sampler's canonical 32 fixed rows, rather than modifying shared sampler code. H2 should be treated as refuted if that focused change passes the original reproducer. No implementation file was edited before this report was written.

## Batch-2 mixed-slot follow-up diagnosis (inspection before edit)

Failure surface: the fourth reduced layer (full attention) reaches
`ttnn.multiply(attention, sigmoid(gate))` and reports invalid subtile broadcast.

### H3: `nlp_concat_heads_decode` exposes the decode tile's 32 physical user rows while the QKV gate retains logical batch 2

- The QKV/gate projection consumes residual shape `[1,1,B,5120]`, so `gate` has logical shape `[1,1,B,1536]` with physical/tiled row extent 32.
- `paged_scaled_dot_product_attention_decode` and `nlp_concat_heads_decode` are decode-specialized ops whose concatenated result uses the full decode tile row contract `[1,1,32,1536]`.
- Elementwise multiply permits the B=1 case as scalar/subtile broadcast, but B=2 is neither equal to 32 nor a broadcast dimension, matching the exact failure signature.

Prediction: padding only `gate` along its decode-user dimension from B to 32 before sigmoid/multiply will make logical and physical row contracts equal. Inactive/padded rows remain zero from the padded residual and are discarded when the layer output is reshaped back to logical B. This does not change TP4 weights, collectives, dtypes, cache policy, or the residual layout.

Focused verification: run the exact reduced B2 mixed-slot capture. If it advances past this multiply and completes capture/feedback, retain the change; also rerun B1 and a construction/static B32 boundary where feasible.
