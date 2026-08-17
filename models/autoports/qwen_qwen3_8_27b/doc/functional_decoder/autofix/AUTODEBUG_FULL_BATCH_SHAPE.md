# AutoDebug: full-attention batched paged-decode shape

## Scope and observation

Source-only inspection; no implementation edits, device opens, or hardware tests. The reported command is:

```text
pytest -q models/autoports/qwen_qwen3_8_27b/tests/test_functional_decoder.py -k batched_paged_decode -s
```

The test supplies public decode input `[B=2, T=1, H=5120]`, a two-row page table, and `current_position[B]`, then observes `[2,2,5120]` instead of `[2,1,5120]`.

## Headline finding

The paged-decode branch in `models/experimental/gated_attention_gated_deltanet/tt/ttnn_gated_attention.py` uses two batch-1-only transposes around the SDPA-decode primitive. This is the exact source-level shape divergence.

Before paged SDPA, `query_states` is explicitly `[B,Hq,T,D]` (lines 251-256). At `T=1`, line 355 does:

```python
q_decode = ttnn.transpose(query_states, 1, 2)
```

This produces `[B,1,Hq,D]`, while `paged_scaled_dot_product_attention_decode` uses the established decode layout `[1,B,Hq,D]`. The comment itself only proves equivalence for `B=1`: `[B,1,Hq,D] == [1,B,Hq,D]` only in that case.

The primitive's output is correspondingly `[1,B,Hq,D]`. Line 371 transposes dimensions 1 and 2 and produces `[1,Hq,B,D]`, not the claimed `[B,Hq,1,D]`. `concatenate_heads` at lines 550-552 consequently yields an attention tensor logically `[1,B,Hq*D]`. The gate retained from line 249 is `[B,1,Hq*D]`. Their multiply at line 559 broadcasts both singleton axes, yielding `[B,B,Hq*D]`; the output projection preserves those leading dimensions, explaining the observed `[2,2,5120]` exactly. With `B=1`, every mistaken layout is shape-equivalent, which explains why single-user tests did not expose it.

The wrapper and decoder layer do not create the extra axis. `FunctionalDecoder.decode_forward` passes `[B,1,H]` through unchanged (functional decoder lines 245-258). The single-device `Qwen36DecoderLayer` passes the norm result to gated attention and later adds the returned attention tensor to the original residual (layer lines 229-247). Broadcasting at that residual add can preserve/propagate an already-expanded `[B,B,H]`, but it is downstream of the first divergence.

## Canonical layout and contract

- Public autoport contract: keep `[B,1,H]` input and output. It is documented in `functional_decoder.py` and asserted by the failing test (test lines 367-378).
- Internal ordinary-attention layout: `[B,H,T,D]`.
- Native TTNN decode-SDPA layout: `[1,B,H,D]` (the repo's common attention path documents decode tensors as sequence-first and passes decode heads directly to the primitive).

Therefore the repair belongs at the experimental gated-attention primitive boundary. Convert `[B,H,1,D] -> [1,B,H,D]` with a real permutation `(2,0,1,3)`, and convert the result back with `(1,2,0,3)` before generic head concatenation. A transpose of only axes 1 and 2 cannot exchange batch and token axes. Keep these as on-device TTNN operations; no torch, `from_torch`, `to_torch`, or host fallback is needed. If the supported TTNN `permute` path is unsuitable for traced execution/layout, use an on-device reshape only after proving the physical ordering is compatible, or use the decode-native `nlp_concat_heads_decode` flow; do not silently reinterpret `[B,1,H,D]` as `[1,B,H,D]`.

## Ranked focused experiments

1. **Verify/refute the exact axis chain (highest value).** Add temporary device-side shape assertions/logging immediately before SDPA, immediately after SDPA, after conversion back, after head concat, and before gate multiply for `B=2`. Prediction for current code: `[2,1,Hq,D] -> [1,2,Hq,D] -> [1,Hq,2,D] -> [1,2,H]`, then broadcast with gate `[2,1,H]` to `[2,2,H]`.
2. **Minimal A/B fix.** Replace only the two transposes with on-device permutations `(2,0,1,3)` and `(1,2,0,3)`. Rerun the original test. Prediction: attention, gate, both residual adds, and final result remain `[2,1,5120]`.
3. **Prove semantic user routing, not shape alone.** Use distinct current positions, disjoint/permuted page-table rows, and distinguishable per-user Q/K/V data. Compare each user's output to a separate batch-1 control. This detects a batch-axis reinterpretation that happens to have the right final shape.
4. **Trace safety.** Capture/replay the repaired `B=2` paged decode using stable input, position, and page-table buffers; measure PCC from replay output and repeat identical input. Confirm no allocations/host conversions are introduced by the layout conversion.
5. **Boundary batches.** Run `B=1`, `B=2`, and the largest supported practical batch. The `B=1` control must stay numerically unchanged; `B>1` must retain `[B,1,H]` throughout.
6. **Residual localization control.** If `[B,B,H]` remains after experiment 2, temporarily inspect/replace attention output at the layer boundary with a correctly shaped TTNN zero tensor. This separates attention/gate output from RMSNorm/MLP residual broadcasting. Source evidence ranks this well below the two transpose errors.

## Other potential issues / remaining uncertainty

- The paged K/V update reshapes `[B,Hkv,1,D]` to `[1,B,Hkv,D]` at lines 339-340. Because `T=1`, reshape preserves the desired flattened element order for the documented update contract, but the batch>1 semantic-routing control above should verify it alongside page-table updates.
- RoPE decode reshapes `cos/sin` using their first dimension at lines 272-276. The failing test appears to use a shared one-token RoPE tensor; distinct per-user positions may require separately gathered per-user RoPE inputs. This does not explain `[B,B,H]` and should not be batched into the primary fix.
- Runtime support/performance of the precise TTNN `permute` configuration must be verified on device. That is implementation evidence still needed, not uncertainty about the source-level causal chain.

## AutoDebug runner note

The required fresh-context `.agents/scripts/autodebug.sh` run was attempted from this artifact directory, but its nested Codex shell could not read the checkout because bubblewrap user namespaces are disabled (`bwrap: No permissions to create a new namespace`). The runner was stopped after repeated identical failures. The findings above were then independently checked by direct read-only source inspection in the parent execution environment.
