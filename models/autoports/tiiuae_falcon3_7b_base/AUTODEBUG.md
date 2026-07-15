# AutoDebug: Falcon3 decode PCC failure

> Historical starting evidence for the AutoFix loop. The explicit-mask hypothesis below was tested
> and refuted: decode PCC remained about 0.8464. The verified defect was the decode-Q memory boundary;
> preserving the emitted DRAM-interleaved Q input raised synthetic decode PCC to 0.99911663 and
> real-weight decode PCC to 0.99882657. See `doc/functional_decoder/work_log.md` for the completed loop.

## Verdict

The leading correctness hypothesis is the decode-SDPA invocation, not KV-cache population. The translated decoder does **not** preserve the emitted graph's attention-selection contract: the emit builds an explicit zero/negative-infinity mask through the current cache position and calls decode SDPA with `attn_mask=<mask>`, `cur_pos_tensor=None`, and `is_causal=False`; the implementation supplies no mask and instead calls with `cur_pos_tensor=cache_position` and `is_causal=True`.

Those modes might look mathematically interchangeable, but they are distinct TTNN kernel interfaces and the compiler chose the explicit-mask mode. This discrepancy predicts the observed shape of the failure: prefill passes, both caches remain correct after decode update, yet the decode result has low PCC.

## Direct observations

- Synthetic prefill passes at sequence 17 (`PCC=0.99893845`) and 128 (`PCC=0.99897713`), while the one-token decode executes but returns only about `PCC=0.8464`.
- TT prefill caches match the Hugging Face layer (K `PCC=0.99991646`, V `PCC=0.99991996`). After the TT decode update at position 17, the cache prefix through length 18 still matches HF (K `PCC=0.99991628`, V `PCC=0.99991987`). This strongly refutes incorrect prefill filling, update index, value update, key RoPE, and key-cache layout as primary causes.
- `tt/functional_decoder.py:537-545` calls `scaled_dot_product_attention_decode` without `attn_mask`, with `cur_pos_tensor=cache_position` and `is_causal=True`.
- The representative emitted layer in `/tmp/tiiuae_falcon3_7b_base/decode.py:991-1011` compares the update position against the cache-index table, selects zero for valid positions and a negative value for future positions, repeats that mask over query heads, and calls SDPA with `is_causal=False`, the explicit `attn_mask`, and `cur_pos_tensor=None`. All 28 emitted layers repeat this form.
- The current and emitted paths otherwise agree structurally on decode head creation, rotary embedding, post-rotary head slicing, KV update, SDPA scale `0.0625`, head concatenation, output projection, residual, and MLP. The emit uses TP-local 3 Q heads/1 KV head and concatenates 3 heads; the dense collapse uses 12 Q heads/4 KV heads and concatenates 12 heads, which is the expected full-weight equivalent.

## Ranked hypotheses and decisive checks

### 1. Decode SDPA is selecting the wrong cache range because the emitted mask contract was replaced

This is the only clear semantic deviation at the first consumer of the already-validated caches.

**Verify:** construct the emit-equivalent device mask (zero for cache indices `<= position`, negative infinity after it), repeat it over all 12 dense query heads, and call SDPA with `attn_mask=mask`, `cur_pos_tensor=None`, and `is_causal=False`. Keep Q, caches, concat, and projections unchanged. The hypothesis is confirmed if decode PCC rises to at least 0.99.

**Refute:** if the attention-head tensor before concatenation is still wrong under the exact emitted mask mode, this hypothesis is insufficient. An A/B debug check of current mode versus explicit-mask mode on identical Q/K/V is more decisive than comparing only final layer outputs.

**Useful control:** compare current `cur_pos_tensor=position` with `position+1`. Improvement only for the latter would expose an inclusive/exclusive convention mismatch, but the final translation should still follow the emitted explicit-mask call.

### 2. Decode-only Q head extraction/slicing is wrong; KV evidence cannot validate Q

The cache checks validate K and V, not Q. The implementation uses the dense form of `nlp_create_qkv_heads_decode`, unpads the rotary result to 12 Q heads, and later calls `nlp_concat_heads_decode` for 12 heads. The emit performs the same sequence for 3 TP-local Q heads. This makes the dense transform plausible, and passing prefill validates the fused dense QKV weight ordering, but neither fact proves the decode primitive's Q layout.

**Verify/refute:** in a debug-only test, compare `query_rotated` to the HF query after norm, Q projection, reshape, and RoPE, before SDPA. Then compare the TT concatenated attention output against `attention_heads.transpose(1, 2).reshape(batch, hidden_size)`. If Q PCC is high and explicit-mask SDPA fixes attention, close this hypothesis. If Q is the first mismatch, inspect the padded decode-head shape and the dimension selected by `_unpad_decode_heads` before changing later operations.

### 3. RoPE/current-position semantics differ for Q only

This is lower confidence. The appended K cache matches HF after decode, so the selected position and cosine/sine values are correct for K. The same tables and rotary primitive are used for Q, making a general RoPE bug unlikely; only a Q-specific padded-layout interaction remains possible.

**Verify/refute:** the pre-SDPA Q comparison above is decisive. Do not change cache-position or RoPE-table logic unless Q itself is shown to diverge.

## Not supported by the evidence

- Rewriting cache fill/update logic: both caches match HF before and after decode.
- Changing dense QKV weight fusion or the residual/MLP path first: prefill passes at two lengths, and the decode mismatch remains localized after correct cache mutation.
- Treating `is_causal=True` plus `cur_pos_tensor` as source-equivalent merely because it is mathematically intended to mask future tokens: the emitted compiler graph explicitly selected a different TTNN API mode.

## Recommended repair order

1. Translate the emitted explicit attention mask and exact SDPA arguments.
2. Measure PCC at SDPA output before head concatenation and at final output.
3. Only if the exact mask fails, compare pre-SDPA Q and post-concat tensors to HF and repair the first divergent transform.

## Investigation note

The repo-local fresh-context AutoDebug runner was attempted first as required, but its nested Codex environment could not launch any filesystem command because `bubblewrap` was unavailable. This replacement report is the bounded source-only fallback, checked directly against the current implementation and generated emit. No TT hardware was used and no implementation, test, or documentation file was changed.
