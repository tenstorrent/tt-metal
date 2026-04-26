"""Decoder layer composition (sliding-window + full).

Phase 1.9.3 absorbs the four legacy parameterized helpers
(_sliding_decoder_layer / _full_decoder_layer × prefill/decode) into
the SlidingDecoderLayer / FullDecoderLayer classes as private body
methods. The op sequence is verbatim — only the module-global
references `_cached__main` and `_LAYER_TABLE` are renamed to
`cached_main` and `layer_table` (passed in as __call__ kwargs).

Each decoder-layer composition is:
  RMSNorm(input_layernorm) → Attention → RMSNorm(post_attn_ln) →
  residual_add → RMSNorm(pre_ff_ln) → FeedForward →
  RMSNorm(post_ff_ln) → residual_add → multiply(layer_scalar)

`is_decode` is fixed at construction time; sliding-vs-full at the
class level. `cached_main` and `layer_table` flow in per-call so the
class never holds stale dict references.
"""
import ttnn

import gemma4


class SlidingDecoderLayer:
    """One sliding-window decoder layer."""

    def __init__(self, layer_idx, *, is_decode):
        self.layer_idx = layer_idx
        self.layer_type = "sliding"
        self._is_decode = is_decode

    def __call__(self, hidden_state, *, sliding_state, full_state, input,
                 cached_main, layer_table):
        del full_state
        if self._is_decode:
            return self._decode_body(
                hidden_state,
                **sliding_state,
                input=input,
                cached_main=cached_main,
                layer_table=layer_table,
            )
        else:
            return self._prefill_body(
                hidden_state,
                **sliding_state,
                input=input,
                cached_main=cached_main,
                layer_table=layer_table,
            )

    @classmethod
    def from_consteval(cls, layer_idx, *, is_decode):
        return cls(layer_idx, is_decode=is_decode)

    def _decode_body(self, hidden_state, *, causal_mask_logical_and, causal_mask_logical_not, sliding_cos_cache, sliding_sin_cache, pos_typecast_11, input, cached_main, layer_table):
        layer_idx = self.layer_idx
        """Parameterized sliding-attention decoder layer. Resolves per-layer
        weights and per-layer runtime input slots from `layer_table[layer_idx]`.
        Shared consteval scalars are looked up directly from `cached_main`.
        The op sequence is bit-identical to the original inlined layer body.
        """
        t = layer_table[layer_idx]
        weights = t["weights"]

        # Per-layer weights (consteval-cached)
        input_layernorm_w = cached_main[f"main_const_eval_{weights['input_layernorm']}"][0]
        fused_qkv_w       = cached_main[f"main_const_eval_{weights['fused_qkv']}"][0]
        o_proj_w          = cached_main[f"main_const_eval_{weights['o_proj']}"][0]
        post_attn_ln_w    = cached_main[f"main_const_eval_{weights['post_attn_ln']}"][0]
        pre_ff_ln_w       = cached_main[f"main_const_eval_{weights['pre_ff_ln']}"][0]
        gate_proj_w       = cached_main[f"main_const_eval_{weights['gate_proj']}"][0]
        up_proj_w         = cached_main[f"main_const_eval_{weights['up_proj']}"][0]
        down_proj_w       = cached_main[f"main_const_eval_{weights['down_proj']}"][0]
        post_ff_ln_w      = cached_main[f"main_const_eval_{weights['post_ff_ln']}"][0]
        layer_scalar_w    = cached_main[f"main_const_eval_{weights['layer_scalar']}"][0]

        # Per-layer raw inputs (q/k norm weights are not consteval-cast)
        q_norm_w = input[t["q_norm_input"]]
        k_norm_w = input[t["k_norm_input"]]

        # Per-layer runtime inputs (KV cache / position id slots)
        rs = t["runtime_inputs"]
        runtime_a = input[rs[0]]
        runtime_b = input[rs[1]]
        runtime_c = input[rs[2]]

        # Shared consteval scalars (same for every layer)
        var_185 = cached_main["main_const_eval_0"][1]
        var_186 = cached_main["main_const_eval_0"][2]
        var_188 = cached_main["main_const_eval_240"][0]
        var_190 = cached_main["main_const_eval_334"][0]
        var_191 = cached_main["main_const_eval_337"][0]
        var_192 = cached_main["main_const_eval_486"][0]
        var_193 = cached_main["main_const_eval_489"][0]

        # Aliases that match the verbatim body op names
        ttnn_multiply_18 = hidden_state
        ttnn_logical_and_0 = causal_mask_logical_and
        ttnn_logical_not_0 = causal_mask_logical_not
        ttnn_typecast_2 = sliding_cos_cache
        ttnn_typecast_3 = sliding_sin_cache
        ttnn_typecast_11 = pos_typecast_11

        ttnn_multiply_21 = gemma4.RMSNorm(input_layernorm_w, var_188)(ttnn_multiply_18)
        ttnn_reshape_46, ttnn_add_15, ttnn_where_8, ttnn_where_10 = gemma4.Attention(
            layer_type="sliding",
            fused_qkv_w=fused_qkv_w,
            q_norm_w=q_norm_w, k_norm_w=k_norm_w, o_proj_w=o_proj_w,
        )(
            ttnn_multiply_21,
            is_decode=True,
            k_cache=runtime_a, v_cache=runtime_b, pos_ids=runtime_c,
            sliding_cos_cache=ttnn_typecast_2, sliding_sin_cache=ttnn_typecast_3,
            pos_typecast_11=ttnn_typecast_11,
            causal_mask_logical_and=ttnn_logical_and_0,
            causal_mask_logical_not=ttnn_logical_not_0,
            var_185=var_185, var_186=var_186,
            var_190=var_190, var_191=var_191, var_192=var_192, var_193=var_193,
        )
        ttnn_multiply_28 = gemma4.RMSNorm(post_attn_ln_w, var_188)(ttnn_reshape_46)
        ttnn.deallocate(ttnn_reshape_46, False)
        ttnn_add_19 = ttnn.add(
            ttnn_multiply_18,
            ttnn_multiply_28,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_28, False)
        ttnn.deallocate(ttnn_multiply_18, False)
        ttnn_multiply_31 = gemma4.RMSNorm(pre_ff_ln_w, var_188)(ttnn_add_19)
        ttnn_reshape_49 = gemma4.FeedForward(gate_proj_w, up_proj_w, down_proj_w)(ttnn_multiply_31)
        ttnn_multiply_35 = gemma4.RMSNorm(post_ff_ln_w, var_188)(ttnn_reshape_49)
        ttnn.deallocate(ttnn_reshape_49, False)
        ttnn_add_22 = ttnn.add(
            ttnn_add_19,
            ttnn_multiply_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_35, False)
        ttnn.deallocate(ttnn_add_19, False)
        ttnn_multiply_36 = ttnn.multiply(
            ttnn_add_22,
            layer_scalar_w,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_22, False)


        return (
            ttnn_multiply_36,
            ttnn_add_15,
            ttnn_where_8,
            ttnn_where_10,
        )

    def _prefill_body(self, hidden_state, *, causal_mask_logical_and, causal_mask_logical_not, sliding_cos_cache, sliding_sin_cache, pos_reshape_15, pos_reshape_16, pos_typecast_11, input, cached_main, layer_table):
        layer_idx = self.layer_idx
        """Parameterized sliding-attention decoder layer. Resolves per-layer
        weights and per-layer runtime input slots from `layer_table[layer_idx]`.
        Shared consteval scalars are looked up directly from `cached_main`.
        The op sequence is bit-identical to the original inlined layer body.
        """
        t = layer_table[layer_idx]
        weights = t["weights"]

        # Per-layer weights (consteval-cached)
        input_layernorm_w = cached_main[f"main_const_eval_{weights['input_layernorm']}"][0]
        fused_qkv_w       = cached_main[f"main_const_eval_{weights['fused_qkv']}"][0]
        o_proj_w          = cached_main[f"main_const_eval_{weights['o_proj']}"][0]
        post_attn_ln_w    = cached_main[f"main_const_eval_{weights['post_attn_ln']}"][0]
        pre_ff_ln_w       = cached_main[f"main_const_eval_{weights['pre_ff_ln']}"][0]
        gate_proj_w       = cached_main[f"main_const_eval_{weights['gate_proj']}"][0]
        up_proj_w         = cached_main[f"main_const_eval_{weights['up_proj']}"][0]
        down_proj_w       = cached_main[f"main_const_eval_{weights['down_proj']}"][0]
        post_ff_ln_w      = cached_main[f"main_const_eval_{weights['post_ff_ln']}"][0]
        layer_scalar_w    = cached_main[f"main_const_eval_{weights['layer_scalar']}"][0]

        # Per-layer raw inputs (q/k norm weights are not consteval-cast)
        q_norm_w = input[t["q_norm_input"]]
        k_norm_w = input[t["k_norm_input"]]

        # Per-layer runtime inputs (causal-mask / scalar slots)
        rs = t["runtime_inputs"]
        k_cache = input[rs[0]]
        v_cache = input[rs[1]]
        pos_ids = input[rs[2]]

        # Shared consteval scalars (same for every layer)
        var_185 = cached_main["main_const_eval_0"][1]
        var_186 = cached_main["main_const_eval_0"][2]
        var_187 = cached_main["main_const_eval_186"][0]
        var_188 = cached_main["main_const_eval_240"][0]
        var_190 = cached_main["main_const_eval_266"][0]
        var_192 = cached_main["main_const_eval_335"][0]
        var_193 = cached_main["main_const_eval_338"][0]

        # `ttnn_multiply_18` is the local name for the residual stream; the
        # later `ttnn.add` step (post-attention residual) and the final
        # `ttnn.deallocate` reference it directly.
        ttnn_multiply_18 = hidden_state

        ttnn_multiply_21 = gemma4.RMSNorm(input_layernorm_w, var_188)(ttnn_multiply_18)
        ttnn_reshape_44, ttnn_add_16, ttnn_where_8, ttnn_where_10 = gemma4.Attention(
            layer_type="sliding",
            fused_qkv_w=fused_qkv_w,
            q_norm_w=q_norm_w, k_norm_w=k_norm_w, o_proj_w=o_proj_w,
        )(
            ttnn_multiply_21,
            is_decode=False,
            k_cache=k_cache, v_cache=v_cache, pos_ids=pos_ids,
            sliding_cos_cache=sliding_cos_cache, sliding_sin_cache=sliding_sin_cache,
            pos_reshape_15=pos_reshape_15, pos_reshape_16=pos_reshape_16,
            pos_typecast_11=pos_typecast_11,
            causal_mask_logical_and=causal_mask_logical_and,
            causal_mask_logical_not=causal_mask_logical_not,
            var_185=var_185, var_186=var_186, var_187=var_187,
            var_190=var_190, var_192=var_192, var_193=var_193,
        )
        ttnn_multiply_28 = gemma4.RMSNorm(post_attn_ln_w, var_188)(ttnn_reshape_44)
        ttnn_add_20 = ttnn.add(
            ttnn_multiply_18,
            ttnn_multiply_28,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_28, False)
        ttnn.deallocate(ttnn_multiply_18, False)
        ttnn_multiply_31 = gemma4.RMSNorm(pre_ff_ln_w, var_188)(ttnn_add_20)
        ttnn_reshape_47 = gemma4.FeedForward(gate_proj_w, up_proj_w, down_proj_w)(ttnn_multiply_31)
        ttnn_multiply_35 = gemma4.RMSNorm(post_ff_ln_w, var_188)(ttnn_reshape_47)
        ttnn_add_23 = ttnn.add(
            ttnn_add_20,
            ttnn_multiply_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_35, False)
        ttnn.deallocate(ttnn_add_20, False)
        ttnn_multiply_36 = ttnn.multiply(
            ttnn_add_23,
            layer_scalar_w,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_23, False)

        return (
            ttnn_multiply_36,
            ttnn_add_16,
            ttnn_where_8,
            ttnn_where_10,
        )


class FullDecoderLayer:
    """One full-attention decoder layer."""

    def __init__(self, layer_idx, *, is_decode):
        self.layer_idx = layer_idx
        self.layer_type = "full"
        self._is_decode = is_decode

    def __call__(self, hidden_state, *, sliding_state, full_state, input,
                 cached_main, layer_table):
        del sliding_state
        if self._is_decode:
            return self._decode_body(
                hidden_state,
                **full_state,
                input=input,
                cached_main=cached_main,
                layer_table=layer_table,
            )
        else:
            return self._prefill_body(
                hidden_state,
                **full_state,
                input=input,
                cached_main=cached_main,
                layer_table=layer_table,
            )

    @classmethod
    def from_consteval(cls, layer_idx, *, is_decode):
        return cls(layer_idx, is_decode=is_decode)

    def _decode_body(self, hidden_state, *, full_cos_cache, full_sin_cache, full_pos_mask, input, cached_main, layer_table):
        layer_idx = self.layer_idx
        """Parameterized full-attention decoder layer. Uses layer_table[layer_idx]
        to resolve per-layer weights and runtime input slots; shared consteval
        scalars are looked up from `cached_main` directly. Op sequence is
        bit-identical to the original inlined layer body (template: layer 11).

        Note: full attention does NOT consume the sliding causal-mask helpers
        (`ttnn_logical_and_0` / `ttnn_logical_not_0`); its mask comes entirely
        from `full_pos_mask` produced by `_full_prelude`.

        Decode-specific: full attention uses `ttnn.experimental.paged_update_cache`
        to write the new K state into the K cache. The `update_idxs_tensor` slot
        is the previous sliding layer's `runtime_c` slot (= the decoder's per-layer
        position-id helper), looked up via layer_table[layer_idx - 1]["runtime_inputs"][2].
        """
        t = layer_table[layer_idx]
        weights = t["weights"]

        input_layernorm_w = cached_main[f"main_const_eval_{weights['input_layernorm']}"][0]
        q_proj_w          = cached_main[f"main_const_eval_{weights['q_proj']}"][0]
        k_proj_w          = cached_main[f"main_const_eval_{weights['k_proj']}"][0]
        o_proj_w          = cached_main[f"main_const_eval_{weights['o_proj']}"][0]
        post_attn_ln_w    = cached_main[f"main_const_eval_{weights['post_attn_ln']}"][0]
        pre_ff_ln_w       = cached_main[f"main_const_eval_{weights['pre_ff_ln']}"][0]
        gate_proj_w       = cached_main[f"main_const_eval_{weights['gate_proj']}"][0]
        up_proj_w         = cached_main[f"main_const_eval_{weights['up_proj']}"][0]
        down_proj_w       = cached_main[f"main_const_eval_{weights['down_proj']}"][0]
        post_ff_ln_w      = cached_main[f"main_const_eval_{weights['post_ff_ln']}"][0]
        layer_scalar_w    = cached_main[f"main_const_eval_{weights['layer_scalar']}"][0]

        q_norm_w = input[t["q_norm_input"]]
        k_norm_w = input[t["k_norm_input"]]

        rs = t["runtime_inputs"]
        runtime_a = input[rs[0]]
        runtime_b = input[rs[1]]
        runtime_c = input[rs[2]]
        update_idxs = input[layer_table[layer_idx - 1]["runtime_inputs"][2]]

        # Shared consteval scalars
        var_185 = cached_main["main_const_eval_0"][1]
        var_188 = cached_main["main_const_eval_240"][0]
        var_191 = cached_main["main_const_eval_337"][0]
        var_193 = cached_main["main_const_eval_489"][0]

        # Aliases that match the verbatim body op names
        ttnn_multiply_198 = hidden_state
        ttnn_typecast_35 = full_cos_cache
        ttnn_typecast_36 = full_sin_cache
        ttnn_typecast_39 = full_pos_mask

        ttnn_multiply_201 = gemma4.RMSNorm(input_layernorm_w, var_188)(ttnn_multiply_198)
        ttnn_reshape_226, ttnn_add_115 = gemma4.Attention(
            layer_type="full",
            q_proj_w=q_proj_w, k_proj_w=k_proj_w, o_proj_w=o_proj_w,
            q_norm_w=q_norm_w, k_norm_w=k_norm_w,
        )(
            ttnn_multiply_201,
            is_decode=True,
            k_cache=runtime_a, v_cache=runtime_b, pos_ids=runtime_c,
            update_idxs=update_idxs,
            full_cos_cache=ttnn_typecast_35, full_sin_cache=ttnn_typecast_36,
            full_pos_mask=ttnn_typecast_39,
            var_185=var_185, var_191=var_191, var_193=var_193,
        )
        ttnn_multiply_208 = gemma4.RMSNorm(post_attn_ln_w, var_188)(ttnn_reshape_226)
        ttnn.deallocate(ttnn_reshape_226, False)
        ttnn_add_119 = ttnn.add(
            ttnn_multiply_198,
            ttnn_multiply_208,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_208, False)
        ttnn.deallocate(ttnn_multiply_198, False)
        ttnn_multiply_211 = gemma4.RMSNorm(pre_ff_ln_w, var_188)(ttnn_add_119)
        ttnn_reshape_229 = gemma4.FeedForward(gate_proj_w, up_proj_w, down_proj_w)(ttnn_multiply_211)
        ttnn_multiply_215 = gemma4.RMSNorm(post_ff_ln_w, var_188)(ttnn_reshape_229)
        ttnn.deallocate(ttnn_reshape_229, False)
        ttnn_add_122 = ttnn.add(
            ttnn_add_119,
            ttnn_multiply_215,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_215, False)
        ttnn.deallocate(ttnn_add_119, False)
        ttnn_multiply_216 = ttnn.multiply(
            ttnn_add_122,
            layer_scalar_w,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_122, False)

        return (
            ttnn_multiply_216,
            ttnn_add_115,
        )

    def _prefill_body(self, hidden_state, *, full_cos_cache, full_sin_cache, full_pos_mask, input, cached_main, layer_table):
        layer_idx = self.layer_idx
        """Parameterized full-attention decoder layer. Uses layer_table[layer_idx]
        to resolve per-layer weights and runtime input slots; shared consteval
        scalars are looked up from `cached_main` directly. Op sequence is
        bit-identical to the original inlined layer body (template: layer 11).

        Note: full attention does NOT consume the sliding causal-mask helpers
        (`ttnn_logical_and_0` / `ttnn_logical_not_0`); its mask comes entirely
        from `full_pos_mask` produced by `_full_prelude`.
        """
        t = layer_table[layer_idx]
        weights = t["weights"]

        input_layernorm_w = cached_main[f"main_const_eval_{weights['input_layernorm']}"][0]
        q_proj_w          = cached_main[f"main_const_eval_{weights['q_proj']}"][0]
        k_proj_w          = cached_main[f"main_const_eval_{weights['k_proj']}"][0]
        o_proj_w          = cached_main[f"main_const_eval_{weights['o_proj']}"][0]
        post_attn_ln_w    = cached_main[f"main_const_eval_{weights['post_attn_ln']}"][0]
        pre_ff_ln_w       = cached_main[f"main_const_eval_{weights['pre_ff_ln']}"][0]
        gate_proj_w       = cached_main[f"main_const_eval_{weights['gate_proj']}"][0]
        up_proj_w         = cached_main[f"main_const_eval_{weights['up_proj']}"][0]
        down_proj_w       = cached_main[f"main_const_eval_{weights['down_proj']}"][0]
        post_ff_ln_w      = cached_main[f"main_const_eval_{weights['post_ff_ln']}"][0]
        layer_scalar_w    = cached_main[f"main_const_eval_{weights['layer_scalar']}"][0]

        q_norm_w = input[t["q_norm_input"]]
        k_norm_w = input[t["k_norm_input"]]

        rs = t["runtime_inputs"]
        k_cache = input[rs[0]]
        v_cache = input[rs[1]]
        pos_ids = input[rs[2]]

        var_185 = cached_main["main_const_eval_0"][1]
        var_187 = cached_main["main_const_eval_186"][0]
        var_188 = cached_main["main_const_eval_240"][0]
        var_193 = cached_main["main_const_eval_338"][0]

        # `ttnn_multiply_198` is the local name for the residual stream; the
        # later `ttnn.add` (post-attention residual) and final `ttnn.deallocate`
        # reference it directly.
        ttnn_multiply_198 = hidden_state

        ttnn_multiply_201 = gemma4.RMSNorm(input_layernorm_w, var_188)(ttnn_multiply_198)
        ttnn_reshape_209, ttnn_add_116 = gemma4.Attention(
            layer_type="full",
            q_proj_w=q_proj_w, k_proj_w=k_proj_w, o_proj_w=o_proj_w,
            q_norm_w=q_norm_w, k_norm_w=k_norm_w,
        )(
            ttnn_multiply_201,
            is_decode=False,
            k_cache=k_cache, v_cache=v_cache, pos_ids=pos_ids,
            full_cos_cache=full_cos_cache,
            full_sin_cache=full_sin_cache,
            full_pos_mask=full_pos_mask,
            var_185=var_185, var_187=var_187, var_193=var_193,
        )
        ttnn_multiply_208 = gemma4.RMSNorm(post_attn_ln_w, var_188)(ttnn_reshape_209)
        ttnn_add_120 = ttnn.add(
            ttnn_multiply_198,
            ttnn_multiply_208,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_208, False)
        ttnn.deallocate(ttnn_multiply_198, False)
        ttnn_multiply_211 = gemma4.RMSNorm(pre_ff_ln_w, var_188)(ttnn_add_120)
        ttnn_reshape_212 = gemma4.FeedForward(gate_proj_w, up_proj_w, down_proj_w)(ttnn_multiply_211)
        ttnn_multiply_215 = gemma4.RMSNorm(post_ff_ln_w, var_188)(ttnn_reshape_212)
        ttnn_add_123 = ttnn.add(
            ttnn_add_120,
            ttnn_multiply_215,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_multiply_215, False)
        ttnn.deallocate(ttnn_add_120, False)
        ttnn_multiply_216 = ttnn.multiply(
            ttnn_add_123,
            layer_scalar_w,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )
        ttnn.deallocate(ttnn_add_123, False)

        return (
            ttnn_multiply_216,
            ttnn_add_116,
        )
