"""Decoder layer composition (sliding-window + full).

Each layer owns its submodules (Attention, FeedForward, four RMSNorms)
and per-layer scalar tensors at __init__. The body methods reference
them via `self.X` — no `cached_main` or `layer_table` indirection.

Composition (for both kinds):
  RMSNorm(input_layernorm) → Attention → RMSNorm(post_attn_ln) →
  residual_add → RMSNorm(pre_ff_ln) → FeedForward →
  RMSNorm(post_ff_ln) → residual_add → multiply(layer_scalar)

`is_decode` is fixed at construction time; sliding-vs-full at the
class level. Runtime tensors (KV caches, position-id helpers, prelude
outputs, shared scalars) flow in per-call.
"""
import torch
from gemma4.attention import Attention
from gemma4.feed_forward import FeedForward
from gemma4.rms_norm import RMSNorm

import ttnn

_DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _load_layer_scalar(state_dict, layer_idx, mesh_device):
    """Load per-layer scalar weight as a [1,1,1] bf16 replicated tensor."""
    hf_key = f"model.language_model.layers.{layer_idx}.layer_scalar"
    torch_ls = state_dict[hf_key].reshape(1, 1, 1).to(torch.bfloat16)
    return ttnn.as_tensor(
        torch_ls,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=mesh_device,
        memory_config=_DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_norms(state_dict, layer_idx, mesh_device, rms_eps_tensor):
    """Construct the four per-layer RMSNorms from HF state_dict."""
    prefix = f"model.language_model.layers.{layer_idx}"
    return {
        "input_layernorm": RMSNorm.from_state_dict(
            state_dict,
            f"{prefix}.input_layernorm.weight",
            eps=rms_eps_tensor,
            mesh_device=mesh_device,
            role="input_layernorm",
        ),
        "post_attention_layernorm": RMSNorm.from_state_dict(
            state_dict,
            f"{prefix}.post_attention_layernorm.weight",
            eps=rms_eps_tensor,
            mesh_device=mesh_device,
            role="post_attention_layernorm",
        ),
        "pre_feedforward_layernorm": RMSNorm.from_state_dict(
            state_dict,
            f"{prefix}.pre_feedforward_layernorm.weight",
            eps=rms_eps_tensor,
            mesh_device=mesh_device,
            role="pre_feedforward_layernorm",
        ),
        "post_feedforward_layernorm": RMSNorm.from_state_dict(
            state_dict,
            f"{prefix}.post_feedforward_layernorm.weight",
            eps=rms_eps_tensor,
            mesh_device=mesh_device,
            role="post_feedforward_layernorm",
        ),
    }


class SlidingDecoderLayer:
    """One sliding-window decoder layer."""

    layer_type = "sliding"

    def __init__(
        self,
        *,
        layer_idx,
        is_decode,
        runtime_slots,
        k_cache,
        v_cache,
        attention,
        feed_forward,
        input_layernorm,
        post_attention_layernorm,
        pre_feedforward_layernorm,
        post_feedforward_layernorm,
        layer_scalar,
    ):
        self.layer_idx = layer_idx
        self._is_decode = is_decode
        # runtime_slots is (k, v, pos) historically; only the pos slot is read
        # now — K and V flow through self.k_cache / self.v_cache.
        self.runtime_slots = runtime_slots
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.attention = attention
        self.feed_forward = feed_forward
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm
        self.layer_scalar = layer_scalar

    def __call__(self, hidden_state, *, sliding_state, full_state, input, shared):
        del full_state
        pos_slot = self.runtime_slots[2]
        kv = (self.k_cache, self.v_cache, input[pos_slot])
        if self._is_decode:
            return self._decode_body(hidden_state, kv=kv, shared=shared, **sliding_state)
        else:
            return self._prefill_body(hidden_state, kv=kv, shared=shared, **sliding_state)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        layer_idx,
        mesh_device,
        *,
        is_decode,
        runtime_slots,
        k_cache,
        v_cache,
        rms_eps_tensor,
        seq_len=19,
    ):
        attention = Attention.from_state_dict_sliding(
            state_dict,
            layer_idx,
            mesh_device,
            seq_len=seq_len,
        )
        feed_forward = FeedForward.from_state_dict(
            state_dict,
            layer_idx,
            mesh_device,
        )
        norms = _build_norms(state_dict, layer_idx, mesh_device, rms_eps_tensor)
        layer_scalar = _load_layer_scalar(state_dict, layer_idx, mesh_device)
        return cls(
            layer_idx=layer_idx,
            is_decode=is_decode,
            runtime_slots=runtime_slots,
            k_cache=k_cache,
            v_cache=v_cache,
            attention=attention,
            feed_forward=feed_forward,
            layer_scalar=layer_scalar,
            **norms,
        )

    def _decode_body(
        self,
        hidden_state,
        *,
        kv,
        shared,
        causal_mask_logical_and,
        causal_mask_logical_not,
        sliding_cos_cache,
        sliding_sin_cache,
        pos_typecast_11,
    ):
        k_cache, v_cache, pos_ids = kv

        # `ttnn_multiply_18` is the local name for the residual stream;
        # the post-attention residual add and the deallocate at the bottom
        # reference it directly.
        ttnn_multiply_18 = hidden_state

        ttnn_multiply_21 = self.input_layernorm(ttnn_multiply_18)
        # The trailing 3 elements (pos-id increment, KV cache writes) are
        # side-effect ops; their tensor values are never read.
        ttnn_reshape_46, _, _, _ = self.attention(
            ttnn_multiply_21,
            is_decode=True,
            k_cache=k_cache,
            v_cache=v_cache,
            pos_ids=pos_ids,
            sliding_cos_cache=sliding_cos_cache,
            sliding_sin_cache=sliding_sin_cache,
            pos_typecast_11=pos_typecast_11,
            causal_mask_logical_and=causal_mask_logical_and,
            causal_mask_logical_not=causal_mask_logical_not,
            var_185=shared["var_185"],
            var_186=shared["var_186"],
            var_190=shared["var_190"],
            var_191=shared["var_191"],
            var_192=shared["var_192"],
            var_193=shared["var_193"],
        )
        ttnn_multiply_28 = self.post_attention_layernorm(ttnn_reshape_46)
        ttnn.deallocate(ttnn_reshape_46, False)
        ttnn_add_19 = ttnn.add(
            ttnn_multiply_18,
            ttnn_multiply_28,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_28, False)
        ttnn.deallocate(ttnn_multiply_18, False)
        ttnn_multiply_31 = self.pre_feedforward_layernorm(ttnn_add_19)
        ttnn_reshape_49 = self.feed_forward(ttnn_multiply_31)
        ttnn_multiply_35 = self.post_feedforward_layernorm(ttnn_reshape_49)
        ttnn.deallocate(ttnn_reshape_49, False)
        ttnn_add_22 = ttnn.add(
            ttnn_add_19,
            ttnn_multiply_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_35, False)
        ttnn.deallocate(ttnn_add_19, False)
        ttnn_multiply_36 = ttnn.multiply(
            ttnn_add_22,
            self.layer_scalar,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_add_22, False)

        return ttnn_multiply_36

    def _prefill_body(
        self,
        hidden_state,
        *,
        kv,
        shared,
        causal_mask_logical_and,
        causal_mask_logical_not,
        sliding_cos_cache,
        sliding_sin_cache,
        pos_reshape_15,
        pos_reshape_16,
        pos_typecast_11,
    ):
        k_cache, v_cache, pos_ids = kv

        ttnn_multiply_18 = hidden_state

        ttnn_multiply_21 = self.input_layernorm(ttnn_multiply_18)
        # Trailing 3 elements are side-effect-only (pos-id increment + KV
        # cache writes); never read by callers.
        ttnn_reshape_44, _, _, _ = self.attention(
            ttnn_multiply_21,
            is_decode=False,
            k_cache=k_cache,
            v_cache=v_cache,
            pos_ids=pos_ids,
            sliding_cos_cache=sliding_cos_cache,
            sliding_sin_cache=sliding_sin_cache,
            pos_reshape_15=pos_reshape_15,
            pos_reshape_16=pos_reshape_16,
            pos_typecast_11=pos_typecast_11,
            causal_mask_logical_and=causal_mask_logical_and,
            causal_mask_logical_not=causal_mask_logical_not,
            var_185=shared["var_185"],
            var_186=shared["var_186"],
            var_187=shared["var_187"],
            var_190=shared["var_190"],
            var_192=shared["var_192"],
            var_193=shared["var_193"],
        )
        ttnn_multiply_28 = self.post_attention_layernorm(ttnn_reshape_44)
        ttnn_add_20 = ttnn.add(
            ttnn_multiply_18,
            ttnn_multiply_28,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_28, False)
        ttnn.deallocate(ttnn_multiply_18, False)
        ttnn_multiply_31 = self.pre_feedforward_layernorm(ttnn_add_20)
        ttnn_reshape_47 = self.feed_forward(ttnn_multiply_31)
        ttnn_multiply_35 = self.post_feedforward_layernorm(ttnn_reshape_47)
        ttnn_add_23 = ttnn.add(
            ttnn_add_20,
            ttnn_multiply_35,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_35, False)
        ttnn.deallocate(ttnn_add_20, False)
        ttnn_multiply_36 = ttnn.multiply(
            ttnn_add_23,
            self.layer_scalar,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_add_23, False)

        return ttnn_multiply_36


class FullDecoderLayer:
    """One full-attention decoder layer."""

    layer_type = "full"

    def __init__(
        self,
        *,
        layer_idx,
        is_decode,
        runtime_slots,
        update_idxs_slot,
        k_cache,
        v_cache,
        attention,
        feed_forward,
        input_layernorm,
        post_attention_layernorm,
        pre_feedforward_layernorm,
        post_feedforward_layernorm,
        layer_scalar,
        is_terminal=False,
    ):
        self.layer_idx = layer_idx
        self._is_decode = is_decode
        # runtime_slots historically held (k, v, pos) (or (k, v) for L59).
        # K and V now flow through self.k_cache / self.v_cache; only the
        # pos slot is still read from input (regular full layer only).
        self.runtime_slots = runtime_slots
        self.update_idxs_slot = update_idxs_slot  # decode-only: previous sliding layer's pos_ids slot
        self.k_cache = k_cache
        self.v_cache = v_cache
        self.attention = attention
        self.feed_forward = feed_forward
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.pre_feedforward_layernorm = pre_feedforward_layernorm
        self.post_feedforward_layernorm = post_feedforward_layernorm
        self.layer_scalar = layer_scalar
        self.is_terminal = is_terminal

    def __call__(self, hidden_state, *, sliding_state, full_state, input, shared):
        del sliding_state
        if self.is_terminal:
            kv = (self.k_cache, self.v_cache, None)
        else:
            pos_slot = self.runtime_slots[2]
            kv = (self.k_cache, self.v_cache, input[pos_slot])
        if self._is_decode:
            update_idxs = input[self.update_idxs_slot] if self.update_idxs_slot is not None else None
            return self._decode_body(hidden_state, kv=kv, update_idxs=update_idxs, shared=shared, **full_state)
        else:
            return self._prefill_body(hidden_state, kv=kv, shared=shared, **full_state)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        layer_idx,
        mesh_device,
        *,
        is_decode,
        runtime_slots,
        update_idxs_slot,
        k_cache,
        v_cache,
        rms_eps_tensor,
        is_terminal=False,
        seq_len=19,
    ):
        attention = Attention.from_state_dict_full(
            state_dict,
            layer_idx,
            mesh_device,
            seq_len=seq_len,
        )
        feed_forward = FeedForward.from_state_dict(
            state_dict,
            layer_idx,
            mesh_device,
        )
        norms = _build_norms(state_dict, layer_idx, mesh_device, rms_eps_tensor)
        layer_scalar = _load_layer_scalar(state_dict, layer_idx, mesh_device)
        return cls(
            layer_idx=layer_idx,
            is_decode=is_decode,
            runtime_slots=runtime_slots,
            update_idxs_slot=update_idxs_slot,
            k_cache=k_cache,
            v_cache=v_cache,
            attention=attention,
            feed_forward=feed_forward,
            layer_scalar=layer_scalar,
            is_terminal=is_terminal,
            **norms,
        )

    def _decode_body(self, hidden_state, *, kv, update_idxs, shared, full_cos_cache, full_sin_cache, full_pos_mask):
        runtime_a, runtime_b, runtime_c = kv

        ttnn_multiply_198 = hidden_state

        ttnn_multiply_201 = self.input_layernorm(ttnn_multiply_198)
        # The pos-id increment (ttnn_add_115) is a side-effect-only op for
        # non-terminal layers; None when is_terminal.
        ttnn_reshape_226, _ = self.attention(
            ttnn_multiply_201,
            is_decode=True,
            k_cache=runtime_a,
            v_cache=runtime_b,
            pos_ids=runtime_c,
            update_idxs=update_idxs,
            full_cos_cache=full_cos_cache,
            full_sin_cache=full_sin_cache,
            full_pos_mask=full_pos_mask,
            var_185=shared["var_185"],
            var_191=shared["var_191"],
            var_193=shared["var_193"],
            compute_position_increment=not self.is_terminal,
        )
        ttnn_multiply_208 = self.post_attention_layernorm(ttnn_reshape_226)
        ttnn.deallocate(ttnn_reshape_226, False)
        ttnn_add_119 = ttnn.add(
            ttnn_multiply_198,
            ttnn_multiply_208,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_208, False)
        ttnn.deallocate(ttnn_multiply_198, False)
        ttnn_multiply_211 = self.pre_feedforward_layernorm(ttnn_add_119)
        ttnn_reshape_229 = self.feed_forward(ttnn_multiply_211)
        ttnn_multiply_215 = self.post_feedforward_layernorm(ttnn_reshape_229)
        ttnn.deallocate(ttnn_reshape_229, False)
        ttnn_add_122 = ttnn.add(
            ttnn_add_119,
            ttnn_multiply_215,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_215, False)
        ttnn.deallocate(ttnn_add_119, False)
        if self.is_terminal:
            return ttnn_add_122
        ttnn_multiply_216 = ttnn.multiply(
            ttnn_add_122,
            self.layer_scalar,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_add_122, False)

        return ttnn_multiply_216

    def _prefill_body(self, hidden_state, *, kv, shared, full_cos_cache, full_sin_cache, full_pos_mask):
        k_cache, v_cache, pos_ids = kv

        ttnn_multiply_198 = hidden_state

        ttnn_multiply_201 = self.input_layernorm(ttnn_multiply_198)
        # ttnn_add_116 is a pos-id increment side-effect op; None for
        # is_terminal.
        ttnn_reshape_209, _ = self.attention(
            ttnn_multiply_201,
            is_decode=False,
            k_cache=k_cache,
            v_cache=v_cache,
            pos_ids=pos_ids,
            full_cos_cache=full_cos_cache,
            full_sin_cache=full_sin_cache,
            full_pos_mask=full_pos_mask,
            var_185=shared["var_185"],
            var_187=shared["var_187"],
            var_193=shared["var_193"],
            compute_position_increment=not self.is_terminal,
        )
        ttnn_multiply_208 = self.post_attention_layernorm(ttnn_reshape_209)
        ttnn_add_120 = ttnn.add(
            ttnn_multiply_198,
            ttnn_multiply_208,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_208, False)
        ttnn.deallocate(ttnn_multiply_198, False)
        ttnn_multiply_211 = self.pre_feedforward_layernorm(ttnn_add_120)
        ttnn_reshape_212 = self.feed_forward(ttnn_multiply_211)
        ttnn_multiply_215 = self.post_feedforward_layernorm(ttnn_reshape_212)
        ttnn_add_123 = ttnn.add(
            ttnn_add_120,
            ttnn_multiply_215,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_multiply_215, False)
        ttnn.deallocate(ttnn_add_120, False)
        if self.is_terminal:
            return ttnn_add_123
        ttnn_multiply_216 = ttnn.multiply(
            ttnn_add_123,
            self.layer_scalar,
            dtype=ttnn.DataType.BFLOAT16,
            memory_config=_DRAM,
        )
        ttnn.deallocate(ttnn_add_123, False)

        return ttnn_multiply_216
