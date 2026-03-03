# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 model implementation using ttml ops and custom autograd functions.

Architecture: HuggingFace Transformers Qwen3 design with:
  - Separate Q, K, V projections (not fused KV)
  - QK-Norm (RMSNorm on Q and K per head_dim)
  - Configurable attention bias
  - SwiGLU MLP
  - RoPE positional encoding
"""

from dataclasses import dataclass

import torch
from tqdm import tqdm
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, ModuleList, Parameter

from utils.checkpoint import (  # noqa: F401 — re-exported for callers
    CheckpointFunction,
    checkpoint,
)
from utils.tensor_utils import (
    get_device as _get_device,
    torch_to_ttml as _torch_to_ttml,
    tile_pad as _tile_pad,
    make_empty_on_device as _make_empty_on_device,
    make_weight as _make_weight,
    make_ones as _make_ones,
    make_zeros as _make_zeros,
)
from utils.param_utils import (  # noqa: F401 — re-exported for callers
    unpermute_proj_rows,
    unpermute_norm_weights,
    repermute_proj_rows,
    repermute_norm_weights,
    build_weight_mapping_single,
)

# Memory tracking utilities
MemoryUsageTracker = ttml.core.utils.MemoryUsageTracker


# =====================================================================
# Configuration
# =====================================================================


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    attention_bias: bool = True
    attention_dropout: float = 0.0
    rope_theta: float = 1000000.0
    rope_scaling_factor: float = 0.0
    rope_original_context_length: int = 0
    rope_high_freq_factor: float = 4.0
    rope_low_freq_factor: float = 1.0


def create_qwen3_config_from_hf(hf_config, max_sequence_length: int) -> Qwen3Config:
    """Create Qwen3Config from a HuggingFace config object."""
    rope_scaling_factor = 0.0
    rope_original_context_length = 0
    rope_high_freq_factor = 4.0
    rope_low_freq_factor = 1.0
    if hasattr(hf_config, "rope_scaling") and hf_config.rope_scaling:
        rs = hf_config.rope_scaling
        rope_scaling_factor = rs.get("factor", 0.0)
        rope_original_context_length = rs.get("original_max_position_embeddings", 0)
        rope_high_freq_factor = rs.get("high_freq_factor", 4.0)
        rope_low_freq_factor = rs.get("low_freq_factor", 1.0)

    return Qwen3Config(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        max_position_embeddings=max_sequence_length,
        rms_norm_eps=hf_config.rms_norm_eps,
        attention_bias=getattr(hf_config, "attention_bias", True),
        rope_theta=getattr(hf_config, "rope_theta", 1000000.0),
        rope_scaling_factor=rope_scaling_factor,
        rope_original_context_length=rope_original_context_length,
        rope_high_freq_factor=rope_high_freq_factor,
        rope_low_freq_factor=rope_low_freq_factor,
    )


from utils.context_managers import is_empty_init


# =====================================================================
# Custom autograd: ConcatLastDim
# =====================================================================


class ConcatLastDim(ttml.autograd.Function):
    """Concatenate two tensors along the last dimension."""

    @staticmethod
    def forward(ctx, a, b):
        a_shape = a.shape()
        b_shape = b.shape()
        ctx.save_for_backward(a_shape, b_shape)
        a_ttnn = a.get_value()
        b_ttnn = b.get_value()
        result = ttnn.concat([a_ttnn, b_ttnn], dim=-1)
        return ttml.autograd.create_tensor(result)

    @staticmethod
    def backward(ctx, grad_output):
        a_shape, b_shape = ctx.saved_tensors
        a_last = a_shape[-1]
        # grad_output is a raw tt_metal::Tensor (from Tensor.get_grad()),
        # NOT an autograd Tensor, so use it directly with ttnn ops.
        grad_a = ttnn.slice(
            grad_output, [0, 0, 0, 0], [a_shape[0], a_shape[1], a_shape[2], a_last]
        )
        grad_b = ttnn.slice(
            grad_output,
            [0, 0, 0, a_last],
            [b_shape[0], b_shape[1], b_shape[2], a_last + b_shape[-1]],
        )
        # Return raw tt_metal::Tensors; the autograd framework handles accumulation.
        return grad_a, grad_b


# =====================================================================
# Custom autograd: MemorySnapshotFunction (identity with memory tracking)
# =====================================================================


class MemorySnapshotFunction(ttml.autograd.Function):
    """Identity op that captures a MemoryUsageTracker snapshot on forward
    and/or backward.  Zero computational overhead — just passes through the
    tensor value and gradient unchanged.
    """

    @staticmethod
    def forward(ctx, x, fwd_label, bwd_label):
        ctx.bwd_label = bwd_label
        if fwd_label:
            MemoryUsageTracker.snapshot(fwd_label)
        return x.get_value()

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.bwd_label:
            MemoryUsageTracker.snapshot(ctx.bwd_label)
        return grad_output


def memory_snapshot(x, fwd_label="", bwd_label=""):
    """Identity wrapper that records memory snapshots during forward/backward.

    Inserts a no-op node into the autograd graph.  When *fwd_label* is set a
    snapshot is taken during the forward pass; when *bwd_label* is set a
    snapshot is taken when the gradient flows back through this point.
    """
    if not fwd_label and not bwd_label:
        return x
    return MemorySnapshotFunction.apply(x, fwd_label, bwd_label)


# =====================================================================
# Custom autograd: RMSNormFunction
# =====================================================================


class RMSNormFunction(ttml.autograd.Function):
    """RMSNorm via rsqrt: y = (x · rsqrt(mean(x², dim=-1) + ε)) · weight.

    Replaces rmsnorm_composite with explicit rsqrt (instead of sqrt +
    division) for direct control of forward/backward on Tenstorrent devices.

    Works with any 4-D shape as long as the last dim matches the weight.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        x_val = x.get_value()
        w_val = weight.get_value()

        # variance = mean(x², dim=-1) + ε  →  rrms = rsqrt(variance)
        x_sq = ttnn.mul(x_val, x_val)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)  # (B, 1, S, 1)
        variance = ttnn.add(mean_sq, eps)  # (B, 1, S, 1)
        rrms = ttnn.rsqrt(variance)  # (B, 1, S, 1)

        # x̂ = x · rrms,  y = x̂ · w
        x_hat = ttnn.mul(x_val, rrms)  # (B, 1, S, D)
        out = ttnn.mul(x_hat, w_val)  # (B, 1, S, D)

        ctx.save_for_backward(x_hat, rrms, w_val)
        return ttml.autograd.create_tensor(out)

    @staticmethod
    def backward(ctx, grad_output):
        x_hat, rrms, w_val = ctx.saved_tensors

        # ---- grad_w: dL/dw = Σ over all dims except last  →  (1, 1, 1, D) ----
        grad_w = ttnn.mul(grad_output, x_hat)
        for d in range(3):
            grad_w = ttnn.sum(grad_w, dim=d, keepdim=True)

        # ---- grad_x = rrms · (g⊙w − x̂ · mean(g⊙w⊙x̂, dim=-1)) ----
        gw = ttnn.mul(grad_output, w_val)  # (B, 1, S, D)
        dot = ttnn.mul(gw, x_hat)  # (B, 1, S, D)
        dot_mean = ttnn.mean(dot, dim=-1, keepdim=True)  # (B, 1, S, 1)
        correction = ttnn.mul(x_hat, dot_mean)  # (B, 1, S, D)
        grad_x = ttnn.mul(ttnn.subtract(gw, correction), rrms)  # (B, 1, S, D)

        return grad_x, grad_w


def linear(x, weight, bias=None):
    # return LinearFunction.apply(x, weight, bias)
    return ttml.ops.linear.linear(x, weight, bias)


# =====================================================================
# Qwen3RMSNorm
# =====================================================================


class Qwen3RMSNorm(AbstractModuleBase):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.weight = Parameter(_make_ones((1, 1, 1, hidden_size)))

    def forward(self, hidden_states):
        # requires for 14B/32B backward, without throws an error ttml::metal::rmsnorm_bw
        # "Statically allocated circular buffers on core range [(x=0,y=0) - (x=4,y=3)]
        # grow to 1764640 B which is beyond max L1 size of 1499136 B"
        return RMSNormFunction.apply(hidden_states, self.weight.tensor, self.eps)
        # return ttml.ops.rmsnorm.rmsnorm(hidden_states, self.weight.tensor, self.eps)


# =====================================================================
# LinearProjection
# =====================================================================


class LinearProjection(AbstractModuleBase):
    """Linear projection layer: y = x @ W^T + optional_bias.

    Weight shape: (1, 1, out_features, in_features).
    Used as the base module for LoRA injection via inject_adapter_in_model.
    """

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False):
        super().__init__()
        self.weight = Parameter(_make_weight((1, 1, out_features, in_features)))
        if has_bias:
            self.bias = Parameter(_make_zeros((1, 1, 1, out_features)))
        else:
            self.bias = None

    def forward(self, x):
        b = self.bias.tensor if self.bias is not None else None
        return linear(x, self.weight.tensor, b)


# =====================================================================
# Qwen3Attention (single device)
# =====================================================================


class Qwen3Attention(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size

        q_out = self.num_heads * self.head_dim
        kv_out = self.num_kv_heads * self.head_dim

        self.q_proj = LinearProjection(
            self.hidden_size, q_out, has_bias=config.attention_bias
        )
        self.k_proj = LinearProjection(
            self.hidden_size, kv_out, has_bias=config.attention_bias
        )
        self.v_proj = LinearProjection(
            self.hidden_size, kv_out, has_bias=config.attention_bias
        )
        self.o_proj = LinearProjection(
            q_out, self.hidden_size, has_bias=config.attention_bias
        )

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_scaling = ttml.ops.rope.RopeScalingParams()
        if (
            config.rope_scaling_factor != 0.0
            and config.rope_original_context_length != 0
        ):
            rope_scaling.original_context_length = config.rope_original_context_length
            rope_scaling.scaling_factor = config.rope_scaling_factor
            rope_scaling.high_freq_factor = config.rope_high_freq_factor
            rope_scaling.low_freq_factor = config.rope_low_freq_factor

        self.rope_params = ttml.ops.rope.build_rope_params(
            sequence_length=config.max_position_embeddings,
            head_dim=self.head_dim,
            theta=config.rope_theta,
            rope_scaling_params=rope_scaling,
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_offset=0,
    ):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_shape = q.shape()
        k_shape = k.shape()
        B, S = q_shape[0], q_shape[2]

        q = ttml.ops.reshape.reshape(q, [B, 1, S * self.num_heads, self.head_dim])
        k = ttml.ops.reshape.reshape(k, [B, 1, S * self.num_kv_heads, self.head_dim])
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = ttml.ops.reshape.reshape(q, q_shape)
        k = ttml.ops.reshape.reshape(k, k_shape)

        kvs = ConcatLastDim.apply(k, v)
        (
            query_heads,
            key_heads,
            value_heads,
        ) = ttml.ops.multi_head_utils.grouped_heads_creation(
            q, kvs, self.num_heads, self.num_kv_heads
        )

        query_heads = ttml.ops.rope.rope(query_heads, self.rope_params, position_offset)
        key_heads = ttml.ops.rope.rope(key_heads, self.rope_params, position_offset)

        # KV cache: append new K/V and use full history for attention
        if past_key_values is not None:
            key_heads, value_heads = past_key_values.update(
                self.layer_idx, key_heads, value_heads
            )

        attn = ttml.ops.attention.scaled_dot_product_attention(
            query_heads, key_heads, value_heads, attention_mask
        )

        attn_output = ttml.ops.multi_head_utils.heads_fusion(attn)
        return self.o_proj(attn_output)


# =====================================================================
# Qwen3MLP (single device)
# =====================================================================


class Qwen3MLP(AbstractModuleBase):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        h = config.hidden_size
        inter = config.intermediate_size
        self.gate_proj = LinearProjection(h, inter)
        self.up_proj = LinearProjection(h, inter)
        self.down_proj = LinearProjection(inter, h)

    def forward(self, x):
        gate = ttml.ops.unary.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(ttml.ops.binary.mul(gate, up))


# =====================================================================
# Qwen3DecoderLayer
# =====================================================================


class Qwen3DecoderLayer(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        position_offset=0,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask,
            past_key_values,
            position_offset=position_offset,
        )
        hidden_states = ttml.ops.binary.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttml.ops.binary.add(residual, hidden_states)
        return hidden_states


# =====================================================================
# Qwen3Model (backbone)
# =====================================================================


class Qwen3Model(AbstractModuleBase):
    def __init__(self, config: Qwen3Config, track_memory=0, use_checkpoint=False):
        super().__init__()
        self.config = config
        self.track_memory = track_memory
        self.use_checkpoint = use_checkpoint
        vocab_size_tiled = ((config.vocab_size + 31) // 32) * 32
        self.embed_tokens = Parameter(
            _make_weight((1, 1, vocab_size_tiled, config.hidden_size))
        )
        self.layers = ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        hidden_states = ttml.ops.embedding.embedding(
            input_ids, self.embed_tokens.tensor
        )
        if self.track_memory:
            hidden_states = memory_snapshot(
                hidden_states, "AFTER_EMBEDDING_FWD", "AFTER_EMBEDDING_BWD"
            )
        position_offset = 0
        if past_key_values is not None:
            position_offset = past_key_values.get_seq_length()
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint:
                hidden_states = checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    position_offset=position_offset,
                )
            if self.track_memory and (i + 1) % self.track_memory == 0:
                hidden_states = memory_snapshot(
                    hidden_states, f"AFTER_LAYER_{i}_FWD", f"AFTER_LAYER_{i}_BWD"
                )
        hidden_states = self.norm(hidden_states)
        return hidden_states


# =====================================================================
# Qwen3ForCausalLM
# =====================================================================


class Qwen3ForCausalLM(AbstractModuleBase):
    def __init__(
        self,
        config: Qwen3Config,
        tie_word_embeddings: bool = False,
        track_memory: int = 0,
        use_checkpoint=False,
    ):
        super().__init__()
        self.create_name("Qwen3ForCausalLM")
        self.config = config
        self.tie_word_embeddings = tie_word_embeddings
        self.track_memory = track_memory
        self.model = Qwen3Model(
            config, track_memory=track_memory, use_checkpoint=use_checkpoint
        )

        if tie_word_embeddings:
            self.lm_head_weight = None
        else:
            vocab_size_tiled = ((config.vocab_size + 31) // 32) * 32
            self.lm_head_weight = Parameter(
                _make_weight((1, 1, vocab_size_tiled, config.hidden_size))
            )

    def forward(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask, past_key_values)
        if self.track_memory:
            hidden_states = memory_snapshot(
                hidden_states, "AFTER_NORM_FWD", "AFTER_NORM_BWD"
            )
        if self.tie_word_embeddings:
            logits = linear(hidden_states, self.model.embed_tokens.tensor, None)
        else:
            logits = linear(hidden_states, self.lm_head_weight.tensor, None)
        if self.track_memory:
            logits = memory_snapshot(logits, "AFTER_LM_HEAD_FWD", "AFTER_LM_HEAD_BWD")
        return logits


# =====================================================================
# Weight loading from HuggingFace
# =====================================================================


def load_weights_from_hf(
    ttml_model: Qwen3ForCausalLM,
    hf_state_dict: dict,
    config: Qwen3Config,
    tie_word_embeddings: bool = False,
    verbose: bool = False,
) -> None:
    """Load HF weights into single-device ttml model."""
    ttml_params = ttml_model.parameters()

    if verbose:
        print("\n  TTML parameter names:")
        for name in sorted(ttml_params.keys()):
            shape = ttml_params[name].shape()
            print(f"    {name}: {list(shape)}")

    any_key = next(iter(ttml_params))
    root_prefix = any_key.split("/")[0]

    mapping, transforms = build_weight_mapping_single(
        config, root_prefix, tie_word_embeddings
    )

    ttml_shapes = {name: list(ttml_params[name].shape()) for name in ttml_params}

    def _prepare_and_transfer(hf_name, ttml_name):
        """CPU prep + host-side conversion + device transfer (pipelined)."""
        if hf_name not in hf_state_dict:
            return None
        if ttml_name not in ttml_shapes:
            return None

        weight = hf_state_dict[hf_name].float()

        if hf_name in transforms:
            tr = transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = ttml_shapes[ttml_name]

        if weight.dim() == 2:
            rows, cols = weight.shape
            tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return _torch_to_ttml(weight)

    from concurrent.futures import ThreadPoolExecutor

    items = list(mapping.items())
    loaded = 0
    skipped = []

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            (hf_name, ttml_name, pool.submit(_prepare_and_transfer, hf_name, ttml_name))
            for hf_name, ttml_name in items
        ]

        for hf_name, ttml_name, future in tqdm(
            futures,
            total=len(items),
            desc="  Loading weights",
            unit="w",
        ):
            new_tensor = future.result()
            if new_tensor is None:
                if ttml_name not in ttml_shapes:
                    print(
                        f"  WARNING: ttml param '{ttml_name}' not found for HF '{hf_name}'"
                    )
                skipped.append(hf_name)
                continue
            ttml_params[ttml_name].assign(new_tensor)
            loaded += 1

    print(f"\n  Weight loading: {loaded} loaded, {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {skipped}")
