# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 specific module implementations for TTNN.

Provides TTNN-accelerated versions of Gemma 4 architectural components:
- TTNNGemma4ScaledEmbedding: Scaled word embedding (multiplies by sqrt(hidden_size))
- TTNNGemma4DecoderLayer: Decoder layer wrapper for on-device residual adds and tracing
"""

import math

import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, TTNNLayerStack
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.gemma4_attention import TTNNGemma4Attention
from models.experimental.tt_symbiote.modules.gemma4_mlp import TTNNGemma4TextMLP, gelu_pytorch_tanh
from models.experimental.tt_symbiote.modules.linear import TTNNLinear
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


@trace_enabled
class TTNNGemma4ScaledEmbedding(TTNNModule):
    """TTNN-accelerated scaled word embedding for Gemma 4 models.

    Wraps TTNNEmbedding and multiplies the output by sqrt(hidden_size),
    replicating the behaviour of Gemma4TextScaledWordEmbedding.

    NOT trace-enabled: forward() converts torch input_ids via ttnn.from_torch
    which allocates new device memory (forbidden during trace capture).
    """

    @classmethod
    def from_torch(cls, embedding):
        """Create from a Gemma4TextScaledWordEmbedding instance.

        Args:
            embedding: HF Gemma4TextScaledWordEmbedding with an ``embed_scale``
                attribute and an underlying ``nn.Embedding`` weight.
        """
        new_layer = cls()
        new_layer._fallback_torch_layer = embedding
        new_layer.embed_scale = float(getattr(embedding, "embed_scale", math.sqrt(embedding.embedding_dim)))
        new_layer.embedder = TTNNEmbedding.from_torch(embedding, scale_factor=new_layer.embed_scale)
        new_layer._typecast_pad_buffers = {}
        return new_layer

    def forward(self, tt_indices):
        # Ensure indices are UINT32 for the TTNN embedding op.
        if isinstance(tt_indices, ttnn.Tensor) and tt_indices.dtype != ttnn.uint32:
            # On-device cast INT32 -> UINT32 using pad-typecast-slice pattern.
            # ttnn.typecast requires padded_shape[-1] % 32 == 0; for decode (seq=1)
            # we concat with a pre-allocated zeros buffer, typecast, then slice back.
            orig_size = tt_indices.shape[-1] if len(tt_indices.shape) > 0 else 1
            if orig_size % 32 != 0:
                pad_amount = 32 - (orig_size % 32)
                if pad_amount not in self._typecast_pad_buffers:
                    pad_torch = torch.zeros(1, pad_amount, dtype=torch.int32)
                    self._typecast_pad_buffers[pad_amount] = ttnn.from_torch(
                        pad_torch,
                        device=self.device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                    )
                tt_indices = ttnn.concat([tt_indices, self._typecast_pad_buffers[pad_amount]], dim=-1)
            tt_indices = ttnn.typecast(tt_indices, ttnn.uint32)
            if orig_size % 32 != 0:
                if len(tt_indices.shape) <= 1:
                    tt_indices = ttnn.slice(tt_indices, [0], [orig_size])
                else:
                    starts = [0] * len(tt_indices.shape)
                    ends = list(tt_indices.shape)
                    ends[-1] = orig_size
                    tt_indices = ttnn.slice(tt_indices, starts, ends)
        out = self.embedder(tt_indices)
        return out


@trace_enabled
class TTNNGemma4DecoderLayer(TTNNModule):
    """Replaces Gemma4TextDecoderLayer to keep residual adds on-device.

    Eliminates host round-trips by using ttnn.add for residual connections.
    This is the primary trace boundary for Gemma4 — @trace_enabled captures
    the full op sequence (norms + attention + FFN) in a single trace.

    HF Gemma4TextDecoderLayer forward flow (31B, no MoE):
        input_layernorm -> self_attn -> post_attention_layernorm -> residual_add
        pre_feedforward_layernorm -> mlp -> post_feedforward_layernorm -> residual_add
        layer_scalar multiplication
    """

    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.pre_feedforward_layernorm = None
        self.post_feedforward_layernorm = None
        self.self_attn = None
        self.mlp = None
        self.layer_scalar = 1.0
        # Per-layer input (PLI) modules — active in 31B (hidden_size_per_layer_input=256)
        self.hidden_size_per_layer_input = 0
        self.per_layer_input_gate = None
        self.per_layer_projection = None
        self.post_per_layer_input_norm = None

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from Gemma4TextDecoderLayer."""
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer

        new_layer.input_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.pre_feedforward_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.pre_feedforward_layernorm)
        new_layer.post_feedforward_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.post_feedforward_layernorm)
        new_layer.self_attn = TTNNGemma4Attention.from_torch(torch_layer.self_attn)
        new_layer.mlp = TTNNGemma4TextMLP.from_torch(torch_layer.mlp)

        # Per-layer input (PLI) modules — gate (5376→256), projection (256→5376), norm.
        # Active when HF config has hidden_size_per_layer_input > 0 (Gemma4 31B: 256).
        # Uses basic TTNNLinear (replicated weights) because the PLI dimension (256) is
        # too small to benefit from cross-device sharding.
        pli_size = getattr(torch_layer, "hidden_size_per_layer_input", 0) or 0
        new_layer.hidden_size_per_layer_input = pli_size
        if pli_size and hasattr(torch_layer, "per_layer_input_gate"):
            new_layer.per_layer_input_gate = TTNNLinear.from_torch(torch_layer.per_layer_input_gate)
            new_layer.per_layer_projection = TTNNLinear.from_torch(torch_layer.per_layer_projection)
            new_layer.post_per_layer_input_norm = TTNNDistributedRMSNorm.from_torch(
                torch_layer.post_per_layer_input_norm
            )

        # layer_scalar is a learnable scalar (nn.Parameter) initialised to 1.0
        scalar = getattr(torch_layer, "layer_scalar", None)
        if scalar is not None:
            new_layer.layer_scalar = float(scalar.item() if hasattr(scalar, "item") else scalar)

        return new_layer

    def post_trace_execute(self, func_args, func_kwargs, result):
        """Update KV cache sequence counters after trace replay.

        During replay, paged_fill/update_on_device Python-side counter
        increments do not execute (baked into device trace). This hook
        ensures counters advance on every replay iteration.
        """
        past_key_values = func_kwargs.get("past_key_values")
        if past_key_values is None or not hasattr(past_key_values, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        layer_idx = self.self_attn.layer_idx
        past_key_values.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

    def forward(
        self,
        hidden_states,
        per_layer_input=None,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        hs = hidden_states

        # Ensure TILE layout and bfloat16 for TTNN ops
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        # ----- Attention block -----
        residual = hs

        hs = self.input_layernorm(hs)

        # Pass cache_position as explicit kwarg so TracedRun can
        # pre-allocate a device buffer for it (matches Ling pattern).
        seq_len = hs.shape[-2]
        is_decode = seq_len == 1
        attn_out, _ = self.self_attn(
            hidden_states=hs,
            position_embeddings=position_embeddings,
            attention_mask=None if is_decode else attention_mask,
            past_key_values=past_key_values,
            cache_position=kwargs.get("cache_position"),
            rope_position_idx=kwargs.get("rope_position_idx"),
        )

        attn_out = self.post_attention_layernorm(attn_out)

        # Residual add ON DEVICE
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)
        # NOTE: Do NOT deallocate residual here — it is the pre-allocated trace
        # input buffer.  ttnn.deallocate inside a traced forward would be
        # replayed by execute_trace, freeing the buffer that
        # _copy_inputs_to_trace_buffer needs on the next replay iteration.

        # ----- FFN block -----
        residual = hs

        hs = self.pre_feedforward_layernorm(hs)
        mlp_out = self.mlp(hs)
        mlp_out = self.post_feedforward_layernorm(mlp_out)

        # Residual add ON DEVICE
        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        # ----- Per-layer input (PLI) block -----
        # Applies a gated projection conditioned on per_layer_input (from model-level
        # embedding + projection).  Active in 31B (hidden_size_per_layer_input=256).
        # HF ref: Gemma4TextDecoderLayer lines 1401-1408.
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            pli_residual = hs
            hs = self.per_layer_input_gate(hs)  # Linear(5376 → 256)
            hs = gelu_pytorch_tanh(hs)  # GELU(tanh approx matching HF)
            hs = ttnn.multiply(hs, per_layer_input)  # Element-wise * PLI
            hs = self.per_layer_projection(hs)  # Linear(256 → 5376)
            hs = self.post_per_layer_input_norm(hs)  # RMSNorm
            hs = ttnn.add(pli_residual, hs)  # Residual add
            ttnn.deallocate(pli_residual)

        # Layer scalar (1.0 in 31B — effectively a no-op)
        if abs(self.layer_scalar - 1.0) > 1e-6:
            hs = ttnn.multiply(hs, self.layer_scalar)

        # HF Gemma4TextDecoderLayer returns a plain tensor (not a tuple).
        return hs


class TTNNGemma4LayerStack(TTNNLayerStack):
    """Gemma4-specific layer stack with per-layer mask/input selection."""

    def __init__(self, layers, layer_types):
        super().__init__(layers)
        self.layer_types = list(layer_types)

    def forward(self, hidden_states, **kwargs):
        causal_mask_mapping = kwargs.pop("attention_mask", {})
        position_embeddings_mapping = kwargs.pop("position_embeddings", {})
        per_layer_inputs = kwargs.pop("per_layer_inputs", None)

        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if per_layer_inputs is not None:
                per_layer_input = (
                    per_layer_inputs[i] if isinstance(per_layer_inputs, list) else per_layer_inputs[:, :, i, :]
                )
            else:
                per_layer_input = None
            hidden_states = layer.forward(
                hidden_states,
                per_layer_input=per_layer_input,
                position_embeddings=position_embeddings_mapping.get(layer_type),
                attention_mask=causal_mask_mapping.get(layer_type),
                **kwargs,
            )
        return hidden_states

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_values = func_kwargs.get("past_key_values")
        if past_key_values is None or not hasattr(past_key_values, "update_seq_length"):
            return
        hidden_states = func_args[0]
        seq_len = hidden_states.shape[-2]
        for layer in self.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer_idx = layer.self_attn.layer_idx
                past_key_values.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)


# Alias for test compatibility: tests import TTNNGemma4FFN from this module.
TTNNGemma4FFN = TTNNGemma4TextMLP
