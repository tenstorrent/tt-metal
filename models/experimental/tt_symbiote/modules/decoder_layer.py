# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Decoder Layers.

Replaces HuggingFace decoder layers to perform residual adds on-device using ttnn.add,
eliminating host round-trips that force device synchronization.
"""


import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.attention import TTNNBailingMoEAttention, LlamaAttention
from models.experimental.tt_symbiote.modules.moe import (
    TTNNBailingMoE,
    TTNNDeepseekV2MoE,
    TTNNDeepseekV2DenseMLP,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm, TTNNRMSNorm


@trace_enabled
class TTNNBailingMoEDecoderLayer(TTNNModule):
    """Replaces BailingMoeV2DecoderLayer to keep residual adds on-device.

    Eliminates 2 host round-trips per layer (one for attention residual,
    one for MoE/MLP residual) by using ttnn.add instead of aten::add.
    """

    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.attention = None
        self.mlp = None
        self._is_dense_layer = False

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from BailingMoeV2DecoderLayer.

        Args:
            torch_layer: HuggingFace BailingMoeV2DecoderLayer instance
        """
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer

        new_layer.input_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNDistributedRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.attention = TTNNBailingMoEAttention.from_torch(torch_layer.attention)

        config = torch_layer.attention.config
        layer_idx = torch_layer.attention.layer_idx
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        is_dense = getattr(config, "num_experts", None) is None or layer_idx < first_k_dense
        new_layer._is_dense_layer = is_dense

        if is_dense:
            from models.experimental.tt_symbiote.modules.moe import TTNNBailingMoeV2MLP

            new_layer.mlp = TTNNBailingMoeV2MLP.from_torch(torch_layer.mlp)
        else:
            new_layer.mlp = TTNNBailingMoE.from_torch(torch_layer.mlp)

        return new_layer

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        position_embeddings=None,
        cache_position=None,
        **kwargs,
    ):
        hs = hidden_states

        # Ensure TILE layout and bfloat16 for TTNN ops
        if hs.layout != ttnn.TILE_LAYOUT:
            hs = ttnn.to_layout(hs, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hs.dtype != ttnn.bfloat16:
            hs = ttnn.typecast(hs, ttnn.bfloat16)

        # Save residual (stays on device as TTNN tensor)
        residual = hs

        # Input layernorm
        hs = self.input_layernorm(hs)

        # Attention — use cache_position (explicit kwarg) if provided,
        # otherwise fall back to position_ids for backward compatibility.
        # Making cache_position a named parameter allows TracedRun to
        # pre-allocate a device buffer for it, so the paged-attention
        # decode path receives a device tensor and avoids host→device
        # writes during trace capture.
        attn_cache_position = cache_position if cache_position is not None else position_ids
        attn_out, self_attn_weights, present_key_value = self.attention(
            hidden_states=hs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=attn_cache_position,
        )

        # Residual add ON DEVICE (replaces aten::add on CPU)
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)
        # NOTE: Do NOT deallocate residual here — it is the pre-allocated trace
        # input buffer.  ttnn.deallocate inside a traced forward would be
        # replayed by execute_trace, freeing the buffer that
        # _copy_inputs_to_trace_buffer needs on the next replay iteration.

        # Save new residual
        residual = hs

        # Post-attention layernorm
        hs_normed = self.post_attention_layernorm(hs)

        # MLP / MoE
        # MLP (layer 0) or MoE (layers 1-19)
        mlp_out = self.mlp(hs_normed)
        router_logits = None
        if isinstance(mlp_out, tuple):
            mlp_out, router_logits = mlp_out

        # Residual add ON DEVICE (replaces aten::add on CPU)
        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        outputs = (hs,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


@trace_enabled
class TTNNDeepseekV2DecoderLayer(TTNNModule):
    """Replaces HF DeepseekV2DecoderLayer to keep residual adds on-device.

    Eliminates 2 host round-trips per layer (one for attention residual,
    one for MoE/MLP residual) by using ttnn.add instead of aten::add.

    Trace-enabled: pre-computes RoPE cos/sin cache on device so the decode
    path runs entirely on device with no host round-trips.  Follows the
    TTNNBailingMoEDecoderLayer pattern from Ling-mini-2.0.
    """

    def __init__(self):
        super().__init__()
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.self_attn = None
        self.mlp = None
        self._is_dense_layer = False
        self._rope_cos_cache = None
        self._rope_sin_cache = None

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from HuggingFace DeepseekV2DecoderLayer."""
        new_layer = cls()
        new_layer._fallback_torch_layer = torch_layer

        new_layer.input_layernorm = TTNNRMSNorm.from_torch(torch_layer.input_layernorm)
        new_layer.post_attention_layernorm = TTNNRMSNorm.from_torch(torch_layer.post_attention_layernorm)
        new_layer.self_attn = LlamaAttention.from_torch(torch_layer.self_attn)

        config = torch_layer.self_attn.config
        layer_idx = torch_layer.self_attn.layer_idx
        first_k_dense = getattr(config, "first_k_dense_replace", 0)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        n_routed = getattr(config, "n_routed_experts", None)
        is_moe = n_routed is not None and layer_idx >= first_k_dense and layer_idx % moe_layer_freq == 0
        new_layer._is_dense_layer = not is_moe

        if is_moe:
            new_layer.mlp = TTNNDeepseekV2MoE.from_torch(torch_layer.mlp)
        else:
            new_layer.mlp = TTNNDeepseekV2DenseMLP.from_torch(torch_layer.mlp)

        return new_layer

    def _init_rope_cache(self):
        """Pre-compute RoPE cos/sin for all positions and store on device.

        This eliminates the CPU RoPE computation inside LlamaAttention.forward
        during trace capture, where host→device transfers are forbidden.
        """
        attn_layer = self.self_attn.torch_layer if hasattr(self.self_attn, "torch_layer") else None
        if attn_layer is None:
            return

        rotary_emb = getattr(attn_layer, "rotary_emb", None)
        if rotary_emb is None:
            return

        config = attn_layer.config
        max_seq_len = getattr(config, "max_position_embeddings", 4096)
        head_dim = getattr(attn_layer, "head_dim", config.hidden_size // config.num_attention_heads)

        position_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
        dummy_val = torch.empty(1, 1, max_seq_len, head_dim, dtype=torch.bfloat16)

        with torch.no_grad():
            try:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding as HFLlamaRotaryEmbedding
            except ImportError:
                HFLlamaRotaryEmbedding = None

            if HFLlamaRotaryEmbedding is not None and isinstance(rotary_emb, HFLlamaRotaryEmbedding):
                cos_cached = rotary_emb.cos_cached if hasattr(rotary_emb, "cos_cached") else None
                sin_cached = rotary_emb.sin_cached if hasattr(rotary_emb, "sin_cached") else None
                if cos_cached is None:
                    cos_all, sin_all = rotary_emb(dummy_val, position_ids)
                else:
                    cos_all = cos_cached[:max_seq_len].unsqueeze(0)
                    sin_all = sin_cached[:max_seq_len].unsqueeze(0)
            else:
                cos_all, sin_all = rotary_emb(dummy_val, position_ids)

        cos_all = cos_all.squeeze(0).to(torch.bfloat16)
        sin_all = sin_all.squeeze(0).to(torch.bfloat16)
        if len(cos_all.shape) == 3:
            cos_all = cos_all.squeeze(0)
        if len(sin_all.shape) == 3:
            sin_all = sin_all.squeeze(0)

        rope_kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        if hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
            rope_kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)

        self._rope_cos_cache = ttnn.from_torch(cos_all, **rope_kw)
        self._rope_sin_cache = ttnn.from_torch(sin_all, **rope_kw)

    def move_weights_to_device_impl(self):
        super().move_weights_to_device_impl()
        if self._rope_cos_cache is None:
            self._init_rope_cache()

    def _get_position_embeddings(self, cache_position, position_ids):
        """Look up pre-computed cos/sin for the current position (all on device).

        Uses ttnn.embedding for trace-compatible indexing: the op structure
        is fixed, only the index values change between replay iterations.
        """
        if self._rope_cos_cache is None:
            return None

        pos = cache_position if cache_position is not None else position_ids
        if pos is None:
            return None

        if isinstance(pos, torch.Tensor) and not isinstance(pos, ttnn.Tensor):
            _is_mesh = hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1
            rope_kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            if _is_mesh:
                rope_kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
            _to_torch_kw = {}
            if _is_mesh:
                _to_torch_kw["mesh_composer"] = ttnn.ConcatMeshToTensor(self.device, dim=0)
            cos_cache = ttnn.to_torch(self._rope_cos_cache, **_to_torch_kw)
            if _is_mesh:
                cos_cache = cos_cache[: self._rope_cos_cache.shape[0]]
            sin_cache = ttnn.to_torch(self._rope_sin_cache, **_to_torch_kw)
            if _is_mesh:
                sin_cache = sin_cache[: self._rope_sin_cache.shape[0]]
            cos_pos = ttnn.from_torch(
                torch.index_select(
                    cos_cache,
                    0,
                    pos.flatten().long(),
                )
                .unsqueeze(0)
                .to(torch.bfloat16),
                **rope_kw,
            )
            sin_pos = ttnn.from_torch(
                torch.index_select(
                    sin_cache,
                    0,
                    pos.flatten().long(),
                )
                .unsqueeze(0)
                .to(torch.bfloat16),
                **rope_kw,
            )
            return (cos_pos, sin_pos)

        if hasattr(pos, "ttnn_tensor") and pos.ttnn_tensor is not None:
            pos = pos.ttnn_tensor
        if not isinstance(pos, ttnn.Tensor):
            return None

        if pos.dtype not in (ttnn.uint32, ttnn.int32):
            if pos.layout != ttnn.TILE_LAYOUT:
                pos = ttnn.to_layout(pos, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            pos = ttnn.typecast(pos, ttnn.uint32)

        if pos.layout != ttnn.ROW_MAJOR_LAYOUT:
            pos = ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if len(pos.shape) > 1:
            total = 1
            for d in pos.shape:
                total *= d
            pos = ttnn.reshape(pos, (total,))

        cos_pos = ttnn.embedding(pos, self._rope_cos_cache, layout=ttnn.TILE_LAYOUT)
        sin_pos = ttnn.embedding(pos, self._rope_sin_cache, layout=ttnn.TILE_LAYOUT)

        cos_pos = ttnn.unsqueeze(cos_pos, 0)
        sin_pos = ttnn.unsqueeze(sin_pos, 0)

        return (cos_pos, sin_pos)

    def _ensure_ttnn(self, t):
        """Convert torch/TorchTTNNTensor to on-device ttnn.Tensor if needed.

        Trace-safe: for TorchTTNNTensor, extracts the inner ttnn_tensor
        (no host→device DMA) instead of calling ttnn.from_torch.  The
        ttnn.from_torch fallback only fires for plain torch.Tensors during
        warmup, before trace capture begins.
        """
        from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor

        if isinstance(t, TorchTTNNTensor):
            if t.ttnn_tensor is not None:
                t = t.ttnn_tensor
            else:
                t = t.to_ttnn
        elif isinstance(t, torch.Tensor) and not isinstance(t, ttnn.Tensor):
            kw = dict(
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if hasattr(self.device, "get_num_devices") and self.device.get_num_devices() > 1:
                kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
            return ttnn.from_torch(t.to(torch.bfloat16), **kw)
        if t.layout != ttnn.TILE_LAYOUT:
            t = ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if t.dtype != ttnn.bfloat16:
            t = ttnn.typecast(t, ttnn.bfloat16)
        return t

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        position_embeddings=None,
        cache_position=None,
        **kwargs,
    ):
        hs = self._ensure_ttnn(hidden_states)

        residual = hs

        hs = self.input_layernorm(hs)

        if position_embeddings is None:
            position_embeddings = self._get_position_embeddings(cache_position, position_ids)

        attn_cache_position = cache_position if cache_position is not None else position_ids
        attn_out, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=attn_cache_position,
            position_embeddings=position_embeddings,
        )

        attn_out = self._ensure_ttnn(attn_out)
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)
        # NOTE: Do NOT deallocate residual here — it is the pre-allocated trace
        # input buffer.  ttnn.deallocate inside a traced forward would be
        # replayed by execute_trace, freeing the buffer that
        # _copy_inputs_to_trace_buffer needs on the next replay iteration.

        residual = hs

        hs_normed = self.post_attention_layernorm(hs)

        mlp_out = self.mlp(hs_normed)
        router_logits = None
        if isinstance(mlp_out, tuple):
            mlp_out, router_logits = mlp_out

        mlp_out = self._ensure_ttnn(mlp_out)
        hs = ttnn.add(residual, mlp_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(residual)

        outputs = (hs,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


def _next_power_of_2(n: int, minimum=256) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    if n <= minimum:
        return minimum
    result = 1 << ((n - 1).bit_length() + 1)  # Shift by one more than the bit length to get the next power of 2
    if result == n * 4:  # If n is already a power of 2, we want to return n, not the next power of 2
        result = n
    return result


class TTNNBailingMoEDecoderLayerPadded(TTNNModule):
    """Decoder layer that pads the input sequence length to the next power of 2.

    Padding to a power-of-2 sequence length reduces the number of unique trace
    cache keys during prefill, since many different prompt lengths map to the
    same padded length. This improves trace reuse across turns.

    The pad is applied before the forward pass and the output is sliced back
    to the original sequence length afterward.
    """

    @classmethod
    def from_torch(cls, torch_layer):
        """Create from BailingMoeV2DecoderLayer.

        Args:
            torch_layer: HuggingFace BailingMoeV2DecoderLayer instance
        """
        new_layer = cls()
        new_layer.layer = TTNNBailingMoEDecoderLayer.from_torch(torch_layer)
        return new_layer

    @staticmethod
    def _pad_dim(tensor, dim, pad_amount, value=0.0):
        """Pad a single dimension of a tensor by ``pad_amount``."""
        rank = len(tensor.shape)
        padding = tuple((0, pad_amount if i == dim else 0) for i in range(rank))
        return ttnn.pad(tensor, padding=padding, value=value)

    @staticmethod
    def _slice_dim(tensor, dim, length):
        """Slice a tensor along ``dim`` to ``length``."""
        starts = [0] * len(tensor.shape)
        ends = list(tensor.shape)
        ends[dim] = length
        return ttnn.slice(tensor, starts, ends)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        output_router_logits=False,
        use_cache=False,
        position_embeddings=None,
        cache_position=None,
        **kwargs,
    ):
        rank = len(hidden_states.shape)
        seq_dim = rank - 2  # sequence length is always second-to-last
        seq_len = hidden_states.shape[seq_dim]
        padded_seq_len = _next_power_of_2(seq_len)
        pad_amount = padded_seq_len - seq_len

        if pad_amount > 0:
            hidden_states = self._pad_dim(hidden_states, seq_dim, pad_amount, value=0.0)

            # attention_mask: [..., seq_len, seq_len] — pad last two dims
            if attention_mask is not None:
                mask_rank = len(attention_mask.shape)
                attention_mask = self._pad_dim(attention_mask, mask_rank - 2, pad_amount, value=float("-inf"))
                attention_mask = self._pad_dim(attention_mask, mask_rank - 1, pad_amount, value=float("-inf"))

            # position_ids: [batch, seq_len] — pad seq dim with 0
            if position_ids is not None:
                pid_seq_dim = len(position_ids.shape) - 1
                position_ids = self._pad_dim(position_ids, pid_seq_dim, pad_amount, value=0)

            # position_embeddings (cos, sin): [batch, seq_len, head_dim]
            if position_embeddings is not None:
                cos, sin = position_embeddings
                cos_seq_dim = len(cos.shape) - 2
                cos = self._pad_dim(cos, cos_seq_dim, pad_amount, value=0.0)
                sin = self._pad_dim(sin, cos_seq_dim, pad_amount, value=0.0)
                position_embeddings = (cos, sin)

        # Pass cache_position explicitly (not padded) so the inner layer's
        # trace infrastructure can pre-allocate a device buffer for it.
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            **kwargs,
        )

        if pad_amount > 0:
            hs = self._slice_dim(outputs[0], seq_dim, seq_len)
            if isinstance(outputs, tuple):
                outputs = (hs,) + outputs[1:]
            else:
                outputs = [hs] + list(outputs[1:])

        return outputs
