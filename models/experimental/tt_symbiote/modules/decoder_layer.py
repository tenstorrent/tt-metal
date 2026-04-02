# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Decoder Layer for BailingMoeV2 (Ling-mini-2.0).

Replaces BailingMoeV2DecoderLayer to perform residual adds on-device using ttnn.add,
eliminating host round-trips that force device synchronization.
"""


import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import trace_enabled
from models.experimental.tt_symbiote.modules.attention import TTNNBailingMoEAttention
from models.experimental.tt_symbiote.modules.moe import TTNNBailingMoE
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm


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
