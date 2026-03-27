# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN Decoder Layer for BailingMoeV2 (Ling-mini-2.0).

Replaces BailingMoeV2DecoderLayer to perform residual adds on-device using ttnn.add,
eliminating host round-trips that force device synchronization.
"""


import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.modules.attention import TTNNBailingMoEAttention
from models.experimental.tt_symbiote.modules.moe import TTNNBailingMoE
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.modules.activation import TTNNSilu


def _to_ttnn(x):
    """Extract raw ttnn.Tensor from TorchTTNNTensor or passthrough."""
    if isinstance(x, TorchTTNNTensor):
        return x.to_ttnn
    return x


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
            # Layer 0: keep BailingMoeV2MLP as nn.Module but replace its children
            # manually since register_module_replacement_dict doesn't recurse into
            # nn.Module attributes of TTNNModule
            from models.experimental.tt_symbiote.utils.module_replacement import register_module_replacement_dict

            mlp = torch_layer.mlp
            register_module_replacement_dict(
                mlp,
                {torch.nn.Linear: TTNNLinearIColShardedWRowSharded, torch.nn.SiLU: TTNNSilu},
            )
            new_layer.mlp = mlp
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

        # Input layernorm (distributed norm needs device_state from module_run)
        input_ndim = len(hs.shape)
        hs = self.input_layernorm(hs)
        hs = _to_ttnn(hs)
        # TTNNDistributedRMSNorm adds a batch dim via unsqueeze(1) for 3D inputs; squeeze back
        if input_ndim == 3 and len(hs.shape) == 4:
            hs = ttnn.reshape(hs, [hs.shape[0], hs.shape[2], hs.shape[3]])

        # Attention
        attn_out, self_attn_weights, present_key_value = self.attention(
            hidden_states=hs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=kwargs.get("cache_position"),
        )
        attn_out = _to_ttnn(attn_out)

        # Residual add ON DEVICE (replaces aten::add on CPU)
        hs = ttnn.add(residual, attn_out)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(residual)

        # Save new residual
        residual = hs

        # Post-attention layernorm (distributed norm needs device_state from module_run)
        post_norm_ndim = len(hs.shape)
        hs_normed = self.post_attention_layernorm(hs)
        hs_normed = _to_ttnn(hs_normed)
        # Squeeze back if distributed norm added a dimension
        if post_norm_ndim == 3 and len(hs_normed.shape) == 4:
            hs_normed = ttnn.reshape(hs_normed, [hs_normed.shape[0], hs_normed.shape[2], hs_normed.shape[3]])

        # MLP / MoE
        # MLP (layer 0) or MoE (layers 1-19)
        mlp_out = self.mlp(hs_normed)
        router_logits = None
        if isinstance(mlp_out, tuple):
            mlp_out, router_logits = mlp_out
        mlp_out = _to_ttnn(mlp_out)

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
