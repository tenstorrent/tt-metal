from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.bge_m3.tt.attention import BgeM3Attention, BgeM3AttentionConfig
from models.demos.wormhole.bge_m3.tt.mlp import BgeM3MLP, BgeM3MLPConfig
from models.demos.wormhole.bge_m3.tt.norm import LayerNorm1D, LayerNorm1DConfig
from models.demos.wormhole.bge_m3.tt.weight_adapter import LayerNormWeights, build_attention_weights, build_mlp_weights


class BgeM3TransformerBlock(LightweightModule):
    """
    Layer-only transformer block that transforms hidden states in-place.
    """

    def __init__(self, args, mesh_device, dtype, state_dict, layer_num):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.layer_num = layer_num

        attention_weights = build_attention_weights(state_dict, layer_num, dtype)
        mlp_weights = build_mlp_weights(state_dict, layer_num, dtype)

        self.attention = BgeM3Attention.from_config(
            BgeM3AttentionConfig(
                wqkv=attention_weights.wqkv,
                bqkv=attention_weights.bqkv,
                wo_weight=attention_weights.wo_weight,
                wo_bias=attention_weights.wo_bias,
                hidden_size=args.dim,
                num_heads=args.n_heads,
                head_dim=args.head_dim,
            )
        )
        self.attention_norm = _build_optional_layer_norm(
            attention_weights.layer_norm,
            eps=args.norm_eps,
            mesh_device=mesh_device,
        )

        self.feed_forward = BgeM3MLP.from_config(
            BgeM3MLPConfig(
                wi_weight=mlp_weights.wi_weight,
                wi_bias=mlp_weights.wi_bias,
                wo_weight=mlp_weights.wo_weight,
                wo_bias=mlp_weights.wo_bias,
                hidden_size=args.dim,
                intermediate_size=args.intermediate_size,
            )
        )
        self.feed_forward_norm = _build_optional_layer_norm(
            mlp_weights.layer_norm,
            eps=args.norm_eps,
            mesh_device=mesh_device,
        )

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        attention_output = self.attention(hidden_states, attention_mask=attention_mask)

        hidden_states = ttnn.add(hidden_states, attention_output)

        if self.attention_norm is not None:
            hidden_states = self.attention_norm(hidden_states)

        mlp_output = self.feed_forward(hidden_states)
        hidden_states = ttnn.add(hidden_states, mlp_output)

        if self.feed_forward_norm is not None:
            hidden_states = self.feed_forward_norm(hidden_states)

        return hidden_states


def _build_optional_layer_norm(
    layer_norm_weights: LayerNormWeights | None,
    eps: float,
    mesh_device,
) -> LayerNorm1D | None:
    if layer_norm_weights is None:
        return None

    return LayerNorm1D.from_config(
        LayerNorm1DConfig(
            weight=layer_norm_weights.weight,
            bias=layer_norm_weights.bias,
            eps=eps,
            mesh_device=mesh_device,
        )
    )


# Backwards-friendly alias for users importing a generic block name.
TransformerBlock = BgeM3TransformerBlock
BgeM3EncoderLayer = BgeM3TransformerBlock
