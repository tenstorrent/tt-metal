# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_cross_attention import cross_attention
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_feedforward import feedforward


def basic_transformer_block(
    hidden_states,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    class_labels=None,
    config=None,
    num_embeds_ada_norm=False,
    norm_type: str = "layer_norm",
    cross_attention_dim: int = None,
    activation_fn: str = "geglu",
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    norm_elementwise_affine: bool = True,
    attention_bias: bool = False,
    attention_head_dim=None,
    *,
    parameters,
    device,
):
    use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
    use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

    if use_ada_layer_norm:
        assert False, "AdaLayerNorm not supported and not used in stable diffusion"
    elif use_ada_layer_norm_zero:
        assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

    else:
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        norm_hidden_states = ttnn.layer_norm(
            hidden_states, epsilon=1e-05, weight=parameters.norm1.weight, bias=parameters.norm1.bias
        )

    # 1. Self-Attention
    cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
    cross_attention_dim = config.cross_attention_dim if cross_attention_dim is None else cross_attention_dim

    attn_output = cross_attention(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if only_cross_attention else None,
        attention_mask=attention_mask,
        cross_attention_kwargs=cross_attention_kwargs,
        cross_attention_dim=cross_attention_dim,
        dim_head=attention_head_dim,
        upcast_attention=upcast_attention,
        parameters=parameters.attn1,
        device=device,
    )

    if use_ada_layer_norm_zero:
        assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

    hidden_states = ttnn.add(attn_output, hidden_states)
    if cross_attention_dim is not None:
        norm_hidden_states = ttnn.layer_norm(
            hidden_states, epsilon=1e-05, weight=parameters.norm2.weight, bias=parameters.norm2.bias
        )

        # 2. Cross-Attention
        attn_output = cross_attention(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            cross_attention_dim=cross_attention_dim,
            dim_head=attention_head_dim,
            upcast_attention=upcast_attention,
            parameters=parameters.attn2,
            device=device,
        )

        hidden_states = ttnn.add(attn_output, hidden_states)

    # 3. Feed-forward
    norm_hidden_states = ttnn.layer_norm(
        hidden_states, epsilon=1e-05, weight=parameters.norm3.weight, bias=parameters.norm3.bias
    )
    if use_ada_layer_norm_zero:
        assert False, "AdaLayerNormZero not supported and not used in stable diffusion"
    # ttnn.dump_device_memory_state(device)
    ff_output = feedforward(config=config, hidden_states=norm_hidden_states, parameters=parameters.ff, device=device)

    hidden_states = ttnn.add(ff_output, hidden_states)

    return hidden_states
