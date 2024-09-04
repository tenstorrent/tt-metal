# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from functools import partial
from loguru import logger

import torch.nn as nn
import torch

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.experimental.stable_diffusion.tt.cross_attention import TtCrossAttention
from models.experimental.stable_diffusion.tt.feedforward import TtFeedForward
from models.experimental.stable_diffusion.sd_utils import make_linear
from models.utility_functions import pad_by_zero, torch2tt_tensor, torch_to_tt_tensor_rm
from models.experimental.stable_diffusion.tt.experimental_ops import Conv2d


class TtBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
        use_fallback_ops=False,
    ):
        super().__init__()
        self.use_fallback_ops = use_fallback_ops
        self.device = device
        self.host = host
        self.only_cross_attention = only_cross_attention
        self.base_address = base_address
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.attn1 = TtCrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            device=self.device,
            host=self.host,
            state_dict=state_dict,
            base_address=f"{base_address}.attn1",
        )

        self.ff = TtFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            device=self.device,
            host=self.host,
            state_dict=state_dict,
            base_address=f"{base_address}.ff",
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = TtCrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                device=self.device,
                host=self.host,
                state_dict=state_dict,
                base_address=f"{base_address}.attn2",
            )
        else:
            self.attn2 = None

        if self.use_ada_layer_norm:
            assert False, "AdaLayerNorm not supported and not used in stable diffusion"
        elif self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"
        else:
            if self.use_fallback_ops:
                norm1_weights = state_dict[f"{base_address}.norm1.weight"]
                norm1_bias = state_dict[f"{base_address}.norm1.bias"]
                self.norm1 = fallback_ops.LayerNorm(
                    weights=norm1_weights,
                    biases=norm1_bias,
                    normalized_shape=dim,
                    elementwise_affine=norm_elementwise_affine,
                )
            else:
                norm1_gamma = torch_to_tt_tensor_rm(
                    state_dict[f"{base_address}.norm1.weight"],
                    self.device,
                    put_on_device=False,
                )
                norm1_beta = torch_to_tt_tensor_rm(
                    state_dict[f"{base_address}.norm1.bias"],
                    self.device,
                    put_on_device=False,
                )

                self.norm1 = partial(
                    ttnn.layer_norm,
                    weight=norm1_gamma,
                    bias=norm1_beta,
                    epsilon=1e-05,
                )

        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            assert not self.use_ada_layer_norm, "AdaLayerNorm is not supported and not used in stable diffusion"

            if self.use_fallback_ops:
                norm2_weights = state_dict[f"{base_address}.norm2.weight"]
                norm2_bias = state_dict[f"{base_address}.norm2.bias"]
                self.norm2 = fallback_ops.LayerNorm(
                    weights=norm2_weights,
                    biases=norm2_bias,
                    normalized_shape=dim,
                    elementwise_affine=norm_elementwise_affine,
                )
            else:
                norm2_gamma = torch_to_tt_tensor_rm(
                    state_dict[f"{base_address}.norm2.weight"],
                    self.device,
                    put_on_device=False,
                )
                norm2_beta = torch_to_tt_tensor_rm(
                    state_dict[f"{base_address}.norm2.bias"],
                    self.device,
                    put_on_device=False,
                )

                self.norm2 = partial(
                    ttnn.layer_norm,
                    weight=norm2_gamma,
                    bias=norm2_beta,
                    epsilon=1e-05,
                )

        else:
            self.norm2 = None

        # 3. Feed-forward
        if self.use_fallback_ops:
            norm3_weight = state_dict[f"{base_address}.norm3.weight"]
            norm3_bias = state_dict[f"{base_address}.norm3.bias"]
            self.norm3 = fallback_ops.LayerNorm(
                weights=norm3_weight,
                biases=norm3_bias,
                normalized_shape=dim,
                elementwise_affine=norm_elementwise_affine,
            )
        else:
            norm3_gamma = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.norm3.weight"],
                self.device,
                put_on_device=False,
            )
            norm3_beta = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.norm3.bias"],
                self.device,
                put_on_device=False,
            )

            self.norm3 = partial(
                ttnn.layer_norm,
                weight=norm3_gamma,
                bias=norm3_beta,
                epsilon=1e-05,
            )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ) -> ttnn.Tensor:
        if self.use_ada_layer_norm:
            assert False, "AdaLayerNorm not supported and not used in stable diffusion"
        elif self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        else:
            norm_hidden_states = self.norm1(hidden_states)

        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

        if self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        hidden_states = ttnn.add(
            attn_output,
            hidden_states,
        )
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )

            hidden_states = ttnn.add(
                attn_output,
                hidden_states,
            )

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)
        if self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"
        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"

        hidden_states = ttnn.add(
            ff_output,
            hidden_states,
        )
        return hidden_states


class TtTransformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        device=None,
        host=None,
        state_dict=None,
        base_address=None,
    ):
        super().__init__()
        self.device = device
        self.host = host

        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None
        assert (
            self.is_input_continuous and (not self.is_input_patches) and (not self.is_input_vectorized)
        ), "we only support continuous input."
        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate(
                "norm_type!=num_embeds_ada_norm",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            norm_type = "ada_norm"

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels
            norm_weights = state_dict[f"{base_address}.norm.weight"]
            norm_bias = state_dict[f"{base_address}.norm.bias"]
            self.norm = fallback_ops.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels,
                eps=1e-6,
                affine=True,
                weights=norm_weights,
                biases=norm_bias,
            )
            if use_linear_projection:
                proj_in_weights = state_dict[f"{base_address}.proj_in.weight"]
                proj_in_bias = state_dict[f"{base_address}.proj_in.bias"]
                self.proj_in = make_linear(
                    in_features=in_channels,
                    out_features=inner_dim,
                    weights=proj_in_weights,
                    bias=proj_in_bias,
                    device=self.device,
                )
            else:
                proj_in_weights = state_dict[f"{base_address}.proj_in.weight"]
                proj_in_bias = state_dict[f"{base_address}.proj_in.bias"]
                self.proj_in = Conv2d(
                    weights=proj_in_weights,
                    biases=proj_in_bias,
                    in_channels=in_channels,
                    out_channels=inner_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
        else:
            assert False, "only continuous input is acceptable for stable diffusion in transformer model"

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TtBasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    device=self.device,
                    host=self.host,
                    state_dict=state_dict,
                    base_address=f"{base_address}.transformer_blocks.{d}",
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continous projections
            if use_linear_projection:
                proj_out_weights = state_dict[f"{base_address}.proj_out.weight"]
                proj_out_bias = state_dict[f"{base_address}.proj_out.bias"]
                self.proj_out = make_linear(
                    in_features=in_channels,
                    out_features=inner_dim,
                    weight=proj_out_weights,
                    bias=proj_out_bias,
                )
            else:
                proj_out_weights = state_dict[f"{base_address}.proj_out.weight"]
                proj_out_bias = state_dict[f"{base_address}.proj_out.bias"]
                self.proj_out = Conv2d(
                    weights=proj_out_weights,
                    biases=proj_out_bias,
                    in_channels=inner_dim,
                    out_channels=in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

        else:
            assert False, "only continuous input is acceptable for stable diffusion in transformer model"

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # 1. Input
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.get_legacy_shape()
            residual = hidden_states

            hidden_states = self.norm(hidden_states)

            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)

                inner_dim = hidden_states.get_legacy_shape()[1]

                hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
                hidden_states = fallback_ops.reshape(hidden_states, 1, batch, height * width, inner_dim)
            else:
                inner_dim = hidden_states.get_legacy_shape()[1]
                hidden_states = ttnn.permute(hidden_states, (0, 2, 3, 1))
                hidden_states = fallback_ops.reshape(hidden_states, 1, batch, height * width, inner_dim)

                hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = fallback_ops.reshape(hidden_states, batch, height, width, inner_dim)
                hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = fallback_ops.reshape(hidden_states, batch, height, width, inner_dim)
                hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))

            output = ttnn.add(
                hidden_states,
                residual,
            )

        if not return_dict:
            return (output,)
        return output
