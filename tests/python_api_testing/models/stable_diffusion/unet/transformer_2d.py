from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")


import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional

from libs import tt_lib as ttl
from utility_functions import torch_to_tt_tensor, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc
from python_api_testing.fused_ops.silu import SiLU as TtSiLU
from python_api_testing.models.stable_diffusion.residual_block import TtResnetBlock2D
from python_api_testing.models.stable_diffusion.attention_block import TtAttentionBlock
from python_api_testing.models.stable_diffusion.cross_attention import TtCrossAttention
from python_api_testing.models.stable_diffusion.fused_ops.feedforward import TtFeedForward
from python_api_testing.models.stable_diffusion.unet.unet_2d_blocks import TtUNetMidBlock2D, TtUpDecoderBlock2D



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
    ):
        super().__init__()
        self.device = device
        self.host = host
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        # if dim==1280:
        #     print("query_dim: ", dim)
        #     print("heads: ", num_attention_heads)
        #     print("droput: ", dropout)
        #     print("bias: ", attention_bias)
        #     print("cross_attention_dim: ", cross_attention_dim if only_cross_attention else None)
        #     print("upcast_attention: ", upcast_attention)


        # 1. Self-Attn
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
            base_address=f"{base_address}.attn1"
        )
        # print("**** ff")
        # print("dim, ", dim)
        # print("dropout: ", dropout)
        # print("act: ", activation_fn)
        # print("final dropout: ", final_dropout)
        # print("**** ff")
        self.ff = TtFeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout, device=self.device,
            host=self.host,
            state_dict=state_dict,
            base_address=f"{base_address}.ff")

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
                base_address=f"{base_address}.attn2"
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.attn2 = None

        if self.use_ada_layer_norm:
            assert False, "AdaLayerNorm not supported and not used in stable diffusion"
            # self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"
            # self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.torch_norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.torch_norm1.weight = nn.Parameter(state_dict[f"{base_address}.norm1.weight"])
            self.torch_norm1.bias = nn.Parameter(state_dict[f"{base_address}.norm1.bias"])


        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            assert not self.use_ada_layer_norm, "AdaLayerNorm not supported and not used in stable diffusion"
            self.torch_norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.torch_norm2.weight = nn.Parameter(state_dict[f"{base_address}.norm2.weight"])
            self.torch_norm2.bias = nn.Parameter(state_dict[f"{base_address}.norm2.bias"])

            # self.norm2 = (
            #     AdaLayerNorm(dim, num_embeds_ada_norm)
            #     if self.use_ada_layer_norm
            #     else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            # )
        else:
            self.norm2 = None

        # 3. Feed-forward
        self.torch_norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.torch_norm3.weight = nn.Parameter(state_dict[f"{base_address}.norm3.weight"])
        self.torch_norm3.bias = nn.Parameter(state_dict[f"{base_address}.norm3.bias"])



    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.use_ada_layer_norm:
            assert False, "AdaLayerNorm not supported and not used in stable diffusion"
            # norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"
            # norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                # hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            # )
        else:
            hidden_states = tt_to_torch_tensor(hidden_states, self.host)
            norm_hidden_states = self.torch_norm1(hidden_states)
            norm_hidden_states = torch_to_tt_tensor(norm_hidden_states, self.device)
            hidden_states = torch_to_tt_tensor(hidden_states, self.device)



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
            # attn_output = gate_msa.unsqueeze(1) * attn_output

        hidden_states = ttl.tensor.add(attn_output, hidden_states)

        if self.attn2 is not None:
            # norm_hidden_states = (
            #     self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            # )
            norm_hidden_states = (
                self.torch_norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.torch_norm2(hidden_states)
            )

            # 2. Cross-Attention
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = ttl.tensor.add(attn_output, hidden_states)


        # 3. Feed-forward
        norm_hidden_states = self.torch_norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"
            # norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            assert False, "AdaLayerNormZero not supported and not used in stable diffusion"
            # ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ttl.tensor.add(ff_output, hidden_states)
        return hidden_states





########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
#################################################################### transformer model #################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################


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
        state_dict = None,
        base_address=None,
    ):
        super().__init__()
        self.device = device
        self.host = host

        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None
        assert self.is_input_continuous and (not self.is_input_patches) and (not self.is_input_vectorized), "we only support continuous input."
        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.torch_norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            self.torch_norm.weight = nn.Parameter(state_dict[f"{base_address}.norm.weight"])
            self.torch_norm.bias = nn.Parameter(state_dict[f"{base_address}.norm.bias"])
            if use_linear_projection:
                weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_in.weight"]))
                bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_in.bias"]))
                self.proj_in = TtLinear(in_channels, innner_dim, weights, bias, self.device)

            else:
                self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
                self.proj_in.weight = nn.Parameter(state_dict[f"{base_address}.proj_in.weight"])
                self.proj_in.bias = nn.Parameter(state_dict[f"{base_address}.proj_in.bias"])

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
                    base_address=f"{base_address}.transformer_blocks.{d}"
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continous projections
            if use_linear_projection:
                weights = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_out.weight"]))
                bias = tilize_to_list(pad_weight(state_dict[f"{base_address}.proj_out.bias"]))
                self.proj_out = TtLinear(in_channels, inner_dim, weights, bias, self.device)
            else:
                self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
                self.proj_out.weight = nn.Parameter(state_dict[f"{base_address}.proj_out.weight"])
                self.proj_out.bias = nn.Parameter(state_dict[f"{base_address}.proj_out.bias"])

        else:
            assert False, "only continuous input is acceptable for stable diffusion in transformer model"

    def forward(
        self,
        hidden_states,
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
            print("input is continuous")
            batch, _, height, width = hidden_states.shape()
            residual = hidden_states

            hidden_states = tt_to_torch_tensor(hidden_states, self.host)
            hidden_states = self.torch_norm(hidden_states)
            hidden_states = torch_to_tt_tensor(hidden_states, self.device)

            if not self.use_linear_projection:
                hidden_states = tt_to_torch_tensor(hidden_states, self.host)
                hidden_states = self.proj_in(hidden_states) # conv
                hidden_states = torch_to_tt_tensor(hidden_states, self.device)
                inner_dim = hidden_states.shape()[1]

                hidden_states = tt_to_torch_tensor(hidden_states, self.host)
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = torch_to_tt_tensor(hidden_states, self.device)
            else:
                inner_dim = hidden_states.shape()[1]

                hidden_states = tt_to_torch_tensor(hidden_states, self.host)
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
                hidden_states = torch_to_tt_tensor(hidden_states, self.device)

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
                hidden_states = tt_to_torch_tensor(hidden_states, self.host)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = torch_to_tt_tensor(hidden_states, self.device)

                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = tt_to_torch_tensor(hidden_states, self.host)
                hidden_states = self.proj_out(hidden_states)
                hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
                hidden_states = torch_to_tt_tensor(hidden_states, self.device)

            output = ttl.tensor.add(hidden_states, residual)

        if not return_dict:
            return (output,)

        # return Transformer2DModelOutput(sample=output)
