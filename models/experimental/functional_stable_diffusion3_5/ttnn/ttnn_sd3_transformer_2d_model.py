# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
from torch import nn
import ttnn

from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_ada_layernorm_continuous import (
    ttnn_AdaLayerNormContinuous,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_joint_transformer_block import (
    ttnn_JointTransformerBlock,
)
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_patch_embed import ttnn_PatchEmbed
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_combined_time_step_text_proj_embeddings import (
    ttnn_CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.transformers.transformer_sd3 import Transformer2DModelOutput

from typing import Tuple, Optional, Dict, Any, Union, List


class ttnn_SD3Transformer2DModel:
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: Optional[str] = None,
        config=None,
        parameters=None,
    ):
        self.config = config  # added by me
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = ttnn_PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
            parameters=parameters["pos_embed"],
        )
        self.time_text_embed = ttnn_CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
            parameters=parameters["time_text_embed"],
        )
        self.context_embedder = ttnn.linear

        self.transformer_blocks = [
            ttnn_JointTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                context_pre_only=i == num_layers - 1,
                qk_norm=qk_norm,
                use_dual_attention=True if i in dual_attention_layers else False,
                parameters=parameters["transformer_blocks"][i],
            )
            for i in range(self.config.num_layers)
        ]

        self.norm_out = ttnn_AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = ttnn.linear

        self.gradient_checkpointing = False

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor = None,
        pooled_projections: ttnn.Tensor = None,
        timestep: ttnn.Tensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        parameters=None,
    ) -> Union[ttnn.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(self, lora_scale)
        # else:
        #     if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
        #         logger.warning(
        #             "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
        #         )

        height, width = hidden_states.shape[-2], hidden_states.shape[-1]

        hidden_states = self.pos_embed(
            hidden_states.device(), hidden_states
        )  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections, device=hidden_states.device())
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, parameters["context_embedder"]["weight"], bias=parameters["context_embedder"]["bias"]
        )

        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            encoder_hidden_states = ttnn.to_memory_config(encoder_hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            temb = ttnn.to_memory_config(temb, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                parameters=parameters["transformer_blocks"][index_block],
            )

            # controlnet residual
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out(hidden_states, temb, parameters=parameters["norm_out"])
        hidden_states = self.proj_out(
            hidden_states, parameters["proj_out"]["weight"], bias=parameters["proj_out"]["bias"]
        )

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = ttnn.reshape(
            hidden_states, (hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        device = hidden_states.device()
        hidden_states = ttnn.to_torch(hidden_states)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        output = ttnn.reshape(
            hidden_states, (hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        # if USE_PEFT_BACKEND:
        #     # remove `lora_scale` from each PEFT layer
        #     unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
