# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from models.experimental.functional_segformer.reference.segformer_overlap_patch_embeddings import (
    SegformerOverlapPatchEmbeddings,
)
from models.experimental.functional_segformer.reference.segformer_layer import SegformerLayer
from typing import Optional, Tuple, Union


class SegformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        patch = []
        # patch embeddings
        for i in range(config.num_encoder_blocks):
            setattr(
                self,
                f"patch_embeddings_{i}",
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                ),
            )
            patch.append(f"patch_embeddings_{i}")
        self.patch_embedding = patch

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                setattr(
                    self,
                    f"block_{i}_{j}",
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    ),
                )
                layers.append(f"block_{i}_{j}")
            blocks.append(layers)

        self.block = blocks

        layer = []
        # Layer norms
        for i in range(config.num_encoder_blocks):
            setattr(self, f"layer_norm_{i}", nn.LayerNorm(config.hidden_sizes[i]))
            layer.append(f"layer_norm_{i}")
        self.layer_norm = layer

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embedding, self.block, self.layer_norm)):
            embedding_layer, norm_layer = getattr(self, x[0]), getattr(self, x[2])
            # first, obtain patch embeddings
            hidden_states, embedding_hw = embedding_layer(hidden_states)
            _, _, height, width = embedding_hw.shape
            # second, send embeddings through blocks
            for i, blk in enumerate(x[1]):
                block_layer = getattr(self, blk)
                layer_outputs = block_layer(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embedding) - 1 or (
                idx == len(self.patch_embedding) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
