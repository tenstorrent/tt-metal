# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.ttnn_segformer_overlap_patch_embeddings import (
    TtSegformerOverlapPatchEmbeddings,
)
from models.experimental.functional_segformer.tt.ttnn_segformer_layer import TtSegformerLayer
from typing import Optional, Tuple, Union

from dataclasses import dataclass


@dataclass
class TtBaseModelOutput:
    last_hidden_state: ttnn.bfloat16 = None
    hidden_states: ttnn.bfloat16 = None
    attentions: ttnn.bfloat16 = None

    def __getitem__(self, idx):
        if idx == 0:
            return self.last_hidden_state
        elif idx == 1:
            return self.hidden_states
        elif idx == 2:
            return self.attentions
        else:
            raise IndexError("Index out of range")


class TtSegformerEncoder:
    def __init__(self, config, parameters, model):
        super().__init__()
        self.config = config

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            setattr(
                self,
                f"patch_embeddings_{i}",
                TtSegformerOverlapPatchEmbeddings(
                    parameters[f"patch_embeddings_{i}"], getattr(model, f"patch_embeddings_{i}")
                ),
            )
            embeddings.append(f"patch_embeddings_{i}")
        self.patch_embeddings = embeddings

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
                    TtSegformerLayer(
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        parameters=parameters[f"block_{i}_{j}"],
                        model=getattr(model, f"block_{i}_{j}"),
                    ),
                )
                layers.append(f"block_{i}_{j}")

            blocks.append(layers)

        self.block = blocks

    def __call__(
        self,
        pixel_values: ttnn.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        parameters=None,
        model=None,
    ) -> Union[Tuple, TtBaseModelOutput]:
        device = pixel_values.device()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]

        hidden_states = pixel_values
        for idx, x in enumerate(zip(self.patch_embeddings, self.block)):
            embedding_layer, block_layer = getattr(self, x[0]), x[1]
            # first, obtain patch embeddings
            hidden_states, hw_tensor = embedding_layer(
                hidden_states, parameters[f"patch_embeddings_{idx}"], model=getattr(model, f"patch_embeddings_{idx}")
            )
            height = hw_tensor.shape[2]
            width = hw_tensor.shape[3]
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                block_layer = getattr(self, blk)
                layer_outputs = block_layer(
                    hidden_states,
                    height,
                    width,
                    parameters[f"block_{idx}_{i}"],
                    output_attentions=output_attentions,
                    device=device,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # third, apply layer norm
            hidden_states = ttnn.layer_norm(
                hidden_states,
                weight=parameters[f"layer_norm_{idx}"]["weight"],
                bias=parameters[f"layer_norm_{idx}"]["bias"],
            )
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
            ):
                hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
                hidden_states = ttnn.reshape(hidden_states, (batch_size, height, width, -1))
                hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.TILE_LAYOUT)
                hidden_states = ttnn.permute(hidden_states, (0, 3, 1, 2))
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return TtBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
