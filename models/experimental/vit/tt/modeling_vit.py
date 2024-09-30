# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch ViT model."""


import math
import collections
import torch

from torch import nn
from typing import Union, Optional, Tuple, Dict
from transformers import ViTForImageClassification

import ttnn
import tt_lib.fallback_ops as fallback_ops

from models.experimental.vit.tt.configuration_vit import ViTConfig
from models.experimental.vit.tt.activations import ACT2FN
from models.experimental.vit.vit_utils import make_address, make_linear
from models.utility_functions import (
    torch_to_tt_tensor,
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


tt_tensor = ttnn.Tensor


class TtViTOutput(nn.Module):
    def __init__(self, config: ViTConfig, base_address, state_dict, device) -> None:
        super().__init__()
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        self.dense = make_linear(
            config.intermediate_size,
            config.hidden_size,
            "dense",
            state_dict,
            base_address,
            device,
            self.out_mem_config_l1,
        )

    def forward(self, hidden_states: ttnn.Tensor, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = ttnn.add(hidden_states, input_tensor, memory_config=self.out_mem_config_l1)
        return hidden_states


class TtViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.device = device
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.recip_sqrt_attention_head_size_tensor = torch.full((1, 1, 32, 32), 1 / math.sqrt(self.attention_head_size))
        self.recip_sqrt_attention_head_size_tensor = torch_to_tt_tensor(
            self.recip_sqrt_attention_head_size_tensor, device
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = make_linear(
            config.hidden_size,
            self.all_head_size,
            "query",
            state_dict,
            base_address,
            device,
            self.out_mem_config_l1,
        )
        self.key = make_linear(
            config.hidden_size,
            self.all_head_size,
            "key",
            state_dict,
            base_address,
            device,
            self.out_mem_config_l1,
        )
        self.value = make_linear(
            config.hidden_size,
            self.all_head_size,
            "value",
            state_dict,
            base_address,
            device,
            self.out_mem_config_l1,
        )

    def transpose_for_scores(self, x: tt_tensor) -> tt_tensor:
        new_x_shape = (x.shape.with_tile_padding()[0], x.shape.with_tile_padding()[2]) + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = tt_lib.fallback_ops.reshape(x, *new_x_shape)
        return ttnn.permute(x, (0, 2, 1, 3))

    def forward(
        self,
        hidden_states,
        head_mask: Optional[tt_tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[tt_tensor, tt_tensor], Tuple[tt_tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer_T = ttnn.transpose(key_layer, -2, -1, self.out_mem_config_l1)
        attention_scores = ttnn.matmul(query_layer, key_layer_T, memory_config=self.out_mem_config_l1)

        attention_scores = ttnn.multiply(
            attention_scores, self.recip_sqrt_attention_head_size_tensor, memory_config=self.out_mem_config_l1
        )

        # Normalize the attention scores to probabilities.
        attention_probs = tt_lib.fallback_ops.softmax(attention_scores, dim=-1)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = ttnn.mul(attention_probs, head_mask, memory_config=self.out_mem_config_l1)

        context_layer = ttnn.matmul(attention_probs, value_layer)

        context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
        new_context_layer_shape = (1,) + tuple(context_layer.shape.with_tile_padding())[:-2] + (self.all_head_size,)
        context_layer = fallback_ops.reshape(context_layer, *new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class TtViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        self.dense = make_linear(
            config.hidden_size,
            config.hidden_size,
            "dense",
            state_dict=state_dict,
            base_address=base_address,
            device=device,
            mem_config=self.out_mem_config_l1,
        )

    def forward(self, hidden_states: tt_tensor, input_tensor: tt_tensor) -> tt_tensor:
        hidden_states = self.dense(hidden_states)

        return hidden_states


class TtViTAttention(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.attention = TtViTSelfAttention(config, f"{base_address}.attention", state_dict, device)
        self.output = TtViTSelfOutput(config, f"{base_address}.output", state_dict, device)

    def forward(
        self,
        hidden_states: tt_tensor,
        head_mask: Optional[tt_tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[tt_tensor, tt_tensor], Tuple[tt_tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TtViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        self.dense = make_linear(
            config.hidden_size,
            config.intermediate_size,
            "dense",
            state_dict,
            base_address,
            device,
            self.out_mem_config_l1,
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: tt_tensor) -> tt_tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TtViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = TtViTAttention(config, f"{base_address}.attention", state_dict, device)
        self.intermediate = TtViTIntermediate(config, f"{base_address}.intermediate", state_dict, device)
        self.output = TtViTOutput(config, f"{base_address}.output", state_dict, device)

        lbw = state_dict[make_address(base_address, "layernorm_before.weight")]
        lbb = state_dict[make_address(base_address, "layernorm_before.bias")]
        self.layernorm_before = fallback_ops.LayerNorm(
            normalized_shape=config.hidden_size,
            weights=lbw,
            biases=lbb,
            eps=config.layer_norm_eps,
        )

        law = state_dict[make_address(base_address, "layernorm_after.weight")]
        lab = state_dict[make_address(base_address, "layernorm_after.bias")]
        self.layernorm_after = fallback_ops.LayerNorm(
            normalized_shape=config.hidden_size,
            weights=law,
            biases=lab,
            eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden_states: tt_tensor,
        head_mask: Optional[tt_tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[tt_tensor, tt_tensor], Tuple[tt_tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = ttnn.add(attention_output, hidden_states, memory_config=self.out_mem_config_l1)

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class TtViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig, base_address: str, state_dict: Dict, device) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [
                TtViTLayer(config, f"{base_address}.layer.{_}", state_dict, device)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: tt_tensor,
        head_mask: Optional[tt_tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                assert False, "TT does not support training yet"
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict or True:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, base_address: str, state_dict: Dict):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.projection.weight = nn.Parameter(state_dict[make_address(base_address, "projection.weight")])
        self.projection.bias = nn.Parameter(state_dict[make_address(base_address, "projection.bias")])

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(
        self,
        config: ViTConfig,
        base_address: str,
        state_dict: Dict,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(state_dict[make_address(base_address, "cls_token")])
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config, make_address(base_address, "patch_embeddings"), state_dict)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(state_dict[make_address(base_address, "position_embeddings")])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class TtViTModel(nn.Module):
    def __init__(
        self,
        config: ViTConfig,
        base_address: str,
        state_dict: Dict,
        device,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.embeddings = ViTEmbeddings(
            config,
            use_mask_token=use_mask_token,
            base_address=make_address(base_address, "embeddings"),
            state_dict=state_dict,
        )
        self.encoder = TtViTEncoder(
            config,
            base_address=f"{base_address}.encoder",
            state_dict=state_dict,
            device=device,
        )

        wln = state_dict[make_address(base_address, "layernorm.weight")]
        bln = state_dict[make_address(base_address, "layernorm.bias")]
        self.layernorm = fallback_ops.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
            weights=wln,
            biases=bln,
        )
        self.pooler = (
            TtViTPooler(
                config,
                base_address=f"{base_address}.pooler",
                state_dict=state_dict,
                device=device,
            )
            if add_pooling_layer
            else None
        )

        # Initialize weights and apply final processing

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def forward(
        self,
        pixel_values: Optional[tt_tensor] = None,
        bool_masked_pos: Optional[tt_tensor] = None,  # torch.booltensor
        head_mask: Optional[tt_tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        pixel_values = tt_to_torch_tensor(pixel_values)
        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        embedding_output = torch_to_tt_tensor_rm(embedding_output, self.device)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        return head_outputs + encoder_outputs[1:]


class TtViTForImageClassification(nn.Module):
    def __init__(
        self,
        config: ViTConfig,
        base_address: str,
        state_dict: Dict,
        device,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.out_mem_config_l1 = ttnn.L1_MEMORY_CONFIG

        self.vit = TtViTModel(
            config,
            base_address=make_address(base_address, "vit"),
            state_dict=state_dict,
            device=device,
            add_pooling_layer=False,
        )

        # Classifier head
        self.classifier = (
            make_linear(
                config.hidden_size,
                config.num_labels,
                "classifier",
                state_dict,
                base_address,
                device,
                self.out_mem_config_l1,
            )
            if config.num_labels > 0
            else None
        )

        # Initialize weights and apply final processing

    def forward(
        self,
        pixel_values: Optional[tt_tensor] = None,
        head_mask: Optional[tt_tensor] = None,
        labels: Optional[tt_tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        assert labels == None, "we do not support training, hence labels should be None"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.classifier is not None:
            # NOTE: keep this here, until we have support for indexing
            # logits = self.classifier(sequence_output[:, 0, :])
            logits = self.classifier(sequence_output)

        loss = None

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


def _vit_for_image_classification(device, config, state_dict, base_address="") -> TtViTForImageClassification:
    tt_model = TtViTForImageClassification(config, base_address=base_address, state_dict=state_dict, device=device)
    return tt_model


def vit_for_image_classification(device) -> TtViTForImageClassification:
    HF_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    config = HF_model.config
    state_dict = HF_model.state_dict()
    tt_model = _vit_for_image_classification(device=device, config=config, state_dict=state_dict)
    tt_model.vit.get_head_mask = HF_model.vit.get_head_mask
    return tt_model
