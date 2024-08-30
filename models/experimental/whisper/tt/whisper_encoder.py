# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import partial
import math
import torch
import torch.nn as nn
import random
from typing import Optional, Tuple, Union

import ttnn

from transformers import WhisperConfig
from dataclasses import dataclass

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.experimental.whisper.tt.whisper_encoder_layer import (
    TtWhisperEncoderLayer,
)

# from tt_lib.fallback_ops import fallback_ops
import tt_lib.fallback_ops as fallback_ops


@dataclass
class TtWhisperEncoderOutput:
    last_hidden_state: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtWhisperEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`TtWhisperEncoderLayer`].

    Args:
        reference_model: WhisperModel
        device: device: ttnn.Device
        config: WhisperConfig
    """

    def __init__(self, state_dict, base_address, device, config: WhisperConfig):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.config = config

        embed_dim = config.d_model
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions

        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        # Init
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv1.weight = nn.Parameter(state_dict[f"{base_address}.conv1.weight"])
        self.conv1.bias = nn.Parameter(state_dict[f"{base_address}.conv1.bias"])

        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.conv2.weight = nn.Parameter(state_dict[f"{base_address}.conv2.weight"])
        self.conv2.bias = nn.Parameter(state_dict[f"{base_address}.conv2.bias"])

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.weight = nn.Parameter(state_dict[f"{base_address}.embed_positions.weight"])

        self.layers = nn.ModuleList(
            [
                TtWhisperEncoderLayer(
                    base_address=f"{base_address}.layers.{ind}",
                    state_dict=state_dict,
                    device=device,
                    embed_dim=embed_dim,
                    num_heads=config.encoder_attention_heads,
                    encoder_ffn_dim=config.encoder_ffn_dim,
                    config=config,
                )
                for ind in range(config.encoder_layers)
            ]
        )

        gamma = torch2tt_tensor(
            self.state_dict[f"{base_address}.layer_norm.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        beta = torch2tt_tensor(
            self.state_dict[f"{base_address}.layer_norm.bias"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.layer_norm = partial(ttnn.layer_norm, weight=gamma, bias=beta, epsilon=1e-05)

        self.gradient_checkpointing = False

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features: ttnn.Tensor,  # bc of shape
        attention_mask: Optional[ttnn.Tensor] = None,  #  NOT used in whisper
        head_mask: Optional[torch.Tensor] = None,  # bc of shape []
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ttnn.Tensor], TtWhisperEncoderOutput]:
        """
        Args:
            input_features (`torch.LongTensor` of shape `(batch_size, feature_size, sequence_length)`):
                Float values of mel features extracted from the raw speech waveform. Raw speech waveform can be
                obtained by loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a
                `numpy.ndarray`, *e.g.* via the soundfile library (`pip install soundfile`). To prepare the array into
                `input_features`, the [`AutoFeatureExtractor`] should be used for extracting the mel features, padding
                and conversion into a tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
            attention_mask (`torch.Tensor`)`, *optional*):
                Whisper does not support masking of the `input_features`, this argument is preserved for compatibility,
                but it is not used. By default the silence in the input log mel spectrogram are ignored.
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_features = tt2torch_tensor(input_features).to(torch.float)
        input_features = torch.squeeze(input_features, 0)
        """PyTorch implementation start"""
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        """PyTorch implementation end"""

        """TT implementation"""
        hidden_states = torch2tt_tensor(hidden_states, self.device, ttnn.ROW_MAJOR_LAYOUT)

        # TODO: Not suppporting dropout at moment
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)

            # TODO: Not supporting training at moment
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                # TODO: Not supporting training at moment
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        None,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        None,
                        layer_head_mask=(
                            torch2tt_tensor(
                                head_mask[idx],
                                self.device,
                                ttnn.ROW_MAJOR_LAYOUT,
                            )
                            if head_mask is not None
                            else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        """
        model's outputs
        Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        return TtWhisperEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )
