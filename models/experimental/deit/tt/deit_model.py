# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Tuple
from torch import nn

import ttnn

from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_embeddings import DeiTEmbeddings
from models.experimental.deit.tt.deit_encoder import TtDeiTEncoder
from models.experimental.deit.tt.deit_pooler import TtDeiTPooler
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor_rm


class TtDeiTModel(nn.Module):
    def __init__(
        self,
        config: DeiTConfig(),
        device,
        state_dict=None,
        base_address="",
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        super().__init__()

        self.config = config
        self.device = device

        self.embeddings = DeiTEmbeddings(
            config,
            state_dict=state_dict,
            base_address=f"{base_address}.embeddings",
            use_mask_token=use_mask_token,
        )
        self.encoder = TtDeiTEncoder(
            config,
            device=device,
            state_dict=state_dict,
            base_address=f"{base_address}.encoder",
        )

        ln_weight = state_dict[f"{base_address}.layernorm.weight"]
        ln_bias = state_dict[f"{base_address}.layernorm.bias"]
        self.layernorm = fallback_ops.LayerNorm(
            normalized_shape=config.hidden_size,
            eps=config.layer_norm_eps,
            weights=ln_weight,
            biases=ln_bias,
        )

        self.pooler = TtDeiTPooler(config, state_dict, f"{base_address}.pooler") if add_pooling_layer else None

    def forward(
        self,
        pixel_values: ttnn.Tensor = None,
        bool_masked_pos: bool = None,
        head_mask: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> Tuple[ttnn.Tensor]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        pixel_values = tt_to_torch_tensor(pixel_values)

        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
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
