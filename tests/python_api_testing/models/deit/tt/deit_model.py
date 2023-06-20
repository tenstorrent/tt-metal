from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import tt_lib
from utility_functions_new import torch_to_tt_tensor_rm, tt_to_torch_tensor, comp_pcc, comp_allclose_and_pcc
from activations import ACT2FN
from deit_config import DeiTConfig
from deit_embeddings import TtDeiTEmbeddings
from deit_patch_embeddings import TtDeiTPatchEmbeddings
from deit_encoder import TtDeiTEncoder
from deit_pooler import TtDeiTPooler
from tt_lib.fallback_ops import fallback_ops


class TtDeiTModel(nn.Module):
    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address="", add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__()

        self.config = config
        self.device = device

        self.embeddings = TtDeiTEmbeddings(config, use_mask_token=use_mask_token, host=host, device=device, state_dict=state_dict, base_address=f"{base_address}.embeddings")
        self.encoder = TtDeiTEncoder(config, host=host, device=device, state_dict=state_dict, base_address=f"{base_address}.encoder")

        wln = state_dict[f"{base_address}.layernorm.weight"]
        bln = state_dict[f"{base_address}.layernorm.bias"]

        self.layernorm = fallback_ops.LayerNorm(normalized_shape= config.hidden_size, eps=config.layer_norm_eps, weights = wln, biases=bln)

        self.pooler = TtDeiTPooler(config, state_dict, f"{base_address}.pooler") if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> TtDeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values = None,
        bool_masked_pos = None,
        head_mask= None,
        output_attentions= None,
        output_hidden_states= None,
    ):
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
        pixel_values = tt_to_torch_tensor(pixel_values, tt_lib.device.GetHost())
        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

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

        # if not return_dict:
        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        return head_outputs + encoder_outputs[1:]
