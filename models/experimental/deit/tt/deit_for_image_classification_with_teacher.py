# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from torch import nn
from typing import Optional, Tuple, Union
from transformers import DeiTForImageClassificationWithTeacher

import ttnn

from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_model import TtDeiTModel
from models.helper_funcs import Linear as TtLinear
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


class TtDeiTForImageClassificationWithTeacher(nn.Module):
    def __init__(self, config: DeiTConfig(), device, state_dict=None, base_address="") -> None:
        super().__init__()
        self.device = device
        self.config = config
        self.num_labels = config.num_labels
        self.deit = TtDeiTModel(
            config,
            device,
            state_dict,
            base_address="deit",
            add_pooling_layer=False,
            use_mask_token=False,
        )

        cls_c_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}cls_classifier.weight"], device)
        cls_c_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}cls_classifier.bias"], device)

        dc_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}distillation_classifier.weight"], device)
        dc_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}distillation_classifier.bias"], device)

        # Classifier heads
        self.cls_classifier = TtLinear(config.hidden_size, config.num_labels, cls_c_weight, cls_c_bias)
        self.distillation_classifier = TtLinear(config.hidden_size, config.num_labels, dc_weight, dc_bias)

    def forward(
        self,
        pixel_values: Optional[ttnnr.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deit(
            pixel_values=pixel_values,
            bool_masked_pos=None,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # move to cpu (no slicing fallbacks yet)
        sequence_output = tt_to_torch_tensor(sequence_output)
        cls_classifier_input = torch_to_tt_tensor_rm(sequence_output[:, :, 0, :], self.device)
        distillation_classifier_input = torch_to_tt_tensor_rm(sequence_output[:, :, 1, :], self.device)

        cls_logits = self.cls_classifier(cls_classifier_input)
        distillation_logits = self.distillation_classifier(distillation_classifier_input)

        # during inference, return the average of both classifier predictions
        logits = ttnn.add(cls_logits, distillation_logits)
        half = ttnn.full(logits.shape.with_tile_padding(), 0.5)
        logits = ttnn.mul(logits, half)

        # if not return_dict:
        output = (logits, cls_logits, distillation_logits) + outputs[1:]
        return output


def _deit_for_image_classification_with_teacher(
    device, config, state_dict, base_address=""
) -> TtDeiTForImageClassificationWithTeacher:
    tt_model = TtDeiTForImageClassificationWithTeacher(
        config, device=device, base_address=base_address, state_dict=state_dict
    )
    return tt_model


def deit_for_image_classification_with_teacher(
    device,
) -> TtDeiTForImageClassificationWithTeacher:
    torch_model = DeiTForImageClassificationWithTeacher.from_pretrained("facebook/deit-base-distilled-patch16-224")
    config = torch_model.config
    state_dict = torch_model.state_dict()
    tt_model = _deit_for_image_classification_with_teacher(device=device, config=config, state_dict=state_dict)
    tt_model.deit.get_head_mask = torch_model.deit.get_head_mask
    return tt_model
