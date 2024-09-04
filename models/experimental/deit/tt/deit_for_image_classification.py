# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
from torch import nn
from transformers import DeiTForImageClassification

import ttnn

from models.experimental.deit.tt.deit_config import DeiTConfig
from models.experimental.deit.tt.deit_model import TtDeiTModel
from models.helper_funcs import Linear as TtLinear
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)


class TtDeiTForImageClassification(nn.Module):
    def __init__(
        self,
        config: DeiTConfig(),
        device,
        base_address: str,
        state_dict: dict,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.deit = TtDeiTModel(
            config,
            device=device,
            state_dict=state_dict,
            base_address=f"{base_address}deit",
            add_pooling_layer=False,
            use_mask_token=False,
        )
        # Classifier head
        self.weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}classifier.weight"], device)
        self.bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}classifier.bias"], device)
        self.classifier = (
            TtLinear(config.hidden_size, config.num_labels, self.weight, self.bias) if config.num_labels > 0 else None
        )

        # Initialize weights and apply final processing

    def forward(
        self,
        pixel_values: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        labels: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple]:
        assert labels == None, "we do not support training, hence labels should be None"
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if self.classifier is not None:
            logits = self.classifier(sequence_output)

        loss = None

        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output


def _deit_for_image_classification(device, config, state_dict, base_address="") -> TtDeiTForImageClassification:
    tt_model = TtDeiTForImageClassification(config, device=device, base_address=base_address, state_dict=state_dict)
    return tt_model


def deit_for_image_classification(device) -> TtDeiTForImageClassification:
    torch_model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")
    config = torch_model.config
    state_dict = torch_model.state_dict()
    tt_model = _deit_for_image_classification(device=device, config=config, state_dict=state_dict)
    tt_model.deit.get_head_mask = torch_model.deit.get_head_mask
    return tt_model
