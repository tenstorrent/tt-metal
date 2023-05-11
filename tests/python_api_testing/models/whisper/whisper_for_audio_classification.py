import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

import random
from typing import Optional, Tuple, Union

from transformers import  WhisperConfig

from python_api_testing.models.whisper.whisper_common import torch2tt_tensor, tt2torch_tensor, create_padded_tensor, create_unpadded_tensor
from python_api_testing.models.whisper.whisper_encoder import TtWhisperEncoder
from python_api_testing.fused_ops.linear import Linear as TtLinear
from python_api_testing.models.whisper.whisper_linear_layer import WhisperPaddedLinear

from libs import tt_lib as ttm

from sweep_tests.comparison_funcs import comp_allclose, comp_pcc

@dataclass
class TtWhisperForAudioClassificationOutput():
    loss: Optional[ttm.tensor.Tensor] = None
    logits: ttm.tensor.Tensor = None
    hidden_states: Optional[Tuple[ttm.tensor.Tensor]] = None
    attentions: Optional[Tuple[ttm.tensor.Tensor]] = None

class TtWhisperForAudioClassification(nn.Module):
    def __init__(
        self,
        state_dict,
        device,
        config
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.config = config

        self.add_padding = True

        self.encoder = TtWhisperEncoder(
            state_dict=state_dict,
            base_address="encoder",
            device=self.device,
            config=config
        )

        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            # Not using this parameter for now
            N, C, H, W = 1, 1, 1, num_layers
            weight_init_const = 1.0 / num_layers
            # TODO: Implement in ttm as soon as full is merged
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            self.layer_weights =  torch2tt_tensor(self.layer_weights, self.device)

        projector_weight = torch2tt_tensor(state_dict[f"projector.weight"], ttm.device.GetHost())
        projector_bias = state_dict[f"projector.bias"]
        projector_bias = create_padded_tensor(list(projector_bias.shape), projector_bias, [1, 1, 32, projector_bias.shape[-1]], 0, ttm.device.GetHost())

        self.projector = TtLinear(in_features=config.hidden_size, out_features=config.classifier_proj_size, weight=projector_weight.data(), bias=projector_bias.data(), device=device)
        self.classifier = WhisperPaddedLinear(config.classifier_proj_size, config.num_labels, state_dict[f"classifier.weight"], state_dict[f"classifier.bias"], device)

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training. Only the projection layers and classification head will be updated.
        """
        self.encoder._freeze_parameters()

    def get_input_embeddings(self) -> nn.Module:
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
        self.encoder.set_input_embeddings(value)

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[ttm.tensor.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TtWhisperForAudioClassificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, WhisperForAudioClassification
        >>> from datasets import load_dataset

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        >>> model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

        >>> ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
        >>> sample = next(iter(ds))

        >>> inputs = feature_extractor(
        ...     sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"], return_tensors="pt"
        ... )
        >>> input_features = inputs.input_features

        >>> with torch.no_grad():
        ...     logits = model(input_features).logits

        >>> predicted_class_ids = torch.argmax(logits).item()
        >>> predicted_label = model.config.id2label[predicted_class_ids]
        >>> predicted_label
        'af_za'
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:

            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            """Not supported for now."""
            # Parameter use_weighted_layer_sum is false and not used in config we are implementing
            # When implementing keep in mind that the size of each individual torch tensor
            # is originaly expected to be 3d

            raise NotImplementedError

            hidden_states = torch.stack(encoder_outputs, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs.last_hidden_state

        # Pad inputs
        add_padding = True
        if add_padding:
            input_tensors_shape = list(hidden_states.shape())
            output_tensor_shape = input_tensors_shape[:]
            output_tensor_shape[-2] = 1504
            hidden_states = create_padded_tensor(input_tensors_shape, hidden_states, output_tensor_shape, pad_value=0, device=self.device)

        # Apply Linear layer
        hidden_states = self.projector(hidden_states)

        # Unpad
        if add_padding:
            # Unpad
            input_tensors_shape = list(hidden_states.shape())
            input_tensors_shape[-2] = 1500
            hidden_states = create_unpadded_tensor(hidden_states, input_tensors_shape)
            # Convert to Torch
            torch_hidden_states = torch.Tensor(hidden_states.data()).reshape(hidden_states.shape())
        else:
            torch_hidden_states = tt2torch_tensor(hidden_states)

        torch_pooled_output = torch_hidden_states.mean(dim=-2)
        # If something changes these dimension -2 should always work

        if add_padding:
            pooled_output = create_padded_tensor(list(torch_pooled_output.size()), torch_pooled_output, [1,1,32, torch_pooled_output.size()[-1]], pad_value=0, device=self.device)
        else:
            pooled_output = torch2tt_tensor(torch_pooled_output, self.device)

        # Apply classifier layer
        logits = self.classifier(pooled_output)

        if add_padding:
            # Unpad
            logits = create_unpadded_tensor(logits, [1, 1, 1, self.config.num_labels])

        loss = None

        if labels is not None:
            """TODO: When implementing Training"""
            raise NotImplementedError
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TtWhisperForAudioClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
