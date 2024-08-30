# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from models.utility_functions import torch2tt_tensor, tt2torch_tensor

from models.experimental.whisper.tt.whisper_common import linear

from models.experimental.whisper.tt.whisper_encoder import TtWhisperEncoder


@dataclass
class TtWhisperForAudioClassificationOutput:
    loss: Optional[ttnn.Tensor] = None
    logits: ttnn.Tensor = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtWhisperForAudioClassification(nn.Module):
    def __init__(self, state_dict, device, config):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.config = config

        self.encoder = TtWhisperEncoder(
            state_dict=state_dict,
            base_address="encoder",
            device=self.device,
            config=config,
        )

        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            # Not using this parameter for now
            N, C, H, W = 1, 1, 1, num_layers
            weight_init_const = 1.0 / num_layers
            self.layer_weights = ttnn.full((1, 1, 1, num_layers), weight_init_const)

        self.projector_weight = torch2tt_tensor(state_dict[f"projector.weight"], self.device, ttnn.ROW_MAJOR_LAYOUT)
        self.projector_bias = torch2tt_tensor(state_dict[f"projector.bias"], self.device, ttnn.ROW_MAJOR_LAYOUT)

        self.classifier_weight = torch2tt_tensor(
            state_dict[f"classifier.weight"],
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        self.classifier_bias = torch2tt_tensor(state_dict[f"classifier.bias"], self.device, ttnn.ROW_MAJOR_LAYOUT)

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
        input_features: Optional[ttnn.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ttnn.Tensor], TtWhisperForAudioClassificationOutput]:
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
            # TODO: Not supported for now.
            # Parameter use_weighted_layer_sum is false and not used in config we are implementing
            # When implementing keep in mind that the size of each individual torch tensor
            # is originaly expected to be 3d

            raise NotImplementedError

            hidden_states = torch.stack(encoder_outputs, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs.last_hidden_state

        # Apply Linear layer
        hidden_states = linear(hidden_states, self.projector_weight, self.projector_bias)

        # Torch mean
        torch_hidden_states = tt2torch_tensor(hidden_states)
        torch_pooled_output = torch_hidden_states.mean(dim=-2)
        # If something changes these dimension -2 should always work
        pooled_output = torch2tt_tensor(torch_pooled_output, self.device, ttnn.ROW_MAJOR_LAYOUT)

        # Apply classifier layer
        logits = linear(pooled_output, self.classifier_weight, self.classifier_bias)

        loss = None

        if labels is not None:
            # TODO: When implementing Training
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
