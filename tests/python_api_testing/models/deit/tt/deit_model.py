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

from activations import ACT2FN
from deit_config import DeiTConfig
from deit_embeddings import DeiTEmbeddings
from deit_encoder import DeiTEncoder

class DeiTModel(nn.Module):
    def __init__(self,
                config,
                add_pooling_layer: bool = True,
                use_mask_token: bool = False,
                state_dict = None,
                base_address = ""):
        super().__init__()

        self.config = config

        self.base_address_with_dot = "" if base_address=="" else f"{base_address}."

        self.embeddings = DeiTEmbeddings(config, use_mask_token=use_mask_token, state_dict=state_dict, base_address=f"{self.base_address_with_dot}embeddings")
        self.encoder = DeiTEncoder(config, state_dict=state_dict, base_address=f"{self.base_address_with_dot}encoder")

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        assign_norm_weight(self.layernorm, state_dict, f"{self.base_address_with_dot}layernorm")

        # self.pooler = DeiTPooler(config, state_dict, f"{self.base_address_with_dot}pooler") if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self) -> DeiTPatchEmbeddings:
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
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
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















# class DeiTPreTrainedModel(nn.Module):
#     """
#     An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
#     models.
#     """

#     config_class = DeiTConfig
#     base_model_prefix = "deit"
#     main_input_name = "pixel_values"
#     supports_gradient_checkpointing = True
#     _no_split_modules = []

#     def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
#         """Initialize the weights"""
#         if isinstance(module, (nn.Linear, nn.Conv2d)):
#             # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
#             # `trunc_normal_cpu` not implemented in `half` issues
#             module.weight.data = nn.init.trunc_normal_(
#                 module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
#             ).to(module.weight.dtype)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def _set_gradient_checkpointing(self, module: DeiTEncoder, value: bool = False) -> None:
#         if isinstance(module, DeiTEncoder):
#             module.gradient_checkpointing = value


# DEIT_START_DOCSTRING = r"""
#     This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
#     as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
#     behavior.

#     Parameters:
#         config ([`DeiTConfig`]): Model configuration class with all the parameters of the model.
#             Initializing with a config file does not load the weights associated with the model, only the
#             configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# """

# DEIT_INPUTS_DOCSTRING = r"""
#     Args:
#         pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
#             Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
#             [`DeiTImageProcessor.__call__`] for details.

#         head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
#             Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

#             - 1 indicates the head is **not masked**,
#             - 0 indicates the head is **masked**.

#         output_attentions (`bool`, *optional*):
#             Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
#             tensors for more detail.
#         output_hidden_states (`bool`, *optional*):
#             Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
#             more detail.
#         return_dict (`bool`, *optional*):
#             Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# """


# class DeiTForMaskedImageModeling(DeiTPreTrainedModel):
#     def __init__(self, config: DeiTConfig) -> None:
#         super().__init__(config)

#         self.deit = DeiTModel(config, add_pooling_layer=False, use_mask_token=True)

#         self.decoder = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=config.hidden_size,
#                 out_channels=config.encoder_stride**2 * config.num_channels,
#                 kernel_size=1,
#             ),
#             nn.PixelShuffle(config.encoder_stride),
#         )

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         bool_masked_pos: Optional[torch.BoolTensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#     ) -> Union[tuple]:
#         r"""
#         bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
#             Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

#         Returns:

#         Examples:
#         ```python
#         >>> from transformers import AutoImageProcessor, DeiTForMaskedImageModeling
#         >>> import torch
#         >>> from PIL import Image
#         >>> import requests

#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
#         >>> model = DeiTForMaskedImageModeling.from_pretrained("facebook/deit-base-distilled-patch16-224")

#         >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
#         >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
#         >>> # create random boolean mask of shape (batch_size, num_patches)
#         >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

#         >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
#         >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
#         >>> list(reconstructed_pixel_values.shape)
#         [1, 3, 224, 224]
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.deit(
#             pixel_values,
#             bool_masked_pos=bool_masked_pos,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         # Reshape to (batch_size, num_channels, height, width)
#         sequence_output = sequence_output[:, 1:-1]
#         batch_size, sequence_length, num_channels = sequence_output.shape
#         height = width = int(sequence_length**0.5)
#         sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

#         # Reconstruct pixel values
#         reconstructed_pixel_values = self.decoder(sequence_output)

#         masked_im_loss = None
#         if bool_masked_pos is not None:
#             size = self.config.image_size // self.config.patch_size
#             bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
#             mask = (
#                 bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
#                 .repeat_interleave(self.config.patch_size, 2)
#                 .unsqueeze(1)
#                 .contiguous()
#             )
#             reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
#             masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

#         if not return_dict:
#             output = (reconstructed_pixel_values,) + outputs[1:]
#             return ((masked_im_loss,) + output) if masked_im_loss is not None else output

#         return MaskedImageModelingOutput(
#             loss=masked_im_loss,
#             reconstruction=reconstructed_pixel_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# class DeiTForImageClassification(DeiTPreTrainedModel):
#     def __init__(self, config: DeiTConfig) -> None:
#         super().__init__(config)

#         self.num_labels = config.num_labels
#         self.deit = DeiTModel(config, add_pooling_layer=False)

#         # Classifier head
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         labels: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#     ) -> Union[tuple]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

#         Returns:

#         Examples:

#         ```python
#         >>> from transformers import AutoImageProcessor, DeiTForImageClassification
#         >>> import torch
#         >>> from PIL import Image
#         >>> import requests

#         >>> torch.manual_seed(3)  # doctest: +IGNORE_RESULT
#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)

#         >>> # note: we are loading a DeiTForImageClassificationWithTeacher from the hub here,
#         >>> # so the head will be randomly initialized, hence the predictions will be random
#         >>> image_processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
#         >>> model = DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224")

#         >>> inputs = image_processor(images=image, return_tensors="pt")
#         >>> outputs = model(**inputs)
#         >>> logits = outputs.logits
#         >>> # model predicts one of the 1000 ImageNet classes
#         >>> predicted_class_idx = logits.argmax(-1).item()
#         >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
#         Predicted class: magpie
#         ```"""
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.deit(
#             pixel_values,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         logits = self.classifier(sequence_output[:, 0, :])
#         # we don't use the distillation token

#         loss = None
#         if labels is not None:
#             labels = labels.to(logits.device)
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"

#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 loss_fct = BCEWithLogitsLoss()
#                 loss = loss_fct(logits, labels)
#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return ((loss,) + output) if loss is not None else output

#         return ImageClassifierOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )


# @dataclass
# class DeiTForImageClassificationWithTeacherOutput(ModelOutput):
#     """
#     Output type of [`DeiTForImageClassificationWithTeacher`].

#     Args:
#         logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
#             Prediction scores as the average of the cls_logits and distillation logits.
#         cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
#             Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
#             class token).
#         distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
#             Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
#             distillation token).
#         hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
#             Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
#             shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
#             plus the initial embedding outputs.
#         attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
#             Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
#             sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
#             the self-attention heads.
#     """

#     logits: torch.FloatTensor = None
#     cls_logits: torch.FloatTensor = None
#     distillation_logits: torch.FloatTensor = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None


# @add_start_docstrings(
#     """
#     DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of
#     the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

#     .. warning::

#            This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
#            supported.
#     """,
#     DEIT_START_DOCSTRING,
# )
# class DeiTForImageClassificationWithTeacher(DeiTPreTrainedModel):
#     def __init__(self, config: DeiTConfig) -> None:
#         super().__init__(config)

#         self.num_labels = config.num_labels
#         self.deit = DeiTModel(config, add_pooling_layer=False)

#         # Classifier heads
#         self.cls_classifier = (
#             nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
#         )
#         self.distillation_classifier = (
#             nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
#         )

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_IMAGE_CLASS_CHECKPOINT,
#         output_type=DeiTForImageClassificationWithTeacherOutput,
#         config_class=_CONFIG_FOR_DOC,
#         expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
#     )
#     def forward(
#         self,
#         pixel_values: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[tuple, DeiTForImageClassificationWithTeacherOutput]:
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.deit(
#             pixel_values,
#             head_mask=head_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         sequence_output = outputs[0]

#         cls_logits = self.cls_classifier(sequence_output[:, 0, :])
#         distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])

#         # during inference, return the average of both classifier predictions
#         logits = (cls_logits + distillation_logits) / 2

#         if not return_dict:
#             output = (logits, cls_logits, distillation_logits) + outputs[1:]
#             return output

#         return DeiTForImageClassificationWithTeacherOutput(
#             logits=logits,
#             cls_logits=cls_logits,
#             distillation_logits=distillation_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
