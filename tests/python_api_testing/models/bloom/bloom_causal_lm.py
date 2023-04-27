from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from abc import abstractmethod
import torch
import math
from torch.nn import functional as F

from libs import tt_lib as ttm
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
import numpy as np
import python_api_testing.models.bloom.bloom_utils as bloom_utils
import python_api_testing.models.bloom.baddbmm
import python_api_testing.models.bloom.bloom_attention as bloom_attention
import python_api_testing.models.bloom.bloom_mlp as bloom_mlp
import python_api_testing.models.bloom.bloom_block as bloom_block
import python_api_testing.model.bloom.pt_bloom_model as pt_bloom_model

from fused_ops.linear import Linear as TtLinear

from fused_ops.layernorm import Layernorm as TtLayernorm

from fused_ops.softmax import softmax as TtSoftmax
from transformers import BloomForCausalLM

from typing import Optional, Tuple, Union

class BloomForCausalLM():

    def __init__(self, hugging_bloom_reference_model, hidden_size, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers)

        state_dict = hugging_bloom_reference_model

        self.transformer = bloom_model.(hugging_bloom_reference_model, hidden_size, n_head, vocab_size, embed_dim, layer_norm_epsilon, num_hidden_layers)

        lm_head_weight = bloom_utils.pt_load_layer_weights("trasnsormer.lm_head.weight", state_dict)

        self.lm_head = nn.Linear(hidden_size, cvocab_size, bias=False)
        self.lm_head.weight = lm_head_weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)


@add_start_docstrings(
    """
    The Bloom Model transformer with a sequence classification head on top (linear layer).
    [`BloomForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    BLOOM_START_DOCSTRING,
)


def run_bloom_causal_lm_inference(device):

    hugging_bloom_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    print(hugging_bloom_reference_model.state_dict())
    tt_bloom_model = TtBloomModel(device, hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)
    pt_bloom_model = BloomModel(hugging_bloom_reference_model, 1024, 32,  250880, 1024, 1e-5, 2)


    # Prepare input
    torch.manual_seed(0)

    input_ids = torch.randint(0, 100, (64, 2048))

    pt_out = pt_bloom_model.forward(input_ids)

    print(pt_out[0])
    print("PT finished")

    tt_out = tt_bloom_model.forward(device, input_ids)

    print("TT finished")

    print(tt_out)

    print(comp_allclose(pt_out[0], tt_out[0]))
    #print(comp_pcc(pt_out, tt_out_converted))

if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)
    run_bloom_causal_lm_inference(device)
    ttm.device.CloseDevice(device)
