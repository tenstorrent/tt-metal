# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional
import copy
import inspect
from typing import Any, Dict, List, Optional, Union
from models.experimental.functional_trocr.tt.ttnn_trocr_causal_lm import trocr_causal_lm
from models.experimental.functional_trocr.tt.ttnn_encoder_vit import vit


class LogitsProcessorList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class StoppingCriteriaList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None


class MaxLengthCriteria:
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


class GenerationMixin:
    def __init__(self, config, model, device, parameters):
        self.model = model
        self.device = device
        self.config = config
        self.parameters = parameters

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        cls_token=None,
        position_embeddings=None,
        head_masks=None,
    ) -> Dict[str, Any]:
        model_kwargs["encoder_outputs"] = vit(
            self.model.encoder.config,
            inputs_tensor,
            attention_mask=head_masks,
            cls_token=cls_token,
            position_embeddings=position_embeddings,
            parameters=self.parameters.encoder,
            device=self.device,
        )
        return model_kwargs

    def _get_stopping_criteria(
        self,
        generation_config,
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            criteria.append(MaxLengthCriteria(max_length=generation_config.max_length))
        return criteria

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config=None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        synced_gpus: Optional[bool] = None,
        cls_token=None,
        position_embeddings=None,
        head_masks=None,
        **kwargs,
    ):
        if generation_config is None:
            generation_config = self.model.generation_config

        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs

        # Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        inputs_tensor = inputs
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
        batch_size = inputs_tensor.shape[0]

        if self.model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor,
                model_kwargs,
                cls_token=cls_token,
                position_embeddings=position_embeddings,
                head_masks=head_masks,
            )

        if self.model.config.is_encoder_decoder:
            decoder_start_token_id = generation_config.decoder_start_token_id
            input_ids = ttnn.ones((batch_size, 1))
            input_ids = ttnn.to_torch(input_ids)
            input_ids = input_ids * decoder_start_token_id

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]

        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        # prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config)

        return self.greedy_search(
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores

        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )
        scores = () if (return_dict_in_generate and output_scores) else None
        unfinished_sequences = ttnn.ones((1, input_ids.shape[0]))
        unfinished_sequences = ttnn.to_torch(unfinished_sequences).to(torch.int32).squeeze(0)

        while True:
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)
            input_ids = ttnn.from_torch(input_ids, device=self.device, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT)
            encoder_hidden_states = model_inputs["encoder_outputs"][0]
            attention_mask = model_inputs["decoder_attention_mask"]

            outputs = trocr_causal_lm(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                config=self.config,
                device=self.device,
                parameters=self.parameters.decoder,
            )
            output = ttnn.to_torch(outputs[0])

            next_token_logits = output[:, -1, :]

            input_ids = ttnn.to_torch(input_ids)

            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break

        return input_ids
