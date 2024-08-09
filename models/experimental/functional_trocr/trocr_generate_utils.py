# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger
from copy import deepcopy
import torch
from models.experimental.functional_trocr.reference import functional_torch_vit
from models.experimental.functional_trocr.reference.functional_torch_trocr import TrOCRForCausalLM
from models.experimental.functional_trocr.trocr_utils import (
    LogitsProcessorList,
    StoppingCriteria,
    MaxLengthCriteria,
    StoppingCriteriaList,
)


def _prepare_model_inputs(
    inputs: Optional[torch.Tensor] = None,
    bos_token_id: Optional[int] = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
    """
    This function extracts the model-specific `inputs` for generation.
    """
    # 1. retrieve all kwargs that are non-None or non-model input related.
    # some encoder-decoder models have different names for model and encoder
    input_name = "pixel_values"

    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

    return inputs, input_name, model_kwargs


def _prepare_attention_mask_for_generation(
    self,
    inputs: torch.Tensor,
    pad_token_id: Optional[int],
    eos_token_id: Optional[Union[int, List[int]]],
) -> torch.LongTensor:
    is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
    is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

    # Check if input is input_ids and padded -> only then is attention_mask defined
    if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
        return inputs.ne(pad_token_id).long()
    else:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)


def _prepare_encoder_decoder_kwargs_for_generation(
    config,
    inputs_tensor: torch.Tensor,
    model_kwargs,
    model_input_name,
    parameters,
    cls_token,
    position_embed,
):
    torch_cls_token = torch.nn.Parameter(cls_token)
    torch_position_embeddings = torch.nn.Parameter(position_embed)

    model_kwargs["return_dict"] = False
    model_kwargs["output_attentions"] = False
    model_kwargs["output_hidden_states"] = False

    encoder_op_list = functional_torch_vit.vit(
        config,
        inputs_tensor,
        torch_position_embeddings,
        torch_cls_token,
        attention_mask=None,
        parameters=parameters,
    )

    model_kwargs["encoder_outputs"] = encoder_op_list[0]
    return model_kwargs


def _get_decoder_start_token_id(
    decoder_start_token_id: Union[int, List[int]] = None,
    bos_token_id: int = None,
    generation_config=None,
) -> int:
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else generation_config.decoder_start_token_id
    )
    bos_token_id = bos_token_id if bos_token_id is not None else generation_config.bos_token_id

    if decoder_start_token_id is not None:
        return decoder_start_token_id
    elif bos_token_id is not None:
        return bos_token_id
    raise ValueError("`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.")


def _merge_criteria_processor_list(
    default_list: Union[LogitsProcessorList, StoppingCriteriaList],
    custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
) -> Union[LogitsProcessorList, StoppingCriteriaList]:
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                raise ValueError(
                    f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                    f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                    " created by passing the corresponding arguments to generate or by the model's config default"
                    f" values. If you just want to change the default values of {object_type} consider passing"
                    f" them as arguments to `.generate()` instead of using a custom {object_type}."
                )
    default_list.extend(custom_list)
    return default_list


def _prepare_decoder_input_ids_for_generation(
    config,
    batch_size: int,
    model_input_name: str,
    model_kwargs: Dict[str, torch.Tensor],
    decoder_start_token_id: Union[int, List[int]] = None,
    bos_token_id: int = None,
    device: torch.device = None,
) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
    """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
    # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
    # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
    if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
        decoder_input_ids = model_kwargs.pop("decoder_input_ids")
    elif "input_ids" in model_kwargs and model_input_name != "input_ids":
        decoder_input_ids = model_kwargs.pop("input_ids")
    else:
        decoder_input_ids = None

    # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
    decoder_start_token_id = _get_decoder_start_token_id(decoder_start_token_id, bos_token_id)

    if isinstance(decoder_start_token_id, list):
        if len(decoder_start_token_id) != batch_size:
            raise ValueError(
                f"`decoder_start_token_id` expcted to have length {batch_size} but got {len(decoder_start_token_id)}"
            )
        decoder_input_ids_start = torch.tensor(decoder_start_token_id, dtype=torch.long)
        decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
    else:
        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long) * decoder_start_token_id

    # no user input -> use decoder_start_token_id as decoder_input_ids
    if decoder_input_ids is None:
        decoder_input_ids = decoder_input_ids_start
    # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
    elif config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
        pass
    elif config.model_type in ["whisper"]:
        pass
    # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
    # decoder_attention_mask if provided)
    elif (
        isinstance(decoder_start_token_id, int) and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
    ) or (
        isinstance(decoder_start_token_id, torch.Tensor)
        and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
    ):
        decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            decoder_attention_mask = torch.cat(
                (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                dim=-1,
            )
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask

    return decoder_input_ids, model_kwargs


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria


def decoder_prepare_inputs(input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if past_key_values:
        past_length = past_key_values[0][0].shape[2]

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
    }


def prepare_inputs_for_generation(
    input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
):
    decoder_inputs = decoder_prepare_inputs(input_ids, past_key_values=past_key_values)
    decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
    input_dict = {
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_input_ids": decoder_inputs["input_ids"],
        "encoder_outputs": encoder_outputs,
        "past_key_values": decoder_inputs["past_key_values"],
        "use_cache": use_cache,
    }
    return input_dict


def _get_stopping_criteria(
    config, generation_config, stopping_criteria: Optional[StoppingCriteriaList]
) -> StoppingCriteriaList:
    criteria = StoppingCriteriaList()
    if generation_config.max_length is not None:
        max_position_embeddings = getattr(config, "max_position_embeddings", None)
        criteria.append(
            MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings,
            )
        )
    if generation_config.max_time is not None:
        criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
    criteria = _merge_criteria_processor_list(criteria, stopping_criteria)
    return criteria


def greedy_search(
    config,
    input_ids: torch.LongTensor,
    generation_config,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    parameters=None,
    **model_kwargs,
):
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else generation_config.output_scores
    output_attentions = output_attentions if output_attentions is not None else generation_config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    while True:
        # prepare model inputs
        model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)

        outputs = TrOCRForCausalLM(
            config=config.decoder,
            input_ids=model_inputs["decoder_input_ids"],
            attention_mask=model_inputs["decoder_attention_mask"],
            encoder_hidden_states=model_inputs["encoder_outputs"],
            use_cache=model_inputs["use_cache"],
            parameters=parameters,
        )

        next_token_logits = outputs[:, -1, :]

        # pre-process distribution
        next_tokens_scores = next_token_logits

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished:
            break

    return input_ids


def generate(
    config,
    generation_config,
    inputs,
    parameters,
    cls_token,
    position_embed,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    **kwargs,
):
    model_kwargs = generation_config.update(**kwargs)

    # Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # Define model inputs
    inputs_tensor, model_input_name, model_kwargs = _prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states

    if not config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = False
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = _prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    if config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
            config.encoder, inputs_tensor, model_kwargs, model_input_name, parameters.encoder, cls_token, position_embed
        )

    if config.is_encoder_decoder:
        input_ids, model_kwargs = _prepare_decoder_input_ids_for_generation(
            config=config,
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
        )

    # Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length

    # prepare stopping criteria
    prepared_stopping_criteria = _get_stopping_criteria(
        config=config, generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    return greedy_search(
        config,
        input_ids,
        generation_config,
        logits_processor=[],
        stopping_criteria=prepared_stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        output_scores=generation_config.output_scores,
        output_logits=generation_config.output_logits,
        return_dict_in_generate=generation_config.return_dict_in_generate,
        parameters=parameters.decoder,
        **model_kwargs,
    )
