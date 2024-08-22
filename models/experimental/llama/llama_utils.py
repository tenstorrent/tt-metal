# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from typing import List, Optional, Tuple, Union
import ttnn
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


def linear(x, weight, bias, device):
    weight = tt_to_torch_tensor(weight)
    x = tt_to_torch_tensor(x)

    weight = torch.transpose(weight, 3, 2)
    x = torch.matmul(x, weight)
    x = torch_to_tt_tensor_rm(x, device, put_on_device=False)

    if bias is not None:
        x = ttnn.add(x, bias)
    return x


def gen_position_ids(input_ids):
    # get positions_ids values
    past_key_values_length = 0
    seq_length = input_ids.shape[1]
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=None,
    )

    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    return position_ids


def _merge_criteria_processor_list(
    default_list,  # Union[LogitsProcessorList, StoppingCriteriaList],
    custom_list,  # Union[LogitsProcessorList, StoppingCriteriaList],
):  # -> Union[LogitsProcessorList, StoppingCriteriaList]:
    if len(custom_list) == 0:
        return default_list

    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                raise ValueError(
                    f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                    f" `generate`, but it has already been created with the values {default}. {default} has been"
                    " created by passing the corresponding arguments to generate or by the model's config default"
                    f" values. If you just want to change the default values of {object_type} consider passing"
                    f" them as arguments to `generate` instead of using a custom {object_type}."
                )
    default_list.extend(custom_list)
    return default_list


def _get_logits_processor(
    generation_config: GenerationConfig,
    input_ids_seq_length: int,
    encoder_input_ids,  # torch.LongTensor
    prefix_allowed_tokens_fn,  # Callable[[int, torch.Tensor], List[int]],
    logits_processor,  # Optional[LogitsProcessorList]
):  # -> LogitsProcessorList:
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    # instantiate processors list
    processors = LogitsProcessorList()

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=generation_config.diversity_penalty,
                num_beams=generation_config.num_beams,
                num_beam_groups=generation_config.num_beam_groups,
            )
        )
    if generation_config.encoder_repetition_penalty is not None and generation_config.encoder_repetition_penalty != 1.0:
        processors.append(
            EncoderRepetitionPenaltyLogitsProcessor(
                penalty=generation_config.encoder_repetition_penalty,
                encoder_input_ids=encoder_input_ids,
            )
        )
    if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
    if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
    if (
        generation_config.encoder_no_repeat_ngram_size is not None
        and generation_config.encoder_no_repeat_ngram_size > 0
    ):
        if self.config.is_encoder_decoder:
            processors.append(
                EncoderNoRepeatNGramLogitsProcessor(generation_config.encoder_no_repeat_ngram_size, encoder_input_ids)
            )
        else:
            raise ValueError("It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture")
    if generation_config.bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id))
    if (
        generation_config.min_length is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_length > 0
    ):
        processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
    if (
        generation_config.min_new_tokens is not None
        and generation_config.eos_token_id is not None
        and generation_config.min_new_tokens > 0
    ):
        processors.append(
            MinNewTokensLengthLogitsProcessor(
                input_ids_seq_length,
                generation_config.min_new_tokens,
                generation_config.eos_token_id,
            )
        )
    if prefix_allowed_tokens_fn is not None:
        processors.append(
            PrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn,
                generation_config.num_beams // generation_config.num_beam_groups,
            )
        )
    if generation_config.forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
    if generation_config.forced_eos_token_id is not None:
        processors.append(
            ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
        )
    if generation_config.remove_invalid_values is True:
        processors.append(InfNanRemoveLogitsProcessor())
    if generation_config.exponential_decay_length_penalty is not None:
        processors.append(
            ExponentialDecayLengthPenalty(
                generation_config.exponential_decay_length_penalty,
                generation_config.eos_token_id,
                input_ids_seq_length,
            )
        )
    if generation_config.suppress_tokens is not None:
        processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
    if generation_config.begin_suppress_tokens is not None:
        begin_index = input_ids_seq_length
        begin_index = (
            begin_index
            if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
            else begin_index + 1
        )
        if generation_config.forced_decoder_ids is not None:
            # generation starts after the last token that is forced
            begin_index += generation_config.forced_decoder_ids[-1][0]
        processors.append(SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index))
    if generation_config.forced_decoder_ids is not None:
        processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids))
    processors = _merge_criteria_processor_list(processors, logits_processor)
    # `LogitNormalization` should always be the last logit processor, when present
    if generation_config.renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors


def get_logits_processor(input_ids, config):
    generation_config = GenerationConfig.from_model_config(config)
    input_ids_seq_length = input_ids.shape[-1]

    logits_processor = _get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList(),
    )

    return logits_processor


def pad_input_32_left(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([pad_tensor, tensor], dim=1)

    return tensor


def prepare_llama_input(prompt, tokenizer, configuration, is_padded=False):
    inputs = tokenizer(prompt, return_tensors="pt")
    print("inputs: ", inputs)

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    position_ids = gen_position_ids(input_ids)

    # pad input tensors
    if is_padded:
        input_ids = pad_input_32_left(input_ids, configuration.pad_token_id)
        attention_mask = pad_input_32_left(attention_mask, configuration.pad_token_id)
        position_ids = gen_position_ids(input_ids)

    return input_ids, attention_mask, position_ids


def get_next_llama_output_token(logits_processor, input_ids, out_tensor, order, model="Pytorch"):
    next_token_logits = out_tensor[:, -1, :]
    # pre-process distribution
    next_tokens_scores = logits_processor(input_ids, next_token_logits)

    # argmax
    next_tokens = torch.argmax(next_tokens_scores, dim=-1)
    logger.debug(f"{model} forward {order+1}-th generated id: {next_tokens}")

    return next_tokens


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full(
        (tgt_len, tgt_len),
        torch.tensor(torch.finfo(dtype).min, device=device),
        device=device,
    )
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def shape_tt(states, batch_size, seq_len, n_heads, head_dim):
    tt_out = ttnn.reshape_on_device(states, batch_size, seq_len, n_heads, head_dim)
    tt_out = ttnn.transpose(tt_out, 1, -2)

    return tt_out
