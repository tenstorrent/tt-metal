# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
from typing import Callable, List, Optional, Union
from transformers.generation.configuration_utils import GenerationConfig

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)

from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
)


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


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_generate(
    input_sentance,
    tokenizer,
    tt_model_constructor,
    device,
    run_tt_model=True,
    log=True,
    comp_pcc=None,
):
    tt_model, hf_reference_model = tt_model_constructor(device)

    # Prepare input
    tokenized = tokenizer(input_sentance, return_tensors="pt")  # Batch size 1

    input_ids = pad_input_32(tokenized.input_ids, hf_reference_model.generation_config.pad_token_id)
    attention_mask = pad_input_32(tokenized.attention_mask, 0)

    if log:
        logger.debug(f"input_ids {input_ids.shape} {input_ids}")
        logger.debug(f"attention_mask {attention_mask.shape} {attention_mask}")

    logits_processor = get_logits_processor(input_ids, hf_reference_model.config)

    decoder_start_values = hf_reference_model.generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)
    decoder_input_ids = hf_reference_model.generation_config.pad_token_id * torch.ones(1, 64).to(torch.long)

    if log:
        logger.debug(f"decoder_input_ids {decoder_input_ids}")

    encoder_outputs = None
    use_cache = False

    for i in range(64):
        # PyTorch forward pass
        pt_out = hf_reference_model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )

        if run_tt_model:
            tt_out = tt_model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                use_cache=use_cache,
            )
            encoder_outputs = tt_out.encoder_outputs
            next_token_logits = tt_out.logits

            if comp_pcc is not None:
                does_pass, pcc_message = comp_pcc(pt_out.logits, tt_out.logits, 0.98)

                if log:
                    logger.info(pcc_message)
        else:
            next_token_logits = pt_out.logits

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if log:
            logger.debug(f"next_tokens {next_tokens}")

        if next_tokens[0][i] == hf_reference_model.generation_config.eos_token_id:
            break

        # We need to expand decoder_input_ids
        if (i + 1) % 32 == 0:
            decoder_input_ids = torch.cat([decoder_input_ids, decoder_start_values], dim=1)

        decoder_input_ids[0][i + 1] = next_tokens[0][i]

        if log:
            logger.debug(f"decoder_input_ids {decoder_input_ids[0]}")

    return tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
