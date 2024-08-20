# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import json
import warnings
import ttnn

from transformers import BloomForCausalLM, BloomTokenizerFast
from transformers.generation.configuration_utils import GenerationConfig
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)

from loguru import logger
import models.experimental.bloom.tt.bloom_causal_lm as bloom_causal_lm
import models.experimental.bloom.bloom_utils as bloom_utils
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
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
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
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


def run_generate(input_sentance, run_tt_model, device):
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    hf_reference_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", torchscript=False)
    hf_reference_model.eval()

    if run_tt_model:
        tt_model = bloom_causal_lm.TtBloomForCausalLM(
            hf_reference_model.config, hf_reference_model.state_dict(), device
        )

    # Prepare input
    tokenized = tokenizer(input_sentance, return_tensors="pt")  # Batch size 1
    generation_config = hf_reference_model.generation_config

    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    logger.debug(f"input_ids {input_ids.shape} {input_ids}")
    input_ids = bloom_utils.pad_input_tensor(tokenized.input_ids, generation_config.pad_token_id, 2)
    logger.debug(f"padded input_ids {input_ids.shape} {input_ids}")

    # Start to generate i'th token
    i = input_ids.shape[1]

    logits_processor = get_logits_processor(input_ids, hf_reference_model.config)

    # Input IDs expansion
    input_ids_expansion = generation_config.pad_token_id * torch.ones(1, 2).to(torch.long)

    while i < 64:
        # PyTorch forward pass
        pt_out = hf_reference_model(input_ids=input_ids)  # , attention_mask=attention_mask)
        pt_next_token_logits = pt_out.logits

        if run_tt_model:
            tt_out = tt_model.forward(device, input_ids=input_ids, return_dict=False)  # attention_mask=attention_mask,
            next_token_logits = tt_out[0]

            does_pass, pcc_message = comp_pcc(pt_next_token_logits, next_token_logits, 0.6)
            logger.debug(pcc_message)
        else:
            next_token_logits = pt_next_token_logits

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)
        pt_next_tokens_scores = logits_processor(input_ids, pt_next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        pt_next_tokens = torch.argmax(pt_next_tokens_scores, dim=-1)

        logger.debug(f"tt_next_tokens {next_tokens}")
        logger.debug(f"pt_next_tokens {pt_next_tokens}")

        if next_tokens[0][i - 1] == generation_config.eos_token_id:
            break

        # We need to expand decoder_input_ids
        if i % 2 == 0:
            input_ids = torch.cat([input_ids, input_ids_expansion], dim=1)

        # Append predicted token
        input_ids[0][i] = next_tokens[0][i - 1]

        logger.debug(f"input_ids {input_ids[0]}")
        i += 1

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    device = ttnn.open_device(0)

    output_sentance_night = run_generate("It was a dark and stormy night", run_tt_model=True, device=device)
    output_sentance_alice = run_generate("Alice stepped through the small door", run_tt_model=True, device=device)

    logger.info(f"Decoded output night: {output_sentance_night}")
    logger.info(f"Decoded output alice: {output_sentance_alice}")

    ttnn.close_device(device)
