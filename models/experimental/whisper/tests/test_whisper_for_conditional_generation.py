# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch

from loguru import logger
from datasets import load_dataset
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from transformers import AutoProcessor, WhisperForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
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

import ttnn

from models.experimental.whisper.tt.whisper_for_conditional_generation import (
    TtWhisperForConditionalGeneration,
)
from models.utility_functions import (
    comp_pcc,
    torch2tt_tensor,
    tt2torch_tensor,
    is_wormhole_b0,
    is_blackhole,
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


# Pre and post process methodes
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


def _prepare_encoder_decoder_kwargs_for_generation(
    model,
    inputs_tensor: torch.Tensor,
    model_kwargs,
    model_input_name: Optional[str] = None,
) -> Dict[str, Any]:
    import inspect

    # 1. get encoder
    encoder = model.get_encoder()

    # 2. Prepare encoder args and encoder kwargs from model kwargs.
    irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
    encoder_kwargs = {
        argument: value
        for argument, value in model_kwargs.items()
        if not any(argument.startswith(p) for p in irrelevant_prefix)
    }
    encoder_signature = set(inspect.signature(encoder.forward).parameters)
    encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
    if not encoder_accepts_wildcard:
        encoder_kwargs = {
            argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
        }

    # 3. make sure that encoder returns `ModelOutput`
    model_input_name = model_input_name if model_input_name is not None else self.main_input_name
    encoder_kwargs["return_dict"] = True
    encoder_kwargs[model_input_name] = inputs_tensor
    model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

    return model_kwargs


def _prepare_decoder_input_ids_for_generation(
    generation_config,
    batch_size: int,
    decoder_start_token_id: int = None,
    bos_token_id: int = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    device: torch.device = None,
) -> torch.LongTensor:
    if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
        return model_kwargs.pop("decoder_input_ids")
    else:
        decoder_start_token_id = _get_decoder_start_token_id(generation_config, decoder_start_token_id, bos_token_id)
        if device is None:
            device = device
        return torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id


def _get_decoder_start_token_id(generation_config, decoder_start_token_id: int = None, bos_token_id: int = None) -> int:
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else generation_config.decoder_start_token_id
    )
    bos_token_id = bos_token_id if bos_token_id is not None else generation_config.bos_token_id

    if decoder_start_token_id is not None:
        return decoder_start_token_id
    elif bos_token_id is not None:
        return bos_token_id
    raise ValueError("`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.")


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


def run_generate(sample, device):
    # Create torch model
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")
    hf_reference_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    hf_reference_model.eval()

    # Setup configs
    model_config = hf_reference_model.config
    generation_config = hf_reference_model.generation_config

    # Create tt model
    tt_model = TtWhisperForConditionalGeneration(
        state_dict=hf_reference_model.state_dict(), config=model_config, device=device
    )
    tt_model.eval()

    # Librispeech dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    logger.info(f"Preparing Input sample...")
    logger.debug(ds[sample]["text"])

    inputs = processor(ds[sample]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features
    logger.debug(f"input_features {input_features.shape} {input_features}")  # 1, 80, 3000

    batch_size = input_features.shape[0]

    torch_model = True
    if torch_model:
        torch_model_kwargs = generation_config.update()
        torch_model_kwargs["output_attentions"] = generation_config.output_attentions
        torch_model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        torch_model_kwargs["use_cache"] = generation_config.use_cache

        # Prepare model args for torch model
        torch_model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
            hf_reference_model, input_features, torch_model_kwargs, "input_features"
        )
        encoder_outputs_last_hidden_state = torch_model_kwargs["encoder_outputs"].last_hidden_state

        input_ids = torch.tensor([[generation_config.decoder_start_token_id]])
        logger.debug(f"Torch input ids: {input_ids}")

        # Create Logits processor for reference model
        torch_logits_processor = get_logits_processor(input_ids, generation_config)
        # Pad decoder inputs to 32
        input_ids = pad_input_32(input_ids, hf_reference_model.config.pad_token_id).to(torch.long)

    run_tt_model = True
    if run_tt_model:
        tt_model_kwargs = generation_config.update()
        tt_model_kwargs["output_attentions"] = generation_config.output_attentions
        tt_model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        tt_model_kwargs["use_cache"] = generation_config.use_cache

        tt_input_features = torch2tt_tensor(input_features, device, ttnn.ROW_MAJOR_LAYOUT)
        # Prepare model args for tt model
        tt_model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
            tt_model, tt_input_features, tt_model_kwargs, "input_features"
        )
        tt_encoder_outputs_last_hidden_state = tt_model_kwargs["encoder_outputs"].last_hidden_state

        tt_input_ids = torch.tensor([[generation_config.decoder_start_token_id]])
        logger.debug(f"TT input ids: {tt_input_ids}")

        # Create Logits processor
        logits_processor = get_logits_processor(tt_input_ids, generation_config)
        # Pad decoder inputs to 32
        tt_input_ids = pad_input_32(tt_input_ids, hf_reference_model.config.pad_token_id).to(torch.long)

    decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)

    for i in range(32):
        if torch_model:
            # generation loop - greedy search implementation
            logger.info(f"Running Torch model forward...")

            decoder_attention_mask = None

            pt_out = hf_reference_model(
                encoder_outputs=(encoder_outputs_last_hidden_state, None, None),
                decoder_input_ids=input_ids,
                use_cache=False,
                return_dict=True,
                output_attentions=generation_config.output_attentions,
                output_hidden_states=generation_config.output_hidden_states,
            )

            torch_next_token_logits = pt_out.logits

            torch_next_token_logits = pt_out.logits[:, i, :]

            # pre-process distribution
            torch_next_tokens_scores = torch_logits_processor(input_features, torch_next_token_logits)

            # argmax
            torch_next_tokens = torch.argmax(torch_next_tokens_scores, dim=-1)

            # We need to expand decoder_input_ids
            if (i + 1) % 32 == 0:
                input_ids = torch.cat([input_ids, decoder_start_values], dim=1)

            input_ids[:, i + 1] = torch_next_tokens[:, None]

            transcription = processor.batch_decode(input_ids, skip_special_tokens=True)[0]

            if not run_tt_model and (torch_next_tokens == generation_config.eos_token_id):
                break

        if run_tt_model:
            logger.info(f"Running TT model forward...")

            # Run model with all inputs
            tt_out = tt_model(
                encoder_outputs=(tt_encoder_outputs_last_hidden_state, None, None),
                decoder_input_ids=tt_input_ids,
                return_dict=True,
                output_attentions=generation_config.output_attentions,
                output_hidden_states=generation_config.output_hidden_states,
            )

            # Convert to Torch
            logits_to_torch = tt2torch_tensor(tt_out.logits)
            logits_to_torch = torch.squeeze(logits_to_torch, 0)

            next_token_logits = logits_to_torch[:, i, :]

            if torch_model:
                # Compare PCC:
                does_pass, pcc_message = comp_pcc(torch_next_token_logits, next_token_logits, 0.98)
                logger.info(pcc_message)
                # assert does_pass

            # pre-process distribution
            next_tokens_scores = logits_processor(input_features, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # We need to expand decoder_input_ids
            if (i + 1) % 32 == 0:
                tt_input_ids = torch.cat([tt_input_ids, decoder_start_values], dim=1)

            tt_input_ids[:, i + 1] = next_tokens[:, None]

            if next_tokens == generation_config.eos_token_id:
                break

            tt_transcription = processor.batch_decode(tt_input_ids, skip_special_tokens=True)[0]

    logger.info(f"Final transcriptions")

    if torch_model:
        torch_transcription = processor.batch_decode(input_ids, skip_special_tokens=True)[0]
        logger.info(f"Torch transcription: {torch_transcription}")

    if run_tt_model:
        tt_transcription = processor.batch_decode(tt_input_ids, skip_special_tokens=True)[0]
        logger.info(f"TT transcription: {tt_transcription}")

        return tt_transcription


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_WhipserForConditionalGeneration_inference(device):
    torch.manual_seed(1234)

    sample = 0
    correct_transcription = " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."

    # sample = 3
    # correct_transcription = " He has grave doubts whether Sir Frederick Leighton's work is really Greek after all, and can discover in it but little of rocky Ithaca."

    tt_transcription = run_generate(sample=sample, device=device)

    logger.info(tt_transcription)
    logger.info(correct_transcription)

    assert tt_transcription == correct_transcription
