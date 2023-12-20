# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from typing import Any, Dict, Optional

from models.generation_utils import pad_input_32, get_logits_processor

import tt_lib

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)


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


def run_generate(processor, sample, hf_reference_model, tt_model, ds, device):
    generation_config = hf_reference_model.generation_config

    inputs = processor(ds[sample]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features

    batch_size = input_features.shape[0]

    tt_model_kwargs = generation_config.update()
    tt_model_kwargs["output_attentions"] = generation_config.output_attentions
    tt_model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    tt_model_kwargs["use_cache"] = generation_config.use_cache

    tt_input_features = torch2tt_tensor(input_features, device, tt_lib.tensor.Layout.ROW_MAJOR)
    # Prepare model args for tt model
    tt_model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
        tt_model, tt_input_features, tt_model_kwargs, "input_features"
    )
    tt_encoder_outputs_last_hidden_state = tt_model_kwargs["encoder_outputs"].last_hidden_state

    tt_input_ids = torch.tensor([[generation_config.decoder_start_token_id]])

    # Create Logits processor
    logits_processor = get_logits_processor(tt_input_ids, generation_config)
    # Pad decoder inputs to 32
    tt_input_ids = pad_input_32(tt_input_ids, hf_reference_model.config.pad_token_id).to(torch.long)

    decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)

    for i in range(32):
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

    tt_transcription = processor.batch_decode(tt_input_ids, skip_special_tokens=True)[0]

    dataset_text = ds[sample]["text"]

    return tt_transcription, dataset_text
