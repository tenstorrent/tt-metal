# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
from models.generation_utils import pad_input_32, get_logits_processor


def run_generate(
    input_sentance,
    tokenizer,
    hf_reference_model,
    tt_model,
    device,
    run_tt_model=True,
    log=True,
):
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
