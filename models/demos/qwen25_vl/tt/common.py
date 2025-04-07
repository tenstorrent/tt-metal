# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger
import math


def merge_vision_tokens(
    input_ids,
    input_embeds,
    image_embeds,
    hf_config,
):
    """
    input_ids are the input ids of the text tokens
    input_embeds are torch embedded text tokens
    image_embeds are torch embedded vision tokens
    """
    n_image_tokens = (input_ids == hf_config.image_token_id).sum().item()
    n_image_features = image_embeds.shape[0]
    if n_image_tokens != n_image_features:
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        )

    mask = input_ids == hf_config.image_token_id
    mask_unsqueezed = mask.unsqueeze(-1)
    mask_expanded = mask_unsqueezed.expand_as(input_embeds)
    image_mask = mask_expanded.to(input_embeds.device)

    input_embeds = input_embeds.masked_scatter(image_mask, image_embeds)
    return input_embeds


def preprocess_inputs_prefill(
    encoded_prompts,
    model_args,
    max_generated_tokens,
    max_prefill_len=None,
):
    """
    Run tokenizer on inputs, and create embeddings for the first token of each input
    """
    # To avoid going out of memory, clip the max prefill length by the maximum number of tokens that will be generated
    if max_prefill_len is None:
        max_prefill_len = model_args.max_seq_len
    max_prefill_len -= max_generated_tokens

    # Print the length of encoded prompts
    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    max_prompt_len = max(len(x) for x in encoded_prompts)
    assert (
        max_prompt_len <= max_prefill_len
    ), f"Max prompt length {max_prompt_len} exceeds max prefill len {max_prefill_len} and clipping and retokenizing is not supported for Qwen2.5 VL"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    # Always prefill the nearest power of 2 for each user. This means that the majority of cases we will prefill more tokens than needed.
    # To avoid issues, we keep track of the decoding position to decode correctly the user's prompt
    for encoded in encoded_prompts:
        # Prefill size is nearest power of 2
        prefill_seq_len = max(2 ** math.ceil(math.log(len(encoded), 2)), 128)

        # Initial prefill tensors full of pad tokens
        input_tokens_prefill_i = torch.full((1, prefill_seq_len), 0, dtype=torch.int32)
        input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
        input_tokens_prefill.append(input_tokens_prefill_i)

        # Keep the correct decoding position of each user
        decoding_pos.append(len(encoded))
        prefill_lens.append(prefill_seq_len)

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )


def generate_position_ids(
    model,
    input_ids,
    inputs_embeds,
    position_ids,
    cache_position,
    image_grid_thw,
    video_grid_thw,
    second_per_grid_ts,
    attention_mask,
):
    assert False, "Not implemented"
    assert position_ids is None and (attention_mask is None or attention_mask.ndim == 2)

    # calculate RoPE index once per generation in the pre-fill stage only
    if (
        (cache_position is not None and cache_position[0] == 0)
        or model.rope_deltas is None
        or (past_key_values is None or past_key_values.get_seq_length() == 0)
    ):
        position_ids, rope_deltas = model.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            attention_mask,
        )
        model.rope_deltas = rope_deltas
    # then use the prev pre-calculated rope-deltas to get the correct position ids
    else:
        batch_size, seq_length, _ = inputs_embeds.shape
        delta = (cache_position[0] + model.rope_deltas).to(inputs_embeds.device) if cache_position is not None else 0
        position_ids = torch.arange(seq_length, device=inputs_embeds.device)
        position_ids = position_ids.view(1, -1).expand(batch_size, -1)
        if cache_position is not None:  # otherwise `deltas` is an int `0`
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
        position_ids = position_ids.add(delta)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
