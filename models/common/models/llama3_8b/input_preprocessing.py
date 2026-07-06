# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Input preprocessing helpers for the TTTv2 Llama-3.1-8B demo."""

import torch
from loguru import logger


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    instruct,
    max_generated_tokens,
    max_prefill_len=128 * 1024,
):
    """Tokenize prompts and build padded prefill tensors for each user."""
    for m_args in model_args:
        assert (
            max_prefill_len <= m_args.max_context_len
        ), f"max_prefill_len {max_prefill_len} cannot exceed max_context_len {m_args.max_context_len}"

    max_prefill_len -= max_generated_tokens
    assert (
        max_prefill_len > 0
    ), f"max_prefill_len ({max_prefill_len + max_generated_tokens}) must be greater than max_generated_tokens ({max_generated_tokens})"

    encoded_prompts = [
        model_args[idx % len(model_args)].encode_prompt(prompt, instruct=instruct)
        for idx, prompt in enumerate(input_prompts)
    ]

    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    if min_prompt_len > max_prefill_len:
        logger.info(f"Left-clipping prompts to {max_prefill_len}")
        if instruct:
            raw_prompts = [
                model_args[idx % len(model_args)].encode_prompt(prompt, instruct=False)
                for idx, prompt in enumerate(input_prompts)
            ]
            overhead = [len(e) - len(r) for e, r in zip(encoded_prompts, raw_prompts)]

            shortened = []
            for idx, (e, o) in enumerate(zip(raw_prompts, overhead)):
                if isinstance(tokenizer, list):
                    shortened_prompt = tokenizer[idx % len(model_args)].decode(e[-(max_prefill_len - o) :])
                else:
                    shortened_prompt = tokenizer.decode(e[-(max_prefill_len - o) :])
                shortened.append(shortened_prompt)

            encoded_prompts = [
                model_args[idx % len(model_args)].encode_prompt(prompt, instruct=instruct)
                for idx, prompt in enumerate(shortened)
            ]
            assert all(
                len(e) == max_prefill_len for e in encoded_prompts
            ), f"Clipped prompts are not of the correct length, expected {max_prefill_len} but got {[len(e) for e in encoded_prompts]}"
        else:
            encoded_prompts = [encoded[-max_prefill_len:] for encoded in encoded_prompts]

        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

    for m_args in model_args:
        assert (
            max_prompt_len <= m_args.max_seq_len
        ), f"Max prompt length {max_prompt_len} exceeds model max seq len {m_args.max_seq_len}"
    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len, f"Minimum prompt length {min_prompt_len} exceeds max len {max_prompt_len}"

    logger.info(f"# of users: {len(encoded_prompts)}")
    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    for encoded in encoded_prompts:
        input_tokens_prefill_i = torch.full((1, max_prompt_len), 0, dtype=torch.int32)
        input_tokens_prefill_i[0, : len(encoded[:])] = torch.tensor(encoded[:]).to(input_tokens_prefill_i)
        input_tokens_prefill.append(input_tokens_prefill_i)
        decoding_pos.append(len(encoded))
        prefill_lens.append(max_prompt_len)

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )
