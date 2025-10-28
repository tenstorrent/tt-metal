# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import create_tt_model, preprocess_inputs_prefill, sample_host
from models.tt_transformers.tt.generator import Generator, SamplingParams
from models.tt_transformers.tt.model_config import DecodersPrecision


def demo_text(
    instruct=True,
    max_seq_len=1024,
    batch_size=1,
    max_generated_tokens=200,
    sampling_params={"temperature": 0, "top_p": 0.08},
    stop_at_eos=True,
    enable_trace=True,
):
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    mesh_device = ttnn.open_device(device_id=0, trace_region_size=30000000)

    """
    Run the TT Transformers demo. Always prompts user for text input.
    """
    # # Always get input from user
    # print("=== TT Transformers Demo ===")
    # print("Enter your prompts (press Enter twice when done):")
    # input_prompts = []
    # while True:
    #     prompt = input("Prompt: ").strip()
    #     if not prompt:
    #         break
    #     input_prompts.append(prompt)

    # if not input_prompts:
    #     print("No prompts entered, using default.")
    #     input_prompts = ["What is the capital of France?"]

    # print(f"\nUsing {len(input_prompts)} prompts:")
    # for i, prompt in enumerate(input_prompts, 1):
    #     print(f"  {i}. {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
    # print()
    input_prompts = ["What is the capital of France?"]

    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    logger.info("Starting inference...")

    state_dict = None

    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=None,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
    )

    # No page table needed for non-paged attention
    page_table = None
    # Host code, safe to reuse tokenizer from the 1st model
    tokenizer = model_args.tokenizer
    processor = model_args.processor

    generator = Generator([model], [model_args], mesh_device, processor=processor, tokenizer=tokenizer)

    # Preprocess initial prompt inputs
    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        input_prompts, tokenizer, [model_args], instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )

    max_encoded_prompt_len = max(len(p) for p in encoded_prompts)
    assert (
        max_generated_tokens + max_encoded_prompt_len <= max_seq_len
    ), f"Prompt prefill tokens ({max_encoded_prompt_len}) + maximum number of decoded iterations ({max_generated_tokens}) needs to be <= than max_seq_len ({max_seq_len})"

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logger.info("Starting prefill warmup...")

    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,  # Prefill warmup for all users, in case some users have different seqlens than others
        page_table=page_table,
        kv_cache=[tt_kv_cache],
        prompt_lens=decoding_pos,
    )
    logger.info("Finished prefill warmup")

    logger.info(f"Starting prefill...")
    logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=[tt_kv_cache],
        prompt_lens=decoding_pos,
    )
    prefilled_token = torch.argmax(logits, dim=-1)
    logger.info(f"Prefill finished")

    # Keep track of generated outputs to print out every iteration
    all_outputs = [encoded_prompts[b][: prefill_lens[b]] for b in range(batch_size)]
    for user in range(batch_size):
        user_tok = int(prefilled_token[user].item())
        all_outputs[user].append(user_tok)

    user_done = [False] * batch_size  # Keeps track when a user reaches EoD token

    # Currently only supporting greedy decoding (temperature=0) on device
    argmax_on_device = sampling_params["temperature"] == 0
    if argmax_on_device:
        device_sampling_params = SamplingParams(temperature=0.0, top_k=-1, top_p=1.0)
    else:
        device_sampling_params = None

    # Initial positions
    current_pos = torch.tensor([decoding_pos[b] for b in range(batch_size)])

    # Start decoding
    iteration = 0
    users_decoding = True
    out_tok = prefilled_token

    logger.info(f"Starting decode loop...")

    # Start decoding
    while users_decoding:
        # Run decode forward
        logits = generator.decode_forward_text(
            out_tok,
            current_pos,
            enable_trace=enable_trace,
            page_table=page_table,
            kv_cache=[tt_kv_cache],
            sampling_params=device_sampling_params,
        )

        # Get the next token
        if device_sampling_params is not None:
            out_tok = logits.unsqueeze(1)
        else:
            # TODO Fix use case with temperature > 0
            _, out_tok = sample_host(
                logits,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                on_host=True,
            )

        # Print iteration info
        logger.debug(f"Iteration {iteration}: Generated token for {batch_size} users")

        current_pos += 1
        # Save output token to print out later
        for user in range(batch_size):
            user_tok = out_tok[user].item()
            if (
                user_tok not in tokenizer.stop_tokens and user_done[user] == False
            ):  # Read until an eos token (e.g. <|eot_id|>); create_tokenizer adds stop_tokens to HF tokenizers
                all_outputs[user].append(user_tok)
            else:
                if (
                    stop_at_eos
                ):  # For performance gathering in CI, we want to sometimes force decoding for a fixed number of iterations
                    user_done[user] = True
                    logger.trace(f"[User {user}] Finished decoding at iteration {iteration}")
                    if all(user_done):
                        users_decoding = False

        # Print out generated outputs for each user at the end of every iteration
        for user in range(batch_size):
            text = "".join(tokenizer.decode(all_outputs[user]))
            if len(text) > 100:
                text = "..." + text[-97:]
            text = text.replace("\n", " ")
            logger.debug("[User {}] {}".format(user, text))

        iteration += 1

        # Upper limit of generated tokens for each user
        if iteration >= max_generated_tokens:
            users_decoding = False

    # Final print
    if not users_decoding:
        logger.info("Finished decoding, printing the final outputs...\n")
        for i, (output, prompt) in enumerate(zip(all_outputs, input_prompts)):
            text = tokenizer.decode(output)
            prompt_including_assistant_tags = tokenizer.decode(model_args.encode_prompt(prompt, instruct=instruct))
            text_after_prompt = text.replace(prompt_including_assistant_tags, "", 1)
            logger.info(f"Output: {text_after_prompt}")

    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    demo_text()
