# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
import tt_lib
import torch
from loguru import logger
from pathlib import Path
import torch.nn as nn


from models.experimental.mistral.tt.mistral_transformer import TtTransformer, TtModelArgs
from models.experimental.mistral.reference.tokenizer import Tokenizer

from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
    tt_to_torch_tensor,
)


# load from json, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["input"])
    return in_prompt


def preprocess_and_validate_inputs(input_prompts, tokenizer):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(input_prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded).to(input_tokens)

    num_users = input_tokens.shape[0]

    return input_tokens, num_users, min_prompt_len


def print_output_prompts(generated, tokenizer, max_tokens, input_prompts, min_prompt_len):
    encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]
    res = []
    if max_tokens > 0:
        generated = torch.cat(generated, 1)

        for i, x in enumerate(encoded_prompts):
            res.append(tokenizer.decode(x[:min_prompt_len] + generated[i].tolist()))
    for i, output_prompt in enumerate(res):
        logger.info(f"Output for input {i}:\n{output_prompt}")


def run_mistral_demo(user_input, model_version, batch_size, num_layers, max_tokens, model_location_generator, device):
    torch.manual_seed(0)

    tt_lib.program_cache.enable()
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")

    base_address = f""
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    profiler.start(f"loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input
    else:
        input_prompts = load_inputs(user_input, batch_size)

    profiler.end(f"loading_inputs")

    # State dict is needed for embeddings
    logger.info("Loading TT model weights")
    profiler.start(f"loading_weights")
    tt_lib.device.Synchronize(device)
    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/Mistral/"
    model_args.max_batch_size = len(input_prompts)

    base_address = f""
    tt_MistralModel = TtTransformer(
        args=model_args, device=device, base_address=base_address, tt_cache_path=tt_cache_path
    )

    tt_lib.device.Synchronize(device)
    logger.info("Loaded TT model weights")
    profiler.end(f"loading_weights")

    logger.info("Tokenizing inputs")
    profiler.start(f"tokenizing_inputs")

    tokenizer = Tokenizer(str(Path(mistral_path) / "tokenizer.model"))
    prefill_ids, num_users, min_prompt_len = preprocess_and_validate_inputs(input_prompts, tokenizer)

    profiler.end(f"tokenizing_inputs")
    profiler.disable()

    ### First prefill run with compile ###
    logger.info("Running 1st run prefill stage with compile...")
    profiler.start(f"first_run_prefill_stage_compile", force_enable=True)
    use_cache = True
    input_mask = prefill_ids != tokenizer.pad_id

    positions = torch.arange(0, min_prompt_len)

    logits = tt_MistralModel.forward(prefill_ids[:, :min_prompt_len], positions)

    logits = tt_to_torch_tensor(logits).squeeze(0)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    tt_lib.device.Synchronize(device)
    logger.info("Finished 1st run prefill stage with compile")
    profiler.end(f"first_run_prefill_stage_compile", force_enable=True)

    ### First run decode stage with compile ###
    logger.info("Running 1st run decode stage with compile...")
    profiler.start(f"first_run_decode_stage_compile", force_enable=True)

    generated = []
    all_logprobs = [
        logprobs[:, :-1, :].gather(2, prefill_ids[:, 1:min_prompt_len, None]).squeeze(-1),
    ]
    cur_pos = min_prompt_len
    for _ in range(max_tokens):
        next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
        if cur_pos < input_mask.shape[1]:
            next_token = torch.where(input_mask[:, cur_pos], prefill_ids[:, cur_pos], next_token)
        all_logprobs.append(
            logprobs[:, -1, :].gather(1, next_token[:, None]),
        )
        generated.append(next_token[:, None])
        logits = tt_MistralModel.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        logits = tt_to_torch_tensor(logits).squeeze(0)
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    tt_lib.device.Synchronize(device)
    logger.info("Finished 1st run decode stage with compile")
    profiler.end(f"first_run_decode_stage_compile", force_enable=True)

    del generated
    del logits
    del logprobs
    del all_logprobs
    del positions
    del input_mask
    del next_token

    ### Second prefill run without compile ###
    profiler.enable()
    enable_persistent_kernel_cache()

    logger.info("Running inference prefill stage...")
    profiler.start(f"second_run_prefill_stage", force_enable=True)

    use_cache = True
    input_mask = prefill_ids != tokenizer.pad_id

    positions = torch.arange(0, min_prompt_len)

    logits = tt_MistralModel.forward(prefill_ids[:, :min_prompt_len], positions)

    logits = tt_to_torch_tensor(logits).squeeze(0)
    logprobs = nn.functional.log_softmax(logits, dim=-1)

    logger.info("Finished inference prefill stage")
    profiler.end(f"second_run_prefill_stage", force_enable=True)
    profiler.disable()

    ### Inference run decode ###
    logger.info("Running inference decode stage...")
    profiler.start(f"second_run_decode_stage", force_enable=True)

    generated = []
    all_logprobs = [
        logprobs[:, :-1, :].gather(2, prefill_ids[:, 1:min_prompt_len, None]).squeeze(-1),
    ]
    cur_pos = min_prompt_len
    for _ in range(max_tokens):
        next_token = torch.argmax(logprobs[:, -1, :], dim=-1)
        if cur_pos < input_mask.shape[1]:
            next_token = torch.where(input_mask[:, cur_pos], prefill_ids[:, cur_pos], next_token)
        all_logprobs.append(
            logprobs[:, -1, :].gather(1, next_token[:, None]),
        )
        generated.append(next_token[:, None])
        logits = tt_MistralModel.forward(next_token[:, None], torch.LongTensor([cur_pos]).to(next_token))
        logits = tt_to_torch_tensor(logits).squeeze(0)
        logprobs = nn.functional.log_softmax(logits, dim=-1)
        cur_pos += 1

    logger.info("Finished inference decode stage")
    profiler.end(f"second_run_decode_stage", force_enable=True)

    generated_ids = generated.copy()

    print_output_prompts(generated, tokenizer, max_tokens, input_prompts, min_prompt_len)

    tt_lib.program_cache.disable_and_clear()

    generated_output = []
    if max_tokens > 0:
        generated_ids = torch.cat(generated_ids, 1)

        encoded_prompts = [tokenizer.encode(prompt) for prompt in input_prompts]
        for i, x in enumerate(encoded_prompts):
            generated_output.append(tokenizer.decode(x[:min_prompt_len] + generated_ids[i].tolist()))

    measurements = {
        "preprocessing": profiler.get("tokenizing_inputs"),
        "compile_prefill": profiler.get("first_run_prefill_stage_compile") - profiler.get("second_run_prefill_stage"),
        "compile_decode": profiler.get("first_run_decode_stage_compile") - profiler.get("second_run_decode_stage"),
        "compile_total": profiler.get("first_run_prefill_stage_compile")
        - profiler.get("second_run_prefill_stage")
        + profiler.get("first_run_decode_stage_compile")
        - profiler.get("second_run_decode_stage"),
        "inference_prefill": profiler.get("second_run_prefill_stage"),
        "inference_decode": profiler.get("second_run_decode_stage"),
        "inference_total": profiler.get("second_run_prefill_stage") + profiler.get("second_run_decode_stage"),
        "inference_prefill_throughput": (batch_size * min_prompt_len) / (profiler.get("second_run_prefill_stage")),
        "inference_decode_throughput": (batch_size * max_tokens) / (profiler.get("second_run_decode_stage")),
    }

    logger.info(f"pre processing duration: {measurements['preprocessing']} s")
    logger.info(f"prefill compile time: {measurements['compile_prefill']} s")
    logger.info(f"decode compile time: {measurements['compile_decode']} s")
    logger.info(f"total compile time: {measurements['compile_total']} s")
    logger.info(f"prefill inference time: {measurements['inference_prefill']} s")
    logger.info(f"decode inference time: {measurements['inference_decode']} s")
    logger.info(f"total inference time: {measurements['inference_total']} s")
    logger.info(f"inference prefill throughput: {measurements['inference_prefill_throughput']} inp/s")
    logger.info(f"inference decode throughput: {measurements['inference_decode_throughput']} inp/s")

    return generated_output, measurements


def test_demo(
    user_input,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"tt_metal/tools/profiler/logs/Mistral")

    return run_mistral_demo(
        user_input=user_input,
        model_version="mistral-7B-v0.1",
        batch_size=16,
        num_layers=32,
        max_tokens=5,
        model_location_generator=model_location_generator,
        device=device,
    )
