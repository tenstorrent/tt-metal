# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from functools import partial
import tt_lib
import torch
from loguru import logger

from transformers import AutoTokenizer

from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM

from models.demos.falcon7b.reference.hf_modeling_falcon import FalconConfig
from models.demos.falcon7b.tt.model_config import get_model_config, get_tt_cache_path, model_config_entries
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
    torch2tt_tensor,
    tt2torch_tensor,
)

import time

END_OF_TEXT = 11
SPACE = 204


# load from jason, return as a list
def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Number of users (batch) must be {batch}!"
        in_prompt = []
        for i in range(batch):
            in_prompt.append(input_data[i]["question"])
        return in_prompt


def post_process(logits, index):
    next_token_logits = logits[:, index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens[:, None]
    return ids


def preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len):
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_inputs = tokenizer(
        input_prompts, padding="max_length", max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )
    prefill_ids = tokenized_inputs["input_ids"]

    tokenized_inputs_nopad = tokenizer(
        input_prompts, padding=False, max_length=max_seq_len, add_special_tokens=False, return_tensors="pt"
    )

    num_users = len(tokenized_inputs_nopad["input_ids"])
    num_input_tokens = len(tokenized_inputs_nopad["input_ids"][0])
    for input_prompt in tokenized_inputs_nopad["input_ids"]:
        assert len(input_prompt) == num_input_tokens
    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    return prefill_ids, num_users, num_input_tokens


def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        k_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        v_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
        tt_k_cache = torch2tt_tensor(k_cache, device)
        tt_v_cache = torch2tt_tensor(v_cache, device)
        kv_cache += ((tt_k_cache, tt_v_cache),)
    return kv_cache


def print_output_prompts(generated_ids, tokenizer, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def run_falcon_demo_kv(
    input_path, model_version, batch_size, num_layers, max_seq_len, model_config, model_location_generator, device
):
    torch.manual_seed(0)

    tt_lib.program_cache.enable()

    tt_cache_path = get_tt_cache_path(model_version)

    configuration = FalconConfig(**model_config_entries)

    profiler.start(f"loading_inputs")
    input_prompts = load_inputs(input_path, batch_size)
    profiler.end(f"loading_inputs")

    # State dict is needed for embeddings
    logger.info("Loading TT model weights")
    profiler.start(f"loading_weights")
    state_dict = {"transformer.word_embeddings.weight": torch.load(tt_cache_path / "embedding.pt")}
    tt_lib.device.Synchronize()

    base_url = ""
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
    )
    tt_lib.device.Synchronize()
    logger.info("Loaded TT model weights")
    profiler.end(f"loading_weights")

    logger.info("Tokenizing inputs")
    profiler.start(f"tokenizing_inputs")

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len)
    profiler.end(f"tokenizing_inputs")

    logger.info("Initializing KV cache")
    profiler.start(f"initializing_KV_cache")
    kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device)
    profiler.end(f"initializing_KV_cache")
    profiler.disable()

    ### First prefill run with compile ###
    logger.info("Running 1st run prefill stage with compile...")
    profiler.start(f"first_run_prefill_stage_compile", force_enable=True)
    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    for user_id in range(num_users):
        prefill_wc_start = time.time()

        (
            tt_prefill_embeddings,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing(
            "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_prefill_embeddings,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache,
            layer_past_len=0,
            use_cache=use_cache,
        )

        tt_prefill_embeddings.deallocate()
        if tt_prefill_attention_mask is not None:
            tt_prefill_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()

        user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        output_ids[user_id] = user_output_ids

        prefill_wc_end = time.time()

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    tt_lib.device.Synchronize()
    logger.info("Finished 1st run prefill stage with compile")
    profiler.end(f"first_run_prefill_stage_compile", force_enable=True)

    ### First run decode stage with compile ###
    logger.info("Running 1st run decode stage with compile...")
    profiler.start(f"first_run_decode_stage_compile", force_enable=True)
    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]
    for output_token_index in range(max_seq_len - num_input_tokens):
        decode_wc_start = time.time()

        (
            tt_decode_embeddings,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_decode_embeddings.deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()

        decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)

        for user_id, user_decode_id in enumerate(decode_ids[:num_users]):
            if user_decode_id == END_OF_TEXT:
                prompt_is_done[user_id] = True
            if prompt_is_done[user_id]:
                decode_ids[user_id] = SPACE

        if all(prompt_is_done):
            break

        generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)
        kv_cache_len += 1

        decode_wc_end = time.time()

    tt_lib.device.Synchronize()
    logger.info("Finished 1st run decode stage with compile")
    profiler.end(f"first_run_decode_stage_compile", force_enable=True)

    del user_output_ids
    del output_ids
    del logits
    del tt_logits
    del tt_prefill_embeddings
    del tt_prefill_attention_mask
    del generated_ids
    del decode_ids
    del user_decode_id
    del tt_decode_embeddings

    ### Second prefill run without compile ###
    profiler.enable()
    enable_persistent_kernel_cache()

    logger.info("Running inference prefill stage...")
    profiler.start(f"second_run_prefill_stage", force_enable=True)

    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    for user_id in range(num_users):
        prefill_start = time.time()

        (
            tt_prefill_embeddings,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing(
            "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_prefill_embeddings,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache,
            layer_past_len=0,
            use_cache=use_cache,
        )

        tt_prefill_embeddings.deallocate()
        if tt_prefill_attention_mask is not None:
            tt_prefill_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()

        user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        output_ids[user_id] = user_output_ids

        prefill_end = time.time()

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    logger.info("Finished inference prefill stage")
    profiler.end(f"second_run_prefill_stage", force_enable=True)
    profiler.disable()

    ### Inference run decode ###
    logger.info("Running inference decode stage...")
    profiler.start(f"second_run_decode_stage", force_enable=True)
    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]
    for output_token_index in range(max_seq_len - num_input_tokens):
        decode_start = time.time()

        (
            tt_decode_embeddings,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_decode_embeddings.deallocate()
        if tt_decode_attention_mask is not None:
            tt_decode_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()

        decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)

        for user_id, user_decode_id in enumerate(decode_ids[:num_users]):
            if user_decode_id == END_OF_TEXT:
                prompt_is_done[user_id] = True
            if prompt_is_done[user_id]:
                decode_ids[user_id] = SPACE

        if all(prompt_is_done):
            break

        generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)
        kv_cache_len += 1

        decode_end = time.time()

    logger.info("Finished inference decode stage")
    profiler.end(f"second_run_decode_stage", force_enable=True)

    print_output_prompts(generated_ids, tokenizer)

    tt_lib.program_cache.disable_and_clear()

    generated_text = tokenizer.batch_decode(generated_ids.tolist())

    measurements = {
        "preprocessing": profiler.get("tokenizing_inputs"),
        "initializing_KV_cache": profiler.get("initializing_KV_cache"),
        "compile_prefill": profiler.get("first_run_prefill_stage_compile") - profiler.get("second_run_prefill_stage"),
        "compile_decode": profiler.get("first_run_decode_stage_compile") - profiler.get("second_run_decode_stage"),
        "compile_total": profiler.get("first_run_prefill_stage_compile")
        - profiler.get("second_run_prefill_stage")
        + profiler.get("first_run_decode_stage_compile")
        - profiler.get("second_run_decode_stage"),
        "inference_prefill": profiler.get("second_run_prefill_stage"),
        "inference_decode": profiler.get("second_run_decode_stage"),
        "inference_total": profiler.get("second_run_prefill_stage") + profiler.get("second_run_decode_stage"),
        "inference_throughput": output_token_index
        / (profiler.get("second_run_prefill_stage") + profiler.get("second_run_decode_stage")),
    }

    logger.info(f"pre processing duration: {measurements['preprocessing']} s")
    logger.info(f"initializing KV cache duration: {measurements['initializing_KV_cache']} s")
    logger.info(f"prefill compile time: {measurements['compile_prefill']} s")
    logger.info(f"decode compile time: {measurements['compile_decode']} s")
    logger.info(f"total compile time: {measurements['compile_total']} s")
    logger.info(f"prefill inference time: {measurements['inference_prefill']} s")
    logger.info(f"decode inference time: {measurements['inference_decode']} s")
    logger.info(f"total inference time: {measurements['inference_total']} s")
    logger.info(f"inference throughput: {measurements['inference_throughput']} inp/s")

    return generated_text, measurements


def test_demo(
    input_path,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    tt_lib.profiler.set_profiler_location(f"falcon7b")

    return run_falcon_demo_kv(
        input_path=input_path,
        model_version="tiiuae/falcon-7b-instruct",
        batch_size=32,
        num_layers=32,
        max_seq_len=256,
        model_config=get_model_config("BFLOAT16-DRAM"),
        model_location_generator=model_location_generator,
        device=device,
    )
