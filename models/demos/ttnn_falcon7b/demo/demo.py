# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import time
from functools import partial

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer, FalconConfig, FalconForCausalLM
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor
from models.demos.ttnn_falcon7b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path, model_config_entries
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    nearest_32,
    profiler,
)

END_OF_TEXT = 11
SPACE = 204


# load from jason, return as a list
def load_inputs(user_input, batch):
    if isinstance(user_input, str):
        with open(user_input, "r") as f:
            user_input = json.load(f)
    assert len(user_input) >= batch, f"Number of users (batch) must be {batch}!"
    in_prompt = []
    for i in range(batch):
        in_prompt.append(user_input[i]["question"])
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

    prefill_ids = prefill_ids[:, : nearest_32(num_input_tokens)]  # only pad up to nearest 32, not max seq len

    return prefill_ids, num_users, num_input_tokens


def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device, model_config):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        k_cache = torch.zeros(batch_size, max_seq_len, head_dim)
        v_cache = torch.zeros(batch_size, max_seq_len, head_dim)
        tt_k_cache = ttnn.from_torch(
            k_cache.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
        )
        tt_v_cache = ttnn.from_torch(
            v_cache.unsqueeze(1), device=device, layout=ttnn.TILE_LAYOUT, dtype=model_config["DEFAULT_DTYPE"]
        )
        kv_cache += ((tt_k_cache, tt_v_cache),)
    return kv_cache


def print_output_prompts(generated_ids, tokenizer, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def run_falcon_demo_kv(
    user_input, model_version, batch_size, num_layers, max_seq_len, model_config, model_location_generator, device
):
    torch.manual_seed(0)

    device.enable_program_cache()

    tt_cache_path = get_tt_cache_path(model_version)

    configuration = FalconConfig(**model_config_entries)

    profiler.start(f"loading_inputs")
    if len(user_input) == 1:
        input_prompts = user_input
    else:
        input_prompts = load_inputs(user_input, batch_size)

    profiler.end(f"loading_inputs")

    # State dict is needed for embeddings
    logger.info("Loading weights...")
    profiler.start(f"loading_weights")
    model_name = model_location_generator(model_version, model_subdir="Falcon")
    model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True).eval()

    logger.info("Loading weights finished!")
    profiler.end(f"loading_weights")

    ttnn.synchronize_device(device)

    logger.info("Moving weights to device; might take some time...")
    profiler.start(f"moving_to_device")

    def convert_to_ttnn(model, name):
        return not isinstance(model, torch.nn.Embedding)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=create_custom_preprocessor(model_config, tt_cache_path=tt_cache_path, device=device),
        convert_to_ttnn=convert_to_ttnn,
    )
    tt_FalconCausalLM_single_layer = TtFalconCausalLM(
        device,
        1,
        configuration,
        max_seq_len,
        model_config,
        parameters,
    )  # single layer only used for compile

    logger.info("Moved weights to device!")
    profiler.end(f"moving_to_device")

    ttnn.synchronize_device(device)

    logger.info("Tokenizing inputs...")
    profiler.start(f"tokenizing_inputs")

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len)

    profiler.end(f"tokenizing_inputs")

    logger.info("Initializing KV cache...")
    profiler.start(f"initializing_KV_cache")
    kv_cache_single_layer = initialize_kv_cache(
        configuration, 1, batch_size, max_seq_len, device, model_config
    )  # only used for compile
    kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, device, model_config)
    profiler.end(f"initializing_KV_cache")
    profiler.disable()

    ### First prefill run with compile ###
    logger.info("Running 1st run prefill stage with compile...")
    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    time_prefill_compile = 0
    for user_id in tqdm(range(num_users)):
        time_prefill_compile_start = time.time()
        (
            tt_prefill_embeddings,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM_single_layer.model_preprocessing(
            "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache_single_layer = tt_FalconCausalLM_single_layer(
            input_embeddings=tt_prefill_embeddings,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache_single_layer,
            layer_past_len=0,
            use_cache=use_cache,
        )
        ttnn.synchronize_device(device)
        time_prefill_compile_end = time.time()
        time_prefill_compile += time_prefill_compile_end - time_prefill_compile_start

        # ttnn.deallocate(tt_prefill_embeddings)
        if tt_prefill_attention_mask is not None:
            ttnn.deallocate(tt_prefill_attention_mask)

        logits = ttnn.to_torch(tt_logits).squeeze(1)
        ttnn.deallocate(tt_logits)

        user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        output_ids[user_id] = user_output_ids

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    ttnn.synchronize_device(device)
    logger.info("Finished 1st run prefill stage with compile!")

    ### First run decode stage with compile ###
    logger.info("Running 1st run decode stage with compile...")
    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)

    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]

    time_decode_compile = 0
    for output_token_index in tqdm(range(max_seq_len - num_input_tokens)):
        time_decode_compile_start = time.time()
        (
            tt_decode_embeddings,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM_single_layer.model_preprocessing(
            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
        )
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache_single_layer = tt_FalconCausalLM_single_layer(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache_single_layer,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        ttnn.synchronize_device(device)
        time_decode_compile_end = time.time()
        time_decode_compile += time_decode_compile_end - time_decode_compile_start

        # ttnn.deallocate(tt_decode_embeddings)
        if tt_decode_attention_mask is not None:
            ttnn.deallocate(tt_decode_attention_mask, force=False)

        logits = ttnn.to_torch(tt_logits).squeeze(1)
        ttnn.deallocate(tt_logits, force=False)

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

    logger.info("Finished 1st run decode stage with compile!")
    ttnn.synchronize_device(device)

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
    del tt_FalconCausalLM_single_layer
    del kv_cache_single_layer

    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        num_layers,
        configuration,
        max_seq_len,
        model_config,
        parameters,
    )

    ### Second prefill run without compile ###
    profiler.enable()
    enable_persistent_kernel_cache()

    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    logger.info("Running inference prefill stage...")
    time_prefill_inference = 0
    for user_id in tqdm(range(num_users)):
        time_prefill_inference_start = time.time()
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
        ttnn.synchronize_device(device)
        time_prefill_inference_end = time.time()
        time_prefill_inference += time_prefill_inference_end - time_prefill_inference_start

        # ttnn.deallocate(tt_prefill_embeddings)

        if tt_prefill_attention_mask is not None:
            ttnn.deallocate(tt_prefill_attention_mask, force=False)

        logits = ttnn.to_torch(tt_logits).squeeze(1)
        ttnn.deallocate(tt_logits, force=False)

        user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        output_ids[user_id] = user_output_ids

    logger.info("Finished inference prefill stage!")

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    profiler.disable()

    ### Inference run decode ###
    logger.info("Running inference decode stage...")

    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]

    time_decode_inference = 0
    for output_token_index in tqdm(range(max_seq_len - num_input_tokens)):
        time_decode_inference_start = time.time()
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
        ttnn.synchronize_device(device)
        time_decode_inference_end = time.time()
        time_decode_inference += time_decode_inference_end - time_decode_inference_start

        # ttnn.deallocate(tt_decode_embeddings)
        if tt_decode_attention_mask is not None:
            ttnn.deallocate(tt_decode_attention_mask, force=False)

        logits = ttnn.to_torch(tt_logits).squeeze(1)
        ttnn.deallocate(tt_logits, force=False)

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

        # TODO: Remove if we don't want to print per generated token
        # os.system("clear")
        # print_output_prompts(generated_ids, tokenizer)

    logger.info("Finished inference decode stage!")
    num_tokens_generated_decode = batch_size * (output_token_index + 1)
    logger.info(f"Total number of tokens generated in decode: {num_tokens_generated_decode}")

    print_output_prompts(generated_ids, tokenizer)

    device.disable_and_clear_program_cache()

    generated_text = tokenizer.batch_decode(generated_ids.tolist())

    measurements = {
        "preprocessing": profiler.get("tokenizing_inputs"),
        "loading_weights": profiler.get("loading_weights"),
        "moving_to_device": profiler.get("moving_to_device"),
        "initializing_KV_cache": profiler.get("initializing_KV_cache"),
        "compile_prefill": time_prefill_compile,
        "compile_decode": time_decode_compile,
        "compile_total": time_prefill_compile + time_decode_compile,
        "inference_prefill": time_prefill_inference,
        "inference_decode": time_decode_inference,
        "inference_total": time_prefill_inference + time_decode_inference,
        "inference_throughput_prefill": num_users / time_prefill_inference,
        "inference_throughput_decode": num_tokens_generated_decode / time_decode_inference,
    }

    logger.info(f"pre processing: {round(measurements['preprocessing'], 5)} s")
    logger.info(f"loading weights (+downloading if not on machine): {round(measurements['loading_weights'], 5)} s")
    logger.info(
        f"conversion to TT (if downloaded) and moving weights to device: {round(measurements['moving_to_device'], 5)} s"
    )
    logger.info(f"initializing KV cache: {round(measurements['initializing_KV_cache'], 5)} s")
    logger.info(f"prefill compile time (single layer): {round(measurements['compile_prefill'],5)} s")
    logger.info(f"decode compile time (single layer): {round(measurements['compile_decode'], 5)} s")
    logger.info(f"total compile time (single layer): {round(measurements['compile_total'], 5)} s")
    logger.info(f"prefill inference time: {round(measurements['inference_prefill'], 5)} s")
    logger.info(f"decode inference time: {round(measurements['inference_decode'], 5)} s")
    logger.info(f"total inference time: {round(measurements['inference_total'], 5)} s")
    logger.info(f"inference throughput prefill: {round(measurements['inference_throughput_prefill'], 5)} users/s")
    logger.info(
        f"inference throughput prefill | seq_len={prefill_ids.shape[1]} : {round(measurements['inference_throughput_prefill']*prefill_ids.shape[1], 5)} tok/s"
    )
    logger.info(f"inference throughput decode: {round(measurements['inference_throughput_decode'], 5)} tok/s")
    logger.info(
        f"inference throughput decode (per user): {round(measurements['inference_throughput_decode']/batch_size, 5)} tok/s/user"
    )

    return generated_text, measurements


def test_demo(
    user_input,
    model_location_generator,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()

    return run_falcon_demo_kv(
        user_input=user_input,
        model_version="tiiuae/falcon-7b-instruct",
        batch_size=32,
        num_layers=32,
        max_seq_len=128,
        # max_seq_len=1024,
        model_config=get_model_config("BFLOAT16-DRAM"),
        model_location_generator=model_location_generator,
        device=device,
    )
