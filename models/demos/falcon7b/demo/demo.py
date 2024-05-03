# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import tt_lib
from loguru import logger
from models.demos.falcon7b.reference.hf_modeling_falcon import FalconConfig, FalconForCausalLM
from models.demos.falcon7b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon7b.tt.model_config import get_model_config, model_config_entries
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    is_wormhole_b0,
    nearest_32,
    profiler,
    torch2tt_tensor,
    tt2torch_tensor,
    tt_tensors_to_torch_tensors,
)
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.generation.utils import top_k_top_p_filtering

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


def preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len, perf_mode=False):
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

    if not perf_mode:
        prefill_ids = prefill_ids[:, : nearest_32(num_input_tokens)]  # only pad up to nearest 32, not max seq len
    else:
        num_input_tokens = max_seq_len - 1

    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    return prefill_ids, num_users, num_input_tokens


def initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, devices):
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    kv_cache = ()
    for _ in range(num_layers):
        kv_cache_cur_layer = []
        for device in devices:
            k_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
            v_cache = torch.zeros(batch_size, 1, max_seq_len, head_dim)
            tt_k_cache = torch2tt_tensor(k_cache, device)
            tt_v_cache = torch2tt_tensor(v_cache, device)
            kv_cache_cur_layer.append((tt_k_cache, tt_v_cache))
        kv_cache += (kv_cache_cur_layer,)
    return kv_cache


def print_output_prompts(generated_ids, tokenizer, batch_size, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        if user_id % batch_size == 0:
            logger.info(f"\n\n=============== Results for device {(user_id // batch_size) + 1} ===============\n")
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def update_model_config(model, model_config_str, prefill_seq_len=0):
    model.model_config.update(get_model_config(model_config_str, prefill_seq_len))


def top_pk_logits(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    next_token_logscores = top_k_top_p_filtering(logits, top_k=k, top_p=p)
    probs = F.softmax(next_token_logscores / temperature, dim=-1)
    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    if return_probs:
        return token, probs
    else:
        return token


def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


def synchronize_devices(devices):
    for device in devices:
        tt_lib.device.Synchronize(device)


def run_falcon_demo_kv(
    user_input,
    batch_size,
    max_seq_len,
    model_config_strs_prefill_decode,
    model_location_generator,
    get_tt_cache_path,
    devices,
    model_version="tiiuae/falcon-7b-instruct",
    num_layers=32,
    perf_mode=False,  # Option to measure perf using max seq length (with invalid outputs)
    greedy_sampling=False,  # Option to use greedy decoding instead of top-k/p
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    num_devices = len(devices)
    global_batch = batch_size * num_devices

    torch.manual_seed(0)

    for device in devices:
        device.enable_program_cache()

    if perf_mode:
        logger.info("Running in performance measurement mode (invalid outputs)!")

    configuration = FalconConfig(**model_config_entries)

    profiler.start(f"loading_inputs")
    if num_devices > 1:
        assert len(user_input) == global_batch, "Number of users must be equal to batch size * number of devices!"
    if len(user_input) == 1:
        input_prompts = user_input
    else:
        input_prompts = load_inputs(user_input, global_batch)

    profiler.end(f"loading_inputs")

    logger.info("Tokenizing inputs...")
    profiler.start(f"tokenizing_inputs")

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(
        input_prompts, tokenizer, max_seq_len, perf_mode
    )
    profiler.end(f"tokenizing_inputs")

    model_config = get_model_config(model_config_strs_prefill_decode[0], nearest_32(num_input_tokens))
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    # State dict is needed for embeddings
    logger.info("Loading weights...")
    profiler.start(f"loading_weights")
    if len(os.listdir(tt_cache_path)) < 337:
        logger.info("Weights not found on machine; downloading weights...")
        model_name = model_location_generator(model_version, model_subdir="Falcon")
        hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        hugging_face_reference_model.eval()
        state_dict = hugging_face_reference_model.state_dict()
    else:
        state_dict = None

    logger.info("Loading weights finished!")
    profiler.end(f"loading_weights")

    synchronize_devices(devices)

    logger.info("Moving weights (single layer) to device...")
    base_url = ""

    tt_FalconCausalLM_singlelayer = TtFalconCausalLM(
        devices,
        state_dict,
        base_url,
        1,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
        nearest_32(num_input_tokens),
    )  # single layer only used for compile
    logger.info("Moved weights (single layer) to device!")

    synchronize_devices(devices)

    logger.info("Initializing KV cache...")
    profiler.start(f"initializing_KV_cache")
    kv_cache_singlelayer = initialize_kv_cache(
        configuration, 1, batch_size, max_seq_len, devices
    )  # only used for compile
    kv_cache = initialize_kv_cache(configuration, num_layers, batch_size, max_seq_len, devices)
    profiler.end(f"initializing_KV_cache")
    profiler.disable()

    ### First prefill run with compile ###
    logger.info("Running 1st run prefill stage with compile...")
    use_cache = True
    time_prefill_compile = 0
    N = num_users // num_devices if not perf_mode else 1
    for user_id in tqdm(range(N)):
        time_prefill_compile_start = time.time()
        (
            tt_prefill_input_ids,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
            "prefill", prefill_ids[user_id::batch_size], 0, num_input_tokens=nearest_32(num_input_tokens)
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
            input_ids=tt_prefill_input_ids,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache_singlelayer,
            layer_past_len=0,
            use_cache=use_cache,
        )
        synchronize_devices(devices)

        for i in range(num_devices):
            tt_prefill_input_ids[i].deallocate()
            if tt_prefill_attention_mask is not None:
                if isinstance(tt_prefill_attention_mask[i], tt_lib.tensor.Tensor):
                    tt_prefill_attention_mask[i].deallocate()
                elif isinstance(tt_prefill_attention_mask[i], list):
                    for tt_attention_mask_element in tt_prefill_attention_mask[i]:
                        tt_attention_mask_element.deallocate()
                else:
                    raise ValueError("Invalid type for tt_attention_mask")
            tt_logits[i].deallocate()

        time_prefill_compile += time.time() - time_prefill_compile_start

    synchronize_devices(devices)
    logger.info("Finished 1st run prefill stage with compile!")

    ### First run decode stage with compile ###
    logger.info("Running 1st run decode stage with compile...")

    # Update model config
    update_model_config(tt_FalconCausalLM_singlelayer, model_config_strs_prefill_decode[1])

    decode_ids = torch.randint(low=0, high=configuration.vocab_size - 1, size=(global_batch, 1), dtype=torch.int64)

    time_decode_compile = 0
    for kv_cache_len in tqdm(range(num_input_tokens, max_seq_len, 32)):
        time_decode_compile_start = time.time()
        (
            tt_decode_input_ids,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
        )
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
            input_ids=tt_decode_input_ids,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache_singlelayer,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        synchronize_devices(devices)

        for i in range(num_devices):
            tt_decode_input_ids[i].deallocate()
            if tt_decode_attention_mask is not None:
                tt_decode_attention_mask[i].deallocate()
            tt_logits[i].deallocate()

        time_decode_compile += time.time() - time_decode_compile_start

    logger.info("Finished 1st run decode stage with compile!")
    synchronize_devices(devices)

    del tt_logits
    del tt_prefill_input_ids
    del tt_prefill_attention_mask
    del decode_ids
    del tt_decode_input_ids
    del tt_FalconCausalLM_singlelayer
    del kv_cache_singlelayer

    logger.info("Moving weights (all layers) to device; might take some time...")
    profiler.start(f"moving_to_device")
    tt_FalconCausalLM = TtFalconCausalLM(
        devices,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        get_model_config(model_config_strs_prefill_decode[0], nearest_32(num_input_tokens)),
        tt_cache_path,
        nearest_32(num_input_tokens),
    )
    logger.info("Moved weights (all layers) to device!")
    profiler.end(f"moving_to_device")

    ### Second prefill run without compile ###
    enable_persistent_kernel_cache()

    post_processor = partial(post_process)
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    logger.info("Running inference prefill stage...")
    time_prefill_inference = 0
    if not perf_mode:
        N = num_users // num_devices
        N_warmup = 0
    else:
        N = 15
        N_warmup = 5
    for i in tqdm(range(N)):
        user_id = i if not perf_mode else 0
        time_prefill_inference_start = time.time()
        (
            tt_prefill_input_ids,
            tt_prefill_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing(
            "prefill", prefill_ids[user_id::batch_size], 0, num_input_tokens=nearest_32(num_input_tokens)
        )
        assert tt_prefill_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_ids=tt_prefill_input_ids,
            llm_mode="prefill",
            attention_mask=tt_prefill_attention_mask,
            user_id=user_id,
            layer_past=kv_cache,
            layer_past_len=0,
            use_cache=use_cache,
        )
        synchronize_devices(devices)

        if tt_prefill_attention_mask is not None:
            for device_id in range(len(tt_prefill_attention_mask)):
                if isinstance(tt_prefill_attention_mask[device_id], tt_lib.tensor.Tensor):
                    tt_prefill_attention_mask[device_id].deallocate()
                elif isinstance(tt_prefill_attention_mask[device_id], list):
                    for tt_attention_mask_element in tt_prefill_attention_mask[device_id]:
                        tt_attention_mask_element.deallocate()
                else:
                    raise ValueError("Invalid type for tt_attention_mask")

        logits = torch.concat(
            [torch_logit.squeeze(1) for torch_logit in tt_tensors_to_torch_tensors(tt_logits)], dim=-2
        )

        for j in range(num_devices):
            tt_prefill_input_ids[j].deallocate()
            tt_logits[j].deallocate()

        user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
        output_ids[user_id::batch_size] = user_output_ids

        if i >= N_warmup:
            time_prefill_inference += time.time() - time_prefill_inference_start

    logger.info("Finished inference prefill stage!")
    num_users_generated_prefill = num_users if not perf_mode else (N - N_warmup) * num_devices

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    ### Inference run decode ###
    logger.info("Running inference decode stage...")

    # Update model config
    update_model_config(tt_FalconCausalLM, model_config_strs_prefill_decode[1])

    decode_ids = torch.zeros(global_batch, 1, dtype=torch.int64)
    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]

    time_decode_inference = 0
    if not perf_mode:
        N = max_seq_len - num_input_tokens
        N_warmup = 0
    else:
        N = 15
        N_warmup = 5
    for output_token_index in range(N):
        time_decode_inference_start = time.time()
        (
            tt_decode_input_ids,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_ids=tt_decode_input_ids,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        synchronize_devices(devices)

        logits = torch.concat(
            [torch_logit.squeeze(1) for torch_logit in tt_tensors_to_torch_tensors(tt_logits)], dim=-2
        )

        for i in range(num_devices):
            tt_decode_input_ids[i].deallocate()
            if tt_decode_attention_mask is not None:
                tt_decode_attention_mask[i].deallocate()
            tt_logits[i].deallocate()

        if greedy_sampling:
            decode_ids = post_processor(logits=logits, index=...).reshape(global_batch, 1)
        else:
            decode_ids = top_pk_logits_efficient(logits.reshape(global_batch, -1)).reshape(global_batch, 1)

        if output_token_index >= N_warmup:
            time_decode_inference += time.time() - time_decode_inference_start

        if not perf_mode:
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
            os.system("clear")
            print_output_prompts(generated_ids, tokenizer, batch_size)

    logger.info("Finished inference decode stage!")
    num_tokens_generated_decode = global_batch * (output_token_index - N_warmup + 1)
    logger.info(f"Total number of tokens generated in decode: {num_tokens_generated_decode}")

    if not perf_mode:
        print_output_prompts(generated_ids, tokenizer, batch_size)

    for device in devices:
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
        "inference_throughput_prefill": num_users_generated_prefill / time_prefill_inference,
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
        f"inference throughput decode (per user): {round(measurements['inference_throughput_decode']/global_batch, 5)} tok/s/user"
    )

    return generated_text, measurements
