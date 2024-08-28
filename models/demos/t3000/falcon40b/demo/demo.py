# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from functools import partial
import ttnn
import torch
import torch.nn.functional as F
from loguru import logger
import time
from pathlib import Path
from transformers import AutoTokenizer
from transformers.generation.utils import top_k_top_p_filtering
import os
from tqdm import tqdm

from models.demos.t3000.falcon40b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import FalconConfig, FalconForCausalLM
from models.demos.t3000.falcon40b.tt.falcon_common import PytorchFalconCausalLM
from models.demos.t3000.falcon40b.tt.model_config import get_model_config, model_config_entries
from models.utility_functions import (
    disable_compilation_reports,
    enable_persistent_kernel_cache,
    profiler,
    torch2tt_tensor,
    tt_tensors_to_torch_tensors,
    nearest_32,
)

END_OF_TEXT = 11
SPACE = 204


# Used for debugging non-deterministic outputs of prefill stage
def save_kv_cache_to_file(device_mesh, kv_cache, kv_cache_path):
    # generate tensor of 60 layers and key and value tensors for each layer where there is 60 layers, key and value and tensor shape (32, 1, 128, 64)
    final_tensor = torch.zeros(60, 2, 32, 1, 128, 512)
    for layer in range(60):
        for type in range(len(kv_cache[layer])):
            # get key tensor from device
            tensor = ttnn.to_torch(
                kv_cache[layer][type], device=device_mesh, mesh_composer=ttnn.ConcatMeshToTensor(device_mesh, dim=-1)
            )
            # save tensor to file
            final_tensor[layer][type] = tensor

    torch.save(final_tensor, kv_cache_path)


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

    num_users = len(input_prompts)
    num_input_tokens = -1
    for prompt in input_prompts:
        tokenized_prompt = tokenizer(prompt, padding=False, add_special_tokens=False, return_tensors="pt")
        num_input_tokens = max(num_input_tokens, len(tokenized_prompt["input_ids"][0]))

    if not perf_mode:
        max_length = nearest_32(num_input_tokens)
    else:
        num_input_tokens = max_seq_len - 1
        max_length = max_seq_len

    tokenized_inputs = tokenizer(
        input_prompts,
        padding="max_length",
        max_length=max_length,
        add_special_tokens=False,
        return_tensors="pt",
    )
    prefill_ids = tokenized_inputs["input_ids"]

    logger.info(f"# of users: {num_users}")
    logger.info(f"# of input tokens per user: {num_input_tokens}")

    return prefill_ids, num_users, num_input_tokens


# TODO: Remove once we have prefill on device
def initialize_and_fill_kv_cache(
    pytorch_FalconCausalLM, model_config, configuration, prefill_ids, num_layers, batch_size, max_seq_len, device_mesh
):
    logger.info("Generating kv cache on host")

    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=prefill_ids, past_key_values=None, use_cache=True
    )

    head_dim = configuration.hidden_size // configuration.num_attention_heads
    q_heads_per_kv_heads = configuration.num_attention_heads // configuration.num_kv_heads
    num_users, kv_cache_len = prefill_ids.shape

    kv_cache = ()
    for i in range(num_layers):
        logger.info(f"Putting kv cache on devices for layer: {i+1}")
        k_cache_repeat_interleaved, v_cache_repeat_interleaved = pytorch_layer_present[i]
        k_cache = k_cache_repeat_interleaved[:, ::q_heads_per_kv_heads, ...]
        v_cache = v_cache_repeat_interleaved[:, ::q_heads_per_kv_heads, ...]

        tt_k_cache_host = torch.zeros(batch_size, configuration.num_kv_heads, max_seq_len, head_dim)
        tt_v_cache_host = torch.zeros(batch_size, configuration.num_kv_heads, max_seq_len, head_dim)
        tt_k_cache_host[:num_users, :, :kv_cache_len, :] = k_cache
        tt_v_cache_host[:num_users, :, :kv_cache_len, :] = v_cache

        tt_k_cache = ttnn.as_tensor(
            tensor=tt_k_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ttnn.ShardTensorToMesh(device_mesh, dim=1),
        )
        tt_v_cache = ttnn.as_tensor(
            tensor=tt_v_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ttnn.ShardTensorToMesh(device_mesh, dim=1),
        )
        kv_cache += ((tt_k_cache, tt_v_cache),)

    return pytorch_out, kv_cache


def print_output_prompts(generated_ids, tokenizer, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def synchronize_devices(devices):
    for device in devices:
        ttnn.device.synchronize_device(device)


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


def run_falcon_demo_kv(
    user_input,
    model_version,
    model_config_str_for_decode,
    model_config_str_for_prefill,
    batch_size,
    num_layers,
    max_seq_len,
    model_location_generator,
    get_tt_cache_path,
    device_mesh,
    prefill_on_host,
    perf_mode=False,
    greedy_sampling=False,
):
    torch.manual_seed(0)

    if perf_mode:
        logger.info("Running in performance measurement mode (invalid outputs)!")

    configuration = FalconConfig(**model_config_entries)
    devices = device_mesh.get_devices()

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
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()

    logger.info("Loading weights finished!")
    profiler.end(f"loading_weights")

    synchronize_devices(devices)

    logger.info("Tokenizing inputs...")
    profiler.start(f"tokenizing_inputs")
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(
        input_prompts, tokenizer, max_seq_len, perf_mode
    )
    profiler.end(f"tokenizing_inputs")

    # Update model_config for prefill
    model_config = get_model_config(model_config_str_for_prefill, "prefill", [1, prefill_ids.shape[1]], len(devices))

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    logger.info("Moving weights (single layer) to devices; might take some time...")
    base_url = ""
    use_global_cos_sin_cache = True
    tt_FalconCausalLM_singlelayer = TtFalconCausalLM(
        device_mesh,
        state_dict,
        base_url,
        1,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache,
    )  # single layer only used for compile
    logger.info("Moved weights (single layer) to devices!")

    synchronize_devices(devices)

    kv_cache_singlelayer = tt_FalconCausalLM_singlelayer.initialize_kv_cache()  # only used for compile

    enable_persistent_kernel_cache()

    ### First prefill run with compile ###
    use_cache = True
    time_prefill_compile = 0
    if not prefill_on_host:
        logger.info("Running 1st run prefill stage with compile...")
        N = num_users if not perf_mode else 1
        for user_id in tqdm(range(N), desc="Filling kv caches for each user"):
            time_prefill_compile_start = time.time()
            (
                tt_prefill_inputs,
                tt_prefill_attention_mask,
            ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
                "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
            )
            assert tt_prefill_attention_mask is not None

            tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
                input_ids=tt_prefill_inputs,
                llm_mode="prefill",
                attention_mask=tt_prefill_attention_mask,
                user_id=user_id,
                layer_past=kv_cache_singlelayer,
                layer_past_len=0,
                use_cache=use_cache,
            )
            synchronize_devices(devices)

            del tt_prefill_inputs
            del tt_prefill_attention_mask
            del tt_logits

            time_prefill_compile += time.time() - time_prefill_compile_start

    synchronize_devices(devices)
    logger.info("Finished 1st run prefill stage with compile!")

    ### First run decode stage with compile ###
    # Update model_config for decode
    model_config = get_model_config(model_config_str_for_decode, "decode", [batch_size, 1], len(devices))
    tt_FalconCausalLM_singlelayer.set_model_config(model_config)

    logger.info("Running 1st run decode stage with compile...")
    decode_ids = torch.randint(low=0, high=configuration.vocab_size - 1, size=(batch_size, 1), dtype=torch.int64)

    time_decode_compile = 0
    for kv_cache_len in tqdm(range(num_input_tokens, max_seq_len, 32)):
        time_decode_compile_start = time.time()
        (
            tt_decode_inputs,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM_singlelayer.model_preprocessing(
            "decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1
        )
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache_singlelayer = tt_FalconCausalLM_singlelayer(
            input_ids=tt_decode_inputs,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache_singlelayer,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        synchronize_devices(devices)

        del tt_decode_inputs
        if tt_decode_attention_mask is not None:
            del tt_decode_attention_mask
        del tt_logits

        time_decode_compile += time.time() - time_decode_compile_start

    logger.info("Finished 1st run decode stage with compile!")
    synchronize_devices(devices)

    del decode_ids
    del tt_FalconCausalLM_singlelayer
    del kv_cache_singlelayer

    # Update model_config for prefill
    model_config = get_model_config(model_config_str_for_prefill, "prefill", [1, prefill_ids.shape[1]], len(devices))

    logger.info("Moving weights (all layers) to device; might take some time...")
    profiler.start(f"moving_to_device")
    tt_FalconCausalLM = TtFalconCausalLM(
        device_mesh,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache,
    )
    logger.info("Moved weights (all layers) to device!")
    profiler.end(f"moving_to_device")

    profiler.start(f"initializing_KV_cache")
    kv_cache = tt_FalconCausalLM.initialize_kv_cache()  # Initialized kv cache for all layers
    profiler.end(f"initializing_KV_cache")

    ### Second prefill run without compile ###
    enable_persistent_kernel_cache()

    post_processor = partial(post_process)
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    logger.info("Running inference prefill stage...")
    time_prefill_inference = 0
    if prefill_on_host:
        pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
        logger.info("Initializing and filling KV cache")
        profiler.start(f"initializing_KV_cache_on_host")
        pt_logits, kv_cache = initialize_and_fill_kv_cache(
            pytorch_FalconCausalLM,
            model_config,
            configuration,
            prefill_ids[:, :num_input_tokens],
            num_layers,
            batch_size,
            max_seq_len,
            device_mesh,
        )
        profiler.end(f"initializing_KV_cache_on_host")

        output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
        for user_id in range(num_users):
            user_output_ids = post_processor(logits=pt_logits[user_id : user_id + 1, :, :], index=num_input_tokens - 1)
            output_ids[user_id] = user_output_ids
    else:
        if not perf_mode:
            N = num_users
            N_warmup = 0
        else:
            N = 15
            N_warmup = 5
        for i in tqdm(range(N), desc="Filling kv caches for each user"):
            user_id = i if not perf_mode else 0
            time_prefill_inference_start = time.time()
            (
                tt_prefill_inputs,
                tt_prefill_attention_mask,
            ) = tt_FalconCausalLM.model_preprocessing(
                "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
            )
            assert tt_prefill_attention_mask is not None

            tt_logits, kv_cache = tt_FalconCausalLM(
                input_ids=tt_prefill_inputs,
                llm_mode="prefill",
                attention_mask=tt_prefill_attention_mask,
                user_id=user_id,
                layer_past=kv_cache,
                layer_past_len=0,
                use_cache=use_cache,
            )
            synchronize_devices(devices)

            del tt_prefill_inputs
            del tt_prefill_attention_mask

            # untilize data first
            if tt_logits.get_layout() == ttnn.TILE_LAYOUT:
                tt_logits = ttnn.untilize(tt_logits, use_multicore=False)

            logits = ttnn.to_torch(
                tt_logits, device=device_mesh, mesh_composer=ttnn.ConcatMeshToTensor(device_mesh, dim=-1)
            ).squeeze(1)
            del tt_logits

            user_output_ids = post_processor(logits=logits, index=num_input_tokens - 1)
            output_ids[user_id] = user_output_ids

            if i >= N_warmup:
                time_prefill_inference += time.time() - time_prefill_inference_start

    logger.info("Finished inference prefill stage!")
    num_users_generated_prefill = num_users if not perf_mode else (N - N_warmup)

    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    ### Inference run decode ###
    logger.info("Running inference decode stage...")

    # Update model_config for decode
    model_config = get_model_config(model_config_str_for_decode, "decode", [batch_size, 1], len(devices))
    tt_FalconCausalLM.set_model_config(model_config)

    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)
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
            tt_decode_inputs,
            tt_decode_attention_mask,
        ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
        assert tt_decode_attention_mask is not None

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_ids=tt_decode_inputs,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        synchronize_devices(devices)

        del tt_decode_inputs
        if tt_decode_attention_mask is not None:
            del tt_decode_attention_mask

        if tt_logits.get_layout() == ttnn.TILE_LAYOUT:
            if tt_logits.memory_config().is_sharded():
                tt_logits = ttnn.sharded_to_interleaved(tt_logits)

            tt_logits = ttnn.untilize(tt_logits, use_multicore=False)

        logits = ttnn.to_torch(
            tt_logits, device=device_mesh, mesh_composer=ttnn.ConcatMeshToTensor(device_mesh, dim=-1)
        ).squeeze(1)

        del tt_logits

        if greedy_sampling:
            decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)
        else:
            decode_ids = top_pk_logits_efficient(logits.reshape(batch_size, -1)).reshape(batch_size, 1)

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
            # TODO: Remove if we don't want to print per generated token
            print_output_prompts(generated_ids, tokenizer)
            kv_cache_len += 1

    logger.info("Finished inference decode stage!")
    num_tokens_generated_decode = batch_size * (output_token_index - N_warmup + 1)
    logger.info(f"Total number of tokens generated in decode: {num_tokens_generated_decode}")

    if not perf_mode:
        print_output_prompts(generated_ids, tokenizer)

    for device in devices:
        device.disable_and_clear_program_cache()

    generated_text = tokenizer.batch_decode(generated_ids.tolist())

    measurements = {
        "preprocessing": profiler.get("tokenizing_inputs"),
        "loading_weights": profiler.get("loading_weights"),
        "moving_to_device": profiler.get("moving_to_device"),
        "initializing_KV_cache": profiler.get("initializing_KV_cache"),
        "compile_prefill": time_prefill_compile if not prefill_on_host else None,
        "compile_decode": time_decode_compile,
        "compile_total": time_prefill_compile + time_decode_compile,
        "inference_prefill": time_prefill_inference,
        "inference_decode": time_decode_inference,
        "inference_total": time_prefill_inference + time_decode_inference,
        "inference_throughput_prefill": (
            num_users_generated_prefill / time_prefill_inference if not prefill_on_host else None
        ),
        "inference_throughput_decode": num_tokens_generated_decode / time_decode_inference,
    }

    logger.info(f"pre processing: {round(measurements['preprocessing'], 5)} s")
    logger.info(f"loading weights (+downloading if not on machine): {round(measurements['loading_weights'], 5)} s")
    logger.info(
        f"conversion to TT (if downloaded) and moving weights to device: {round(measurements['moving_to_device'], 5)} s"
    )
    logger.info(f"initializing KV cache: {round(measurements['initializing_KV_cache'], 5)} s")
    if not prefill_on_host:
        logger.info(f"prefill compile time: {round(measurements['compile_prefill'],5)} s")
    logger.info(f"decode compile time: {round(measurements['compile_decode'], 5)} s")
    logger.info(f"total compile time: {round(measurements['compile_total'], 5)} s")
    logger.info(f"prefill inference time: {round(measurements['inference_prefill'], 5)} s")
    logger.info(f"decode inference time: {round(measurements['inference_decode'], 5)} s")
    logger.info(f"total inference time: {round(measurements['inference_total'], 5)} s")
    if not prefill_on_host:
        logger.info(f"inference throughput prefill: {round(measurements['inference_throughput_prefill'], 5)} users/s")
        logger.info(
            f"inference throughput prefill | seq_len={prefill_ids.shape[1]}: {round(measurements['inference_throughput_prefill']*prefill_ids.shape[1], 5)} tok/s"
        )
    logger.info(f"inference throughput decode: {round(measurements['inference_throughput_decode'], 5)} tok/s")
    logger.info(
        f"inference throughput decode (per user): {round(measurements['inference_throughput_decode']/batch_size, 5)} tok/s/user"
    )

    return generated_text, measurements


@pytest.mark.parametrize("perf_mode", (False,))  # Option to measure perf using max seq length (with invalid outputs)
@pytest.mark.parametrize("greedy_sampling", (False,))
@pytest.mark.parametrize("max_seq_len", (128,))
def test_demo(
    perf_mode,
    greedy_sampling,
    max_seq_len,
    user_input,
    model_location_generator,
    get_tt_cache_path,
    t3k_device_mesh,
    use_program_cache,
):
    # disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_falcon_demo_kv(
        user_input=user_input,
        model_version=model_config_entries["_name_or_path"],
        model_config_str_for_decode="BFLOAT8_B-SHARDED",  # Decode model config
        model_config_str_for_prefill="BFLOAT8_B-DRAM",  # Prefill model config
        batch_size=32,
        num_layers=model_config_entries["num_hidden_layers"],
        max_seq_len=max_seq_len,
        model_location_generator=model_location_generator,
        get_tt_cache_path=get_tt_cache_path,
        device_mesh=t3k_device_mesh,
        prefill_on_host=False,
        perf_mode=perf_mode,
        greedy_sampling=greedy_sampling,
    )
