# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pytest
from functools import partial
import tt_lib
import torch
from loguru import logger
import time
from pathlib import Path
from transformers import AutoTokenizer
import os

from models.demos.falcon40b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.falcon40b.reference.hf_modeling_falcon import FalconConfig, FalconForCausalLM
from models.demos.falcon40b.tt.falcon_common import PytorchFalconCausalLM
from models.demos.falcon40b.tt.model_config import get_model_config, model_config_entries
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
    torch2tt_tensor,
    tt2torch_tensor,
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

    return prefill_ids, num_users, num_input_tokens


# TODO: Review for falcon40b on multi-chip once we have prefill
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


# TODO: Remove once we have prefill on device
def initialize_and_fill_kv_cache(
    pytorch_FalconCausalLM, model_config, configuration, prefill_ids, num_layers, batch_size, max_seq_len, devices
):
    logger.info("Generating kv cache on host")

    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=prefill_ids, past_key_values=None, use_cache=True
    )

    head_dim = configuration.hidden_size // configuration.num_attention_heads
    q_heads_per_kv_heads = configuration.num_attention_heads // configuration.num_kv_heads
    num_users, kv_cache_len = prefill_ids.shape

    # TODO: Remove this debug code; uncomment to use dummy cache
    # pytorch_out = torch.rand(num_users, kv_cache_len, 65024)
    # single_layer_cache = (torch.rand(num_users, 128, kv_cache_len, 64), torch.rand(num_users, 128, kv_cache_len, 64))
    # pytorch_layer_present = (single_layer_cache,) * 60

    kv_cache = ()
    # tt_layer_past_host = () # TODO: Not needed?
    for i in range(num_layers):
        logger.info(f"Putting kv cache on devices for layer: {i+1}")
        k_cache_repeat_interleaved, v_cache_repeat_interleaved = pytorch_layer_present[i]
        k_cache = k_cache_repeat_interleaved[:, ::q_heads_per_kv_heads, ...]
        v_cache = v_cache_repeat_interleaved[:, ::q_heads_per_kv_heads, ...]

        tt_k_cache_host = torch.zeros(batch_size, configuration.num_kv_heads, max_seq_len, head_dim)
        tt_v_cache_host = torch.zeros(batch_size, configuration.num_kv_heads, max_seq_len, head_dim)
        tt_k_cache_host[:num_users, :, :kv_cache_len, :] = k_cache
        tt_v_cache_host[:num_users, :, :kv_cache_len, :] = v_cache
        tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
        tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)
        # tt_layer_past_host += ((tt_k_cache_host, tt_v_cache_host),) # TODO: Not needed?

        tt_k_cache = []
        tt_v_cache = []
        for j in range(len(devices)):
            tt_k_cache.append(
                torch2tt_tensor(
                    tt_k_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
            tt_v_cache.append(
                torch2tt_tensor(
                    tt_v_cache_host[j],
                    devices[j],
                    tt_lib.tensor.Layout.TILE,
                    model_config["KV_CACHE_MEMCFG"],
                    model_config["KV_CACHE_DTYPE"],
                )
            )
        kv_cache += ((tt_k_cache, tt_v_cache),)

    return pytorch_out, kv_cache


def print_output_prompts(generated_ids, tokenizer, num_users_to_display=None):
    output_prompts = tokenizer.batch_decode(generated_ids.tolist())
    for user_id, output_prompt in enumerate(output_prompts[:num_users_to_display]):
        logger.info(f"Output for user {user_id}:\n{output_prompt}")


def run_falcon_demo_kv(
    user_input,
    model_version,
    batch_size,
    num_layers,
    max_seq_len,
    model_config,
    model_location_generator,
    tt_cache_path,
    devices,
):
    torch.manual_seed(0)

    tt_lib.program_cache.enable()

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
    if (tt_cache_path == Path(f"models/demos/falcon40b/datasets/{model_version}")) and (
        len(os.listdir(f"models/demos/falcon40b/datasets/{model_version}")) < 260
    ):
        logger.info("Weights not found on machine; downloading weights...")
        model_name = model_location_generator(model_version, model_subdir="Falcon")
        hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        hugging_face_reference_model.eval()
        state_dict = hugging_face_reference_model.state_dict()
        torch.save(state_dict["transformer.word_embeddings.weight"], tt_cache_path / "embedding.pt")
    else:
        state_dict = None

    logger.info("Loading weights finished!")
    profiler.end(f"loading_weights")

    for device in devices:
        tt_lib.device.Synchronize(device)

    logger.info("Moving weights to devices; might take some time...")
    profiler.start(f"moving_to_device")

    base_url = ""
    use_global_cos_sin_cache = True
    tt_FalconCausalLM = TtFalconCausalLM(
        devices,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_seq_len,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache,
    )

    logger.info("Moved weights to devices!")
    profiler.end(f"moving_to_device")

    for device in devices:
        tt_lib.device.Synchronize(device)

    # TODO: Remove pytorch model once prefill is on device
    logger.info("Loading PyTorch model for prefill")
    model_name = model_location_generator(model_version, model_subdir="Falcon")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)

    logger.info("Tokenizing inputs...")
    profiler.start(f"tokenizing_inputs")

    tokenizer = AutoTokenizer.from_pretrained(model_version)
    prefill_ids, num_users, num_input_tokens = preprocess_and_validate_inputs(input_prompts, tokenizer, max_seq_len)

    profiler.end(f"tokenizing_inputs")

    logger.info("Initializing and filling KV cache on host...")
    profiler.start(f"initializing_KV_cache")
    pt_logits, kv_cache = initialize_and_fill_kv_cache(
        pytorch_FalconCausalLM,
        model_config,
        configuration,
        prefill_ids[:, :num_input_tokens],
        num_layers,
        batch_size,
        max_seq_len,
        devices,
    )
    profiler.end(f"initializing_KV_cache")
    profiler.disable()
    # TODO: Is this safe? Disabling kernel caching disable program caching as well?
    enable_persistent_kernel_cache()

    ### First prefill run with compile ###
    # TODO: Add prefill once it works on device
    logger.info("Running 1st run prefill stage with compile...")
    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    time_prefill_compile = 0
    for user_id in range(num_users):
        time_prefill_compile_start = time.time()
        # (
        #     tt_prefill_embeddings,
        #     tt_prefill_attention_mask,
        # ) = tt_FalconCausalLM.model_preprocessing(
        #     "prefill", prefill_ids[user_id : user_id + 1], 0, num_input_tokens=num_input_tokens
        # )
        # assert tt_prefill_attention_mask is not None

        # tt_logits, kv_cache = tt_FalconCausalLM(
        #     input_embeddings=tt_prefill_embeddings,
        #     llm_mode="prefill",
        #     attention_mask=tt_prefill_attention_mask,
        #     user_id=user_id,
        #     layer_past=kv_cache,
        #     layer_past_len=0,
        #     use_cache=use_cache,
        # )
        time_prefill_compile_end = time.time()
        time_prefill_compile += time_prefill_compile_end - time_prefill_compile_start

        # tt_prefill_embeddings.deallocate()
        # if tt_prefill_attention_mask is not None:
        #     tt_prefill_attention_mask.deallocate()

        # logits = tt2torch_tensor(tt_logits).squeeze(1)
        # tt_logits.deallocate()

        user_output_ids = post_processor(logits=pt_logits[user_id : user_id + 1, :, :], index=num_input_tokens - 1)
        output_ids[user_id] = user_output_ids

    # TODO: Should the concat be removed since output token for prefill shouldn't be used
    generated_ids = torch.concat((prefill_ids[..., :num_input_tokens], output_ids), dim=1)

    for device in devices:
        tt_lib.device.Synchronize(device)
    logger.info("Finished 1st run prefill stage with compile!")

    ### First run decode stage with compile ###
    attention_mask_memconfig = model_config["ATTN_MASK_MEMCFG"]
    if attention_mask_memconfig.is_sharded():
        attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
        attn_mask_shard_shape[-1] = max_seq_len
        attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

    logger.info("Running 1st run decode stage with compile...")
    decode_ids = torch.zeros(batch_size, 1, dtype=torch.int64)

    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    kv_cache_len = num_input_tokens  # This will increment by one after each decode
    prompt_is_done = [False for _ in range(num_users)]

    time_decode_compile = 0
    for output_token_index in range(max_seq_len - num_input_tokens):
        logger.info(f"Generating token: {output_token_index + num_input_tokens + 1}")
        time_decode_compile_start = time.time()
        (
            tt_decode_embeddings_host,
            tt_decode_attention_mask_host,
        ) = tt_FalconCausalLM.model_preprocessing("decode", decode_ids, kv_cache_len, num_input_tokens=kv_cache_len + 1)
        assert tt_decode_attention_mask_host is not None

        tt_decode_embeddings = [
            tt_decode_embeddings_host[i].to(devices[i], model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
            for i in range(len(devices))
        ]
        tt_decode_attention_mask = [
            tt_decode_attention_mask_host[i].to(devices[i], attention_mask_memconfig) for i in range(len(devices))
        ]

        tt_logits, kv_cache = tt_FalconCausalLM(
            input_embeddings=tt_decode_embeddings,
            llm_mode="decode",
            attention_mask=tt_decode_attention_mask,
            layer_past=kv_cache,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        time_decode_compile_end = time.time()
        time_decode_compile += time_decode_compile_end - time_decode_compile_start

        del tt_decode_embeddings
        if tt_decode_attention_mask is not None:
            del tt_decode_attention_mask

        tt_outs = []
        for i in range(len(devices)):
            tt_outs.append(tt2torch_tensor(tt_logits[0]).squeeze(1))
        logits = torch.concat(tt_outs, dim=-1)
        del tt_logits

        decode_ids = post_processor(logits=logits, index=...).reshape(batch_size, 1)

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

    logger.info("Finished 1st run decode stage with compile!")
    for device in devices:
        tt_lib.device.Synchronize(device)

    logger.info(f"Total number of tokens generated in decode: {batch_size*(kv_cache_len)}")

    print_output_prompts(generated_ids, tokenizer)

    tt_lib.program_cache.disable_and_clear()

    del user_output_ids
    del output_ids
    del logits
    del generated_ids
    del decode_ids
    del user_decode_id

    return

    # TODO: Add second run back with measurements?
    """
    ### Second prefill run without compile ###
    profiler.enable()
    enable_persistent_kernel_cache()

    post_processor = partial(post_process)
    use_cache = True
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    logger.info("Running inference prefill stage...")
    time_prefill_inference = 0
    for user_id in range(num_users):
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
        time_prefill_inference_end = time.time()
        time_prefill_inference += time_prefill_inference_end - time_prefill_inference_start

        tt_prefill_embeddings.deallocate()
        if tt_prefill_attention_mask is not None:
            tt_prefill_attention_mask.deallocate()

        logits = tt2torch_tensor(tt_logits).squeeze(1)
        tt_logits.deallocate()

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
    for output_token_index in range(max_seq_len - num_input_tokens):
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
        time_decode_inference_end = time.time()
        time_decode_inference += time_decode_inference_end - time_decode_inference_start

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

    logger.info("Finished inference decode stage!")
    logger.info(f"Total number of tokens generated in decode: {batch_size*(kv_cache_len)}")

    print_output_prompts(generated_ids, tokenizer)

    tt_lib.program_cache.disable_and_clear()

    generated_text = tokenizer.batch_decode(generated_ids.tolist())

    measurements = {
        "preprocessing": profiler.get("tokenizing_inputs"),
        "loading_weights": profiler.get("loading_weights"),
        "moving_to_device": profiler.get("moving_to_device"),
        "initializing_KV_cache": profiler.get("initializing_KV_cache"),
        "compile_prefill": time_prefill_compile - time_prefill_inference,
        "compile_decode": time_decode_compile - time_decode_inference,
        "compile_total": time_prefill_compile - time_prefill_inference + time_decode_compile - time_decode_inference,
        "inference_prefill": time_prefill_inference,
        "inference_decode": time_decode_inference,
        "inference_total": time_prefill_inference + time_decode_inference,
        "inference_throughput_prefill": num_users / time_prefill_inference,
        "inference_throughput_decode": batch_size / time_decode_inference,
    }

    logger.info(f"pre processing: {round(measurements['preprocessing'], 5)} s")
    logger.info(f"loading weights (+downloading if not on machine): {round(measurements['loading_weights'], 5)} s")
    logger.info(
        f"conversion to TT (if downloaded) and moving weights to device: {round(measurements['moving_to_device'], 5)} s"
    )
    logger.info(f"initializing KV cache: {round(measurements['initializing_KV_cache'], 5)} s")
    logger.info(f"prefill compile time: {round(measurements['compile_prefill'],5)} s")
    logger.info(f"decode compile time: {round(measurements['compile_decode'], 5)} s")
    logger.info(f"total compile time: {round(measurements['compile_total'], 5)} s")
    logger.info(f"prefill inference time: {round(measurements['inference_prefill'], 5)} s")
    logger.info(f"decode inference time: {round(measurements['inference_decode'], 5)} s")
    logger.info(f"total inference time: {round(measurements['inference_total'], 5)} s")
    logger.info(f"inference throughput prefill: {round(measurements['inference_throughput_prefill'], 5)} 1/s")
    logger.info(
        f"inference throughput prefill | seq_len={num_input_tokens}: {round(measurements['inference_throughput_prefill']*num_input_tokens, 5)} tok/sec"
    )
    logger.info(f"inference throughput decode: {round(measurements['inference_throughput_decode'], 5)} 1/s")
    logger.info(
        f"end-to-end throughput | seq_len={num_input_tokens}: {round((batch_size*(kv_cache_len)/measurements['inference_total'])/num_users, 5)} tok/sec/user"
    )

    return generated_text, measurements
    """


def test_demo(
    user_input,
    model_location_generator,
    get_tt_cache_path,
    pcie_devices,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    tt_lib.profiler.set_profiler_location(f"tt_metal/tools/profiler/logs/falcon40b")

    # TODO: Prefill and decode will likely require different configs for sharding
    # Currently, prefill is on host so only generate for decode
    llm_mode = "decode"
    num_devices = 4
    model_config = get_model_config("BFLOAT8_B-SHARDED", llm_mode, num_devices)
    model_version = model_config_entries["_name_or_path"]
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    return run_falcon_demo_kv(
        user_input=user_input,
        model_version=model_version,
        batch_size=32,
        num_layers=model_config_entries["num_hidden_layers"],
        max_seq_len=128,  # 1024,
        model_config=model_config,
        model_location_generator=model_location_generator,
        tt_cache_path=tt_cache_path,
        devices=pcie_devices[: model_config["NUM_DEVICES"]],
    )
