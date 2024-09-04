# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from functools import partial
from loguru import logger
from pathlib import Path
import torch
import ttnn

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)
from models.demos.t3000.llama2_70b.tt.llama_common import get_llama_path, MAX_SEQ_LEN, BASE_URL, load_llama_state_dict
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    nearest_32,
    skip_for_grayskull,
    get_devices_for_t3000,
)
from models.perf.perf_utils import prep_perf_report
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from tqdm import tqdm


def load_prompts_file(tokenizer, prefill_length, generation_length=128, gap=64):
    with open("models/demos/t3000/llama2_70b/demo/data/a_tale_of_two_cities.txt", encoding="utf-8-sig") as f:
        tokenized = tokenizer.encode(f.read(), bos=True, eos=False)

    token_windows = []
    ground_truth_texts = []
    for i in range(0, len(tokenized) - prefill_length + 1, prefill_length + gap):
        token_windows.append(tokenized[i : i + prefill_length])
        ground_truth_text = tokenizer.decode(tokenized[i : i + generation_length + 1])
        ground_truth_texts.append(ground_truth_text)
        if len(token_windows) == 32:
            return token_windows, ground_truth_texts

    return token_windows, ground_truth_texts


def post_process(logits, index):
    next_token_logits = logits[:, index, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    ids = next_tokens.reshape(-1)
    return ids


def intialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    # pad the model to maximum length
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")

    input_text_mask = tokens != pad_id  # use prefill token if that token is not masked
    return tokens, input_text_mask


def print_output_prompts(
    generated_ids,
    tokenizer,
    num_users_to_display=6,
    output_file="models/demos/t3000/llama2_70b/demo/data/output_prompts.txt",
):
    output_prompts = tokenizer.decode(generated_ids.tolist())

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for user_id, output_prompt in enumerate(output_prompts):
            if user_id < num_users_to_display:
                logger.info(f"Output for user {user_id}:\n{output_prompt}")
            f.write(f"Output for user {user_id}:\n{output_prompt}\n")


def run_test_LlamaModel_end_to_end(
    mesh_device,
    batch,
    seq_len,
    model_config,
    n_layers,
    n_devices,
    prefill_length,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    emulated,
    num_users,
):
    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(mesh_device, model_config, n_devices, emulated)
    logger.info(f"Running num_layer: {n_layers}")

    generator = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=num_users,
        n_layers=1,
        skip_model_load=False,
    )
    hugging_face_reference_model, tokenizer = generator.model, generator.tokenizer
    hugging_face_reference_model.eval()
    state_dict = load_llama_state_dict(ckpt_dir, n_layers=n_layers)
    configuration = hugging_face_reference_model.params

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params

    prefill_ids, prefill_texts = load_prompts_file(tokenizer, prefill_length)
    prefill_ids, _ = intialize_inputs(tokenizer, prefill_ids, num_users, prefill_length)

    # Clear global profiler state before starting measurements
    profiler.clear()

    # Set up model -----------------------------------------------------------------------
    logger.info("Moving weights to devices; might take some time...")
    profiler.start("TT_llama_model_setup")
    tt_model = TtLlamaModel_optimized(
        mesh_device,
        state_dict,
        BASE_URL,
        n_layers,
        model_config,
        configuration,
        batch,
        emulated=emulated,
        cache_path=cache_path,
        read_cache=False,
    )

    for i in mesh_device.get_device_ids():
        device = mesh_device.get_device(i)
        ttnn.synchronize_device(device)

    del state_dict

    logger.info("Running 1st run prefill stage with compile...")
    output_ids = torch.zeros(num_users, 1, dtype=torch.int64)
    post_processor = partial(post_process)

    for user_id in tqdm(range(num_users), desc="Prefill to 2k upto 32 users", colour="blue"):
        logger.info(f"Filling kv cache for user {user_id + 1}")
        if user_id == 0 or user_id == 25:
            profiler.start(f"processing_of_prefill_input_{user_id}")

        tt_inp_emb, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(
            prefill_ids[user_id : user_id + 1], start_pos=0
        )
        if user_id == 0 or user_id == 25:
            profiler.end(f"processing_of_prefill_input_{user_id}")
            profiler.start(f"model_run_for_prefill_{user_id}")

        tt_logits = tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            user_id=user_id,
        )
        if user_id == 0 or user_id == 25:
            profiler.end(f"model_run_for_prefill_{user_id}")

        del tt_inp_emb
        del rot_mat
        del attn_mask

        logits = torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_logits], -1)
        logits = logits[..., : configuration.vocab_size].float()
        del tt_logits

        user_output_ids = post_processor(logits=logits, index=prefill_length - 1)
        output_ids[user_id] = user_output_ids

    generated_ids = torch.concat((prefill_ids[..., :prefill_length], output_ids), dim=1)
    print_output_prompts(generated_ids, tokenizer)

    for device in devices:
        ttnn.synchronize_device(device)
    logger.info("Finished 1st run prefill stage with compile!")

    batch, seq_len = 32, 1
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)
    tt_model.set_model_config(model_config)

    logger.info("Running 1st run decode stage with compile...")
    decode_ids = torch.zeros(batch, 1, dtype=torch.int64)

    for user_id, output_id in enumerate(output_ids):
        decode_ids[user_id] = output_id

    for cur_pos in range(generation_length):
        start_pos = prefill_length + cur_pos
        logger.info(f"Generating token: {start_pos + 1}")

        if cur_pos == 0 or cur_pos == 35:  # Skip the first few iterations to warm up
            profiler.start(f"processing_of_decode_input_{cur_pos}")

        tt_inp_emb, start_pos, rot_mat, attn_mask, cache_idxs = tt_model.prepare_inputs(decode_ids, start_pos)

        tt_inp_emb = ttnn.to_device(tt_inp_emb, mesh_device, memory_config=tt_model.model_config["DRAM_MEMCFG"])
        tt_inp_emb = tt_model.tt_embd(tt_inp_emb)
        tt_inp_emb = ttnn.interleaved_to_sharded(tt_inp_emb, tt_model.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])

        rot_mat = ttnn.to_device(rot_mat, mesh_device, memory_config=tt_model.model_config["ROT_MAT_MM_IN1_MEMCFG"])
        cache_idxs = ttnn.to_device(cache_idxs, mesh_device, memory_config=tt_model.model_config["DRAM_MEMCFG"])

        if cur_pos == 0 or cur_pos == 35:  # Skip the first few iterations to warm up
            profiler.end(f"processing_of_decode_input_{cur_pos}")
            profiler.start(f"model_run_for_inference_{cur_pos}")

        tt_logits = tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            cache_idxs=cache_idxs,
        )

        if cur_pos == 0 or cur_pos == 35:  # Skip the first few iterations to warm up
            profiler.end(f"model_run_for_inference_{cur_pos}")

        del tt_inp_emb
        del rot_mat
        del attn_mask

        for i in mesh_device.get_device_ids():
            device = mesh_device.get_device(i)
            ttnn.synchronize_device(device)

        logits = ttnn.to_torch(tt_logits, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
        logits = logits[..., : configuration.vocab_size].float()  # [1, batch, vocab_size]
        del tt_logits

        decode_ids = post_processor(logits=logits, index=...).reshape(batch, 1)

        generated_ids = torch.concat((generated_ids, decode_ids[:num_users]), dim=1)

        # TODO: Remove if we don't want to print per generated token
        print_output_prompts(generated_ids, tokenizer)

    logger.info("Finished 1st run decode stage with compile!")
    profiler.print()

    print_output_prompts(generated_ids, tokenizer)

    comment = f"num_layers={n_layers}_n_devices={n_devices}_emulated={emulated}"
    cpu_time = profiler.get("hugging_face_reference_model")

    prefill_compile_time = profiler.get(f"model_run_for_prefill_{0}")
    prefill_time = profiler.get(f"model_run_for_prefill_{25}")
    decode_compile_time = profiler.get(f"model_run_for_inference_{0}")
    decode_time = profiler.get(f"model_run_for_inference_{35}")

    prep_perf_report(
        model_name=f"Llama_{comment}",
        batch_size=batch,
        inference_and_compile_time=decode_compile_time,
        inference_time=decode_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"Prefill tokens per second for user 25 { 1 / prefill_time}")
    logger.info(f"llama {comment} inference time: {decode_time}")
    tokens_per_s_per_user = 1 / decode_time
    tokens_per_s_overall = tokens_per_s_per_user * batch * seq_len

    logger.info(f"Time per iteration: {decode_time}")
    logger.info(f"Tokens per s per user: {tokens_per_s_per_user}")
    logger.info(f"Tokens per s overall: {tokens_per_s_overall}")

    # This script will assert since this is not a part of regular perf pipeline
    # assert second_iter_time <= expected_inference_time
    # assert compile_time <= expected_compile_time


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(240000)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "prefill_length, generation_length, expected_compile_time, expected_inference_time",
    (
        (128, 128, 60, 0.22),
        (128, 2048, 60, 0.22),
        (2048, 128, 60, 0.22),
    ),
    ids=["short-short", "short-long", "long-short"],
)
def test_Llama_perf_host(
    prefill_length,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    t3k_mesh_device,
    n_layers=80,
    n_devices=8,
    emulated=False,
    num_users=32,
):
    batch, seq_len = 1, prefill_length
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)

    if t3k_mesh_device.get_num_devices() < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_mesh_device.get_device(0).compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    for i in t3k_mesh_device.get_device_ids():
        device = t3k_mesh_device.get_device(i)
        device.enable_program_cache()
        device.enable_async(True)
    disable_compilation_reports()

    run_test_LlamaModel_end_to_end(
        t3k_mesh_device,
        batch,
        seq_len,
        model_config,
        n_layers,
        n_devices,
        prefill_length,
        generation_length,
        expected_compile_time,
        expected_inference_time,
        emulated,
        num_users,
    )


@pytest.mark.timeout(240000)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize(
    "batch, seq_len, expected_perf",
    ((32, 1, 300000), (1, 128, 300000), (1, 2048, 300000)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_Llama_perf_device(batch, seq_len, expected_perf):
    subdir = "llama2_70b"
    margin = 0.03  # 0.5

    dir_path = "generated/profiler/reports/" + subdir
    os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Checking existence of directory: {dir_path}")
    if not os.path.exists(dir_path):
        logger.info("Directory does not exist. Attempting to create.")
        os.makedirs(dir_path, exist_ok=True)
    else:
        logger.info("Directory exists.")

    seq_len_str = "2k" if seq_len == 2048 else str(seq_len)
    llm_mode = "decode" if seq_len == 1 else f"prefill_{seq_len_str}"
    command = f"pytest models/demos/t3000/llama2_70b/tests/test_llama_model.py::test_LlamaModel_inference[{llm_mode}-8chip-T3000-2L]"

    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, UNIT_TEST_GENERATION_LENGTH, cols, batch)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)
    prep_device_perf_report(
        model_name=f"llama_70b_{batch}batch_{seq_len}seq_len",
        batch_size=batch,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        # comments=test.replace("/", "_"),
    )
