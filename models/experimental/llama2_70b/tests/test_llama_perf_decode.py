# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import os
from functools import partial
from loguru import logger
from pathlib import Path
import torch
from torch import nn
import tt_lib
import ttnn

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized
from models.experimental.llama2_70b.tt.model_config import (
    get_model_config,
)
from models.experimental.llama2_70b.tt.llama_common import (
    get_llama_path,
    MAX_SEQ_LEN,
    BASE_URL,
)
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


def load_prompts_file(tokenizer, prefill_length, gap=64):
    # Load prompts
    prompts = open("models/demos/llama2_70b/demo/data/a_tale_of_two_cities.txt", encoding="utf-8-sig").read()
    tokenized = tokenizer.encode(prompts, bos=True, eos=False)

    token_ids = []
    for i in range(0, len(tokenized) - prefill_length + 1, prefill_length + gap):
        window = tokenized[i : i + prefill_length]
        if len(token_ids) == 32:
            return token_ids, tokenizer.decode(token_ids)
        token_ids.append(window)


def prepare_next_input(tokenizer, tokens, input_text_mask, cur_pos, next_token):
    # only replace token if prompt has already been generated
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token

    eos_reached = (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
    prev_pos = cur_pos

    return tokens, eos_reached, prev_pos


def intialize_inputs(tokenizer, prompt_tokens, bsz, total_len):
    # pad the model to maximum length
    pad_id = tokenizer.pad_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cpu")
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cpu")

    input_text_mask = tokens != pad_id  # use prefill token if that token is not masked
    return tokens, input_text_mask


def run_test_LlamaModel_end_to_end(
    devices,
    batch,
    seq_len,
    model_config,
    n_layers,
    n_devices,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    emulated,
    num_users,
):
    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(devices, model_config, n_devices, emulated)
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
    state_dict = hugging_face_reference_model.state_dict()
    configuration = hugging_face_reference_model.params

    # Prepare input -----------------------------------------------------------------------
    torch.manual_seed(0)
    total_len = min(MAX_SEQ_LEN, generation_length + 1)
    prefill_ids, prefill_texts = load_prompts_file(tokenizer, prefill_length=32 if generation_length > 32 else 20)
    tokens, input_text_mask = intialize_inputs(tokenizer, prefill_ids, num_users, total_len)
    # Clear global profiler state before starting measurements
    profiler.clear()

    # Set up model -----------------------------------------------------------------------
    logger.info("Moving weights to devices; might take some time...")
    profiler.start("TT_llama_model_setup")
    tt_model = TtLlamaModel_optimized(
        devices,
        state_dict,
        BASE_URL,
        n_layers,
        model_config,
        configuration,
        batch,
        emulated=emulated,
        cache_path=cache_path,
    )
    for device in devices:
        tt_lib.device.Synchronize(device)
    profiler.end("TT_llama_model_setup")

    del state_dict

    logger.info("Running 1st run decode stage with compile...")

    start_pos = 0
    prev_pos = start_pos
    for cur_pos in range(start_pos + 1, generation_length):
        logger.info(f"Generating token: {cur_pos}")

        if cur_pos == start_pos + 1 or cur_pos >= generation_length - 10:  # Skip the first few iterations to warm up
            enable_persistent_kernel_cache()
            profiler.start(f"processing_of_decode_input_{cur_pos}")

        tt_inp_emb, prev_pos, rot_mat, attn_mask = tt_model.prepare_inputs(tokens[:, prev_pos:cur_pos], prev_pos)

        if cur_pos == start_pos + 1 or cur_pos >= generation_length - 10:  # Skip the first few iterations to warm up
            profiler.end(f"processing_of_decode_input_{cur_pos}")
            profiler.start(f"model_run_for_inference_{cur_pos}")

        tt_logits = tt_model(
            tt_inp_emb,
            rot_mat,
            prev_pos,
            attn_mask,
        )

        if cur_pos == start_pos + 1 or cur_pos >= generation_length - 10:  # Skip the first few iterations to warm up
            profiler.end(f"model_run_for_inference_{cur_pos}")

        del tt_inp_emb
        del rot_mat
        del attn_mask

        for device in devices:
            tt_lib.device.Synchronize(device)

        logits = torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_logits], -1)
        logits = logits[..., : configuration.vocab_size].float()  # [1, batch, vocab_size]
        del tt_logits

        next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)

        tokens, eos_reached, prev_pos = prepare_next_input(tokenizer, tokens, input_text_mask, cur_pos, next_token)

        if all(eos_reached):
            break

        for user_id in range(3):
            text = tokenizer.decode(tokens[user_id, : cur_pos + 1].tolist())
            logger.info(f"Loop {cur_pos} user {user_id}: {text}\n")

    logger.info("Finished 1st run decode stage with compile!")
    profiler.print()

    comment = f"num_layers={n_layers}_n_devices={n_devices}_emulated={emulated}"
    cpu_time = profiler.get("hugging_face_reference_model")

    decode_compile_time = profiler.get(f"model_run_for_inference_{start_pos + 1}")
    decode_time = profiler.get(f"model_run_for_inference_{generation_length - 10}")

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
    "generation_length, expected_compile_time, expected_inference_time",
    (
        (32, 552, 1.2),
        (128, 552, 1.3),
        (2048, 552, 1.5),
    ),
    ids=["quick", "short", "long"],
)
def test_Llama_perf_host(
    generation_length,
    expected_compile_time,
    expected_inference_time,
    all_devices,
    use_program_cache,
    n_layers=80,
    n_devices=8,
    emulated=False,
    num_users=32,
):
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    batch, seq_len = 32, 1
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    disable_compilation_reports()

    run_test_LlamaModel_end_to_end(
        devices,
        batch,
        seq_len,
        model_config,
        n_layers,
        n_devices,
        generation_length,
        expected_compile_time,
        expected_inference_time,
        emulated,
        num_users,
    )
