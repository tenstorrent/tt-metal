# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
from ttnn import ConcatMeshToTensor

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized
from models.demos.t3000.llama2_70b.tests.test_llama_perf import (
    load_prompts_file,
    intialize_inputs,
)
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)
from models.demos.t3000.llama2_70b.tt.llama_common import get_llama_path, MAX_SEQ_LEN, BASE_URL, load_llama_state_dict
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    skip_for_grayskull,
)
from tqdm import tqdm


def prepare_next_input(tokenizer, tokens, input_text_mask, cur_pos, next_token):
    # only replace token if prompt has already been generated
    next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
    tokens[:, cur_pos] = next_token

    eos_reached = (~input_text_mask[:, cur_pos]) & (next_token == tokenizer.eos_id)
    prev_pos = cur_pos

    return tokens, eos_reached, prev_pos


def run_test_LlamaModel_stress_test(
    mesh_device,
    batch,
    seq_len,
    model_config,
    n_layers,
    n_devices,
    generation_length,
    emulated,
):
    devices, ckpt_dir, tokenizer_path, cache_path = get_llama_path(devices, model_config, n_devices, emulated)
    logger.info(f"Running num_layer: {n_layers}")

    generator = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=batch,
        n_layers=1,
        skip_model_load=False,
    )
    hugging_face_reference_model, tokenizer = generator.model.eval(), generator.tokenizer
    # state_dict = hugging_face_reference_model.state_dict()
    state_dict = load_llama_state_dict(ckpt_dir, n_layers=n_layers)
    configuration = hugging_face_reference_model.params

    # Prepare input -----------------------------------------------------------------------
    torch.manual_seed(0)
    total_len = min(MAX_SEQ_LEN, generation_length + 1)
    prefill_ids, ground_truth_texts = load_prompts_file(
        tokenizer, prefill_length=32 if generation_length > 32 else 20, generation_length=generation_length
    )

    # Set up model -----------------------------------------------------------------------
    logger.info("Moving weights to devices; might take some time...")
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
        read_cache=True,
    )
    for i in mesh_device.get_device_ids():
        device = mesh_device.get_device(i)
        ttnn.synchronize_device(device)

    del state_dict

    logger.info("Starting stress test...")
    # enable_persistent_kernel_cache()
    for stress_test_iteration in tqdm(range(10), desc="Stress Test Progress", colour="blue"):
        tokens, input_text_mask = intialize_inputs(tokenizer, prefill_ids, batch, total_len)

        start_pos = 0
        prev_pos = start_pos
        for cur_pos in tqdm(range(start_pos + 1, total_len), desc="Decode to 2k Progress", leave=False, colour="green"):
            tt_inp_emb, prev_pos, rot_mat, attn_mask, cache_idxs = tt_model.prepare_inputs(
                tokens[:, prev_pos:cur_pos], prev_pos
            )
            tt_inp_emb = ttnn.to_device(tt_inp_emb, mesh_device, memory_config=model_config["DRAM_MEMCFG"])
            tt_inp_emb = tt_model.tt_embd(tt_inp_emb)
            tt_inp_emb = ttnn.interleaved_to_sharded(tt_inp_emb, model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
            rot_mat = ttnn.to_device(rot_mat, mesh_device, memory_config=model_config["ROT_MAT_MM_IN1_MEMCFG"])
            cache_idxs = ttnn.to_device(cache_idxs, mesh_device, memory_config=model_config["DRAM_MEMCFG"])

            tt_logits = tt_model(tt_inp_emb, rot_mat, prev_pos, attn_mask, cache_idxs=cache_idxs)

            del tt_inp_emb, rot_mat, attn_mask

            logits = ttnn.to_torch(tt_logits, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
            logits = logits[..., : configuration.vocab_size].float()
            del tt_logits

            next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            tokens, eos_reached, prev_pos = prepare_next_input(tokenizer, tokens, input_text_mask, cur_pos, next_token)

    logger.info("Completed all stress test iterations.")


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(24000000)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "generation_length",
    (2048,),
    ids=["long"],
)
def test_Llama_stress_test(
    generation_length,
    t3k_mesh_device,
    n_layers=80,
    n_devices=8,
    emulated=False,
):
    batch, seq_len = 32, 1

    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)

    if t3k_mesh_device.get_num_devices() < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_mesh_device.get_device(0).compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    for i in t3k_mesh_device.get_device_ids():
        device = t3k_mesh_device.get_device(i)
        device.enable_program_cache()
    disable_compilation_reports()
    run_test_LlamaModel_stress_test(
        devices,
        batch,
        seq_len,
        model_config,
        n_layers,
        n_devices,
        generation_length,
        emulated,
        num_users,
    )
