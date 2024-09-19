# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import pytest
from loguru import logger
import torch
from torch import nn
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


import scipy
from sklearn.metrics import top_k_accuracy_score
import numpy as np

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration


from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
    extract_pcc_from_log,
    MAX_SEQ_LEN,
    BASE_URL,
    UNIT_TEST_START_POS,
    UNIT_TEST_GENERATION_LENGTH,
    comp_pcc,
    should_skip_model_load,
    check_kv_cache,
)

from models.demos.t3000.llama2_70b.demo.demo import (
    build_generator,
    load_prompts_file,
    intialize_inputs,
    prepare_next_input,
    get_sampling_func,
    construct_arg,
    DemoArgs,
)


def run_test_generation(args):
    model_args = args.model
    tt_args = args.tt
    data_args = args.data
    generator = Llama.build(
        ckpt_dir=model_args.ckpt_dir,
        tokenizer_path=model_args.tokenizer_path,
        max_seq_len=model_args.max_seq_len,
        max_batch_size=model_args.max_batch_size,
        skip_model_load=model_args.skip_model_load,
        n_layers=model_args.num_layers,
    )

    tt_model = TtLlamaModelForGeneration(
        configuration=generator.model.params,
        state_dict=generator.model.state_dict(),
        model_args=model_args,
        tt_args=tt_args,
    )

    pt_model = generator.model

    tokenizer = generator.tokenizer
    prompt_tokens, prompts = load_prompts_file(model_args, data_args, tokenizer)
    sampling_func = get_sampling_func(data_args.top_k, data_args.top_p, data_args.temperature)

    all_tests_pass = True
    all_pccs, all_top1, all_top5 = [], [], []

    # decode arguments
    bsz = model_args.max_batch_size
    max_gen_len = data_args.num_tokens

    min_prompt_len = min(len(t) for t in prompt_tokens) if not tt_args.decode_only else 1
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= model_args.max_kv_context_len
    total_len = min(model_args.max_kv_context_len, max_gen_len + max_prompt_len)
    assert total_len <= model_args.max_kv_context_len

    # prepare inputs
    tokens, input_text_mask, eos_reached = intialize_inputs(tokenizer, prompt_tokens, bsz, total_len)
    prev_pos = 0

    # some profiling and logging

    for cur_pos in range(min_prompt_len, total_len):
        input_tokens = tokens[:, prev_pos:cur_pos]
        # Print all relevant details
        logger.info(f"Input idx {cur_pos}: input_tokens shape: {input_tokens.shape}, prev_pos: {prev_pos}")
        tt_logits = tt_model.forward(input_tokens, prev_pos)
        pt_logits = pt_model.forward(input_tokens, prev_pos)
        # expects logits to be of shape (bsz, 1, vocab_size)

        next_token = sampling_func(pt_logits)

        tokens, eos_reached, prev_pos = prepare_next_input(tokenizer, tokens, input_text_mask, cur_pos, next_token)

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pt_logits, tt_logits, 0.99)
        logger.info(f"Output idx {cur_pos}: {output_pcc}")


@pytest.mark.timeout(240000)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("decode_only", (True, False), ids=["decode_only", "prefill_decode"])
@pytest.mark.parametrize("num_layers", (1, 5), ids=["1L", "5L"])
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices, llama_version",
    [
        ("tt", False, 8, "llama3"),
    ],
    ids=["tt-70b-T3000"],
)
@pytest.mark.parametrize(
    "num_tokens, prompts_file, output_at_end, top_p, top_k, temperature",
    [
        (128, "models/demos/t3000/llama2_70b/demo/data/multi_prompt.json", True, 1, 1, 1.0),
    ],
    ids=["greedy"],
)
def test_LlamaModel_inference(
    implementation,
    skip_model_load,
    num_layers,
    # Generation args
    num_tokens,
    prompts_file,
    output_at_end,
    top_p,
    top_k,
    temperature,
    # TT args
    # all_devices,
    t3k_mesh_device,
    n_devices,
    llama_version,
    decode_only,
):
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )

    if t3k_mesh_device.get_num_devices() < n_devices:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    for i in t3k_mesh_device.get_device_ids():
        device = t3k_mesh_device.get_device(i)
        device.enable_program_cache()

    args = construct_arg(
        implementation=implementation,
        llama_version=llama_version,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        num_layers=num_layers,
        num_tokens=num_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        mesh_device=t3k_mesh_device,
        n_devices=n_devices,
        cache_path=cache_path,
        decode_only=decode_only,
    )
    run_test_generation(args)
