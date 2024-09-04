# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger


from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
)
from models.demos.t3000.llama2_70b.demo.demo import main, construct_arg


@pytest.mark.timeout(240000)
@pytest.mark.parametrize(
    "llama_version",
    (("llama3"),),
)
@pytest.mark.parametrize(
    "chat, prompts_file",
    (
        (True, "models/demos/t3000/llama2_70b/demo/data/multi_prompt_chat.json"),
        (False, "models/demos/t3000/llama2_70b/demo/data/multi_prompt.json"),
    ),
    ids=("chat_completion", "text_completion"),
)
@pytest.mark.parametrize("decode_only", (True, False), ids=("decode_only", "prefill_decode"))
@pytest.mark.parametrize("num_layers", (1, 2, 10, 80), ids=("1L", "2L", "10L", "80L"))
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices",
    (
        (
            "tt",
            False,
            8,
        ),
        (
            "meta",
            False,
            8,
        ),
    ),
    ids=("tt-70b-T3000", "meta-70b"),
)
@pytest.mark.parametrize(
    "max_output_tokens, output_at_end, top_p, top_k, temperature",
    (
        (128, True, 1, 1, 1.0),
        (128, True, 0.9, 10, 1.0),
    ),
    ids=("greedy", "sampling"),
)
@pytest.mark.parametrize(
    "ground_truth",
    ("models/demos/t3000/llama2_70b/demo/data/llama3_ground_truth.json", None),
    ids=("check_enabled", "check_disabled"),
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    (
        (32, 2048),
        (16, 8192),
    ),
    ids=("short_context", "long_context"),
)
def test_LlamaModel_demo(
    # model args
    implementation,
    skip_model_load,
    num_layers,
    # Generation args
    max_output_tokens,
    prompts_file,
    output_at_end,
    top_p,
    top_k,
    temperature,
    chat,
    # TT args
    t3k_mesh_device,
    n_devices,
    decode_only,
    llama_version,
    ground_truth,
    max_batch_size,
    max_context_len,
    use_program_cache,
):
    logger.info("Running LlamaModel demo")
    ## Get model config

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
    )

    check_mesh_device(t3k_mesh_device, model_config)

    for i in t3k_mesh_device.get_device_ids():
        device = t3k_mesh_device.get_device(i)
        device.enable_async(True)

    args = construct_arg(
        implementation=implementation,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        max_batch_size=max_batch_size,
        max_kv_context_len=max_context_len,
        num_layers=num_layers,
        max_output_tokens=max_output_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        chat=chat,
        mesh_device=t3k_mesh_device,
        n_devices=n_devices,
        cache_path=cache_path,
        decode_only=decode_only,
        llama_version=llama_version,
        ground_truth=ground_truth,
    )
    main(args)
