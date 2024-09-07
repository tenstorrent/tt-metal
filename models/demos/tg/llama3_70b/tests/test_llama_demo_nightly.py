# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import os
import json
import torch
import torch.nn.functional as F

from time import time
import pytest
from loguru import logger
from models.utility_functions import skip_for_grayskull
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
)
from models.demos.tg.llama3_70b.demo.demo import run_demo, construct_arg


@pytest.mark.timeout(240000)
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "cluster_shape, mesh_device", [pytest.param((4, 8), (8, 4), id="4x8_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "llama_version",
    (("llama3-tg"),),
)
@pytest.mark.parametrize(
    "chat, prompts_file",
    ((False, "models/demos/t3000/llama2_70b/demo/data/multi_prompt.json"),),
    ids=("text_completion",),
)
@pytest.mark.parametrize("decode_only", (True,), ids=("decode_only",))
@pytest.mark.parametrize("num_layers", (80,), ids=("80L",))
@pytest.mark.parametrize(
    "implementation, skip_model_load, n_devices",
    (
        (
            "tt",
            False,
            8,
        ),
    ),
    ids=("tt-70b-glx",),
)
@pytest.mark.parametrize(
    "max_output_tokens, output_at_end, top_p, top_k, temperature",
    ((128, True, 1, 1, 1.0),),
    ids=("greedy",),
)
@pytest.mark.parametrize(
    "ground_truth",
    (None,),
    ids=("check_disabled",),
)
@pytest.mark.parametrize(
    "max_batch_size, max_context_len",
    ((32, 2048),),
    ids=("short_context",),
)
def test_llama3_tg_nightly_demo(
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
    mesh_device,
    cluster_shape,
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

    check_mesh_device(mesh_device, model_config)

    # TODO: Renable when issue #11089 is resolved
    for i in mesh_device.get_device_ids():
        device = mesh_device.get_device(i)
        device.enable_async(True)

    args = construct_arg(
        implementation=implementation,
        llama_version=llama_version,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        skip_model_load=skip_model_load,
        num_layers=num_layers,
        max_batch_size=max_batch_size,
        max_kv_context_len=max_context_len,
        max_output_tokens=max_output_tokens,
        prompts_file=prompts_file,
        output_at_end=output_at_end,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        chat=chat,
        mesh_device=mesh_device,
        cluster_shape=cluster_shape,
        n_devices=n_devices,
        cache_path=cache_path,
        decode_only=decode_only,
        ground_truth=ground_truth,
        print_output_as_generated=False,
        print_output_at_end=True,
    )
    run_demo(args)
