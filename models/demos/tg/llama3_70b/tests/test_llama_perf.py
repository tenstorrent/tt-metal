# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.tg.llama3_70b.tt.llama_model_galaxy import TtLlamaModel_galaxy
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
    BASE_URL,
    should_skip_model_load,
    ConcatMesh2DToTensor,
)

from models.utility_functions import (
    profiler,
    disable_compilation_reports,
    skip_for_grayskull,
    profiler,
    enable_persistent_kernel_cache,
)
from models.demos.tg.llama3_70b.tt.llama_common import PytorchLlamaModel


def get_reference_model(
    ckpt_dir,
    tokenizer_path,
    n_layers,
    model_config,
    llama_version,
    batch,
):
    skip_model_load = should_skip_model_load()
    hf_reference = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=model_config[llama_version],
        max_batch_size=batch,
        n_layers=n_layers,
        skip_model_load=skip_model_load,
    )
    hf_reference_model = hf_reference.model
    hf_reference_model.eval()
    state_dict = hf_reference_model.state_dict()
    configuration = hf_reference_model.params
    pytorch_model = PytorchLlamaModel(hf_reference_model)

    return pytorch_model, state_dict, configuration


def prepare_inputs_for_tt_model(
    tt_model, batch, seq_len, vocab_size, generation_start_pos, attn_mask=None, mode="decode"
):
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    tt_input_ids = input_ids.clone()
    # Push inputs to device
    tt_inp_emb, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(
        tt_input_ids, generation_start_pos + 1, attn_mask=attn_mask, mode=mode
    )

    return tt_inp_emb, start_pos, rot_mat, attn_mask


def run_test_LlamaModel_end_to_end(
    mesh_device,
    cluster_shape,
    batch,
    seq_len,
    model_config,
    n_layers,
    llama_version,
    ckpt_dir,
    tokenizer_path,
    cache_path,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    num_iterations,
):
    # Load HuggingFace model
    torch_model, state_dict, configuration = get_reference_model(
        ckpt_dir,
        tokenizer_path,
        1,
        model_config,
        llama_version,
        batch,
    )
    # Load TT 1 layer model for compilation
    tt_model_single_layer = TtLlamaModel_galaxy(
        mesh_device,
        cluster_shape,
        state_dict,
        BASE_URL,
        1,
        model_config,
        configuration,
        cache_path,
        read_cache=True,
    )

    mode = "decode" if seq_len == 1 else "prefill"

    if mode == "prefill":
        generation_start_pos = 0
    else:
        # Decode mode not supported
        raise ValueError(f"Unsupported LLM_MODE: {mode}")

    # Prepare inputs for TT model
    tt_inp_emb, start_pos, rot_mat, attn_mask = prepare_inputs_for_tt_model(
        tt_model_single_layer, batch, seq_len, configuration.vocab_size, generation_start_pos, mode=mode
    )
    # Forward pass of single layer model
    tt_out = tt_model_single_layer(tt_inp_emb, rot_mat, start_pos, attn_mask, mode=mode)
    del tt_inp_emb, rot_mat, tt_out

    # Load TT model with n_layers
    tt_model = TtLlamaModel_galaxy(
        mesh_device,
        cluster_shape,
        state_dict,
        BASE_URL,
        n_layers,
        model_config,
        configuration,
        cache_path,
        read_cache=True,
    )
    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device)
    # Run prefill num_iterations times and measure time
    enable_persistent_kernel_cache()
    for iter_idx in range(num_iterations):
        # Prepare inputs for TT model
        profiler.start(f"prefill_iteration_{iter_idx}")
        tt_inp_emb, start_pos, rot_mat, attn_mask = prepare_inputs_for_tt_model(
            tt_model, batch, seq_len, configuration.vocab_size, generation_start_pos, attn_mask=attn_mask, mode=mode
        )
        # Forward pass of TT model
        tt_out = tt_model(tt_inp_emb, rot_mat, start_pos, attn_mask, mode=mode)
        # Retrieve output from device
        tt_out_cpu = ttnn.to_torch(
            tt_out, mesh_composer=ConcatMesh2DToTensor(mesh_device, dims=(1, 3), cluster_shape=cluster_shape)
        )

        profiler.end(f"prefill_iteration_{iter_idx}")
        for device in mesh_device.get_devices():
            ttnn.synchronize_device(device)

    # Calculate average time for prefill
    profiler.print()
    total_prefill_time = 0
    for iter_idx in range(num_iterations):
        prefill_iter_time = profiler.get(f"prefill_iteration_{iter_idx}")
        total_prefill_time += prefill_iter_time

    avg_prefill_time = total_prefill_time / num_iterations
    logger.info(f"Average prefill time: {avg_prefill_time}")
    tokens_per_s_prefill = seq_len / avg_prefill_time
    logger.info(f"Tokens per second prefill: {tokens_per_s_prefill}")


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(240000)
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "prefill_length, generation_length, num_iterations, expected_compile_time, expected_inference_time",
    ((256, 1, 30, 60, 0.22),),
    ids=["short_context"],
)
@pytest.mark.parametrize("llama_version", ("llama3-tg",))
@pytest.mark.parametrize(
    "cluster_shape, mesh_device", [pytest.param((4, 8), (8, 4), id="4x8_grid")], indirect=["mesh_device"]
)
def test_Llama_perf_host(
    prefill_length,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    llama_version,
    mesh_device,
    cluster_shape,
    num_iterations,
    n_layers=80,
):
    batch, seq_len = 1, prefill_length

    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        seq_len=seq_len,
    )

    check_mesh_device(mesh_device, model_config)

    for device in mesh_device.get_devices():
        device.enable_program_cache()
        device.enable_async(True)
    disable_compilation_reports()

    run_test_LlamaModel_end_to_end(
        mesh_device,
        cluster_shape,
        batch,
        seq_len,
        model_config,
        n_layers,
        llama_version,
        ckpt_dir,
        tokenizer_path,
        cache_path,
        generation_length,
        expected_compile_time,
        expected_inference_time,
        num_iterations,
    )
