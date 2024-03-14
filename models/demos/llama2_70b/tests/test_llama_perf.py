# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
from pathlib import Path
import torch
from torch import nn
import tt_lib
import ttnn

from models.demos.llama2_70b.reference.llama.llama import Llama
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)
from models.demos.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized

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


class PytorchLlamaModel(torch.nn.Module):
    def __init__(self, hf_reference_model):
        super().__init__()
        self.model = hf_reference_model

        # Disable dropout
        self.model.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def forward(self, x, start_pos):
        """
        x: (batch, seq)
        start_pos: int

        return: (batch, seq, hidden_dim)
        """
        return self.model(x, start_pos)


# TODO: Replace this with actual Llama application-level tests
def run_test_LlamaModel_end_to_end(
    devices,
    batch,
    seq_len,
    model_config,
    n_layers,
    n_devices,
    expected_compile_time,
    expected_inference_time,
    inference_iterations,
    emulated=False,
):
    if emulated:
        ckpt_dir = "/proj_sw/user_dev/llama-data-repacked-2/llama-2-70b/"
        tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"
        cache_path = Path("/proj_sw/user_dev/llama-data-cache/weights-cache")
        device = devices[0]
        devices = [device for _ in range(n_devices)]  # Emulate fracturing on N chips
    else:
        ckpt_dir = model_config["DEFAULT_CKPT_DIR"]
        tokenizer_path = model_config["DEFAULT_TOKENIZER_PATH"]
        cache_path = model_config["DEFAULT_CACHE_PATH"]

    print(f"Running emulated: {emulated}")
    print(f"Running on {n_devices} devices")
    print(f"Running with {n_layers} layers")

    max_seq_len = 4096
    # Clear global profiler state before starting measurements
    profiler.clear()

    profiler.start("llama_reference_model_setup")

    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=batch,
        n_layers=1,
        skip_model_load=False,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    pytorch_model = PytorchLlamaModel(hugging_face_reference_model)
    profiler.end("llama_reference_model_setup")

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = "layers"
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
    start_pos = 0
    tt_inp_ids = pt_inp_ids.clone()

    for device in devices:
        tt_lib.device.Synchronize(device)

    # Prepare output -----------------------------------------------------------------------
    profiler.start("llama_reference_model_run")
    pytorch_out = pytorch_model(
        pt_inp_ids,
        start_pos,
    )
    profiler.end("llama_reference_model_run")
    del pytorch_out
    del pytorch_model

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    profiler.start("TT_llama_model_setup")
    tt_model = TtLlamaModel_optimized(
        devices,
        state_dict,
        base_url,
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

    start_pos = 0
    enable_persistent_kernel_cache()
    profiler.start("warmup_processing_of_input")
    tt_inp_emb, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(tt_inp_ids, start_pos)
    profiler.end("warmup_processing_of_input")

    profiler.start("processing_of_input")
    tt_inp_emb, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs_profile(tt_inp_ids, start_pos)
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Llama model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    # tt_embeddings = [
    #         tt_embeddings_host[i].to(devices[i], model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
    #         for i in range(len(devices))
    #     ]
    tt_out = tt_model(
        tt_inp_emb,
        rot_mat,
        start_pos,
        attn_mask,
    )
    tt_out = [tt_o.cpu() for tt_o in tt_out]
    profiler.end("first_model_run_with_compile", force_enable=True)
    for device in devices:
        tt_lib.device.Synchronize(device)
    logger.info(f"Finished first Llama model with compile")

    del tt_out
    del rot_mat
    del attn_mask

    # Second run for perf ----------------------------------------------------------------
    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    # enable_persistent_kernel_cache()

    def run_inference():
        inference_start_pos = 1
        for i in range(inference_iterations - 1):
            start_pos = inference_start_pos + i
            tt_inp_emb, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(tt_inp_ids, start_pos)

            tt_out = tt_model(
                tt_inp_emb,
                rot_mat,
                start_pos,
                attn_mask,
            )

            tt_out = [tt_o.cpu() for tt_o in tt_out]

    profiler.start(f"model_warmup_run_for_inference")
    run_inference()
    profiler.end(f"model_warmup_run_for_inference")
    for device in devices:
        tt_lib.device.Synchronize(device)

    logger.info(f"Finished Llama model warm up run for inference")

    profiler.start(f"model_run_for_inference")
    run_inference()
    profiler.end(f"model_run_for_inference")
    for device in devices:
        tt_lib.device.Synchronize(device)

    logger.info(f"Finished Llama model run for inference")

    profiler.print()

    comment = f"num_layers={n_layers}_n_devices={n_devices}_emulated={emulated}"
    cpu_time = profiler.get("hugging_face_reference_model")
    first_iter_time = profiler.get("first_model_run_with_compile")
    prepare_inputs_time = profiler.get("processing_of_input")
    second_iter_time = profiler.get("model_run_for_inference") / inference_iterations
    prep_perf_report(
        model_name=f"Llama_{comment}",
        batch_size=batch,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time
    logger.info(f"llama {comment} inference time: {second_iter_time}")
    logger.info(f"llama {comment} compile time: {compile_time}")

    tokens_per_s_per_user = 1 / second_iter_time
    tokens_per_s_overall = tokens_per_s_per_user * batch * seq_len
    logger.info(f"{inference_iterations} Iterations inference time: {profiler.get('model_run_for_inference')}")
    logger.info(f"Time per iteration: {second_iter_time}")

    logger.info(f"Tokens per s per user: {tokens_per_s_per_user}")
    logger.info(f"Tokens per s overall: {tokens_per_s_overall}")

    # This script will assert since this is not a part of regular perf pipeline
    # assert second_iter_time <= expected_inference_time
    # assert compile_time <= expected_compile_time


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "n_layers",
    (1, 2, 4, 8),
)
@pytest.mark.parametrize("emulated", (False, True))
@pytest.mark.parametrize("n_devices", (4, 8))
@pytest.mark.parametrize(
    "batch, seq_len, expected_compile_time, expected_inference_time, inference_iterations",
    (
        (32, 1, 60, 0.22, 10),
        # ("decode", 32, 1, 1024, 0.35, 10),
        # ("decode", 32, 1, 2047, 0.48, 10),
    ),
    ids=[
        "decode_batch32",
        # "decode_batch32_1024",
        # "decode_batch32_2047",
    ],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_perf_bare_metal(
    batch,
    seq_len,
    model_config_str,
    n_layers,
    n_devices,
    expected_compile_time,
    expected_inference_time,
    inference_iterations,
    request,
    all_devices,
    emulated,
):
    devices = get_devices_for_t3000(all_devices, num_devices=n_devices if not emulated else 1)
    model_config = get_model_config(model_config_str, num_devices=n_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"llama2_70b_{request.node.callspec.id}")

    run_test_LlamaModel_end_to_end(
        devices,
        batch,
        seq_len,
        model_config,
        n_layers,
        n_devices,
        expected_compile_time,
        expected_inference_time,
        inference_iterations,
        emulated,
    )
