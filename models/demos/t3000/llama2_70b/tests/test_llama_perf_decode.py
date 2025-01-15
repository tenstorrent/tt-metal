# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
import ttnn
from ttnn import ConcatMeshToTensor
import time

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama

from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized
from models.demos.t3000.llama2_70b.tt.llama_common import (
    setup_llama_env,
    check_mesh_device,
    MAX_SEQ_LEN,
    BASE_URL,
    load_llama_state_dict,
    should_skip_model_load,
)
from models.utility_functions import (
    profiler,
    disable_compilation_reports,
    skip_for_grayskull,
    is_wormhole_b0,
)
from models.perf.perf_utils import prep_perf_report

from collections import defaultdict


def get_decode_time(profiler, start_token, end_token):
    total_time = 0
    num_tokens = end_token - start_token + 1

    for token in range(start_token, end_token + 1):
        total_time += profiler.get(f"model_run_for_inference_{token}")

    average_time = total_time / num_tokens
    return average_time


# Define a dictionary to hold the profiling ranges for each generation length
profiling_ranges = {32: [(20, 30)], 128: [(20, 30), (116, 126)], 2048: [(20, 30), (116, 126), (2036, 2046)]}


def is_in_profiling_range(cur_pos, generation_length, profiling_ranges):
    if generation_length in profiling_ranges:
        for start, end in profiling_ranges[generation_length]:
            if start <= cur_pos <= end:
                return True
    return False


def calculate_decode_times(profiler, generation_length):
    times = {}
    for start, end in profiling_ranges[generation_length]:
        label = f"decode_time_{end+2}"
        times[label] = get_decode_time(profiler, start, end)
    return times, times[f"decode_time_{generation_length}"]


def run_test_LlamaModel_end_to_end(
    mesh_device,
    llama_version,
    batch,
    seq_len,
    max_context_len,
    model_config,
    n_layers,
    n_devices,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    ckpt_dir,
    tokenizer_path,
    cache_path,
):
    # Prepare paths and devices
    skip_model_load = should_skip_model_load()

    logger.info(f"Running num_layer: {n_layers}")

    generator = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=max_context_len,
        max_batch_size=batch,
        n_layers=1,
        skip_model_load=skip_model_load,
    )
    hugging_face_reference_model, tokenizer = generator.model, generator.tokenizer
    hugging_face_reference_model.eval()
    # state_dict = hugging_face_reference_model.state_dict()
    state_dict = load_llama_state_dict(ckpt_dir, n_layers=n_layers)
    configuration = hugging_face_reference_model.params

    # Prepare input -----------------------------------------------------------------------
    torch.manual_seed(0)
    total_len = min(max_context_len, generation_length + 1)
    n_iters = 100  # Number of iterations to run in order to get a perf estimate
    tokens = torch.randint(0, 10000, (batch, 1), dtype=torch.long)
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
        cache_path=cache_path,
        read_cache=True,
    )

    for i in mesh_device.get_device_ids():
        device = mesh_device.get_device(i)
        ttnn.synchronize_device(device)

    profiler.end("TT_llama_model_setup")

    del state_dict

    ##### Prepare Inputs #####
    prev_pos = total_len - 1
    tt_inp_emb, prev_pos, rot_mat, cache_idxs, _ = tt_model.prepare_device_inputs_decode(tokens, prev_pos)

    ##### Compile Model #####
    logger.info("Compiling model")
    profiler.start(f"compile_time")
    tt_logits = tt_model(tt_inp_emb, rot_mat, prev_pos, cache_idxs=cache_idxs, mode="decode")
    tt_logits = ttnn.all_gather(tt_logits, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_logits_tensors = ttnn.get_device_tensors(tt_logits)
    logits_rm = ttnn.to_layout(tt_logits_tensors[0], ttnn.ROW_MAJOR_LAYOUT)
    logits = ttnn.to_torch(logits_rm)
    profiler.end(f"compile_time")
    profiler.print()
    compile_iter_time = profiler.get("compile_time")
    logger.info(f"decode with compile time, single iter latency: {compile_iter_time}")

    ##### Capture Trace #####
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    tt_logits = tt_model(tt_inp_emb, rot_mat, prev_pos, cache_idxs=cache_idxs, mode="decode")
    tt_logits = ttnn.all_gather(tt_logits, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_logits_tensors = ttnn.get_device_tensors(tt_logits)
    logits_rm = ttnn.to_layout(tt_logits_tensors[0], ttnn.ROW_MAJOR_LAYOUT)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ##### Execute Trace #####
    logger.info("Executing trace")
    profiler.start(f"end_to_end_inference")
    for i in range(n_iters):
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        logits = ttnn.to_torch(logits_rm)
    profiler.end(f"end_to_end_inference")
    ttnn.release_trace(mesh_device, trace_id)

    profiler.print()
    loop_time = profiler.get("end_to_end_inference")
    iter_time = loop_time / n_iters
    logger.info(f"decode cached, single iter latency: {iter_time}")

    comment = f"num_layers={n_layers}L_n_devices={n_devices}"

    prep_perf_report(
        model_name=f"{llama_version}_70b_{comment}",
        batch_size=batch,
        inference_and_compile_time=compile_iter_time,
        inference_time=iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )

    tokens_per_s_per_user = 1 / iter_time
    tokens_per_s_overall = tokens_per_s_per_user * batch

    logger.info(f"Time per iteration: {iter_time}")
    logger.info(f"Tokens per s per user: {tokens_per_s_per_user}")
    logger.info(f"Tokens per s overall: {tokens_per_s_overall}")

    # assert compile_time <= expected_compile_time
    assert iter_time <= expected_inference_time


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(4500)
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize(
    "llama_version",
    (("llama3"),),
)
@pytest.mark.parametrize(
    "generation_length, expected_compile_time, expected_inference_time, batch, seq_len, max_context_len",
    (
        (32, 10000, 0.0653 + 0.01, 32, 1, 4096),
        (128, 10000, 0.0655 + 0.01, 32, 1, 4096),
        (2048, 10000, 0.0771 + 0.01, 32, 1, 4096),
        (8192, 10000, 0.0825 + 0.01, 16, 1, 8192),
        (128 * 1024, 10000, 0.0918 + 0.01, 1, 1, 128 * 1024),
    ),
    ids=["gen32", "gen128", "gen2k", "gen8k", "gen128k"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
def test_Llama_perf_host(
    generation_length,
    expected_compile_time,
    expected_inference_time,
    batch,
    seq_len,
    max_context_len,
    t3k_mesh_device,
    llama_version,
    use_program_cache,
    n_layers=80,
    n_devices=8,
):
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        max_batch_size=batch,
        max_context_len=max_context_len,
    )

    check_mesh_device(t3k_mesh_device, model_config)

    t3k_mesh_device.enable_async(True)

    disable_compilation_reports()

    run_test_LlamaModel_end_to_end(
        t3k_mesh_device,
        llama_version,
        batch,
        seq_len,
        max_context_len,
        model_config,
        n_layers,
        n_devices,
        generation_length,
        expected_compile_time,
        expected_inference_time,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )


def run_test_LlamaModel_end_to_end_hybrid_data_tensor_parallel(
    mesh_device,
    llama_version,
    batch,
    seq_len,
    max_context_len,
    model_config,
    n_layers,
    n_devices,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    ckpt_dir,
    tokenizer_path,
    cache_path,
):
    # Prepare paths and devices
    skip_model_load = should_skip_model_load()

    logger.info(f"Running num_layer: {n_layers}")

    generator = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=max_context_len,
        max_batch_size=batch,
        n_layers=1,
        skip_model_load=skip_model_load,
    )
    hugging_face_reference_model, tokenizer = generator.model, generator.tokenizer
    hugging_face_reference_model.eval()
    # state_dict = hugging_face_reference_model.state_dict()
    state_dict = load_llama_state_dict(ckpt_dir, n_layers=n_layers)
    configuration = hugging_face_reference_model.params

    # Prepare input -----------------------------------------------------------------------
    torch.manual_seed(0)
    total_len = min(max_context_len, generation_length + 1)
    n_iters = 100  # Number of iterations to run in order to get a perf estimate
    tokens = torch.randint(0, 10000, (batch, 1), dtype=torch.long)
    # Clear global profiler state before starting measurements
    profiler.clear()

    submesh_to_metadata = defaultdict(dict)
    submeshes = mesh_device.create_submeshes((2, 4), ttnn.MeshType.Ring)
    for submesh in submeshes:
        # Set up model -----------------------------------------------------------------------
        logger.info("Moving weights to devices; might take some time...")
        profiler.start("TT_llama_model_setup")
        tt_model = TtLlamaModel_optimized(
            submesh,
            state_dict,
            BASE_URL,
            n_layers,
            model_config,
            configuration,
            cache_path=cache_path,
            read_cache=True,
        )

        for i in submesh.get_device_ids():
            device = submesh.get_device(i)
            ttnn.synchronize_device(device)

        profiler.end("TT_llama_model_setup")

        ##### Prepare Inputs #####
        prev_pos = total_len - 1
        tt_inp_emb, prev_pos, rot_mat, cache_idxs, _ = tt_model.prepare_device_inputs_decode(
            tokens, prev_pos, mode="decode"
        )

        ##### Compile Model #####
        logger.info("Compiling model")
        profiler.start(f"compile_time")
        tt_logits = tt_model(tt_inp_emb, rot_mat, prev_pos, cache_idxs=cache_idxs, mode="decode")
        tt_logits = ttnn.all_gather(tt_logits, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_logits_tensors = ttnn.get_device_tensors(tt_logits)
        logits_rm = ttnn.to_layout(tt_logits_tensors[0], ttnn.ROW_MAJOR_LAYOUT)
        logits = ttnn.to_torch(logits_rm)
        profiler.end(f"compile_time")
        profiler.print()
        compile_iter_time = profiler.get("compile_time")
        logger.info(f"decode with compile time, single iter latency: {compile_iter_time}")

        submesh_to_metadata[submesh.id()] = {
            "submesh": submesh,
            "logits_rm": logits_rm,
            "tt_model": tt_model,
            "prev_pos": prev_pos,
            "tt_inp_emb": tt_inp_emb,
            "rot_mat": rot_mat,
            "cache_idxs": cache_idxs,
        }

    ##### Capture Trace #####
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

    for submesh in submeshes:
        mesh_id = submesh.id()
        tt_model = submesh_to_metadata[mesh_id]["tt_model"]
        tt_inp_emb = submesh_to_metadata[mesh_id]["tt_inp_emb"]
        rot_mat = submesh_to_metadata[mesh_id]["rot_mat"]
        cache_idxs = submesh_to_metadata[mesh_id]["cache_idxs"]
        prev_pos = submesh_to_metadata[mesh_id]["prev_pos"]

        tt_logits = tt_model(tt_inp_emb, rot_mat, prev_pos, cache_idxs=cache_idxs, mode="decode")
        tt_logits = ttnn.all_gather(tt_logits, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_logits_tensors = ttnn.get_device_tensors(tt_logits)
        logits_rm = ttnn.to_layout(tt_logits_tensors[0], ttnn.ROW_MAJOR_LAYOUT)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ##### Execute Trace #####
    logger.info("Executing trace")
    profiler.start(f"end_to_end_inference")
    for i in range(n_iters):
        ttnn.execute_trace(mesh_device, trace_id, blocking=False)
        logits = ttnn.to_torch(logits_rm)
    profiler.end(f"end_to_end_inference")
    ttnn.release_trace(mesh_device, trace_id)

    profiler.print()
    loop_time = profiler.get("end_to_end_inference")
    iter_time = loop_time / n_iters
    logger.info(f"decode cached, single iter latency: {iter_time}")

    comment = f"num_layers={n_layers}L_n_devices={n_devices}"

    prep_perf_report(
        model_name=f"{llama_version}_70b_{comment}",
        batch_size=batch,
        inference_and_compile_time=compile_iter_time,
        inference_time=iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
    )

    tokens_per_s_per_user = 1 / iter_time
    tokens_per_s_overall = tokens_per_s_per_user * batch * len(submeshes)

    logger.info(f"Time per iteration: {iter_time}")
    logger.info(f"Tokens per s per user: {tokens_per_s_per_user}")
    logger.info(f"Tokens per s overall: {tokens_per_s_overall}")

    # assert compile_time <= expected_compile_time
    assert iter_time <= expected_inference_time


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.timeout(4500)
@pytest.mark.model_perf_tg
@pytest.mark.parametrize(
    "llama_version",
    (("llama3"),),
)
@pytest.mark.parametrize(
    "generation_length, expected_compile_time, expected_inference_time, batch, seq_len, max_context_len",
    (
        (32, 10000, 0.0653 + 0.01, 32, 1, 4096),
        (128, 10000, 0.0655 + 0.01, 32, 1, 4096),
        (2048, 10000, 0.0771 + 0.01, 32, 1, 4096),
        (8192, 10000, 0.0825 + 0.01, 16, 1, 8192),
        (128 * 1024, 10000, 0.0918 + 0.01, 1, 1, 128 * 1024),
    ),
    ids=["gen32", "gen128", "gen2k", "gen8k", "gen128k"],
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 20000000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
def test_Llama_perf_hybrid_data_tensor_parallel(
    mesh_device,
    generation_length,
    expected_compile_time,
    expected_inference_time,
    batch,
    seq_len,
    max_context_len,
    llama_version,
    use_program_cache,
    n_layers=80,
    n_devices=8,
):
    model_config, ckpt_dir, tokenizer_path, cache_path = setup_llama_env(
        llama_version=llama_version,
        max_batch_size=batch,
        max_context_len=max_context_len,
    )

    check_mesh_device(mesh_device, model_config)
    mesh_device.enable_async(True)

    disable_compilation_reports()

    run_test_LlamaModel_end_to_end_hybrid_data_tensor_parallel(
        mesh_device,
        llama_version,
        batch,
        seq_len,
        max_context_len,
        model_config,
        n_layers,
        n_devices,
        generation_length,
        expected_compile_time,
        expected_inference_time,
        ckpt_dir,
        tokenizer_path,
        cache_path,
    )
