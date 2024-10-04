# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import torch
import pytest
from loguru import logger

import ttnn
from ttnn import ConcatMeshToTensor

from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.t3000.falcon40b.tt.falcon_causallm import TtFalconCausalLM

from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)

from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    skip_for_grayskull,
)
from models.perf.perf_utils import prep_perf_report


# TODO: Replace this with actual Falcon application-level tests
def run_test_FalconCausalLM_end_to_end(
    mesh_device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    model_config,
    model_config_str,
    tt_cache_path,
    model_location_generator,
    expected_compile_time,
    expected_inference_time,
    warmup_iterations,
    is_ci_env,
):
    if not is_ci_env:  # Enable tracy signpost support in local runs only
        from tracy import signpost

    # Clear global profiler state before starting measurements
    profiler.clear()
    devices = mesh_device.get_devices()
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    profiler.start("hugging_face_model_setup")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=num_layers
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()
    profiler.end("hugging_face_model_setup")

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    use_cache = True
    use_global_cos_sin_cache = True

    if True:
        model_input = torch.randint(0, seq_len * batch, (batch, seq_len))
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.randint(0, seq_len)] * batch).reshape(batch, seq_len)

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")
    for device in devices:
        ttnn.synchronize_device(device)

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        mesh_device,
        state_dict,
        base_url,
        num_layers,
        configuration,
        max_position_embeddings,
        model_config,
        tt_cache_path,
        use_global_cos_sin_cache,
    )
    for device in devices:
        ttnn.synchronize_device(device)
    profiler.end("TtFalcon_model_setup")

    del state_dict

    # Initialize past layer values
    tt_layer_past = tt_FalconCausalLM.initialize_kv_cache()

    profiler.start("processing_of_input")
    if llm_mode == "prefill":
        model_inputs = torch.split(model_input, 1)
        tt_inputs, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_inputs, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)

    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("COMPILE_RUN")

    if llm_mode == "prefill":
        tt_outs = []
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_inputs[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)

        tt_outs = [
            ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))
            for tt_out in tt_outs
        ]
        tt_out = tt_outs

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_inputs,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_out = ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    profiler.end("first_model_run_with_compile", force_enable=True)
    for device in devices:
        ttnn.synchronize_device(device)

    del tt_out
    del tt_layer_present
    del tt_inputs
    del tt_attention_mask

    # Run warmup interations - profiler still disabled
    profiler.start(f"model_warmup_run_for_inference")

    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("WARMUP_RUNS")

    for _ in range(warmup_iterations):
        for device in devices:
            ttnn.DumpDeviceProfiler(device)
        if llm_mode == "prefill":
            model_inputs = torch.split(model_input, 1)
            tt_inputs, tt_attention_mask = zip(
                *[
                    tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                    for m_i in model_inputs
                ]
            )
            tt_outs = []
            for user_id in range(batch):
                tt_out, tt_layer_present = tt_FalconCausalLM(
                    input_ids=tt_inputs[user_id],
                    llm_mode=llm_mode,
                    attention_mask=tt_attention_mask[user_id],
                    user_id=user_id,
                    layer_past=tt_layer_past,
                    layer_past_len=kv_cache_len,
                    use_cache=use_cache,
                )
                tt_outs.append(tt_out)

            tt_outs = [
                ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))
                for tt_out in tt_outs
            ]
        elif llm_mode == "decode":
            tt_inputs, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
                llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
            )
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_inputs,
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs = ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))

    profiler.end(f"model_warmup_run_for_inference")
    for device in devices:
        ttnn.synchronize_device(device)

    # Run for perf iteration - profiler enabled
    for device in devices:
        ttnn.DumpDeviceProfiler(device)
    profiler.enable()
    enable_persistent_kernel_cache()
    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.start(f"model_run_for_inference")

    if not is_ci_env:  # Enable tracy signpost support in local runs only
        signpost("PERF_RUN")

    if llm_mode == "prefill":
        # Push inputs to device and do preprocessing
        model_inputs = torch.split(model_input, 1)
        tt_inputs, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
        # Prefill - forward pass
        tt_outs = []
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_inputs[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)

    elif llm_mode == "decode":
        # Prepare inputs
        tt_inputs, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        # Decode - forward pass
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_inputs,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        # TODO: Return token id to simulate real situation in decode
        _ = ttnn.to_torch(tt_FalconCausalLM.perf_e2e_test_tile_tensor)

    for device in devices:
        ttnn.synchronize_device(device)

    profiler.end(f"model_run_for_inference")

    profiler.print()

    comment = f"kv_cache_len={kv_cache_len}_seq_len={seq_len}_num_layers={num_layers}_config={model_config_str}"
    cpu_time = profiler.get("hugging_face_reference_model")
    first_iter_time = profiler.get("first_model_run_with_compile")
    second_iter_time = profiler.get("model_run_for_inference")
    prep_perf_report(
        model_name=f"Falcon_{llm_mode}_{comment}",
        batch_size=batch,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comment,
        inference_time_cpu=cpu_time,
    )

    compile_time = first_iter_time - second_iter_time
    logger.info(f"falcon {comment} inference time: {second_iter_time}")
    logger.info(f"falcon {comment} compile time: {compile_time}")

    tokens_per_s_per_user = 1 / second_iter_time
    tokens_per_s_overall = tokens_per_s_per_user * batch * seq_len
    logger.info(f"Time per iteration: {second_iter_time}")
    if llm_mode == "prefill":
        logger.info(f"Prompt per s per user: {tokens_per_s_per_user}")
    elif llm_mode == "decode":
        logger.info(f"Tokens per s per user: {tokens_per_s_per_user}")
    logger.info(f"Tokens per s overall: {tokens_per_s_overall}")

    # This script does not asser the expected vs actual time since this is done based on the perf report and as part of the perf pipeline


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.model_perf_t3000
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_compile_time, expected_inference_time, num_layers, model_config_str",
    (
        ("prefill", 1, 32, 0, 62, 0.37 + 0.04, 60, "BFLOAT8_B-DRAM"),
        ("prefill", 1, 128, 0, 60, 0.39 + 0.04, 60, "BFLOAT8_B-DRAM"),
        ("prefill", 1, 2048, 0, 60, 0.94 + 0.1, 60, "BFLOAT8_B-DRAM"),
        ("prefill", 1, 32, 0, 60, 0.42 + 0.04, 60, "BFLOAT16-DRAM"),
        ("prefill", 1, 128, 0, 60, 0.46 + 0.04, 60, "BFLOAT16-DRAM"),
        ("prefill", 1, 2048, 0, 60, 1.18 + 0.1, 60, "BFLOAT16-DRAM"),
        ("decode", 32, 1, 128, 60, 0.21 + 0.02, 60, "BFLOAT8_B-SHARDED"),
    ),
    ids=[
        "prefill_seq32_bfp8",
        "prefill_seq128_bfp8",
        "prefill_seq2048_bfp8",
        "prefill_seq32_fp16",
        "prefill_seq128_fp16",
        "prefill_seq2048_fp16",
        "decode_batch32",
    ],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize(
    "async_mode",
    (True,),
)
def test_perf_bare_metal(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_compile_time,
    expected_inference_time,
    num_layers,
    request,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
    is_ci_env,
    async_mode,
):
    if llm_mode == "prefill" and (model_config_str not in ["BFLOAT8_B-DRAM", "BFLOAT16-DRAM"] or num_devices != 8):
        pytest.skip("Prefill is only supported for DRAM memory config and 8 chips!")
    if llm_mode == "decode" and model_config_str not in ["BFLOAT8_B-SHARDED"]:
        pytest.skip("Decode is only supported for SHARDED memory config!")

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    t3k_mesh_device.enable_async(async_mode)
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        t3k_mesh_device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        model_config,
        model_config_str,
        tt_cache_path,
        model_location_generator,
        expected_compile_time,
        expected_inference_time,
        warmup_iterations=10,
        is_ci_env=is_ci_env,
    )


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_compile_time, expected_inference_time, num_layers, model_config_str",
    (
        ("prefill", 1, 128, 0, 60, 0.39 + 0.04, 1, "BFLOAT8_B-DRAM"),
        ("prefill", 1, 2048, 0, 60, 0.94 + 0.1, 1, "BFLOAT8_B-DRAM"),
    ),
    ids=[
        "prefill_seq128_bfp8_layers1",
        "prefill_seq2048_bfp8_layers1",
    ],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
def test_device_perf_bare_metal(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_compile_time,
    expected_inference_time,
    num_layers,
    request,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
    is_ci_env,
):
    if llm_mode == "prefill" and (model_config_str not in ["BFLOAT8_B-DRAM", "BFLOAT16-DRAM"] or num_devices != 8):
        pytest.skip("Prefill is only supported for DRAM memory config and 8 chips!")
    if llm_mode == "decode" and model_config_str not in ["BFLOAT8_B-SHARDED"]:
        pytest.skip("Decode is only supported for SHARDED memory config!")

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = t3k_mesh_device.get_devices()
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        t3k_mesh_device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        model_config,
        model_config_str,
        tt_cache_path,
        model_location_generator,
        expected_compile_time,
        expected_inference_time,
        warmup_iterations=10,
        is_ci_env=is_ci_env,
    )
