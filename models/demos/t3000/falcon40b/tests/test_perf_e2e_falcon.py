# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import time
import ttnn
import tt_lib
from ttnn import ConcatMeshToTensor
from loguru import logger

from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.t3000.falcon40b.tt.falcon_causallm import TtFalconCausalLM

from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)

from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    skip_for_grayskull,
)


# TODO: Replace this with actual Falcon application-level tests
def run_test_FalconCausalLM_end_to_end(
    device_mesh,
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
    inference_iterations,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")
    devices = device_mesh.get_devices()
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=num_layers
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    use_cache = True
    use_global_cos_sin_cache = True

    if True:
        model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.arange(seq_len)] * batch).reshape(batch, seq_len)

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
        ttnn.device.synchronize_device(device)

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    # profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        device_mesh,
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
        ttnn.device.synchronize_device(device)

    del state_dict

    # Initialize past layer values
    tt_layer_past = tt_FalconCausalLM.initialize_kv_cache()

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

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches")

    # Use force enable to only record this profiler call while others are disabled
    compile_time_start = time.time()
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
            ttnn.to_torch(tt_out, device=device_mesh, mesh_composer=ConcatMeshToTensor(device_mesh, dim=-1))
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
        tt_out = ttnn.to_torch(tt_out, device=device_mesh, mesh_composer=ConcatMeshToTensor(device_mesh, dim=-1))
    compile_duration = time.time() - compile_time_start
    for device in devices:
        ttnn.device.synchronize_device(device)

    del tt_out
    del tt_layer_present
    del tt_inputs
    del tt_attention_mask

    # Prepare inputs
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

    # Run warmup interations - profiler still disabled
    for _ in range(inference_iterations - 1):
        for device in devices:
            tt_lib.device.DumpDeviceProfiler(device)
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
            model_inputs = torch.split(model_input, 1)
            tt_inputs, tt_attention_mask = zip(
                *[
                    tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                    for m_i in model_inputs
                ]
            )
            tt_outs = [
                ttnn.to_torch(tt_out, device=device_mesh, mesh_composer=ConcatMeshToTensor(device_mesh, dim=-1))
                for tt_out in tt_outs
            ]

        elif llm_mode == "decode":
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_inputs,
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_out = ttnn.to_torch(tt_out, device=device_mesh, mesh_composer=ConcatMeshToTensor(device_mesh, dim=-1))
    for device in devices:
        ttnn.device.synchronize_device(device)

    for device in devices:
        tt_lib.device.DumpDeviceProfiler(device)
    logger.info(f"Enable binary and compile cache, and start timing.")
    enable_persistent_kernel_cache()
    start_time = time.time()

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
            # Forward pass
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
        # Return single tile to synchronize device (sync_device on all 8 devices is slow)
        # _ = ttnn.to_torch(tt_FalconCausalLM.perf_e2e_test_tile_tensor)
    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_inputs,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        # Return single tile to synchronize device (sync_device on all 8 devices is slow)
        # _ = ttnn.to_torch(tt_FalconCausalLM.perf_e2e_test_tile_tensor)

    for device in devices:
        ttnn.device.synchronize_device(device)

    inference_duration = time.time() - start_time

    logger.info(f"falcon 40b compile time: {compile_duration}")
    logger.info(f"falcon 40b inference time: {inference_duration}")

    tokens_per_s_per_user = 1 / inference_duration
    tokens_per_s_overall = tokens_per_s_per_user * batch * seq_len
    logger.info(f"{inference_iterations} Iterations inference time: {inference_duration}")
    logger.info(f"Time per iteration: {inference_duration}")
    if llm_mode == "prefill":
        logger.info(f"Prompt per s per user: {tokens_per_s_per_user}")
    elif llm_mode == "decode":
        logger.info(f"Tokens per s per user: {tokens_per_s_per_user}")
    logger.info(f"Tokens per s overall: {tokens_per_s_overall}")

    # This script will assert since this is not a part of regular perf pipeline
    # assert second_iter_time <= expected_inference_time
    # assert compile_time <= expected_compile_time


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_compile_time, expected_inference_time, inference_iterations",
    (
        ("prefill", 1, 32, 0, 60, 0.22, 10),
        ("prefill", 1, 128, 0, 60, 0.30, 10),
        ("prefill", 1, 2048, 0, 60, 0.30, 10),
        ("decode", 32, 1, 128, 60, 0.22, 10),
    ),
    ids=[
        "prefill_seq32",
        "prefill_seq128",
        "prefill_seq2048",
        "decode_batch32",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    (
        1,
        60,
    ),
    ids=["layers_1", "layers_60"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT8_B-SHARDED", "BFLOAT8_B-DRAM", "BFLOAT16-DRAM"))
def test_perf_bare_metal(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_compile_time,
    expected_inference_time,
    inference_iterations,
    num_layers,
    request,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    t3k_device_mesh,
    use_program_cache,
):
    if llm_mode == "prefill" and (model_config_str not in ["BFLOAT8_B-DRAM", "BFLOAT16-DRAM"] or num_devices != 8):
        pytest.skip("Prefill is only supported for DRAM memory config and 8 chips!")
    if llm_mode == "decode" and model_config_str not in ["BFLOAT8_B-SHARDED"]:
        pytest.skip("Decode is only supported for SHARDED memory config!")

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = t3k_device_mesh.get_devices()
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    for i in t3k_device_mesh.get_device_ids():
        device = t3k_device_mesh.get_device(i)
        device.enable_program_cache()
        device.enable_async(True)

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        t3k_device_mesh,
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
        inference_iterations,
    )
