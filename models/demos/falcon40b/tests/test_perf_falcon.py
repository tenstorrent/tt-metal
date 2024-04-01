# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon40b.tt.falcon_causallm import TtFalconCausalLM

# TODO: Remove this?
from models.demos.falcon40b.tt.falcon_common import (
    PytorchFalconCausalLM,
)

from models.demos.falcon40b.tt.model_config import (
    get_model_config,
)

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    is_e75,
    nearest_32,
    skip_for_grayskull,
    get_devices_for_t3000,
)
from models.perf.perf_utils import prep_perf_report


# TODO: Replace this with actual Falcon application-level tests
def run_test_FalconCausalLM_end_to_end(
    devices,
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
    # Clear global profiler state before starting measurements
    profiler.clear()

    model_name = model_location_generator(model_version, model_subdir="Falcon")

    profiler.start("hugging_face_model_setup")
    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=num_layers
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()
    pytorch_FalconCausalLM = PytorchFalconCausalLM(hugging_face_reference_model, num_layers)
    profiler.end("hugging_face_model_setup")

    # Prepare input ------------------------------------------------------------------------
    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    num_attention_heads = configuration.num_attention_heads
    num_kv_heads = configuration.num_kv_heads
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

        past_key_values = None
        tt_layer_past = ()
        tt_k_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
        tt_v_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
        tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
        tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)

        for _ in range(num_layers):
            tt_k_cache = []
            tt_v_cache = []
            for j in range(len(devices)):
                tt_k_cache.append(
                    torch2tt_tensor(
                        tt_k_cache_host[j],
                        devices[j],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
                )
                tt_v_cache.append(
                    torch2tt_tensor(
                        tt_v_cache_host[j],
                        devices[j],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
                )
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        past_key_values = ()
        tt_layer_past = ()
        for i in range(num_layers):
            k_cache = torch.rand(batch, num_kv_heads, kv_cache_len, head_dim)
            v_cache = torch.rand(batch, num_kv_heads, kv_cache_len, head_dim)
            past_key_values += (
                (
                    torch.repeat_interleave(k_cache, num_attention_heads // num_kv_heads, 1),
                    (torch.repeat_interleave(v_cache, num_attention_heads // num_kv_heads, 1)),
                ),
            )

            tt_k_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
            tt_v_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
            tt_k_cache_host[:, :, :kv_cache_len, :] = k_cache
            tt_v_cache_host[:, :, :kv_cache_len, :] = v_cache
            tt_k_cache_host = torch.chunk(tt_k_cache_host, len(devices), 1)
            tt_v_cache_host = torch.chunk(tt_v_cache_host, len(devices), 1)

            tt_k_cache = []
            tt_v_cache = []
            for j in range(len(devices)):
                tt_k_cache.append(
                    torch2tt_tensor(
                        tt_k_cache_host[j],
                        devices[j],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
                )
                tt_v_cache.append(
                    torch2tt_tensor(
                        tt_v_cache_host[j],
                        devices[j],
                        tt_lib.tensor.Layout.TILE,
                        model_config["KV_CACHE_MEMCFG"],
                        model_config["KV_CACHE_DTYPE"],
                    )
                )
            tt_layer_past += ((tt_k_cache, tt_v_cache),)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")
    for device in devices:
        tt_lib.device.Synchronize(device)

    # Prepare output -----------------------------------------------------------------------
    profiler.start("hugging_face_reference_model")
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )
    profiler.end("hugging_face_reference_model")
    del past_key_values
    del pytorch_layer_present
    del pytorch_out
    del pytorch_FalconCausalLM

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    profiler.start("TtFalcon_model_setup")
    tt_FalconCausalLM = TtFalconCausalLM(
        devices,
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
        tt_lib.device.Synchronize(device)
    profiler.end("TtFalcon_model_setup")

    del state_dict

    profiler.start("processing_of_input")
    if llm_mode == "prefill":
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings_host, tt_attention_mask_host = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        attention_mask_memconfig = model_config["ATTN_MASK_MEMCFG"]
        num_max_tokens = nearest_32(kv_cache_len + 1)
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = num_max_tokens
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
    profiler.end("processing_of_input")

    # First run to fill compile cache ----------------------------------------------------
    logger.info(f"Running Falcon model once to fill caches -> disable profiler")
    profiler.disable()

    # Use force enable to only record this profiler call while others are disabled
    profiler.start("first_model_run_with_compile", force_enable=True)
    if llm_mode == "prefill":
        tt_outs = []
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)
        tt_outs = [[tt_o.cpu() for tt_o in tt_out] for tt_out in tt_outs]
        tt_out = tt_outs

    elif llm_mode == "decode":
        tt_embeddings = [
            tt_embeddings_host[i].to(devices[i], model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
            for i in range(len(devices))
        ]
        tt_attention_mask = [
            tt_attention_mask_host[i].to(devices[i], attention_mask_memconfig) for i in range(len(devices))
        ]
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
        tt_out = [tt_o.cpu() for tt_o in tt_out]
    profiler.end("first_model_run_with_compile", force_enable=True)
    for device in devices:
        tt_lib.device.Synchronize(device)

    del tt_out
    del tt_layer_present
    del tt_embeddings
    del tt_attention_mask

    # Second run for perf ----------------------------------------------------------------
    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()

    def run_inference():
        if llm_mode == "prefill":
            model_inputs = torch.split(model_input, 1)
            tt_embeddings, tt_attention_mask = zip(
                *[
                    tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                    for m_i in model_inputs
                ]
            )
        elif llm_mode == "decode":
            tt_embeddings = [
                tt_embeddings_host[i].to(devices[i], model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
                for i in range(len(devices))
            ]
            tt_attention_mask = [
                tt_attention_mask_host[i].to(devices[i], attention_mask_memconfig) for i in range(len(devices))
            ]
        for _ in range(inference_iterations - 1):
            if llm_mode == "prefill":
                tt_outs = []
                for user_id in range(batch):
                    tt_out, tt_layer_present = tt_FalconCausalLM(
                        input_embeddings=tt_embeddings[user_id],
                        llm_mode=llm_mode,
                        attention_mask=tt_attention_mask[user_id],
                        user_id=user_id,
                        layer_past=tt_layer_past,
                        layer_past_len=kv_cache_len,
                        use_cache=use_cache,
                    )
                    tt_outs.append(tt_out)
                model_inputs = torch.split(model_input, 1)
                tt_embeddings, tt_attention_mask = zip(
                    *[
                        tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                        for m_i in model_inputs
                    ]
                )
                tt_outs = [[tt_o.cpu() for tt_o in tt_out] for tt_out in tt_outs]

            elif llm_mode == "decode":
                tt_out, tt_layer_present = tt_FalconCausalLM(
                    input_embeddings=tt_embeddings,
                    llm_mode=llm_mode,
                    attention_mask=tt_attention_mask,
                    layer_past=tt_layer_past,
                    layer_past_len=kv_cache_len,
                    use_cache=use_cache,
                )
                tt_embeddings = [
                    tt_embeddings_host[i].to(devices[i], model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
                    for i in range(len(devices))
                ]
                tt_attention_mask = [
                    tt_attention_mask_host[i].to(devices[i], attention_mask_memconfig) for i in range(len(devices))
                ]
                tt_out = [tt_o.cpu() for tt_o in tt_out]

        if llm_mode == "prefill":
            tt_outs = []
            for user_id in range(batch):
                tt_out, tt_layer_present = tt_FalconCausalLM(
                    input_embeddings=tt_embeddings[user_id],
                    llm_mode=llm_mode,
                    attention_mask=tt_attention_mask[user_id],
                    user_id=user_id,
                    layer_past=tt_layer_past,
                    layer_past_len=kv_cache_len,
                    use_cache=use_cache,
                )
                tt_outs.append(tt_out)
            tt_outs = [[tt_o.cpu() for tt_o in tt_out] for tt_out in tt_outs]

        elif llm_mode == "decode":
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings,
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=use_cache,
            )
            tt_out = [tt_o.cpu() for tt_o in tt_out]

    profiler.start(f"model_warmup_run_for_inference")
    run_inference()
    profiler.end(f"model_warmup_run_for_inference")
    for device in devices:
        tt_lib.device.Synchronize(device)

    profiler.start(f"model_run_for_inference")
    run_inference()
    profiler.end(f"model_run_for_inference")
    for device in devices:
        tt_lib.device.Synchronize(device)

    profiler.print()

    comment = f"kv_cache_len={kv_cache_len}_seq_len={seq_len}_num_layers={num_layers}_config={model_config_str}"
    cpu_time = profiler.get("hugging_face_reference_model")
    first_iter_time = profiler.get("first_model_run_with_compile")
    second_iter_time = profiler.get("model_run_for_inference") / inference_iterations
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
    logger.info(f"{inference_iterations} Iterations inference time: {profiler.get('model_run_for_inference')}")
    logger.info(f"Time per iteration: {second_iter_time}")
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
@pytest.mark.parametrize("num_devices", (4, 8), ids=["4chips", "8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_compile_time, expected_inference_time, inference_iterations",
    (
        ("prefill", 1, 32, 0, 60, 0.22, 10),
        # ("prefill", 1, 128, 0, 0.30),
        # ("prefill", 1, 256, 0, 0.44),
        ("decode", 32, 1, 128, 60, 0.22, 10),
        # ("decode", 32, 1, 1024, 0.35, 10),
        # ("decode", 32, 1, 2047, 0.48, 10),
    ),
    ids=[
        "prefill_seq32",
        # "prefill_seq128",
        # "prefill_seq256",
        "decode_batch32",
        # "decode_batch32_1024",
        # "decode_batch32_2047",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    (60,),
    ids=["layers_60"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT8_B-SHARDED",))
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
    all_devices,
    use_program_cache,
):
    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = get_devices_for_t3000(all_devices, num_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    disable_persistent_kernel_cache()
    disable_compilation_reports()

    run_test_FalconCausalLM_end_to_end(
        devices,
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
