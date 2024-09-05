# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from ttnn import ConcatMeshToTensor
from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.t3000.falcon40b.tt.falcon_causallm import TtFalconCausalLM

from models.demos.t3000.falcon40b.tt.falcon_common import (
    PytorchFalconCausalLM,
)

from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc, comp_and_get_pcc
from models.utility_functions import (
    profiler,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
    disable_compilation_reports,
    skip_for_grayskull,
)


# TODO: Replace this with actual Falcon application-level tests
def run_test_FalconCausalLM_end_to_end(
    mesh_device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    out_pcc,
    k_cache_pcc,
    v_cache_pcc,
    token_pcc,
    model_config,
    num_loops,
    tt_cache_path,
    model_location_generator,
):
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

    if 1:
        model_input = torch.randint(0, seq_len * batch, (batch, seq_len))
    else:
        # batch identical sequences for debugging
        model_input = torch.stack([torch.randint(0, seq_len)] * batch).reshape(batch, seq_len)

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        past_key_values = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        past_key_values = ()
        for i in range(num_layers):
            k_cache = torch.zeros(batch, num_kv_heads, kv_cache_len, head_dim)
            v_cache = torch.zeros(batch, num_kv_heads, kv_cache_len, head_dim)
            past_key_values += (
                (
                    torch.repeat_interleave(k_cache, num_attention_heads // num_kv_heads, 1),
                    (torch.repeat_interleave(v_cache, num_attention_heads // num_kv_heads, 1)),
                ),
            )

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    # Prepare output -----------------------------------------------------------------------
    logger.info("Running HF reference model")
    profiler.start("hugging_face_reference_model")
    pytorch_out, pytorch_layer_present = pytorch_FalconCausalLM(
        input_ids=model_input, past_key_values=past_key_values, use_cache=use_cache
    )
    profiler.end("hugging_face_reference_model")
    logger.info("Done running HF reference model")

    # NOTE: Passing in pytorch tensor here instead of ll buda tensor
    # since we don't yet have embedding support on device
    # device, state_dict, base_url, max_position_embeddings, config, num_decoders
    logger.info("Loading TT Falcon Model")
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
    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device)
    profiler.end("TtFalcon_model_setup")
    logger.info("Done loading TT Falcon Model")

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
    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device)

    del tt_out
    del tt_inputs
    del tt_attention_mask

    # Warmup loops
    for i in range(num_loops - 1):
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
        logger.info(f"Running Falcon model warmup loop {i}")

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
                if tt_out.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
                    tt_out = ttnn.untilize(tt_out, use_multicore=False)
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
        for device in mesh_device.get_devices():
            ttnn.synchronize_device(device)

        del tt_out
        del tt_inputs
        del tt_attention_mask

    # Second run for perf ----------------------------------------------------------------
    logger.info(f"Enable profiler and enable binary and compile cache")
    profiler.enable()
    enable_persistent_kernel_cache()

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
    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device)
    profiler.start(f"model_run_for_inference")

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

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_inputs,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )
    profiler.end(f"model_run_for_inference")
    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device)

    if llm_mode == "prefill":
        tensors = [
            ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))
            for tt_out in tt_outs
        ]
        tt_out = torch.vstack(tensors)
    elif llm_mode == "decode":
        tt_out = ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=-1))
        tt_out = tt_out.squeeze(1).transpose(0, 1)
    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, out_pcc)
    logger.info(f"Output: {output_pcc}")

    min_k_cache_pcc = 1.0
    min_v_cache_pcc = 1.0

    for i in range(num_layers):
        # Only check every 4 layers for full model
        if num_layers == 60 and i % 4 > 0:
            continue

        pytorch_layer_pres = pytorch_layer_present[i]
        tt_layer_pres = (
            ttnn.to_torch(
                tt_layer_present[i][0], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)
            ),
            ttnn.to_torch(
                tt_layer_present[i][1], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)
            ),
        )
        tt_layer_pres = (
            torch.repeat_interleave(
                tt_layer_pres[0][:batch, :, :kv_len, :],
                configuration.num_attention_heads // configuration.num_kv_heads,
                1,
            ),
            torch.repeat_interleave(
                tt_layer_pres[1][:batch, :, :kv_len, :],
                configuration.num_attention_heads // configuration.num_kv_heads,
                1,
            ),
        )

        does_pass2, output_pcc, calc_pcc = comp_and_get_pcc(pytorch_layer_pres[0], tt_layer_pres[0], k_cache_pcc)
        logger.info(f"K Cache Layer {i}: {output_pcc}")
        if calc_pcc < min_k_cache_pcc:
            min_k_cache_pcc = calc_pcc

        does_pass = does_pass and does_pass2

        does_pass2, output_pcc, calc_pcc = comp_and_get_pcc(pytorch_layer_pres[1], tt_layer_pres[1], v_cache_pcc)
        logger.info(f"V Cache Layer {i}: {output_pcc}")
        if calc_pcc < min_v_cache_pcc:
            min_v_cache_pcc = calc_pcc

        does_pass = does_pass and does_pass2

        if llm_mode == "decode":
            does_pass2, output_pcc, calc_pcc = comp_and_get_pcc(
                pytorch_layer_pres[0][:, :, kv_len - 1 : kv_len, :],
                tt_layer_pres[0][:, :, kv_len - 1 : kv_len, :],
                token_pcc,
            )
            logger.info(f"K Cache Layer {i} new token: {output_pcc}")
            if calc_pcc < min_k_cache_pcc:
                min_k_cache_pcc = calc_pcc

            does_pass = does_pass and does_pass2

            does_pass2, output_pcc, calc_pcc = comp_and_get_pcc(
                pytorch_layer_pres[1][:, :, kv_len - 1 : kv_len, :],
                tt_layer_pres[1][:, :, kv_len - 1 : kv_len, :],
                token_pcc,
            )
            logger.info(f"V Cache Layer {i} new token: {output_pcc}")
            if calc_pcc < min_v_cache_pcc:
                min_v_cache_pcc = calc_pcc

            does_pass = does_pass and does_pass2

    logger.info(f"Min K Cache pcc: {min_k_cache_pcc}")
    logger.info(f"Min V Cache pcc: {min_v_cache_pcc}")

    profiler.print()

    if does_pass:
        logger.info("Falcon Model End-to-End Passed!")
    else:
        logger.warning("Falcon Model End-to-End Failed!")
        assert does_pass


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 32, 0),
        ("prefill", 2, 32, 0),
        ("prefill", 1, 128, 0),
        ("prefill", 1, 2048, 0),
        ("decode", 32, 1, 128),
    ),
    ids=[
        "prefill_seq32",
        "prefill_seq32_batch2",
        "prefill_seq128",
        "prefill_seq2048",
        "decode_batch32",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    (
        1,
        4,
        12,
        60,
    ),
    ids=["layers_1", "layers_4", "layers_12", "layers_60"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize(
    "data_type, memcfg",
    (
        (
            "BFLOAT8_B",
            "SHARDED",
        ),
        (
            "BFLOAT8_B",
            "DRAM",
        ),
        (
            "BFLOAT16",
            "DRAM",
        ),
    ),
)
@pytest.mark.parametrize(
    "async_mode",
    (True,),
)
def test_FalconCausalLM_end_to_end_with_program_cache(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    request,
    data_type,
    memcfg,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
    async_mode,
):
    model_config_str = f"{data_type}-{memcfg}"
    if llm_mode == "prefill" and memcfg != "DRAM" or num_devices != 8:
        pytest.skip("Prefill is only supported for DRAM memory config and 8 chips!")
    if llm_mode == "decode" and memcfg != "SHARDED":
        pytest.skip("Decode is only supported for SHARDED memory config!")

    out_pcc = 0.99
    k_cache_pcc = 0.99
    v_cache_pcc = 0.99
    token_pcc = 0.99

    if llm_mode == "prefill":
        if num_layers == 60:
            if data_type == "BFLOAT8_B":
                if seq_len == 32:
                    out_pcc = 0.986
                    k_cache_pcc = 0.978
                    v_cache_pcc = 0.934
                    token_pcc = 0.99
                elif seq_len == 128:
                    out_pcc = 0.990
                    k_cache_pcc = 0.988
                    v_cache_pcc = 0.940
                    token_pcc = 0.99
                elif seq_len == 2048:
                    out_pcc = 0.992
                    k_cache_pcc = 0.990
                    v_cache_pcc = 0.967
                    token_pcc = 0.99
            elif data_type == "BFLOAT16":
                if seq_len == 32:
                    out_pcc = 0.981
                    k_cache_pcc = 0.978
                    v_cache_pcc = 0.929
                    token_pcc = 0.99
                elif seq_len == 128:
                    out_pcc = 0.991
                    k_cache_pcc = 0.993
                    v_cache_pcc = 0.976
                    token_pcc = 0.99
                elif seq_len == 2048:
                    out_pcc = 0.992
                    k_cache_pcc = 0.989
                    v_cache_pcc = 0.972
                    token_pcc = 0.99
        elif num_layers == 12:
            out_pcc = 0.99
            k_cache_pcc = 0.98
            v_cache_pcc = 0.98
    else:  # Decode
        if num_layers == 60:
            out_pcc = 0.92
            k_cache_pcc = 0.99
            v_cache_pcc = 0.99
            token_pcc = 0.85

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, llm_mode, input_shape, num_devices)
    devices = t3k_mesh_device.get_devices()
    # Set async mode
    for device in devices:
        device.enable_async(async_mode)
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
        out_pcc,
        k_cache_pcc,
        v_cache_pcc,
        token_pcc,
        model_config,
        1,
        tt_cache_path,
        model_location_generator,
    )
