# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from models.demos.t3000.falcon40b.reference.hf_modeling_falcon import FalconForCausalLM, FalconConfig
from models.demos.t3000.falcon40b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.t3000.falcon40b.tt.model_config import get_model_config, model_config_entries
from models.utility_functions import skip_for_grayskull
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)


def run_test_falcon_prefill_end_to_end_determinism(
    mesh_device,
    model_version,
    generate_weights,
    batch,
    seq_len,
    kv_cache_len,
    iterations,
    num_layers,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    if generate_weights:
        logger.info("Loading PyTorch Falcon model...")
        model_name = model_location_generator(model_version, model_subdir="Falcon")
        hugging_face_reference_model = FalconForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, num_hidden_layers=num_layers
        )
        hugging_face_reference_model.eval()
        configuration = hugging_face_reference_model.config
        state_dict = hugging_face_reference_model.state_dict()
        logger.info("Done loading PyTorch Falcon model")
    else:
        configuration = FalconConfig(**model_config_entries)
        state_dict = None

    torch.manual_seed(0)
    base_url = ""
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads
    num_kv_heads = configuration.num_kv_heads
    use_cache = True
    use_global_cos_sin_cache = True

    # Generate dummy kv_cache --------------------------------------------------------------
    q_len = seq_len
    assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
    assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

    tt_layer_past = ()
    tt_k_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)
    tt_v_cache_host = torch.zeros(batch, num_kv_heads, max_position_embeddings, head_dim)

    for _ in range(num_layers):
        tt_k_cache = ttnn.as_tensor(
            tensor=tt_k_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )
        tt_v_cache = ttnn.as_tensor(
            tensor=tt_v_cache_host,
            dtype=model_config["KV_CACHE_DTYPE"],
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=model_config["KV_CACHE_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=1),
        )
        tt_layer_past += ((tt_k_cache, tt_v_cache),)

    logger.info("Loading TT Falcon Model...")
    # NOTE: Passing in pytorch tensor here instead of tt tensor
    # since we don't yet have embedding support on device
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
    logger.info("Done loading TT Falcon Model")

    # Prepare inputs -----------------------------------------------------------------------
    model_input = torch.randint(0, seq_len * batch, (batch, seq_len))
    model_inputs = torch.split(model_input, 1)

    # First run to get reference output ----------------------------------------------------
    tt_inputs, tt_attention_mask = zip(
        *[
            tt_FalconCausalLM.model_preprocessing("prefill", m_i, kv_cache_len, num_input_tokens=seq_len)
            for m_i in model_inputs
        ]
    )

    logger.info("Running TT Falcon model once to get reference output...")

    tt_outs = []
    for user_id in range(batch):
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_ids=tt_inputs[user_id],
            llm_mode="prefill",
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

    logger.info("Done running TT Falcon model")

    for device in mesh_device.get_devices():
        ttnn.synchronize_device(device)

    reference_out = torch.vstack(tt_outs)
    reference_kv_cache = []
    for i in range(num_layers):
        tt_layer_pres = (
            ttnn.to_torch(
                tt_layer_present[i][0], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)
            ),
            ttnn.to_torch(
                tt_layer_present[i][1], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)
            ),
        )
        reference_kv_cache.append(tt_layer_pres)

    del tt_out
    del tt_layer_present
    del tt_inputs
    del tt_attention_mask

    # Determinism runs ---------------------------------------------------------------------
    does_pass = True
    expected_pcc = 1
    logger.info(f"Running {iterations} iterations...")
    for i in range(iterations):
        logger.info(f"Iteration {i}")
        tt_inputs, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing("prefill", m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )

        tt_outs = []
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_ids=tt_inputs[user_id],
                llm_mode="prefill",
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

        # Check outputs --------------------------------------------------------------------
        pt_tt_out = torch.vstack(tt_outs)
        output_passes, output_pcc = comp_pcc(reference_out, pt_tt_out, expected_pcc)
        logger.info(f"Output: {output_pcc}")

        does_pass = does_pass and output_passes

        for i in range(num_layers):
            pytorch_layer_pres = reference_kv_cache[i]
            tt_layer_pres = (
                ttnn.to_torch(
                    tt_layer_present[i][0], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)
                ),
                ttnn.to_torch(
                    tt_layer_present[i][1], device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=1)
                ),
            )

            k_cache_passes, output_pcc = comp_pcc(pytorch_layer_pres[0], tt_layer_pres[0], expected_pcc)
            logger.info(f"K Cache Layer {i}: {output_pcc}")

            does_pass = does_pass and k_cache_passes

            v_cache_passes, output_pcc = comp_pcc(pytorch_layer_pres[1], tt_layer_pres[1], expected_pcc)
            logger.info(f"V Cache Layer {i}: {output_pcc}")

            does_pass = does_pass and v_cache_passes

        del tt_out
        del tt_layer_present
        del tt_inputs
        del tt_attention_mask

    assert does_pass, "Determinism test failed"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "generate_weights", (True, False), ids=["generate_weights_if_not_cached", "load_cached_weights"]
)
@pytest.mark.parametrize("enable_program_cache", (True, False), ids=["enable_program_cache", "disable_program_cache"])
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "batch, seq_len, kv_cache_len, iterations",
    (
        (1, 32, 0, 30),
        (1, 128, 0, 30),
        (1, 2048, 0, 30),
    ),
    ids=[
        "prefill_seq32",
        "prefill_seq128",
        "prefill_seq2048",
    ],
)
@pytest.mark.parametrize(
    "num_layers",
    (1, 60),
    ids=["layers_1", "layers_60"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-40b-instruct",),
    ids=["falcon_40b"],
)
@pytest.mark.parametrize(
    "model_config_str",
    (
        "BFLOAT8_B-DRAM",
        "BFLOAT16-DRAM",
    ),
)
def test_falcon_prefill_end_to_end_determinism(
    generate_weights,
    enable_program_cache,
    num_devices,
    model_version,
    batch,
    seq_len,
    kv_cache_len,
    iterations,
    num_layers,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
):
    num_devices = 8

    input_shape = [batch, seq_len]
    model_config = get_model_config(model_config_str, "prefill", input_shape, num_devices)
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    if enable_program_cache:
        t3k_mesh_device.enable_program_cache()

    run_test_falcon_prefill_end_to_end_determinism(
        t3k_mesh_device,
        model_version,
        generate_weights,
        batch,
        seq_len,
        kv_cache_len,
        iterations,
        num_layers,
        model_config,
        tt_cache_path,
        model_location_generator,
    )

    if enable_program_cache:
        t3k_mesh_device.disable_and_clear_program_cache()
