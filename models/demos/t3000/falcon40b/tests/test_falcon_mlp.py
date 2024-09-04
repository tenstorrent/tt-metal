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
from models.demos.t3000.falcon40b.tt.falcon_mlp import TtFalconMLP
from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import skip_for_grayskull


class PytorchFalconMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.transformer.h[layer_num].mlp

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def run_test_FalconMLP_inference(
    mesh_device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(
        model_name, low_cpu_mem_usage=True, num_hidden_layers=1
    )
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    if llm_mode == "decode":
        input_shape = [seq_len, 1, batch, configuration.hidden_size]
    else:
        input_shape = [batch, 1, seq_len, configuration.hidden_size]
    mlp_input = (torch.rand(input_shape) * 2) - 1
    layer_num = 0
    base_url = "transformer.h"

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconMLP_model = PytorchFalconMLPModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_FalconMLP_model(mlp_input)
    # TT hardware execution -------------------------------------------------------------

    tt_FalconMLP_model = TtFalconMLP(
        mesh_device,
        state_dict,
        base_url,
        layer_num,
        configuration.hidden_size,
        model_config,
        tt_cache_path,
    )

    tt_mlp_input = ttnn.as_tensor(
        mlp_input,
        dtype=model_config["LN_MLP_OUTPUT_DTYPE"],
        device=mesh_device,
        memory_config=model_config["LN_MLP_OUTPUT_MEMCFG"],
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = tt_FalconMLP_model(tt_mlp_input, llm_mode)
    tt_out_tensor = ttnn.to_torch(tt_out, device=mesh_device, mesh_composer=ConcatMeshToTensor(mesh_device, dim=3))
    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out_tensor, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", (8,), ids=["8chips"])
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len",
    (
        ("decode", 32, 1),
        ("prefill", 1, 32),
        ("prefill", 1, 128),
        ("prefill", 1, 2048),
    ),
    ids=(
        "decode_batch32",
        "prefill_seq32",
        "prefill_seq128",
        "prefill_seq2048",
    ),
)
@pytest.mark.parametrize(
    "model_version",
    (("tiiuae/falcon-40b-instruct"),),
    ids=("falcon_40b",),
)
@pytest.mark.parametrize(
    "model_config_str, pcc",
    [
        ("BFLOAT8_B-SHARDED", 0.9985),
        ("BFLOAT16-SHARDED", 0.9985),
        ("BFLOAT8_B-DRAM", 0.9983),
        ("BFLOAT16-DRAM", 0.9986),
    ],
    ids=("BFLOAT8_B-SHARDED", "BFLOAT16-SHARDED", "BFLOAT8_B-DRAM", "BFLOAT16-DRAM"),
)
def test_FalconMLP_inference(
    num_devices,
    model_version,
    llm_mode,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    t3k_mesh_device,
    use_program_cache,
):
    if llm_mode == "prefill" and (model_config_str not in ["BFLOAT8_B-DRAM", "BFLOAT16-DRAM"] or num_devices != 8):
        pytest.skip("Prefill is only supported for DRAM memory config and 8 chips!")
    if llm_mode == "decode" and model_config_str not in ["BFLOAT8_B-SHARDED", "BFLOAT16-SHARDED"]:
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
    run_test_FalconMLP_inference(
        t3k_mesh_device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
