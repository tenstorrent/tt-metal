# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.falcon40b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon40b.tt.falcon_mlp import TtFalconMLP
from models.demos.falcon40b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000


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
    devices,
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
        devices,
        state_dict,
        base_url,
        layer_num,
        configuration.hidden_size,
        model_config,
        tt_cache_path,
    )

    tt_mlp_input_host = torch2tt_tensor(mlp_input, None, tt_dtype=model_config["LN_MLP_OUTPUT_DTYPE"])
    tt_mlp_input = []
    for device in devices:
        tt_mlp_input.append(tt_mlp_input_host.to(device, model_config["LN_MLP_OUTPUT_MEMCFG"]))

    tt_out = tt_FalconMLP_model(tt_mlp_input)
    tt_out = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_out], -1)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("num_devices", (4, 8))
@pytest.mark.parametrize(
    "llm_mode, batch, seq_len",
    (
        ("decode", 32, 1),
        ("prefill", 1, 32),
        ("prefill", 1, 64),
    ),
)
@pytest.mark.parametrize(
    "model_version",
    (("tiiuae/falcon-40b-instruct"),),
)
@pytest.mark.parametrize("model_config_str, pcc", [("BFLOAT8_B-SHARDED", 0.9987), ("BFLOAT16-SHARDED", 0.9987)])
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
    all_devices,
    # use_program_cache, # TODO: enable program cache as soon as multi chip correctness is verified
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

    run_test_FalconMLP_inference(
        devices,
        model_version,
        llm_mode,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
