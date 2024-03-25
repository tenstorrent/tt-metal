# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.falcon7b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon7b.tt.falcon_mlp import TtFalconMLP
from models.demos.falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, get_devices_for_t3000


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
    batch,
    seq_len,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    num_devices = len(devices)
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    mlp_input = (torch.rand(batch * num_devices, 1, seq_len, configuration.hidden_size) * 2) - 1
    logger.info(f"MLP input shape: {mlp_input.shape}")
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

    tt_mlp_input = []
    for i in range(num_devices):
        tt_mlp_input.append(torch2tt_tensor(mlp_input[batch * i : batch * (i + 1)], devices[i]))

    tt_out = tt_FalconMLP_model(tt_mlp_input)
    for i in range(num_devices):
        tt_out[i] = tt2torch_tensor(tt_out[i])
    tt_out = torch.concat(tt_out)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize("num_devices", (1, 2, 4))
@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            1,
            128,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconMLP_inference(
    num_devices,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
    all_devices,
):
    devices = get_devices_for_t3000(all_devices, num_devices)

    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    run_test_FalconMLP_inference(
        devices,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
