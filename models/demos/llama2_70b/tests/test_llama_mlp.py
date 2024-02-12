# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.llama2_70b.reference.llama import Llama
from models.demos.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized
from models.demos.llama2_70b.tt.llama_mlp import TtLlamaMLP
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
    # get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchLlamaMLPModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.mlp = hf_reference_model.layers[layer_num].feed_forward

        # Disable dropout
        self.mlp.eval()

    def forward(self, x):
        result = self.mlp(x)
        return result


def run_test_LlamaMLP_inference(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config,
    # tt_cache_path,
    # model_location_generator,
):
    # model_name = model_location_generator(model_version, model_subdir="Falcon")

    ckpt_dir = "/proj_sw/user_dev/llama-data-repacked/llama-2-70b/"
    tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, seq_len, batch, n_layers=1, skip_model_load=False
    ).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()

    devices = [device for _ in range(8)]  # Emulate fracturing on N chips

    # Prepare input
    torch.manual_seed(0)
    if seq_len == 1:
        input_shape = [seq_len, 1, batch, configuration.dim]
    else:
        input_shape = [batch, 1, seq_len, configuration.dim]

    mlp_input = (torch.rand(input_shape) * 2) - 1
    layer_num = 0

    print(state_dict.keys())
    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaMLP_model = PytorchLlamaMLPModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaMLP_model(mlp_input)
    compute_grid_size = device.compute_with_storage_grid_size()
    print(f"Compute grid size: {compute_grid_size}")
    print(f"Compute grid size x: {compute_grid_size.x}")
    print(f"Compute grid size y: {compute_grid_size.y}")

    mlp_optimized = False
    if mlp_optimized:
        # TT hardware execution -------------------------------------------------------------
        tt_LlamaMLP_model = TtLlamaMLP_optimized(
            device,
            state_dict,
            base_url,
            layer_num,
            configuration.dim,
            model_config,
            tt_cache_path=None,
        )
        # Put input sharded in L1
        tt_mlp_input = torch2tt_tensor(
            mlp_input,
            device,
            tt_memory_config=model_config["LN_MLP_OUTPUT_MEMCFG"],
            tt_dtype=model_config["LN_MLP_OUTPUT_DTYPE"],
        )
    else:
        # TT hardware execution -------------------------------------------------------------
        tt_LlamaMLP_model = TtLlamaMLP(
            devices,
            state_dict,
            base_url,
            layer_num,
            configuration.dim,
            model_config,
        )
        tt_mlp_input = [torch2tt_tensor(mlp_input.clone(), device) for device in devices]

    tt_out = tt_LlamaMLP_model(tt_mlp_input)
    if len(devices) > 1:
        assert len(tt_out) == len(devices)
        tt_outs = [tt2torch_tensor(o) for o in tt_out]
        for i in range(len(devices)):
            logger.info(comp_allclose(pytorch_out, tt_outs[i]))

            does_pass, output_pcc = comp_pcc(pytorch_out, tt_outs[i], pcc)
            logger.info(f"PCC value: {output_pcc}")
    else:
        tt_out = tt2torch_tensor(tt_out[0])
        # check outputs ----------------------------------------------------------------------
        logger.info(comp_allclose(pytorch_out, tt_out))

        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "llama-2-70B",
            32,
            1,
            0.96,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-SHARDED", "BFLOAT8_B-SHARDED"))
def test_LlamaMLP_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    # model_location_generator,
    device,
    use_program_cache,
):
    model_config = get_model_config(model_config_str)
    # tt_cache_path = get_tt_cache_path(model_version)

    run_test_LlamaMLP_inference(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
