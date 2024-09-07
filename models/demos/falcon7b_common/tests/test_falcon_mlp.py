# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh
from models.demos.falcon7b_common.tt.falcon_mlp import TtFalconMLPDecode, TtFalconMLPPrefill
from models.demos.falcon7b_common.tt.model_config import get_model_config
from models.demos.falcon7b_common.tests.test_utils import load_hf_model, tt_from_torch, get_num_devices
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc
from models.utility_functions import tt_tensors_to_torch_tensors


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
    max_seq_len=2048,
):
    num_devices = get_num_devices(mesh_device)

    hugging_face_reference_model, state_dict = load_hf_model(model_location_generator, model_version)
    configuration = hugging_face_reference_model.config

    # Prepare input
    torch.manual_seed(0)
    mlp_input = (torch.rand(batch * num_devices, 1, seq_len, configuration.hidden_size) * 2) - 1
    logger.info(f"MLP input shape: {mlp_input.shape}")
    layer_num = 0
    base_url = "transformer.h"

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconMLP_model = PytorchFalconMLPModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_FalconMLP_model(mlp_input)

    if llm_mode == "prefill":
        ttFalconMLP = TtFalconMLPPrefill
    elif llm_mode == "decode":
        ttFalconMLP = TtFalconMLPDecode
    else:
        raise ValueError(f"Unknown llm_mode: {llm_mode}")

    # TT hardware execution -------------------------------------------------------------
    tt_FalconMLP_model = ttFalconMLP(
        mesh_device,
        state_dict,
        base_url,
        layer_num,
        configuration.hidden_size,
        max_seq_len,
        model_config,
        tt_cache_path,
    )

    tt_mlp_input = tt_from_torch(
        mlp_input,
        dtype=model_config["DEFAULT_DTYPE"],
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=0),
    )

    tt_out = tt_FalconMLP_model(tt_mlp_input)
    tt_out = tt_tensors_to_torch_tensors(tt_out, mesh_device, concat_dim=0).to(pytorch_out.dtype)

    # check outputs ----------------------------------------------------------------------
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize("mesh_device", (1, 2, 4, (8, 4)), indirect=True, ids=["1chip", "2chip", "4chip", "32chipTG"])
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
@pytest.mark.parametrize(
    "model_version, llm_mode, batch, seq_len, pcc",
    (
        (
            "tiiuae/falcon-7b-instruct",
            "prefill",
            1,
            2048,
            0.98,
        ),
        (
            "tiiuae/falcon-7b-instruct",
            "prefill",
            1,
            1024,
            0.98,
        ),
        (
            "tiiuae/falcon-7b-instruct",
            "prefill",
            1,
            128,
            0.98,
        ),
        (
            "tiiuae/falcon-7b-instruct",
            "decode",
            1,
            32,
            0.98,
        ),
    ),
    ids=["prefill_seq2048", "prefill_seq1024", "prefill_seq128", "decode_batch32"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconMLP_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
    get_tt_cache_path,
    mesh_device,
    enable_async_mode,
):
    model_config = get_model_config(model_config_str, seq_len, batch)
    tt_cache_path = get_tt_cache_path(
        model_version, model_subdir="Falcon", default_dir=model_config["DEFAULT_CACHE_PATH"]
    )

    run_test_FalconMLP_inference(
        mesh_device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
        max_seq_len=2048,
    )
