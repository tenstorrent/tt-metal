# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import torch
from torch import nn
import tt_lib
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor

from models.experimental.llama2_70b.reference.llama.llama import Llama
from models.experimental.llama2_70b.tt.llama_mlp_optimized import TtLlamaMLP_optimized
from models.experimental.llama2_70b.tt.llama_mlp_galaxy import TtLlamaMLP_galaxy
from models.experimental.llama2_70b.tt.model_config import (
    get_model_config,
)

from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.experimental.llama2_70b.tt.llama_common import (
    get_llama_path,
    MAX_SEQ_LEN,
    BASE_URL,
    UNIT_TEST_N_LAYER,
    UNIT_TEST_LAYER_NUM,
    comp_pcc,
    should_skip_model_load,
)


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
    t3k_device_mesh,
    batch,
    seq_len,
    pcc,
    model_config,
    n_devices,
    emulated=False,
):
    # Prepare paths and devices
    t3k_device_mesh, ckpt_dir, tokenizer_path, cache_path = get_llama_path(
        t3k_device_mesh, model_config, n_devices, emulated
    )
    skip_model_load = should_skip_model_load()

    # Prepare configs
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=batch,
        n_layers=UNIT_TEST_N_LAYER,
        skip_model_load=skip_model_load,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    torch.manual_seed(0)
    configuration = hugging_face_reference_model.params

    # Prepare input
    pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
    pt_inp = hugging_face_reference_model.tok_embeddings(pt_inp_ids)
    pt_inp_normed = hugging_face_reference_model.layers[UNIT_TEST_LAYER_NUM].ffn_norm(pt_inp)
    if model_config["LLM_MODE"] == "decode":
        # shape should be (1, seq_len, batch, dim)
        pt_inp_normed = pt_inp_normed.unsqueeze(1).permute(2, 1, 0, 3)
    else:
        pt_inp_normed = pt_inp_normed.unsqueeze(0)

    tt_inp = pt_inp_normed.clone()

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaMLP_model = PytorchLlamaMLPModel(hugging_face_reference_model, UNIT_TEST_LAYER_NUM)
    pytorch_out = pytorch_LlamaMLP_model(pt_inp_normed)

    # TT hardware execution -------------------------------------------------------------
    tt_LlamaMLP_model = TtLlamaMLP_optimized(
        t3k_device_mesh,
        state_dict,
        BASE_URL,
        UNIT_TEST_LAYER_NUM,
        configuration.dim,
        model_config,
        emulated=emulated,
        cache_path=cache_path,
    )

    tt_mlp_input = tt_LlamaMLP_model.prepare_inputs(tt_inp)

    tt_out = tt_LlamaMLP_model(tt_mlp_input)
    tt_out = ttnn.from_device(tt_out)
    tt_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=3))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Llama MLP output Passed!")
    else:
        logger.warning("Llama MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "n_devices, emulated",
    (
        (8, False),
        (8, True),
    ),
    ids=(
        "8chip-T3000",
        "8chip-emulated",
    ),
)
@pytest.mark.parametrize(
    "batch, seq_len, pcc",
    ((32, 1, 0.9998), (1, 128, 0.998), (1, 2048, 0.998)),
    ids=("decode", "prefill_128", "prefill_2k"),
)
def test_LlamaMLP_inference(
    batch,
    seq_len,
    pcc,
    n_devices,
    t3k_device_mesh,
    emulated,
):
    model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices, seq_len=seq_len)

    if t3k_device_mesh.get_num_devices() < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")

    compute_grid_size = t3k_device_mesh.get_device(0).compute_with_storage_grid_size()
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaMLP_inference(
        t3k_device_mesh,
        batch,
        seq_len,
        pcc,
        model_config,
        n_devices,
        emulated,
    )
