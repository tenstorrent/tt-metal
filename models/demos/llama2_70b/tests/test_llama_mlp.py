# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
from pathlib import Path

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
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, get_devices_for_t3000


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
    devices,
    batch,
    seq_len,
    pcc,
    model_config,
    optimized,
    n_devices,
    emulated=False,
):
    if emulated:
        ckpt_dir = "/proj_sw/user_dev/llama-data-repacked-2/llama-2-70b/"
        tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"
        cache_path = Path("/proj_sw/user_dev/llama-data-cache/weights-cache")
        device = devices[0]
        devices = [device for _ in range(n_devices)]  # Emulate fracturing on N chips
    else:
        ckpt_dir = model_config["DEFAULT_CKPT_DIR"]
        tokenizer_path = model_config["DEFAULT_TOKENIZER_PATH"]
        cache_path = model_config["DEFAULT_CACHE_PATH"]

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, seq_len, batch, n_layers=1, skip_model_load=False
    ).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()
    # devices = [device for _ in range(n_devices)]  # Emulate fracturing on N chips
    layer_num = 0

    # Prepare input
    torch.manual_seed(0)
    pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
    pt_inp = hugging_face_reference_model.tok_embeddings(pt_inp_ids)
    pt_inp_normed = hugging_face_reference_model.layers[layer_num].ffn_norm(pt_inp)

    pt_inp_normed = pt_inp_normed.unsqueeze(1).permute(2, 1, 0, 3)
    tt_inp = pt_inp_normed.clone()

    print(state_dict.keys())
    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaMLP_model = PytorchLlamaMLPModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaMLP_model(pt_inp_normed)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    print(f"Compute grid size: {compute_grid_size}")
    print(f"Running optimized: {optimized}")
    print(f"Running emulated: {emulated}")
    print(f"Running on {n_devices} devices")

    if optimized:
        # TT hardware execution -------------------------------------------------------------
        tt_LlamaMLP_model = TtLlamaMLP_optimized(
            devices,
            state_dict,
            base_url,
            layer_num,
            configuration.dim,
            model_config,
            emulated=emulated,
            cache_path=cache_path,
        )
        # Put input sharded in L1
        tt_mlp_input = [
            torch2tt_tensor(
                tt_inp.clone(),
                device,
            )
            for device in devices
        ]
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
        tt_mlp_input = [torch2tt_tensor(tt_inp.clone(), device) for device in devices]

    tt_out = tt_LlamaMLP_model(tt_mlp_input)
    assert len(tt_out) == len(devices)
    tt_outs = [tt2torch_tensor(o) for o in tt_out]
    tt_out = torch.cat(tt_outs, dim=-1)
    logger.info(comp_allclose(pytorch_out, tt_out))

    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Llama MLP output Passed!")
    else:
        logger.warning("Llama MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "batch, seq_len, pcc, optimized, n_devices, emulated",
    (
        (32, 1, 0.9999, True, 4, False),
        (32, 1, 0.9999, True, 8, False),
        (32, 1, 0.9999, True, 4, True),
        (32, 1, 0.9999, True, 8, True),
        (32, 1, 0.9999, False, 4, False),
        (32, 1, 0.9999, False, 8, False),
        (32, 1, 0.9999, False, 4, True),
        (32, 1, 0.9999, False, 8, True),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_LlamaMLP_inference(
    batch,
    seq_len,
    pcc,
    optimized,
    model_config_str,
    n_devices,
    all_devices,
    emulated,
):
    devices = get_devices_for_t3000(all_devices, n_devices)

    model_config = get_model_config(model_config_str, num_devices=n_devices)
    compute_grid_size = devices[0].compute_with_storage_grid_size()
    if len(all_devices) < n_devices and not emulated:
        pytest.skip(f"Requires at {n_devices} devices to run")
    if compute_grid_size.x < model_config["MAX_GRID_SIZE"][0] or compute_grid_size.y < model_config["MAX_GRID_SIZE"][1]:
        pytest.skip(f"Requires grid size of at least {model_config['MAX_GRID_SIZE']} to run")

    run_test_LlamaMLP_inference(
        devices,
        batch,
        seq_len,
        pcc,
        model_config,
        optimized,
        n_devices,
        emulated,
    )
