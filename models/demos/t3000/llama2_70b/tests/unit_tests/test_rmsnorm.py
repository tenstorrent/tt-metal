# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn

from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


class TtLlamaRMSNorm(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
        configuration,
        batch,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        attn_norm_str = f"{layer_name}.attention_norm.weight"

        self.norm_eps = configuration.norm_eps

        # Must create weight like this to ensure it is in row-major order
        # or we get an error that weight.shape[3] doesn't match inp.shape[3]
        attn_norm = ttnn.Tensor(
            self.state_dict[attn_norm_str].reshape([1, 1, -1, 32]), self.model_config["LN_ATTN_WEIGHTS_DTYPE"]
        )
        self.attn_norm = attn_norm.to(device, self.model_config["LN_ATTN_WEIGHTS_MEMCFG"])

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = ttnn.interleaved_to_sharded(x, self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"])
        x_attn_norm = ttnn.rms_norm(
            x,
            epsilon=self.norm_eps,
            weight=self.attn_norm,
            memory_config=self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"],
            program_config=self.model_config["LN_F_PROGCFG"],
        )
        return x_attn_norm


class PytorchNormModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.norm = hf_reference_model.layers[layer_num].attention_norm

        # Disable dropout
        self.norm.eval()

    def forward(self, x):
        return self.norm(x)


def run_test_LlamaNorm(
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
        ckpt_dir, tokenizer_path, max_seq_len=4096, max_batch_size=batch, n_layers=1
    ).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    inp = (torch.rand(seq_len, 1, batch, configuration.dim) * 2) - 1
    layer_num = 0

    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_model = PytorchNormModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_model(inp)

    # TT hardware execution -------------------------------------------------------------
    tt_model = TtLlamaRMSNorm(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration.dim,
        model_config,
        tt_cache_path=None,
        configuration=configuration,
        batch=batch,
    )

    tt_inp = torch2tt_tensor(inp, device)

    tt_out_tensor = tt_model(tt_inp)
    tt_out = tt2torch_tensor(tt_out_tensor)

    # check outputs ----------------------------------------------------------------------

    logger.info(comp_allclose(pytorch_out, tt_out))

    out_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if out_pass:
        logger.info("Llama RMS output Passed!")
    else:
        logger.warning("Llama RMS output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "llama-2-70B",
            32,
            1,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_LlamaNorm_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    # model_location_generator,
    device,
):
    model_config = get_model_config(model_config_str)
    # tt_cache_path = get_tt_cache_path(model_version)

    run_test_LlamaNorm(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
