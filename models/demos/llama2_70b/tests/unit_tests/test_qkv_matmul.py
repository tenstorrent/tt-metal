# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.llama2_70b.reference.llama.llama import Llama
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtLlamaQKV(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        model_config,
        tt_cache_path,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.device = device
        self.hidden_size = hidden_size
        self.model_config = model_config

        layer_name = f"{base_url}.{layer_num}"

        wq_str = f"{layer_name}.attention.wq.weight"
        wk_str = f"{layer_name}.attention.wk.weight"
        wv_str = f"{layer_name}.attention.wv.weight"

        self.wq = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wq_str],
                -2,
                -1,
            ),
            self.device,
        )
        self.wk = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wk_str],
                -2,
                -1,
            ),
            self.device,
        )
        self.wv = torch2tt_tensor(
            torch.transpose(
                self.state_dict[wv_str],
                -2,
                -1,
            ),
            self.device,
        )

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        wq_out = tt_lib.tensor.matmul(
            x,
            self.wq,
        )

        wk_out = tt_lib.tensor.matmul(
            x,
            self.wk,
        )

        wv_out = tt_lib.tensor.matmul(
            x,
            self.wv,
        )

        return wq_out, wk_out, wv_out


class PytorchLlamaQKVModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attn = hf_reference_model.layers[layer_num].attention

        # Disable dropout
        self.attn.eval()

    def forward(self, x):
        xq = self.attn.wq(x)
        xk = self.attn.wk(x)
        xv = self.attn.wv(x)
        return xq, xk, xv


def run_test_LlamaQKV(
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

    hugging_face_reference_model = Llama.build(ckpt_dir, tokenizer_path, seq_len, batch, n_layers=1).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    inp = (torch.rand(batch, 1, seq_len, configuration.dim) * 2) - 1
    layer_num = 0

    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaQKV_model = PytorchLlamaQKVModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaQKV_model(inp)

    # TT hardware execution -------------------------------------------------------------
    tt_LlamaQKV_model = TtLlamaQKV(
        device,
        state_dict,
        base_url,
        layer_num,
        configuration.dim,
        model_config,
        tt_cache_path=None,
    )

    tt_inp = torch2tt_tensor(inp, device)

    tt_out = tt_LlamaQKV_model(tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]

    # check outputs ----------------------------------------------------------------------

    for i in range(3):
        logger.info(comp_allclose(pytorch_out[i], tt_out[i]))

    does_pass = True
    for i in range(3):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], pcc)
        logger.info(f"PCC value: {output_pcc}")
        does_pass = does_pass and out_pass

    if does_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        (
            "llama-2-70B",
            1,
            128,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_LlamaQKV_inference(
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

    run_test_LlamaQKV(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
