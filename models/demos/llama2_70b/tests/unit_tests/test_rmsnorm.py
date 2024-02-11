# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.llama2_70b.reference.llama import Llama
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
    # get_tt_cache_path,
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

        self.attn_norm = torch2tt_tensor(self.state_dict[attn_norm_str].unsqueeze(0).expand(batch, -1), self.device)

        # Pad by zero to get a `32x8k` weight tensor
        # self.attn_norm = pad_by_zero(self.state_dict[attn_norm_str], self.device)[0]

    def rms_decomp(self, x):
        squared = tt_lib.tensor.pow(x, 2)
        # mean_squared = tt_lib.tensor.mean(squared, )
        sum_squared = tt_lib.tensor.reduce(
            squared, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.W, scaler=1.0
        )
        # Tensor is 1,1,32,1+31 now
        mean_squared = tt_lib.tensor.div_unary(sum_squared, x.shape()[-1])
        mean_squared_eps = tt_lib.tensor.add_unary(mean_squared, self.norm_eps)
        rms = tt_lib.tensor.pow(mean_squared_eps, 0.5)
        rms_recip = tt_lib.tensor.recip(rms)
        normed_x = tt_lib.tensor.bcast(
            x, rms_recip, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W
        )
        norm_out = tt_lib.tensor.mul(normed_x, self.attn_norm)
        return norm_out

    def forward(self, x: tt_lib.tensor.Tensor) -> tt_lib.tensor.Tensor:
        # x_attn_norm = tt_lib.tensor.rmsnorm(
        #     x, self.norm_eps, self.attn_norm,
        # )
        #  (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * weight
        x_attn_norm = self.rms_decomp(x)
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
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-SHARDED",))
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
