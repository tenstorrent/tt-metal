# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.llama2_70b.reference.llama import Llama
from models.demos.llama2_70b.reference.llama.model import apply_rotary_emb, precompute_freqs_cis
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtLlamaRotary(torch.nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_size: int,
        n_heads,
        n_kv_heads,
        model_config,
        tt_cache_path,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

    def forward(self, xq, xk, xv, freqs_cis):
        bsz = xq.shape()[0]
        seqlen = xq.shape()[2]

        xqkv_fused = tt_lib.tensor.concat([xq, xk, xv], dim=-1)
        (
            q_heads,
            k_heads,
            v_heads,
        ) = tt_lib.tensor.nlp_create_qkv_heads(
            xqkv_fused, num_heads=self.n_heads, num_kv_heads=self.n_kv_heads, transpose_k_heads=False
        )
        # xq = tt_lib.tensor.reshape(xq, bsz, seqlen, self.n_heads, self.head_dim)
        # xk = tt_lib.tensor.reshape(xk, bsz, seqlen, self.n_kv_heads, self.head_dim)
        # xv = tt_lib.tensor.reshape(xv, bsz, seqlen, self.n_kv_heads, self.head_dim)

        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        return q_heads, k_heads, v_heads


class PytorchLlamaRotaryModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.n_heads = hf_reference_model.params.n_heads
        self.n_kv_heads = hf_reference_model.params.n_kv_heads
        self.head_dim = hf_reference_model.params.dim // self.n_heads

    def forward(self, xq, xk, xv, freqs_cis):
        bsz = xq.size(0)
        seqlen = xq.size(2)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

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

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, seq_len, batch, n_layers=1, skip_model_load=True
    ).model

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    state_dict = hugging_face_reference_model.state_dict()

    # Prepare input
    torch.manual_seed(0)
    inp = [
        (torch.rand(batch, 1, seq_len, hidden_dim) * 2) - 1,
        (torch.rand(batch, 1, seq_len, int(head_dim * n_kv_heads)) * 2) - 1,
        (torch.rand(batch, 1, seq_len, int(head_dim * n_kv_heads)) * 2) - 1,
    ]
    freqs_cis = precompute_freqs_cis(
        # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
        # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
        hidden_dim // n_heads,
        configuration.params.max_seq_len * 2,
    )
    inp.append(freqs_cis)
    layer_num = 0

    base_url = "layers"

    # PyTorch output --------------------------------------------------------------------
    pytorch_model = PytorchLlamaRotaryModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_model(*inp)

    # TT hardware execution -------------------------------------------------------------
    tt_model = TtLlamaRotary(
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_dim,
        n_heads,
        n_kv_heads,
        model_config,
        tt_cache_path=None,
    )

    tt_inp = [torch2tt_tensor(i, device) for i in inp]

    tt_out = tt_model(*tt_inp)
    tt_out = [tt2torch_tensor(tt_out_tensor) for tt_out_tensor in tt_out]

    # check outputs ----------------------------------------------------------------------

    for i in range(3):
        logger.info(comp_allclose(pytorch_out[i], tt_out[i]))

    does_pass = True
    for i in range(3):
        out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], pcc)
        # Check each shape matches
        assert pytorch_out[i].shape == tt_out[i].shape
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
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
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
