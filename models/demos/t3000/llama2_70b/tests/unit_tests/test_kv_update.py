# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.reference.llama.llama import Llama
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
    # get_tt_cache_path,
)

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero


class TtLlamaKVUpdate(torch.nn.Module):
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
        layer_past,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        layer_past = [torch2tt_tensor(lp, device) for lp in layer_past]
        self.layer_past = layer_past

    def forward(self, xk, xv, layer_past_len):
        ttnn.update_cache(self.layer_past[0], xk, layer_past_len)
        ttnn.update_cache(self.layer_past[1], xv, layer_past_len)

        return self.layer_past


class PytorchLlamaKVUpdateModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_past):
        super().__init__()
        self.n_heads = hf_reference_model.params.n_heads
        self.n_kv_heads = hf_reference_model.params.n_kv_heads
        self.head_dim = hf_reference_model.params.dim // self.n_heads
        self.layer_past = layer_past

    def forward(self, xk, xv, layer_past_len):
        """
        Take k_new, v_new. Return updated layer_past.
        """
        # xk = [seq_len, num_kv_heads, batch, dhead]
        # xv = [seq_len, num_kv_heads, batch, dhead]
        seqlen = xk.shape[0]
        key_past = self.layer_past[0]  # [batch, num_kv_heads, seq_len, dhead]
        value_past = self.layer_past[1]  # [batch, num_kv_heads, seq_len, dhead]

        xk = xk.permute(2, 1, 0, 3)  # [batch, num_kv_heads, seq_len, dhead]
        xv = xv.permute(2, 1, 0, 3)  # [batch, num_kv_heads, seq_len, dhead]
        key_past[:, :, layer_past_len : layer_past_len + seqlen, :] = xk
        value_past[:, :, layer_past_len : layer_past_len + seqlen, :] = xv

        return [tensor.clone() for tensor in self.layer_past]


def run_test_LlamaKVUpdate(
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
        ckpt_dir, tokenizer_path, max_seq_len=4096, max_batch_size=1, n_layers=1, skip_model_load=True
    ).model

    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    state_dict = hugging_face_reference_model.state_dict()

    num_iterations = 20
    n_kv_heads = 1

    # Prepare input
    torch.manual_seed(0)
    kv_inps = [
        [
            (torch.rand(seq_len, n_kv_heads, batch, head_dim) * 100) - 50,
            (torch.rand(seq_len, n_kv_heads, batch, head_dim) * 100) - 50,
        ]
        for i in range(num_iterations)
    ]

    layer_past_pytorch = (
        torch.zeros(batch, n_kv_heads, configuration.max_seq_len, head_dim),
        torch.zeros(batch, n_kv_heads, configuration.max_seq_len, head_dim),
    )
    layer_past_tt = (
        torch.zeros(batch, n_kv_heads, configuration.max_seq_len, head_dim),
        torch.zeros(batch, n_kv_heads, configuration.max_seq_len, head_dim),
    )

    layer_num = 0

    base_url = "layers"

    layer_past_len = 120

    # PyTorch output --------------------------------------------------------------------
    pytorch_model = PytorchLlamaKVUpdateModel(hugging_face_reference_model, layer_past_pytorch)
    pytorch_outs = [pytorch_model(*kv_inp, layer_past_len + i) for i, kv_inp in enumerate(kv_inps)]

    # TT hardware execution -------------------------------------------------------------
    tt_model = TtLlamaKVUpdate(
        device,
        state_dict,
        base_url,
        layer_num,
        hidden_dim,
        n_heads,
        n_kv_heads,
        model_config,
        tt_cache_path=None,
        layer_past=layer_past_tt,
    )

    tt_outs = []

    for i, kv_inp in enumerate(kv_inps):
        if batch < 32:
            tt_inp = [pad_by_zero(i, device)[0] for i in kv_inp]
        else:
            tt_inp = [torch2tt_tensor(i, device) for i in kv_inp]

        tt_out = tt_model(*tt_inp, layer_past_len + i)
        tt_out = [tt2torch_tensor(tt_out_tensor).clone() for tt_out_tensor in tt_out]
        tt_outs.append(tt_out)

    # check outputs ----------------------------------------------------------------------

    for pytorch_out, tt_out in zip(pytorch_outs, tt_outs):
        for i in range(len(pytorch_out)):
            logger.info(comp_allclose(pytorch_out[i], tt_out[i]))

        all_tests_pass = True
        for i in range(len(pytorch_out)):
            out_pass, output_pcc = comp_pcc(pytorch_out[i], tt_out[i], pcc)
            # Check each shape matches
            assert pytorch_out[i].shape == tt_out[i].shape
            if out_pass:
                logger.info(f"PCC value: {output_pcc}")
            else:
                logger.warning(f"PCC value {output_pcc} is lower than {pcc}")
                all_tests_pass = False

            mae = torch.mean(torch.abs(pytorch_out[i] - tt_out[i]))
            logger.info(f"MAE: {mae}")

            max_incorrect = torch.max(torch.abs(pytorch_out[i] - tt_out[i]))
            logger.info(f"Max incorrect: {max_incorrect}")

            max_gt = torch.max(torch.abs(pytorch_out[i]))
            logger.info(f"Max ground truth: {max_gt}")

    if all_tests_pass:
        logger.info("Llama QKV output Passed!")
    else:
        logger.warning("Llama QKV output Failed!")
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
        (
            "llama-2-70B",
            8,
            1,
            0.98,
        ),
    ),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))
def test_LlamaKVUpdate_inference(
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

    run_test_LlamaKVUpdate(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
