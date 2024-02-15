# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import scipy
from sklearn.metrics import top_k_accuracy_score
import numpy as np

import tt_lib

from models.demos.llama2_70b.reference.llama import Llama

from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.demos.llama2_70b.tt.llama_model import TtLlamaModel


class PytorchLlamaModel(torch.nn.Module):
    def __init__(self, hf_reference_model):
        super().__init__()
        self.model = hf_reference_model

        # Disable dropout
        self.model.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def forward(self, x, start_pos):
        """
        x: (batch, seq)
        start_pos: int

        return: (batch, seq, hidden_dim)
        """
        return self.model(x, start_pos)


def run_test_LlamaModel_inference(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    pcc,
    model_config,
    n_layers,
    n_devices
    # tt_cache_path,
    # model_location_generator,
):
    # model_name = model_location_generator(model_version, model_subdir="Falcon")

    ckpt_dir = "/proj_sw/user_dev/llama-data-repacked/llama-2-70b/"
    tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"
    max_seq_len = 4096
    hugging_face_reference_model = Llama.build(
        ckpt_dir,
        tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=batch,
        n_layers=n_layers,
        skip_model_load=False,
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    print(state_dict.keys())

    # Prepare configs
    devices = [device for _ in range(n_devices)]  # Emulate fracturing on N chips

    torch.manual_seed(0)
    base_url = "layers"
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    # PyTorch model --------------------------------------------------------------------
    pytorch_model = PytorchLlamaModel(hugging_face_reference_model)
    # TT model -------------------------------------------------------------
    tt_model = TtLlamaModel(devices, state_dict, base_url, n_layers, model_config, configuration, batch)

    generation_start_pos = 0
    generation_length = 1
    all_tests_pass = True
    for i in range(generation_length):
        # Prepare input
        pt_inp_ids = torch.randint(0, configuration.vocab_size, (batch, seq_len))
        tt_inp_ids = pt_inp_ids.clone()
        start_pos = generation_start_pos + i

        # TT hardware execution -------------------------------------------------------------
        tt_inp_emb, start_pos, rot_mat, attn_mask = tt_model.prepare_inputs(tt_inp_ids, start_pos)

        tt_out = tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
        )

        # PyTorch output --------------------------------------------------------------------
        pytorch_out = pytorch_model(
            pt_inp_ids,
            start_pos,
        )

        tt_out = tt2torch_tensor(tt_out)

        tt_out = tt_out.permute(2, 1, 0, 3).squeeze()  # [batch, hidden_dim]
        tt_out = tt_out.float()
        pytorch_out = pytorch_out.squeeze()  # [batch, hidden_dim]

        # check outputs ----------------------------------------------------------------------
        does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
        logger.info(f"Output: {output_pcc}")

        kl_divs = scipy.stats.entropy(
            torch.nn.functional.softmax(pytorch_out, dim=-1), torch.nn.functional.softmax(tt_out, dim=-1), axis=-1
        )
        logger.info(f"Mean KL Divergence: {kl_divs.mean()}")

        # Write the code to check top-5 and top-1 accuracy. It should show the
        # percentage where the top-1 prediction in pytorch was in the top-5
        # predictions in tt.
        reference_top1 = np.argmax(pytorch_out, axis=-1)
        top1_acc = top_k_accuracy_score(reference_top1, tt_out, k=1, labels=np.arange(tt_out.shape[-1]))
        top5_acc = top_k_accuracy_score(reference_top1, tt_out, k=5, labels=np.arange(tt_out.shape[-1]))

        logger.info(f"Mean Top-1: {top1_acc}")
        logger.info(f"Mean Top-5: {top5_acc}")

        if does_pass:
            logger.info(f"[start_pos={start_pos}] Llama2-70b Decoder output Passed!")
        else:
            logger.warning(f"[start_pos={start_pos}] Llama2-70b Decoder output Failed! PCC value is lower than {pcc}")
            all_tests_pass = False

    # # Check kv cache
    # # PyTorch output --------------------------------------------------------------------
    # pytorch_layer_present = [
    #     pytorch_LlamaDecoder_model.decoder.attention.cache_k.clone().permute(
    #         0, 2, 1, 3
    #     ),  # [batch, n_kv_heads, seq, head_dim]
    #     pytorch_LlamaDecoder_model.decoder.attention.cache_v.clone().permute(
    #         0, 2, 1, 3
    #     ),  # [batch, n_kv_heads, seq, head_dim]
    # ]
    # # TT hardware execution -------------------------------------------------------------
    # tt_layer_present = []
    # for layer_past in tt_LlamaDecoder_model.attention.layer_past_list:
    #     tt_layer_present.append([tt2torch_tensor(cache) for cache in layer_past])
    # # concat the pasts by heads
    # tt_layer_present = [
    #     torch.cat([tt_cache for tt_cache in tt_cache_head], dim=1) for tt_cache_head in zip(*tt_layer_present)
    # ]

    # for cache_pt, cache_tt in zip(pytorch_layer_present, tt_layer_present):
    #     cache_length_to_check = generation_start_pos + generation_length + 1
    #     cache_pt = cache_pt[:, :, generation_start_pos:cache_length_to_check, :]
    #     cache_tt = cache_tt[:, :, generation_start_pos:cache_length_to_check, :]
    #     does_pass, output_pcc = comp_pcc(cache_pt, cache_tt, pcc)
    #     logger.info(f"Output: {output_pcc}")

    #     if does_pass:
    #         logger.info(f"KV Cache Passed!")
    #     else:
    #         logger.warning(f"KV Cache Failed! PCC value is lower than {pcc}")
    #         all_tests_pass = False

    if all_tests_pass:
        logger.info("Llama2 Decoder output Passed!")
    else:
        logger.warning("Llama2 Decoder output Failed!")
        assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, n_layers, n_devices",
    (("decode", 32, 1, 128, 1, 8),),
    ids=["decode_batch32_layers1_devices8"],
)
@pytest.mark.parametrize(
    "model_version, pcc",
    (("llama-2-70B", 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM",))  # , "BFLOAT8_B-SHARDED"))
def test_LlamaModel_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    pcc,
    model_config_str,
    n_layers,
    n_devices,
    # model_location_generator,
    device,
):
    model_config = get_model_config(model_config_str, num_devices=n_devices)
    # tt_cache_path = get_tt_cache_path(model_version)

    run_test_LlamaModel_inference(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        pcc,
        model_config,
        n_layers,
        n_devices
        # tt_cache_path,
        # model_location_generator,
    )
