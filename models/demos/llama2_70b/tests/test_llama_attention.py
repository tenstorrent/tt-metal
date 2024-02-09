# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib

from models.demos.llama2_70b.reference.llama import Llama
from models.demos.llama2_70b.reference.llama.model import precompute_freqs_cis

# from models.demos.llama2_70b.tt.llama_attention import TtLlamaAttention
from models.demos.falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchLlamaAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.layers[layer_num].attention

        # Disable dropout
        self.attention.eval()

        configuration = hf_reference_model.params
        self.n_heads = configuration.n_heads
        hidden_dim = configuration.dim
        self.head_dim = hidden_dim // self.n_heads
        self.max_seq_len = configuration.max_seq_len

    def prepare_inputs(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        """
        batch = x.size(0)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.max_seq_len * 2)
        freqs_cis = freqs_cis[start_pos : start_pos + 1]

        attn_mask = torch.zeros(batch, 1, 1, start_pos + 1)
        attn_mask[:, :, :, : start_pos + 1] = -1e9
        attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)

        return x, start_pos, freqs_cis, attn_mask

    def forward(self, x, start_pos, freqs_cis, mask):
        """
        x: (batch, seq, hidden_dim)
        start_pos: int
        freqs_cis: ?
        mask: ?

        return: (batch, seq, hidden_dim)
        """
        result = self.attention(
            x,
            start_pos,
            freqs_cis,
            mask,
        )
        return result


def run_test_LlamaAttention_inference(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    pcc,
    model_config,
    # tt_cache_path,
    # model_location_generator,
):
    # model_name = model_location_generator(model_version, model_subdir="Falcon")

    ckpt_dir = "/proj_sw/user_dev/llama-data-repacked/llama-2-70b/"
    tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"
    max_seq_len = 4096
    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, max_seq_len=max_seq_len, max_batch_size=batch, n_layers=1, skip_model_load=False
    ).model
    hugging_face_reference_model.eval()
    state_dict = hugging_face_reference_model.state_dict()
    print(state_dict.keys())

    # Prepare input
    torch.manual_seed(0)
    layer_num = 0

    base_url = "layers"

    # max_position_embeddings = 2048
    configuration = hugging_face_reference_model.params
    n_heads = configuration.n_heads
    n_kv_heads = configuration.n_kv_heads
    hidden_dim = configuration.dim
    head_dim = hidden_dim // n_heads

    pt_attention_input = (torch.rand(batch, seq_len, configuration.dim) * 2) - 1
    tt_attention_input = pt_attention_input.clone()

    start_pos = 127

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaAttention_model = PytorchLlamaAttentionModel(hugging_face_reference_model, layer_num)
    attention_input, start_pos, freqs_cis, attn_mask = pytorch_LlamaAttention_model.prepare_inputs(
        pt_attention_input, start_pos
    )

    pytorch_out = pytorch_LlamaAttention_model(
        attention_input,
        start_pos,
        freqs_cis,
        attn_mask,
    )

    # TT hardware execution -------------------------------------------------------------
    # tt_FalconAttention_model = TtFalconAttention(
    #     device,
    #     state_dict,
    #     # None,
    #     base_url,
    #     layer_num,
    #     configuration.hidden_size,
    #     configuration.n_head,
    #     # 4544,
    #     # 71,
    #     max_position_embeddings,
    #     model_config,
    #     tt_cache_path,
    # )

    # tt_out, tt_layer_present = tt_FalconAttention_model(
    #     tt_attention_input,
    #     alibi=None,
    #     attention_mask=tt_attention_mask,
    #     llm_mode=llm_mode,
    #     user_id=user_id,
    #     layer_past=tt_layer_past,
    #     layer_past_len=kv_cache_len,
    #     use_cache=use_cache,
    # )
    # tt_out = tt2torch_tensor(tt_out).squeeze(1)
    # tt_layer_present = (
    #     tt2torch_tensor(tt_layer_present[0]).squeeze(1),
    #     tt2torch_tensor(tt_layer_present[1]).squeeze(1),
    # )

    # if llm_mode == "decode":
    #     tt_out = tt_out.transpose(0, 1)
    # tt_layer_present = (
    #     tt_layer_present[0][:, :kv_len, :],
    #     tt_layer_present[1][:, :kv_len, :],
    # )

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, pytorch_out, pcc)
    logger.info(f"Output: {output_pcc}")

    if does_pass:
        logger.info("Llama2-70b Attention output Passed!")
    else:
        logger.warning("Llama2-70b Attention output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"

    # does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    # logger.info(f"Output: {output_pcc}")

    # does_pass2, output_pcc = comp_pcc(
    #     pytorch_layer_present[0], tt_layer_present[0], pcc
    # )
    # logger.info(f"K Cache: {output_pcc}")

    # does_pass = does_pass and does_pass2

    # does_pass2, output_pcc = comp_pcc(
    #     pytorch_layer_present[1], tt_layer_present[1], pcc
    # )
    # logger.info(f"V Cache: {output_pcc}")

    # does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Llama2 Attention output Passed!")
    else:
        logger.warning("Llama2 Attention output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128", "decode_batch32"],
)
@pytest.mark.parametrize(
    "model_version, pcc",
    (("llama-2-70B", 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_LlamaAttention_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    pcc,
    model_config_str,
    # model_location_generator,
    device,
):
    model_config = get_model_config(model_config_str)
    # tt_cache_path = get_tt_cache_path(model_version)

    run_test_LlamaAttention_inference(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        pcc,
        model_config,
        # tt_cache_path,
        # model_location_generator,
    )
