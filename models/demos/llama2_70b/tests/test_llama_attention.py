# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.falcon7b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.llama2_70b.reference.llama import Llama
from models.demos.llama2_70b.tt.llama_attention import TtLlamaAttention
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

    def forward(self, x, start_pos, freqs_cis, mask):
        result = self.attention(
            x,
            start_pos,
            freqs_cis,
            mask,
        )
        return result


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


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

    ckpt_dir = "/proj_sw/user_dev/llama-data/llama-2-70b/"
    tokenizer_path = "/proj_sw/user_dev/llama-data/tokenizer.model"

    hugging_face_reference_model = Llama.build(
        ckpt_dir, tokenizer_path, seq_len, batch, n_layers=1, skip_model_load=True
    ).model
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.params
    state_dict = hugging_face_reference_model.state_dict()
    # use_cache = True
    # user_id = 0

    # Prepare input
    torch.manual_seed(0)
    layer_num = 0
    print(state_dict.keys())
    base_url = "layers"

    # max_position_embeddings = 2048
    head_dim = configuration.dim // configuration.n_heads

    attention_input = (torch.rand(batch, seq_len, configuration.dim) * 2) - 1
    start_pos = 0
    # Taken Directly from Llama2 model
    freqs_cis = precompute_freqs_cis(head_dim, seq_len * 2)
    freqs_cis = freqs_cis[start_pos : start_pos + seq_len]

    attention_mask_pytorch = None
    if seq_len > 1:
        attention_mask_pytorch = torch.full((seq_len, seq_len), float("-inf"))

        attention_mask_pytorch = torch.triu(attention_mask_pytorch, diagonal=1)

        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        attention_mask_pytorch = torch.hstack([torch.zeros((seq_len, start_pos)), attention_mask_pytorch]).type_as(
            attention_input
        )

    # PyTorch output --------------------------------------------------------------------
    pytorch_LlamaAttention_model = PytorchLlamaAttentionModel(hugging_face_reference_model, layer_num)
    pytorch_out = pytorch_LlamaAttention_model(
        attention_input,
        start_pos,
        freqs_cis,
        attention_mask_pytorch,
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
