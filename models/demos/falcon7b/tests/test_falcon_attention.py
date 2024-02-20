# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import tt_lib
from models.demos.falcon7b.reference.hf_modeling_falcon import (
    FalconForCausalLM,
)
from models.demos.falcon7b.tt.falcon_attention import TtFalconAttention
from models.demos.falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class PytorchFalconAttentionModel(torch.nn.Module):
    def __init__(self, hf_reference_model, layer_num):
        super().__init__()
        self.attention = hf_reference_model.transformer.h[layer_num].self_attention

        # Disable dropout
        self.attention.eval()

    def forward(self, x, alibi, attention_mask, layer_past, use_cache):
        result = self.attention(
            hidden_states=x,
            alibi=alibi,
            attention_mask=attention_mask,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        return result


def run_test_FalconAttention_inference(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
):
    model_name = model_location_generator(model_version, model_subdir="Falcon")

    hugging_face_reference_model = FalconForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    hugging_face_reference_model.eval()
    configuration = hugging_face_reference_model.config
    state_dict = hugging_face_reference_model.state_dict()
    use_cache = True
    user_id = 0

    # Prepare input
    torch.manual_seed(0)
    layer_num = 0
    base_url = "transformer.h"
    max_position_embeddings = 2048
    head_dim = configuration.hidden_size // configuration.num_attention_heads

    # Generate input, attention_mask, and kv_cache --------------------------------------
    # TODO: Generate attention_mask on device
    if llm_mode == "prefill":
        q_len, kv_len = seq_len, seq_len
        assert batch == 1, "For prefill, batch must be 1!"
        assert q_len % 32 == 0, "For prefill, seq_len must be multiple of 32!"
        assert kv_cache_len == 0, "For prefill, no kv_cache is passed in!"

        attention_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1
        attention_mask_bool = torch.ones(batch, 1, q_len, kv_len, dtype=bool).triu(diagonal=1)
        layer_past = None

        tt_attention_input = torch2tt_tensor(attention_input.unsqueeze(1), device)
        tt_attention_mask = torch2tt_tensor(
            (attention_mask_bool * -100000).expand(-1, configuration.num_attention_heads, -1, -1),
            device,
        )
        tt_k_cache = torch.zeros(batch, max_position_embeddings, head_dim)
        tt_v_cache = torch.zeros(batch, max_position_embeddings, head_dim)
        tt_k_cache = torch2tt_tensor(tt_k_cache.unsqueeze(1), device)
        tt_v_cache = torch2tt_tensor(tt_v_cache.unsqueeze(1), device)
        tt_layer_past = (tt_k_cache, tt_v_cache)

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        attention_input = (torch.rand(batch, q_len, configuration.hidden_size) * 2) - 1
        # attention_input = (torch.rand(batch, q_len, 4544) * 2) - 1
        attention_mask_bool = torch.zeros(batch, 1, q_len, kv_len, dtype=bool)
        k_cache = torch.rand(batch, kv_cache_len, head_dim)
        v_cache = torch.rand(batch, kv_cache_len, head_dim)
        layer_past = (k_cache, v_cache)

        tt_attention_input = torch2tt_tensor(attention_input.unsqueeze(1).transpose(0, 2), device)

        kv_len_padded = (kv_len + 31) // 32 * 32
        attention_mask_bool_padded = torch.cat(
            (
                attention_mask_bool,
                torch.ones(batch, 1, q_len, kv_len_padded - kv_len, dtype=bool),
            ),
            dim=-1,
        )
        tt_attention_mask = torch2tt_tensor(
            (attention_mask_bool_padded.transpose(0, 2) * -100000).expand(
                -1,
                configuration.num_attention_heads,
                -1,
                -1
                # -1, 71, -1, -1
            ),
            device,
        )
        tt_k_cache = torch.zeros(batch, max_position_embeddings, head_dim)
        tt_v_cache = torch.zeros(batch, max_position_embeddings, head_dim)
        tt_k_cache[:, :kv_cache_len, :] = k_cache
        tt_v_cache[:, :kv_cache_len, :] = v_cache
        tt_k_cache = torch2tt_tensor(tt_k_cache.unsqueeze(1), device)
        tt_v_cache = torch2tt_tensor(tt_v_cache.unsqueeze(1), device)
        tt_layer_past = (tt_k_cache, tt_v_cache)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    # PyTorch output --------------------------------------------------------------------
    pytorch_FalconAttention_model = PytorchFalconAttentionModel(hugging_face_reference_model, layer_num)
    pytorch_out, pytorch_layer_present = pytorch_FalconAttention_model(
        attention_input,
        alibi=None,
        attention_mask=attention_mask_bool,
        layer_past=layer_past,
        use_cache=use_cache,
    )

    # TT hardware execution -------------------------------------------------------------
    tt_FalconAttention_model = TtFalconAttention(
        device,
        state_dict,
        # None,
        base_url,
        layer_num,
        configuration.hidden_size,
        configuration.num_attention_heads,
        # 4544,
        # 71,
        max_position_embeddings,
        model_config,
        tt_cache_path,
    )

    tt_out, tt_layer_present = tt_FalconAttention_model(
        tt_attention_input,
        alibi=None,
        attention_mask=tt_attention_mask,
        llm_mode=llm_mode,
        user_id=user_id,
        layer_past=tt_layer_past,
        layer_past_len=kv_cache_len,
        use_cache=use_cache,
    )
    tt_out = tt2torch_tensor(tt_out).squeeze(1)
    tt_layer_present = (
        tt2torch_tensor(tt_layer_present[0]).squeeze(1),
        tt2torch_tensor(tt_layer_present[1]).squeeze(1),
    )

    if llm_mode == "decode":
        tt_out = tt_out.transpose(0, 1)
    tt_layer_present = (
        tt_layer_present[0][:, :kv_len, :],
        tt_layer_present[1][:, :kv_len, :],
    )

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output: {output_pcc}")

    does_pass2, output_pcc = comp_pcc(pytorch_layer_present[0], tt_layer_present[0], pcc)
    logger.info(f"K Cache: {output_pcc}")

    does_pass = does_pass and does_pass2

    does_pass2, output_pcc = comp_pcc(pytorch_layer_present[1], tt_layer_present[1], pcc)
    logger.info(f"V Cache: {output_pcc}")

    does_pass = does_pass and does_pass2

    if does_pass:
        logger.info("Falcon Attention output Passed!")
    else:
        logger.warning("Falcon Attention output Failed!")
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
    (("tiiuae/falcon-7b-instruct", 0.98),),
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_FalconAttention_inference(
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    pcc,
    model_config_str,
    model_location_generator,
    device,
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    run_test_FalconAttention_inference(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
