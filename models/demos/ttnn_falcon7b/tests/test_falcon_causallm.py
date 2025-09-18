# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor, create_kv_cache
from models.demos.ttnn_falcon7b.tt.falcon_causallm import TtFalconCausalLM
from models.demos.ttnn_falcon7b.tt.model_config import get_model_config, get_tt_cache_path
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len",
    (
        ("prefill", 1, 128, 0),
        ("decode", 32, 1, 128),
    ),
    ids=["prefill_seq128_batch32", "decode_batch32"],
)
@pytest.mark.parametrize(
    "num_layers, expected_pcc",
    (
        (1, 0.98),
        (2, 0.98),
        (32, 0.60),
    ),
    ids=[
        "layers_1",
        "layers_2",
        "layers_32",
    ],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-DRAM", "BFLOAT16-L1"))
def test_falcon_causal_lm(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    expected_pcc,
    model_config_str,
):
    torch.manual_seed(0)
    configuration = transformers.FalconConfig.from_pretrained(model_version)
    configuration.num_hidden_layers = num_layers
    model = transformers.models.falcon.modeling_falcon.FalconForCausalLM.from_pretrained(
        model_version, config=configuration
    ).eval()
    model_config = get_model_config(model_config_str)
    dtype = model_config["DEFAULT_DTYPE"]
    kv_len = seq_len if llm_mode == "prefill" else kv_cache_len + 1

    model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)

    if llm_mode == "prefill":
        past_key_values = None
        tt_layer_past = ()
        for i in range(num_layers):
            _, tt_current_layer_past = create_kv_cache(llm_mode, dtype, batch, kv_cache_len, configuration, device)
            tt_layer_past += (tt_current_layer_past,)

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"

        past_key_values = ()
        tt_layer_past = ()
        for i in range(num_layers):
            current_layer_past, tt_current_layer_past = create_kv_cache(
                llm_mode, dtype, batch, kv_cache_len, configuration, device
            )
            past_key_values += (current_layer_past,)
            tt_layer_past += (tt_current_layer_past,)

    else:
        raise NotImplementedError(f"Llm mode {llm_mode} is not supported! Must be one of prefill or decode.")

    pytorch_out, pytorch_layer_present = model(
        input_ids=model_input,
        attention_mask=None,  # when attention_mask is None, a causal mask is created under the hood
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=False,
    )

    def convert_to_ttnn(model, name):
        return not isinstance(model, torch.nn.Embedding)

    tt_cache_path = get_tt_cache_path(model_version)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=create_custom_preprocessor(model_config, tt_cache_path=tt_cache_path, device=device),
        convert_to_ttnn=convert_to_ttnn,
    )
    tt_FalconCausalLM = TtFalconCausalLM(
        device,
        num_layers,
        configuration,
        configuration.max_position_embeddings,
        model_config,
        parameters,
    )
    # TODO: Generate embeddings and attention_mask on device
    if llm_mode == "prefill":
        tt_outs = []
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
        for user_id in range(batch):
            tt_out, tt_layer_present = tt_FalconCausalLM(
                input_embeddings=tt_embeddings[user_id],
                llm_mode=llm_mode,
                attention_mask=tt_attention_mask[user_id],
                user_id=user_id,
                layer_past=tt_layer_past,
                layer_past_len=kv_cache_len,
                use_cache=True,
            )
            tt_outs.append(ttnn.to_torch(tt_out).squeeze(1))
        tt_out = torch.vstack(tt_outs)

    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=True,
        )
        tt_out = ttnn.to_torch(tt_out).squeeze(1)
        tt_out = tt_out.transpose(0, 1)

    passed, pcc = assert_with_pcc(pytorch_out, tt_out.to(pytorch_out.dtype), expected_pcc)
    logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")

    for i in range(num_layers):
        tt_layer_pres = (
            ttnn.to_torch(tt_layer_present[i][0]),
            ttnn.to_torch(tt_layer_present[i][1]),
        )
        if llm_mode == "prefill":
            pytorch_layer_pres = pytorch_layer_present[i]
            tt_layer_pres = (
                tt_layer_pres[0][:, :, :kv_len, :],
                tt_layer_pres[1][:, :, :kv_len, :],
            )
        elif llm_mode == "decode":
            pytorch_layer_pres = (
                pytorch_layer_present[i][0][:, :, kv_cache_len, :],
                pytorch_layer_present[i][1][:, :, kv_cache_len, :],
            )
            tt_layer_pres = (
                tt_layer_pres[0][:, :, kv_cache_len, :],
                tt_layer_pres[1][:, :, kv_cache_len, :],
            )

        passed, pcc = assert_with_pcc(
            pytorch_layer_pres[0], tt_layer_pres[0].to(pytorch_layer_pres[0].dtype), expected_pcc
        )
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")
        passed, pcc = assert_with_pcc(
            pytorch_layer_pres[1], tt_layer_pres[1].to(pytorch_layer_pres[1].dtype), expected_pcc
        )
        logger.success(f"Passed: pcc: {pcc}, expected: {expected_pcc}")

    logger.info("Falcon CausalLM Passed!")
