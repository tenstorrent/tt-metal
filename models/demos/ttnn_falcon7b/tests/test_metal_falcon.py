# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import transformers
import pytest
from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.ttnn_falcon7b.tt.common import create_custom_preprocessor

from models.demos.ttnn_falcon7b.tt.falcon_causallm import TtFalconCausalLM

from models.demos.ttnn_falcon7b.tt.model_config import (
    get_model_config,
    get_tt_cache_path,
)
from models.demos.ttnn_falcon7b.tt.common import (
    create_custom_preprocessor,
    create_kv_cache,
)

from models.utility_functions import (
    is_e75,
)
import ttnn


def run_test_FalconCausalLM_end_to_end(
    device,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    num_layers,
    pcc,
    model_config,
    tt_cache_path,
    model_location_generator,
    expected_inference_time,
):
    torch.manual_seed(0)

    configuration = transformers.FalconConfig.from_pretrained(model_version)
    configuration.num_hidden_layers = num_layers
    model = transformers.models.falcon.modeling_falcon.FalconForCausalLM.from_pretrained(
        model_version, config=configuration
    ).eval()

    head_dim = configuration.hidden_size // configuration.num_attention_heads
    use_cache = True
    dtype = model_config["DEFAULT_DTYPE"]
    kv_len = seq_len if llm_mode == "prefill" else kv_cache_len + 1

    model_input = torch.arange(seq_len * batch).reshape(batch, seq_len)

    # Generate dummy kv_cache --------------------------------------------------------------
    if llm_mode == "prefill":
        past_key_values = None
        tt_layer_past = ()
        for i in range(num_layers):
            _, tt_current_layer_past = create_kv_cache(llm_mode, dtype, batch, kv_cache_len, configuration, device)
            tt_layer_past += (tt_current_layer_past,)
        attention_mask = None

    elif llm_mode == "decode":
        q_len, kv_len = seq_len, kv_cache_len + 1
        assert batch % 32 == 0, "For decode, batch must be multiple of 32!"
        assert q_len == 1, "For decode, q_len must be 1!"
        attention_mask = torch.ones(batch, 1, seq_len, kv_len, dtype=int)
        attention_mask = attention_mask.triu(diagonal=1)

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
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )

    if llm_mode == "prefill":
        tt_layer_past = ()
        for i in range(num_layers):
            _, tt_current_layer_past = create_kv_cache(llm_mode, dtype, batch, kv_cache_len, configuration, device)
            tt_layer_past += (tt_current_layer_past,)
        attention_mask = None

    elif llm_mode == "decode":
        tt_layer_past = ()
        for i in range(num_layers):
            current_layer_past, tt_current_layer_past = create_kv_cache(
                llm_mode, dtype, batch, kv_cache_len, configuration, device
            )
            tt_layer_past += (tt_current_layer_past,)

    if llm_mode == "prefill":
        model_inputs = torch.split(model_input, 1)
        tt_embeddings, tt_attention_mask = zip(
            *[
                tt_FalconCausalLM.model_preprocessing(llm_mode, m_i, kv_cache_len, num_input_tokens=seq_len)
                for m_i in model_inputs
            ]
        )
    elif llm_mode == "decode":
        tt_embeddings, tt_attention_mask = tt_FalconCausalLM.model_preprocessing(
            llm_mode, model_input, kv_cache_len, num_input_tokens=kv_len
        )

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
                use_cache=use_cache,
            )
            tt_outs.append(tt_out)

    elif llm_mode == "decode":
        tt_out, tt_layer_present = tt_FalconCausalLM(
            input_embeddings=tt_embeddings,
            llm_mode=llm_mode,
            attention_mask=tt_attention_mask,
            layer_past=tt_layer_past,
            layer_past_len=kv_cache_len,
            use_cache=use_cache,
        )


@pytest.mark.parametrize(
    "llm_mode, batch, seq_len, kv_cache_len, expected_inference_time",
    (
        ("prefill", 1, 128, 0, 0.30),
        ("prefill", 1, 256, 0, 0.44),
        ("decode", 32, 1, 128, 0.27),
        ("decode", 32, 1, 1024, 0.35),
        ("decode", 32, 1, 2047, 0.48),
    ),
    ids=[
        "prefill_seq128",
        "prefill_seq256",
        "decode_batch32",
        "decode_batch32_1024",
        "decode_batch32_2047",
    ],
)
@pytest.mark.parametrize(
    "num_layers, pcc",
    ((32, 0.86),),
    ids=["layers_32"],
)
@pytest.mark.parametrize(
    "model_version",
    ("tiiuae/falcon-7b-instruct",),
    ids=["falcon_7b"],
)
@pytest.mark.parametrize("model_config_str", ("BFLOAT16-L1",))
def test_perf_bare_metal(
    device,
    use_program_cache,
    model_version,
    llm_mode,
    batch,
    seq_len,
    kv_cache_len,
    expected_inference_time,
    num_layers,
    pcc,
    request,
    model_config_str,
    model_location_generator,
):
    if is_e75(device) and batch == 32:
        pytest.skip("Falcon batch 32 is not supported on E75")

    ttnn.device.EnableMemoryReports()

    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    run_test_FalconCausalLM_end_to_end(
        device,
        model_version,
        llm_mode,
        batch,
        seq_len,
        kv_cache_len,
        num_layers,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
        expected_inference_time,
    )
