# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import transformers
from models.utility_functions import torch_random, is_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.squeezebert.tt import ttnn_functional_squeezebert


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_squeezebert_attention(device, model_name, batch_size, sequence_size, torch_dtype, reset_seeds):
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    model = transformers.models.squeezebert.modeling_squeezebert.SqueezeBertSelfAttention(
        config, cin=config.hidden_size, q_groups=config.q_groups, k_groups=config.k_groups, v_groups=config.v_groups
    ).eval()
    state_dict = model.state_dict()
    model = model.to(torch_dtype)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch_dtype)
    torch_hidden_states = torch_hidden_states.permute(0, 2, 1)

    torch_attention_mask = torch.ones(batch_size, sequence_size, dtype=torch_dtype)
    torch_attention_mask = torch_attention_mask[:, None, None, :]

    torch_output = model(torch_hidden_states, attention_mask=torch_attention_mask, output_attentions=False)

    ttnn_attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_squeezebert.squeezebert_attention(
        config,
        hidden_states,
        attention_mask=ttnn_attention_mask,
        state_dict=state_dict,
        base_addr=f"",
        parameters=parameters,
        device=device,
    )

    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output["context_layer"], output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_squeezebert_intermediate(device, model_name, batch_size, sequence_size, torch_dtype, reset_seeds):
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    model = transformers.models.squeezebert.modeling_squeezebert.ConvActivation(
        cin=config.hidden_size, cout=config.intermediate_size, groups=config.intermediate_groups, act=config.hidden_act
    ).eval()
    state_dict = model.state_dict()
    model = model.to(torch_dtype)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch_dtype)
    torch_hidden_states = torch_hidden_states.permute(0, 2, 1)

    torch_output = model(torch_hidden_states)

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_squeezebert.squeezebert_intermediate(
        config=config,
        hidden_states=hidden_states,
        state_dict=state_dict,
        base_addr=f"",
        parameters=parameters,
        device=device,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_squeezebert_output(device, model_name, batch_size, sequence_size, torch_dtype, reset_seeds):
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    model = transformers.models.squeezebert.modeling_squeezebert.ConvDropoutLayerNorm(
        cin=config.intermediate_size,
        cout=config.hidden_size,
        groups=config.output_groups,
        dropout_prob=config.hidden_dropout_prob,
    ).eval()
    state_dict = model.state_dict()
    model = model.to(torch_dtype)

    torch_hidden_states = torch_random(
        (batch_size, sequence_size, config.intermediate_size), -0.1, 0.1, dtype=torch_dtype
    )
    torch_hidden_states = torch_hidden_states.permute(0, 2, 1)

    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch_dtype)
    torch_residual = torch_residual.permute(0, 2, 1)

    torch_output = model(torch_hidden_states, torch_residual)

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_squeezebert.squeezebert_conv_layernorm(
        config=config,
        hidden_states=hidden_states,
        input_tensor=residual,
        state_dict=state_dict,
        base_addr=f"",
        parameters=parameters,
        device=device,
        cin=config.intermediate_size,
        cout=config.hidden_size,
        groups=config.output_groups,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output.to(torch_output.dtype), 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_squeezebert_layer(device, model_name, batch_size, sequence_size, torch_dtype, reset_seeds):
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    model = transformers.models.squeezebert.modeling_squeezebert.SqueezeBertModule(config).eval()
    state_dict = model.state_dict()
    model = model.to(torch_dtype)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch_dtype)
    torch_hidden_states = torch_hidden_states.permute(0, 2, 1)

    torch_attention_mask = torch.ones(batch_size, sequence_size, dtype=torch_dtype)
    torch_attention_mask = torch_attention_mask[:, None, None, :]

    torch_output = model(torch_hidden_states, attention_mask=torch_attention_mask, output_attentions=False)

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_squeezebert.squeezebert_layer(
        config,
        hidden_states,
        attention_mask=ttnn_attention_mask,
        state_dict=state_dict,
        base_addr=f"",
        parameters=parameters,
        device=device,
    )

    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output["feature_map"], output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("torch_dtype", [torch.bfloat16])
def test_squeezebert_encoder(device, model_name, batch_size, sequence_size, torch_dtype, reset_seeds):
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    model = transformers.models.squeezebert.modeling_squeezebert.SqueezeBertEncoder(config).eval()
    state_dict = model.state_dict()
    model = model.to(torch_dtype)

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch_dtype)
    torch_attention_mask = torch.ones(batch_size, sequence_size, dtype=torch_dtype)
    torch_attention_mask = torch_attention_mask[:, None, None, :]

    torch_output = model(torch_hidden_states, attention_mask=torch_attention_mask).last_hidden_state

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
        device=device,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attention_mask = ttnn.from_torch(torch_attention_mask, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_functional_squeezebert.squeezebert_encoder(
        config,
        hidden_states,
        attention_mask=ttnn_attention_mask,
        state_dict=state_dict,
        base_addr=f"",
        parameters=parameters,
        device=device,
    )

    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_squeezebert_model(device, model_name, batch_size, sequence_size, reset_seeds):
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    model = transformers.SqueezeBertModel.from_pretrained(model_name, config=config).eval()
    state_dict = model.state_dict()
    model = model.to(torch.bfloat16)

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(1, sequence_size, dtype=torch.bfloat16)

    torch_output = model(
        torch_input_ids,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    ).last_hidden_state

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
    )

    ttnn_bert_inputs = ttnn_functional_squeezebert.preprocess_inputs(
        torch_input_ids,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        device=device,
    )

    output = ttnn_functional_squeezebert.squeezebert(
        config,
        *ttnn_bert_inputs,
        state_dict=state_dict,
        base_addr=f"",
        parameters=parameters,
        device=device,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.98 if is_grayskull() else 0.99)


@pytest.mark.parametrize("model_name", ["squeezebert/squeezebert-uncased"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_squeezebert_for_question_answering(device, model_name, batch_size, sequence_size, reset_seeds):
    rf_model = transformers.SqueezeBertForQuestionAnswering.from_pretrained(model_name)
    config = transformers.SqueezeBertConfig.from_pretrained(model_name)
    state_dict = rf_model.state_dict()

    torch_squeezebert_input = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(batch_size, sequence_size)

    torch_output = rf_model(
        input_ids=torch_squeezebert_input,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    )

    tt_model_name = f"ttnn_{model_name}_optimized"

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: rf_model,
        custom_preprocessor=ttnn_functional_squeezebert.custom_preprocessor,
        device=device,
    )

    ttnn_squeezebert_inputs = ttnn_functional_squeezebert.preprocess_inputs(
        torch_squeezebert_input,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        device=device,
    )

    tt_output = ttnn_functional_squeezebert.squeezebert_for_question_answering(
        config,
        *ttnn_squeezebert_inputs,
        state_dict=state_dict,
        base_addr=f"transformer.",
        parameters=parameters,
        device=device,
    )

    tt_output = ttnn.to_torch(tt_output)

    tt_start_logits = tt_output[..., :, 0]
    tt_end_logits = tt_output[..., :, 1]

    assert_with_pcc(torch_output.start_logits, tt_start_logits, 0.83 if is_grayskull() else 0.88)
    assert_with_pcc(torch_output.end_logits, tt_end_logits, 0.84 if is_grayskull() else 0.93)
