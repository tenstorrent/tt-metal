# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import transformers
import pytest
from models.experimental.functional_sentence_bert.reference.sentence_bert import (
    BertEmbeddings,
    BertOutput,
    BertIntermediate,
    BertSelfOutput,
    BertPooler,
    BertSdpaSelfAttention,
    BertAttention,
    BertLayer,
    BertEncoder,
    BertModel,
    custom_extended_mask,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_sentence_bert.ttnn.ttnn_sentence_bert import (
    ttnn_BertEmbeddings,
    ttnn_BertOutput,
    ttnn_BertIntermediate,
    ttnn_BertSelfOutput,
    ttnn_BertPooler,
    ttnn_BertSelfAttention,
    ttnn_BertAttention,
    ttnn_BertLayer,
    ttnn_BertEncoder,
    ttnn_BertModel,
    preprocess_inputs,
)
from transformers import BertConfig
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 8]],
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_Bert_Embeddings(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).embeddings.eval()
    config = BertConfig.from_pretrained(inputs[0])
    input_ids = torch.randint(low=0, high=config.vocab_size - 1, size=inputs[1], dtype=torch.int64)
    attention_mask = torch.randint(low=0, high=inputs[1][1], size=inputs[1], dtype=torch.int64)
    token_type_ids = torch.zeros(inputs[1], dtype=torch.int64)
    position_ids = torch.arange(0, inputs[1][1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertEmbeddings(config)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertEmbeddings(parameters)
    ttnn_input_ids, ttnn_token_type_ids, ttnn_position_ids, _ = preprocess_inputs(
        input_ids, token_type_ids, position_ids, attention_mask, device
    )
    ttnn_out = ttnn_module(
        input_ids=ttnn_input_ids, token_type_ids=ttnn_token_type_ids, position_ids=ttnn_position_ids, device=device
    )
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(reference_out, ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 8, 3072], [2, 8, 768]],
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 3072], [2, 32, 768]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_output(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.layer[0].output.eval()
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    input_tensor = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertOutput(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(hidden_states, input_tensor)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertOutput(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(ttnn_hidden_states, ttnn_input_tensor)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(reference_out, ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 8, 768]],
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 768]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_intermediate(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.layer[0].intermediate.eval()
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    reference_module = BertIntermediate(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(hidden_states)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertIntermediate(parameters=parameters)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(ttnn_hidden_states)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(reference_out, ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 8, 768], [2, 8, 768]],
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 768], [2, 32, 768]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_self_output(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.layer[0].attention.output.eval()
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    input_tensor = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertSelfOutput(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(hidden_states, input_tensor)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertSelfOutput(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_tensor = ttnn.from_torch(input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(ttnn_hidden_states, ttnn_input_tensor)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(reference_out, ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 8, 768]],
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 768]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_pooler(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).pooler.eval()
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    reference_module = BertPooler(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(hidden_states)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertPooler(parameters=parameters)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(ttnn_hidden_states)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(reference_out, ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 768], [2, 1, 32, 32]]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_self_attention(device, inputs):  # 0.86-real_wts
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.layer[0].attention.self
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertSdpaSelfAttention(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(
        hidden_states,
        attention_mask,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertSelfAttention(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attention_mask = ttnn.from_torch(attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(
        ttnn_hidden_states,
        ttnn_attention_mask,
        device=device,
    )
    ttnn_out = ttnn.to_torch(ttnn_out[0])
    assert_with_pcc(reference_out[0], ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 768], [2, 1, 32, 32]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_attention(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.layer[0].attention.eval()
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertAttention(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(hidden_states, attention_mask)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertAttention(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attention_mask = ttnn.from_torch(attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(
        ttnn_hidden_states,
        ttnn_attention_mask,
        device=device,
    )
    ttnn_out = ttnn.to_torch(ttnn_out[0])
    assert_with_pcc(reference_out[0], ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 768], [2, 1, 32, 32]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_layer(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.layer[0].eval()
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertLayer(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(
        hidden_states,
        attention_mask,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertLayer(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attention_mask = ttnn.from_torch(attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(
        ttnn_hidden_states,
        ttnn_attention_mask,
        device=device,
    )
    ttnn_out = ttnn.to_torch(ttnn_out[0])
    assert_with_pcc(reference_out[0], ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32, 768], [2, 1, 32, 32]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_encoder(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.eval()
    config = BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertEncoder(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(
        hidden_states,
        attention_mask,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertEncoder(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(hidden_states, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_attention_mask = ttnn.from_torch(attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_out = ttnn_module(
        ttnn_hidden_states,
        ttnn_attention_mask,
        device=device,
    )
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(reference_out.last_hidden_state, ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [2, 32], [2, 1, 32, 32]]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_model(device, inputs):  #
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).eval()
    config = BertConfig.from_pretrained(inputs[0])
    input_ids = torch.randint(low=0, high=config.vocab_size - 1, size=inputs[1], dtype=torch.int64)
    attention_mask = torch.randint(0, inputs[1][0], size=inputs[1], dtype=torch.int64)
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = torch.zeros(inputs[1], dtype=torch.int64)
    position_ids = torch.arange(0, inputs[1][1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(
        input_ids, attention_mask=extended_mask, token_type_ids=token_type_ids, position_ids=position_ids
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertModel(parameters=parameters, config=config)
    ttnn_input_ids, ttnn_token_type_ids, ttnn_position_ids, ttnn_attention_mask = preprocess_inputs(
        input_ids, token_type_ids, position_ids, extended_mask, device
    )
    ttnn_out = ttnn_module(ttnn_input_ids, ttnn_attention_mask, ttnn_token_type_ids, ttnn_position_ids, device=device)
    ttnn_out = ttnn.to_torch(ttnn_out[0])
    assert_with_pcc(reference_out.last_hidden_state, ttnn_out, 1.0)


@pytest.mark.parametrize(
    "inputs",
    [
        [
            "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            [
                "Bugün hava çok güzel, güneş parlıyor ve insanlar dışarıda yürüyüş yaparak, doğanın tadını çıkarıyorlar, parklarda çocuklar oynuyor, insanlar kahve içiyor.",
                "Yarın tatilde olacağım, ailemle birlikte şehir dışında birkaç gün geçireceğiz, doğa yürüyüşleri yapacağız, yeni yerler keşfedeceğiz, eğlenceli bir tatil olacak.",
            ],
        ]
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_model_real_inputs(device, inputs):  #
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).eval()
    config = BertConfig.from_pretrained(inputs[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(inputs[0])
    encoded_input = tokenizer(inputs[1], padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = encoded_input["token_type_ids"]
    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(
        input_ids, attention_mask=extended_mask, token_type_ids=token_type_ids, position_ids=position_ids
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertModel(parameters=parameters, config=config)
    ttnn_input_ids, ttnn_token_type_ids, ttnn_position_ids, ttnn_attention_mask = preprocess_inputs(
        input_ids, token_type_ids, position_ids, extended_mask, device
    )
    ttnn_out = ttnn_module(ttnn_input_ids, ttnn_attention_mask, ttnn_token_type_ids, ttnn_position_ids, device=device)
    ttnn_out = ttnn.to_torch(ttnn_out[0])
    assert_with_pcc(reference_out.last_hidden_state, ttnn_out, 1.0)
