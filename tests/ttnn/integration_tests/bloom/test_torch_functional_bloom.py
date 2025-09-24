# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
from transformers.models import bloom

from models.demos.grayskull.functional_bloom.reference import torch_functional_bloom
from models.common.utility_functions import is_wormhole_b0, is_blackhole
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_random(shape, low, high, dtype):
    if dtype in {torch.bool, torch.int64}:
        return torch.randint(low, high, shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype).uniform_(low, high)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_bloom_gelu():
    torch.manual_seed(0)

    torch_input = torch_random((1, 1, 1), -0.1, 0.1, dtype=torch.float32)
    model = bloom.modeling_bloom.BloomGelu().eval()
    torch_output = model(torch_input)

    output = torch_functional_bloom.bloom_gelu(torch_input)

    assert_with_pcc(torch_output, output, pcc=0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_attention(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch_hidden_states.dtype
    )

    torch_output, *_ = model(
        torch_hidden_states,
        torch_residual,
        torch_alibi,
        torch_attention_mask,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_bloom.custom_preprocessor,
    )

    alibi = torch_functional_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch_hidden_states.dtype
    )

    output = torch_functional_bloom.bloom_attention(
        config,
        torch_hidden_states,
        torch_residual,
        alibi,
        torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, pcc=0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_mlp(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomMLP(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)

    torch_output = model(torch_hidden_states, torch_residual)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )

    output = torch_functional_bloom.bloom_mlp(
        torch_hidden_states,
        torch_residual,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, pcc=0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_block(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomBlock(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch_hidden_states.dtype
    )

    torch_output, *_ = model(
        torch_hidden_states,
        torch_alibi,
        torch_attention_mask,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_bloom.custom_preprocessor,
    )

    alibi = torch_functional_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch_hidden_states.dtype
    )

    output = torch_functional_bloom.bloom_block(
        config,
        torch_hidden_states,
        alibi,
        torch_attention_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, pcc=0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomModel.from_pretrained(model_name, config=config).eval().to(torch.float32)

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_output = model(input_ids=torch_input_ids, attention_mask=torch_attention_mask).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_bloom.custom_preprocessor,
    )

    alibi = torch_functional_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, torch.bfloat16)
    causal_mask = torch_functional_bloom.make_causal_mask(torch_attention_mask, torch_input_ids.shape)

    output = torch_functional_bloom.bloom(
        config,
        torch_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output, output, pcc=0.994)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_for_question_answering(model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    config.position_embedding_type = "none"
    model = bloom.modeling_bloom.BloomForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_output = model(input_ids=torch_input_ids, attention_mask=torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_bloom.custom_preprocessor,
    )

    alibi = torch_functional_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, torch.bfloat16)
    causal_mask = torch_functional_bloom.make_causal_mask(torch_attention_mask, torch_input_ids.shape)

    output = torch_functional_bloom.bloom_for_question_answering(
        config,
        torch_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )

    start_logits = output[..., 0]
    end_logits = output[..., 1]

    assert_with_pcc(torch_output.start_logits, start_logits, 0.985)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.9914)
