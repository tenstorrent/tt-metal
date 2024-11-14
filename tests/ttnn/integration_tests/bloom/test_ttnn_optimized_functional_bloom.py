# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
from transformers.models import bloom

from models.demos.grayskull.functional_bloom.tt import ttnn_functional_bloom, ttnn_optimized_functional_bloom
from models.utility_functions import is_wormhole_b0, is_blackhole
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


def torch_random(shape, low, high, dtype):
    if dtype in {torch.bool, torch.int64}:
        return torch.randint(low, high, shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype).uniform_(low, high)


@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_attention(device, model_name, batch_size, sequence_size, reset_seeds):
    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.float32)
    input_embeds = torch.randn((1, 384, 1024), dtype=torch.bfloat16)

    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask=torch_attention_mask,
        input_shape=(1, 384),
        inputs_embeds=input_embeds,
        past_key_values_length=0,
    )
    causal_mask_torch = causal_mask.bool()

    torch_output, *_ = model(torch_hidden_states, torch_residual, torch_alibi, causal_mask_torch)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
    )
    causal_mask = ttnn.from_torch(causal_mask, dtype=ttnn.bfloat16)
    causal_mask = ttnn.to_layout(causal_mask, ttnn.TILE_LAYOUT)
    causal_mask = ttnn.to_device(causal_mask, device)

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_attention_mask.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    alibi = ttnn_optimized_functional_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch.bfloat16
    )
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_functional_bloom.bloom_attention(
        config,
        hidden_states,
        residual,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9988780065080113)


@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_mlp(device, model_name, batch_size, sequence_size, reset_seeds):
    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomMLP(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)

    torch_output = model(torch_hidden_states, torch_residual)

    parameters = preprocess_model_parameters(initialize_model=lambda: model, device=device)

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_functional_bloom.bloom_mlp(
        hidden_states,
        residual,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9972341722786722)


@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_block(device, model_name, batch_size, sequence_size, reset_seeds):
    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomBlock(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.float32)
    input_embeds = torch.randn((1, 384, 1024), dtype=torch.bfloat16)

    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask=torch_attention_mask,
        input_shape=(1, 384),
        inputs_embeds=input_embeds,
        past_key_values_length=0,
    )
    causal_mask_torch = causal_mask.bool()

    torch_output, *_ = model(
        torch_hidden_states,
        torch_alibi,
        causal_mask_torch,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
    )
    causal_mask = ttnn.from_torch(causal_mask, dtype=ttnn.bfloat16)
    causal_mask = ttnn.to_layout(causal_mask, ttnn.TILE_LAYOUT)
    causal_mask = ttnn.to_device(causal_mask, device)

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    alibi = ttnn_optimized_functional_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch.bfloat16
    )
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_functional_bloom.bloom_block(
        config,
        hidden_states,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.9516447916123273)


@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom(device, model_name, batch_size, sequence_size, reset_seeds):
    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomModel.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_output = model(input_ids=torch_input_ids, attention_mask=torch_attention_mask).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
    )

    padded_input_ids, alibi, causal_mask = ttnn_optimized_functional_bloom.preprocess_inputs(
        input_ids=torch_input_ids,
        attention_mask=torch_attention_mask,
        num_heads=config.n_head,
        device=device,
    )

    output = ttnn_optimized_functional_bloom.bloom(
        config,
        padded_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=1)


@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_for_question_answering(device, model_name, batch_size, sequence_size, reset_seeds):
    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    config.position_embedding_type = "none"
    model = bloom.modeling_bloom.BloomForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_output = model(input_ids=torch_input_ids, attention_mask=torch_attention_mask)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
    )
    padded_input_ids, alibi, causal_mask = ttnn_optimized_functional_bloom.preprocess_inputs(
        input_ids=torch_input_ids,
        attention_mask=torch_attention_mask,
        num_heads=config.n_head,
        device=device,
    )

    output = ttnn_optimized_functional_bloom.bloom_for_question_answering(
        config,
        padded_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    start_logits, end_logits = output.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    assert_with_pcc(torch_output.start_logits, start_logits, 0.8242853701503637)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.8860940511364798)


@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_for_causal_lm(device, model_name, batch_size, sequence_size, reset_seeds):
    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    config.position_embedding_type = "none"
    model = bloom.modeling_bloom.BloomForCausalLM.from_pretrained(model_name, config=config).eval()

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_output = model(input_ids=torch_input_ids, attention_mask=None)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
        convert_to_ttnn=lambda model, name: name != "lm_head",
    )
    padded_input_ids, alibi, causal_mask = ttnn_optimized_functional_bloom.preprocess_inputs(
        input_ids=torch_input_ids,
        attention_mask=None,
        num_heads=config.n_head,
        device=device,
    )
    output = ttnn_optimized_functional_bloom.bloom_for_causal_lm(
        config,
        padded_input_ids,
        alibi,
        causal_mask,
        parameters=parameters,
    )

    assert_with_pcc(torch_output.logits, output, 0.9160810251128927)  # test is killed for bs>1
