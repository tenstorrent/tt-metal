# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import ttnn
from transformers.models import bloom

from models.demos.grayskull.functional_bloom.tt import ttnn_optimized_functional_bloom
from models.utility_functions import is_wormhole_b0, skip_for_grayskull, is_blackhole
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_random(shape, low, high, dtype):
    if dtype in {torch.bool, torch.int64}:
        return torch.randint(low, high, shape, dtype=dtype)
    return torch.zeros(shape, dtype=dtype).uniform_(low, high)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@skip_for_grayskull(reason_str="#10797: OOM")
@pytest.mark.parametrize("model_name", ["bigscience/bloom-560m"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_bloom_attention(device, model_name, batch_size, sequence_size):
    torch.manual_seed(0)

    config = bloom.configuration_bloom.BloomConfig.from_pretrained(model_name)
    model = bloom.modeling_bloom.BloomAttention(config).eval()

    torch_hidden_states = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_residual = torch_random((batch_size, sequence_size, config.hidden_size), -0.1, 0.1, dtype=torch.float32)
    torch_attention_mask = torch_random((batch_size, sequence_size), 0, 2, dtype=torch.bool)
    torch_alibi = bloom.modeling_bloom.build_alibi_tensor(torch_attention_mask, config.n_head, dtype=torch.float32)

    torch_causal_mask = torch.empty((sequence_size, sequence_size), dtype=torch.bool)
    torch_seq_ids = torch.arange(sequence_size)
    torch_causal_mask[:, 0:] = torch_seq_ids[:, None] < torch_seq_ids[None, :]
    torch_causal_mask = torch_causal_mask[None, None, :, :].expand(
        batch_size, config.n_head, sequence_size, sequence_size
    )

    torch_output, *_ = model(
        torch_hidden_states,
        torch_residual,
        torch_alibi,
        torch_causal_mask,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        custom_preprocessor=ttnn_optimized_functional_bloom.custom_preprocessor,
    )

    hidden_states = ttnn.from_torch(torch_hidden_states.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    attention_mask = ttnn.from_torch(torch_causal_mask.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    alibi = ttnn_optimized_functional_bloom.build_alibi_tensor(
        torch_attention_mask, config.n_head, dtype=torch.bfloat16
    )
    alibi = ttnn.from_torch(alibi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_optimized_functional_bloom.bloom_attention(
        config,
        hidden_states,
        residual,
        alibi,
        attention_mask,
        parameters=parameters,
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, pcc=0.993)
