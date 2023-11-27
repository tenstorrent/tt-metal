# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn.functional as F

import ttnn

from models.experimental.functional_bert.reference.torch_functional_bert import torch_feedforward as torch_model
from models.experimental.functional_bert.tt.ttnn_functional_bert import ttnn_feedforward as ttnn_model
from models.utility_functions import torch_random

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


# Note that our reshape requires the width and height to both be multiples of 32
# so the number of heads must be 32
@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [2 * 32])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [32])
def test_feedforward(device, use_program_cache, batch_size, sequence_size, num_heads, head_size):
    torch.manual_seed(0)

    hidden_size = num_heads * head_size
    intermediate_size = hidden_size * 4

    torch_hidden_states = torch_random((batch_size, sequence_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)

    torch_ff1_weight = torch_random((hidden_size, intermediate_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff1_bias = torch_random((intermediate_size,), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff2_weight = torch_random((intermediate_size, hidden_size), -0.1, 0.1, dtype=torch.bfloat16)
    torch_ff2_bias = torch_random((hidden_size,), -0.1, 0.1, dtype=torch.bfloat16)

    torch_output = torch_model(
        torch_hidden_states,
        torch_ff1_weight,
        torch_ff1_bias,
        torch_ff2_weight,
        torch_ff2_bias,
    )

    assert torch_output.shape == (
        batch_size,
        sequence_size,
        hidden_size,
    ), f"Expected output shape to be {batch_size, sequence_size, hidden_size}, got {torch_output.shape}"

    hidden_states = ttnn.from_torch(torch_hidden_states)
    ff1_weight = ttnn.from_torch(torch_ff1_weight)
    ff1_bias = ttnn.from_torch(torch_ff1_bias)
    ff2_weight = ttnn.from_torch(torch_ff2_weight)
    ff2_bias = ttnn.from_torch(torch_ff2_bias)

    hidden_states = ttnn.to_device(hidden_states, device)
    ff1_weight = ttnn.to_device(ff1_weight, device)
    ff1_bias = ttnn.to_device(ff1_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)
    ff2_weight = ttnn.to_device(ff2_weight, device)
    ff2_bias = ttnn.to_device(ff2_bias, device, memory_config=ttnn.L1_MEMORY_CONFIG)

    tt_output = ttnn_model(
        hidden_states,
        ff1_weight,
        ff1_bias,
        ff2_weight,
        ff2_bias,
    )

    assert tt_output.shape == [
        batch_size,
        sequence_size,
        hidden_size,
    ], f"Expected output shape to be {batch_size, sequence_size, hidden_size}, got {tt_output.shape}"

    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.999)
