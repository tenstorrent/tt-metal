# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.bge_large_en.common import load_torch_model
from models.demos.bge_large_en.ttnn.ttnn_bge_output import TtnnBGEOutput
from models.demos.sentence_bert.reference.sentence_bert import BertOutput
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "inputs",
    [
        ["BAAI/bge-large-en-v1.5", [8, 384, 4096], [8, 384, 1024]],
        ["BAAI/bge-large-en-v1.5", [8, 512, 4096], [8, 512, 1024]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": BGE_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_bge_output(device, inputs, model_location_generator):
    """Test BGE output (FFN second part) layer PCC."""
    target_prefix = f"encoder.layer.{0}.output."

    config = transformers.BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)  # [batch, seq, intermediate_size=4096]
    input_tensor = torch.randn(inputs[2], dtype=torch.bfloat16)  # [batch, seq, hidden_size=1024]
    reference_module = BertOutput(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix=target_prefix, model_location_generator=model_location_generator
    )
    reference_out = reference_module(hidden_states, input_tensor)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnBGEOutput(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_input_tensor = ttnn.from_torch(
        input_tensor.unsqueeze(dim=1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    sharded_hidden_states = ttnn.to_memory_config(
        ttnn_hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_hidden_states.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),  # BGE uses (8, 8) grid
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    sharded_input_tens = ttnn.to_memory_config(
        ttnn_input_tensor,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_input_tensor.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),  # BGE uses (8, 8) grid
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ttnn_out = ttnn_module(sharded_hidden_states, sharded_input_tens)
    ttnn_out = ttnn.to_torch(ttnn_out).squeeze(dim=1)
    assert_with_pcc(reference_out, ttnn_out, 0.99)
