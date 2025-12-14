# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.bge_large_en.common import load_torch_model
from models.demos.bge_large_en.ttnn.ttnn_bge_self_attention import TtnnBGESelfAttention
from models.demos.sentence_bert.reference.sentence_bert import BertSelfAttention
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, BGE_SEQ_LENGTH, custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "inputs",
    [["BAAI/bge-large-en-v1.5", [8, BGE_SEQ_LENGTH, 1024], [8, 1, 1, BGE_SEQ_LENGTH]]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": BGE_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_bge_self_attention(device, inputs, model_location_generator):
    """Test BGE self-attention layer PCC."""
    target_prefix = f"encoder.layer.{0}.attention.self"

    config = transformers.BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertSelfAttention(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix=target_prefix, model_location_generator=model_location_generator
    )
    reference_out = reference_module(
        hidden_states,
        attention_mask,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnBGESelfAttention(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    sharded_input = ttnn.to_memory_config(
        ttnn_hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_hidden_states.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),  # BGE uses (8, 8) grid
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ttnn_attention_mask = ttnn.from_torch(
        attention_mask,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_out = ttnn_module(
        sharded_input,
        ttnn_attention_mask,
        device=device,
    )
    ttnn_out = ttnn.to_torch(ttnn_out).squeeze(dim=1)
    assert_with_pcc(reference_out[0], ttnn_out, 0.99)
