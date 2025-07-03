# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import transformers
import pytest
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.sentence_bert.ttnn.common import custom_preprocessor
from models.demos.sentence_bert.reference.sentence_bert import BertSelfOutput
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_self_output import TtnnSentenceBertSelfOutput


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [8, 384, 768], [8, 384, 768]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_self_output(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).encoder.layer[0].attention.output.eval()
    config = transformers.BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    input_tensor = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertSelfOutput(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(hidden_states, input_tensor)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnSentenceBertSelfOutput(parameters=parameters, config=config)
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
            core_grid=ttnn.CoreGrid(y=8, x=6),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    sharded_input_tens = ttnn.to_memory_config(
        ttnn_input_tensor,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_hidden_states.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ttnn_out = ttnn_module(sharded_hidden_states, sharded_input_tens)
    ttnn_out = ttnn.to_torch(ttnn_out).squeeze(dim=1)
    assert_with_pcc(reference_out, ttnn_out, 0.99)
