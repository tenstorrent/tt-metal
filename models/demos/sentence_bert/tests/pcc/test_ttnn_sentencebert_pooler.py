# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.sentence_bert.common import load_torch_model
from models.demos.sentence_bert.reference.sentence_bert import BertPooler
from models.demos.sentence_bert.ttnn.common import custom_preprocessor
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_pooler import TtnnSentenceBertPooler
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "inputs",
    [
        ["emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", [8, 384, 768]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_ttnn_sentence_bert_pooler(device, inputs, model_location_generator):
    target_prefix = f"pooler."

    config = transformers.BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    reference_module = BertPooler(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix=target_prefix, model_location_generator=model_location_generator
    )
    reference_out = reference_module(hidden_states)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnSentenceBertPooler(parameters=parameters)
    ttnn_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_out = ttnn_module(ttnn_hidden_states)
    ttnn_out = ttnn.to_torch(ttnn_out)
    assert_with_pcc(reference_out, ttnn_out, 0.99)
