# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import transformers

from models.demos.bert.reference import torch_bert
from models.demos.bert.tt import ttnn_optimized_bert

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@pytest.mark.parametrize("model_name", ["phiyodr/bert-large-finetuned-squad2"])
@pytest.mark.parametrize("per_device_batch_size", [5])
@pytest.mark.parametrize("sequence_size", [384])
@pytest.mark.parametrize(
    "device_mesh",
    [
        8,
    ],
    indirect=True,
)
def test_bert_for_question_answering(device_mesh, model_name, per_device_batch_size, sequence_size):
    batch_size = per_device_batch_size * device_mesh.get_num_devices()
    torch.manual_seed(1234)

    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2

    # TODO(arakhmati): re-enable the line below once the issue with ttnn.embedding is fixed
    # torch_bert_input = torch.randint(0, config.config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_bert_input = torch.randint(0, 1, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(1, sequence_size)

    torch_parameters = preprocess_model_parameters(
        model_name=f"torch_{model_name}",
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            model_name, config=config
        ).eval(),
        convert_to_ttnn=lambda *_: False,
    )

    torch_output = torch_bert.bert_for_question_answering(
        config,
        torch_bert_input,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        parameters=torch_parameters,
    )

    tt_model_name = f"ttnn_{model_name}_optimized"

    with ttnn.default_mesh_mapper(ttnn.ReplicateTensorToMesh(device_mesh)):
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
                model_name,
                config=config,
            ).eval(),
            custom_preprocessor=ttnn_optimized_bert.custom_preprocessor,
            device=device_mesh,
        )

    with ttnn.default_mesh_mapper(ttnn.ShardTensorToMesh(device_mesh, dim=0)):
        ttnn_bert_inputs = ttnn_optimized_bert.preprocess_inputs(
            torch_bert_input,
            torch_token_type_ids,
            torch_position_ids,
            torch_attention_mask,
            device=device_mesh,
        )

    tt_output = ttnn_optimized_bert.bert_for_question_answering(
        config,
        *ttnn_bert_inputs,
        parameters=parameters,
    )
    with ttnn.default_mesh_composer(ttnn.ConcatMeshToTensor(device_mesh, dim=0)):
        tt_output = ttnn.to_torch(tt_output)

    tt_output = tt_output[..., :2]

    assert_with_pcc(torch_output, tt_output, 0.9999)
