# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import tt_lib
import transformers

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import RobertaForQuestionAnswering, RobertaConfig
from models.demos.wormhole.roberta.tt import ttnn_optimized_roberta
from models.utility_functions import skip_for_wormhole_b0, is_grayskull, run_for_wormhole_b0, is_wormhole_b0


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


@run_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["deepset/roberta-large-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_roberta(mesh_device, use_program_cache, reset_seeds, model_name, batch_size, sequence_size):
    config = transformers.RobertaConfig.from_pretrained(model_name)
    model = transformers.RobertaModel.from_pretrained(model_name)
    config.use_dram = True

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = batch_size * 2 if mesh_device_flag else batch_size

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(batch_size, sequence_size)
    torch_position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=config.pad_token_id)
    torch_output = model(
        input_ids=input_ids,
        attention_mask=torch_attention_mask,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
    )
    torch_output = torch_output.last_hidden_state
    tt_model_name = f"ttnn_{model_name}_optimized"

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: transformers.RobertaModel.from_pretrained(model_name, torchscript=False),
            custom_preprocessor=ttnn_optimized_roberta.custom_preprocessor,
            device=mesh_device,
        )

    ttnn_roberta_inputs = ttnn_optimized_roberta.preprocess_inputs(
        input_ids,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        device=mesh_device,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )

    tt_output = ttnn_optimized_roberta.bert(
        config,
        *ttnn_roberta_inputs,
        parameters=parameters,
    )

    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    assert_with_pcc(torch_output, tt_output, 0.94)


@run_for_wormhole_b0()
@pytest.mark.parametrize("model_name", ["deepset/roberta-large-squad2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [384])
def test_roberta_for_question_answering(
    mesh_device, use_program_cache, reset_seeds, model_name, batch_size, sequence_size
):
    config = RobertaConfig.from_pretrained(model_name)
    model = RobertaForQuestionAnswering.from_pretrained(model_name)
    config.use_dram = True

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = batch_size * 2 if mesh_device_flag else batch_size

    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(batch_size, sequence_size)
    torch_position_ids = create_position_ids_from_input_ids(input_ids=input_ids, padding_idx=config.pad_token_id)
    torch_output = model(
        input_ids=input_ids,
        attention_mask=torch_attention_mask,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
    )
    torch_output_start_logits = torch_output.start_logits
    torch_output_end_logits = torch_output.end_logits

    tt_model_name = f"ttnn_{model_name}_optimized"
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            model_name=tt_model_name,
            initialize_model=lambda: transformers.RobertaForQuestionAnswering.from_pretrained(
                model_name, torchscript=False
            ),
            custom_preprocessor=ttnn_optimized_roberta.custom_preprocessor,
            device=mesh_device,
        )

    ttnn_roberta_inputs = ttnn_optimized_roberta.preprocess_inputs(
        input_ids,
        torch_token_type_ids,
        torch_position_ids,
        torch_attention_mask,
        device=mesh_device,
        inputs_mesh_mapper=inputs_mesh_mapper,
    )

    tt_output = ttnn_optimized_roberta.bert_for_question_answering(
        config,
        *ttnn_roberta_inputs,
        parameters=parameters,
        name="roberta",
    )
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    tt_output_start_logits = tt_output[..., :, 0]
    tt_output_end_logits = tt_output[..., :, 1]

    assert_with_pcc(torch_output_start_logits, tt_output_start_logits, 0.89)
    assert_with_pcc(torch_output_end_logits, tt_output_end_logits, 0.88)
