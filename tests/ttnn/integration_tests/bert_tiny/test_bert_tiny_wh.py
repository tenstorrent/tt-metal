# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn

from transformers import BertForQuestionAnswering, BertConfig
from models.demos.wormhole.bert_tiny.tt.bert_tiny import (
    bert_for_question_answering,
    bert_attention,
    bert_intermediate,
    bert_output,
    bert_layer,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull, is_wormhole_b0


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bert_attention_inference(
    model_location_generator,
    mesh_device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    encoder_idx = 0
    pytorch_attention_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx].attention
    config = hugging_face_reference_model.config
    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_attention_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    torch_attention_mask = torch.zeros(1, 128)
    pytorch_out = pytorch_attention_model(input.squeeze(1), attention_mask=torch_attention_mask)[0]

    tt_input = ttnn.from_torch(
        input.squeeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
    )
    tt_attention_mask = ttnn.from_torch(
        torch_attention_mask,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_output = bert_attention(
        config=config,
        hidden_states=tt_input,
        attention_mask=tt_attention_mask,
        parameters=parameters,
        device=mesh_device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    assert_with_pcc(pytorch_out, tt_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bert_intermediate_inference(
    model_location_generator,
    mesh_device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    encoder_idx = 0
    pytorch_intermediate_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx].intermediate

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_intermediate_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_intermediate_model(input).squeeze(0)

    tt_input = ttnn.from_torch(
        input.squeeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )

    tt_output = bert_intermediate(
        hidden_states=tt_input,
        parameters=parameters,
        device=mesh_device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
    assert_with_pcc(pytorch_out.squeeze(1), tt_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bert_output_inference(
    model_location_generator,
    mesh_device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    encoder_idx = 0
    config = hugging_face_reference_model.config
    pytorch_output_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx].attention.output

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_output_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    hidden_state = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_output_model(hidden_state, input).squeeze(0)

    ttnn_input = ttnn.from_torch(
        input.squeeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )
    ttnn_hidden_state = ttnn.from_torch(
        hidden_state.squeeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
    )

    tt_output = bert_output(
        config=config,
        hidden_states=ttnn_hidden_state,
        residual=ttnn_input,
        parameters=parameters,
        device=mesh_device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
    assert_with_pcc(pytorch_out.squeeze(1), tt_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bert_layer_inference(
    model_location_generator,
    mesh_device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    encoder_idx = 0
    config = hugging_face_reference_model.config
    pytorch_layer_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx]

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: pytorch_layer_model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_layer_model(input.squeeze(1))[0]

    ttnn_input = ttnn.from_torch(
        input.squeeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
    )

    tt_output = bert_layer(
        config=config,
        hidden_states=ttnn_input,
        attention_mask=None,
        parameters=parameters,
        device=mesh_device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
    assert_with_pcc(pytorch_out.squeeze(1), tt_output, 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize("model_name", ["mrm8488/bert-tiny-finetuned-squadv2"])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_hidden_layers", [1])
def test_bert_for_question_answering(mesh_device, model_name, sequence_size, num_hidden_layers, reset_seeds):
    inputs_mesh_mapper = None
    output_mesh_composer = None
    parameters = None

    config = BertConfig.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = 16 if mesh_device_flag else 8
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: model,
            device=mesh_device,
            convert_to_ttnn=lambda *_: True,
        )

    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.zeros((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.zeros(1, sequence_size)
    torch_output = model(
        torch_input_ids,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    )

    ttnn_bert_inputs_ids = ttnn.from_torch(
        torch_input_ids, dtype=ttnn.uint32, mesh_mapper=inputs_mesh_mapper, device=mesh_device
    )
    ttnn_token_type_ids = ttnn.from_torch(
        torch_token_type_ids, dtype=ttnn.uint32, mesh_mapper=inputs_mesh_mapper, device=mesh_device
    )
    ttnn_position_ids = ttnn.from_torch(
        torch_position_ids, dtype=ttnn.uint32, mesh_mapper=inputs_mesh_mapper, device=mesh_device
    )
    ttnn_attention_mask = ttnn.from_torch(
        torch_attention_mask,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        device=mesh_device,
    )

    output = bert_for_question_answering(
        config,
        ttnn_bert_inputs_ids,
        ttnn_token_type_ids,
        ttnn_position_ids,
        ttnn_attention_mask,
        parameters=parameters,
        device=mesh_device,
    )
    output = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
    start_logits = output[..., 0]
    end_logits = output[..., 1]

    assert_with_pcc(torch_output.start_logits, start_logits, 0.94)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.95)
