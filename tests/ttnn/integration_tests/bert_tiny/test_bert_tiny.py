# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn

from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig
from models.demos.bert_tiny.tt.bert_tiny import (
    bert_for_question_answering,
    bert_attention,
    bert_intermediate,
    bert_output,
    bert_layer,
    bert,
)
from ttnn.model_preprocessing import preprocess_model_parameters
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
@pytest.mark.parametrize(
    "sequence_size",
    ((128),),
)
def test_bert_inference(
    batch_size,
    sequence_size,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    config = hugging_face_reference_model.config
    tokenizer_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    torch_input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size)).to(torch.int32)
    torch_token_type_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_position_ids = torch.ones((batch_size, sequence_size), dtype=torch.int32)
    torch_attention_mask = torch.ones(1, sequence_size)

    ttnn_input_ids = ttnn.from_torch(torch_input_ids, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_token_type_ids = ttnn.from_torch(
        torch_token_type_ids, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn_position_ids = ttnn.from_torch(
        torch_position_ids, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn_attention_mask = ttnn.from_torch(
        torch_attention_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    torch_position_ids = torch.zeros((1, 128), dtype=torch.int32)
    ttnn_position_ids = ttnn.from_torch(torch_position_ids, device=device, dtype=ttnn.uint32)
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    state_dict = hugging_face_reference_model.state_dict()

    pytorch_bert_model = hugging_face_reference_model.bert
    pytorch_out = pytorch_bert_model(
        torch_input_ids,
        token_type_ids=torch_token_type_ids,
        position_ids=torch_position_ids,
        attention_mask=torch_attention_mask,
    ).last_hidden_state

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_bert_model,
        device=device,
        convert_to_ttnn=lambda *_: True,
    )

    tt_output = bert(
        config=config,
        input_ids=ttnn_input_ids,
        token_type_ids=ttnn_token_type_ids,
        position_ids=ttnn_position_ids,
        attention_mask=ttnn_attention_mask,
        device=device,
        parameters=parameters,
    )
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(pytorch_out, tt_output, 0.68)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
def test_bert_attention_inference(
    batch_size,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    state_dict = hugging_face_reference_model.state_dict()
    encoder_idx = 0
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    pytorch_attention_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx].attention
    config = hugging_face_reference_model.config

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_attention_model,
        device=device,
        convert_to_ttnn=lambda *_: True,
    )

    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    torch_attention_mask = torch.zeros(1, 128)
    pytorch_out = pytorch_attention_model(input.squeeze(1), attention_mask=torch_attention_mask)[0]

    tt_input = ttnn.from_torch(input.squeeze(1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_attention_mask = ttnn.from_torch(
        torch_attention_mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    tt_output = bert_attention(
        config=config,
        hidden_states=tt_input,
        attention_mask=tt_attention_mask,
        parameters=parameters,
        device=device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(pytorch_out, tt_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
def test_bert_intermediate_inference(
    batch_size,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    state_dict = hugging_face_reference_model.state_dict()
    encoder_idx = 0
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    config = hugging_face_reference_model.config
    pytorch_intermediate_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx].intermediate

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_intermediate_model,
        device=device,
        convert_to_ttnn=lambda *_: True,
    )

    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_intermediate_model(input).squeeze(0)

    tt_input = ttnn.from_torch(input.squeeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = bert_intermediate(
        hidden_states=tt_input,
        parameters=parameters,
        device=device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(pytorch_out.squeeze(1), tt_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
def test_bert_output_inference(
    batch_size,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    state_dict = hugging_face_reference_model.state_dict()
    encoder_idx = 0
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    config = hugging_face_reference_model.config
    pytorch_output_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx].attention.output

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_output_model,
        device=device,
        convert_to_ttnn=lambda *_: True,
    )
    hidden_state = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_output_model(hidden_state, input).squeeze(0)

    ttnn_input = ttnn.from_torch(input.squeeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_hidden_state = ttnn.from_torch(
        hidden_state.squeeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_output = bert_output(
        config=config,
        hidden_states=ttnn_hidden_state,
        residual=ttnn_input,
        parameters=parameters,
        device=device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(pytorch_out.squeeze(1), tt_output, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
def test_bert_layer_inference(
    batch_size,
    model_location_generator,
    device,
    reset_seeds,
):
    model_name = str(model_location_generator("mrm8488/bert-tiny-finetuned-squadv2", model_subdir="Bert"))
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)

    state_dict = hugging_face_reference_model.state_dict()
    encoder_idx = 0
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    config = hugging_face_reference_model.config
    pytorch_layer_model = hugging_face_reference_model.bert.encoder.layer[encoder_idx]

    parameters = preprocess_model_parameters(
        initialize_model=lambda: pytorch_layer_model,
        device=device,
        convert_to_ttnn=lambda *_: True,
    )
    input = (torch.rand(batch_size, 1, 128, hugging_face_reference_model.config.hidden_size) * 2) - 1
    pytorch_out = pytorch_layer_model(input.squeeze(1))[0]

    ttnn_input = ttnn.from_torch(input.squeeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = bert_layer(
        config=config,
        hidden_states=ttnn_input,
        attention_mask=None,
        parameters=parameters,
        device=device,
    )
    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)
    assert_with_pcc(pytorch_out.squeeze(1), tt_output, 0.99)


@pytest.mark.parametrize("model_name", ["mrm8488/bert-tiny-finetuned-squadv2"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_hidden_layers", [1])
def test_bert_for_question_answering(device, model_name, batch_size, sequence_size, num_hidden_layers):
    torch.manual_seed(0)

    config = BertConfig.from_pretrained(model_name)
    model = BertForQuestionAnswering.from_pretrained(model_name, config=config).eval()

    if num_hidden_layers is not None:
        config.num_hidden_layers = num_hidden_layers

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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        device=device,
        convert_to_ttnn=lambda *_: True,
    )

    ttnn_bert_inputs_ids = ttnn.from_torch(torch_input_ids, dtype=ttnn.uint32, device=device)
    ttnn_token_type_ids = ttnn.from_torch(torch_token_type_ids, dtype=ttnn.uint32, device=device)
    ttnn_position_ids = ttnn.from_torch(torch_position_ids, dtype=ttnn.uint32, device=device)
    ttnn_attention_mask = ttnn.from_torch(torch_attention_mask, dtype=ttnn.bfloat16, device=device)

    output = bert_for_question_answering(
        config,
        ttnn_bert_inputs_ids,
        ttnn_token_type_ids,
        ttnn_position_ids,
        ttnn_attention_mask,
        parameters=parameters,
        device=device,
    )
    output = ttnn.to_torch(output)
    start_logits = output[..., 0]
    end_logits = output[..., 1]

    assert_with_pcc(torch_output.start_logits, start_logits, 0.92)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.94)
