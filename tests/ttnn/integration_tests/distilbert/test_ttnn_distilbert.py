# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters
from transformers import (
    DistilBertForQuestionAnswering as HF_DistilBertForQuestionAnswering,
)
from transformers import AutoTokenizer
from models.demos.distilbert.tt import ttnn_optimized_distilbert

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.distilbert.tt.distilbert_preprocessing import (
    preprocess_linear_weight,
    preprocess_linear_bias,
    preprocess_layernorm_parameter,
    preprocess_embedding_weight,
    preprocess_attn_weight,
    preprocess_attn_bias,
)


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [768])
def test_bert_for_question_answering(model_name, batch_size, sequence_size, reset_seeds, mesh_device):
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)
    parameters = {}
    attn_weight = []
    for name, parameter in HF_model.state_dict().items():
        if "_embeddings.weight" in name:
            parameters[name] = preprocess_embedding_weight(parameter, weights_mesh_mapper, mesh_device)
        elif "LayerNorm" in name or "_layer_norm" in name:
            parameters[name] = preprocess_layernorm_parameter(parameter, weights_mesh_mapper, mesh_device)
        elif "q_lin" in name or "k_lin" in name or "v_lin" in name or "LayerNorm" in name:
            attn_weight.append(name)
        elif "out_lin" in name or "lin1" in name or "lin2" in name or "qa_outputs" in name:
            if "weight" in name:
                parameters[name] = preprocess_linear_weight(parameter, weights_mesh_mapper, mesh_device)
            elif "bias" in name:
                parameters[name] = preprocess_linear_bias(parameter, weights_mesh_mapper, mesh_device)
        else:
            parameters[name] = ttnn.from_torch(
                parameter,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=weights_mesh_mapper,
                device=mesh_device,
            )
    for i in range(6):
        parameters[
            "distilbert.transformer.layer." + str(i) + ".attention.query_key_value.weight"
        ] = preprocess_attn_weight(
            HF_model.state_dict()["distilbert.transformer.layer." + str(i) + ".attention.q_lin.weight"],
            HF_model.state_dict()["distilbert.transformer.layer." + str(i) + ".attention.k_lin.weight"],
            HF_model.state_dict()["distilbert.transformer.layer." + str(i) + ".attention.v_lin.weight"],
            weights_mesh_mapper,
            mesh_device,
        )

        parameters["distilbert.transformer.layer." + str(i) + ".attention.query_key_value.bias"] = preprocess_attn_bias(
            HF_model.state_dict()["distilbert.transformer.layer." + str(i) + ".attention.q_lin.bias"],
            HF_model.state_dict()["distilbert.transformer.layer." + str(i) + ".attention.k_lin.bias"],
            HF_model.state_dict()["distilbert.transformer.layer." + str(i) + ".attention.v_lin.bias"],
            weights_mesh_mapper,
            mesh_device,
        )
    model = HF_model.eval()
    config = HF_model.config

    question = batch_size * ["Where do I live?"]
    context = batch_size * ["My name is Merve and I live in İstanbul."]
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        padding="max_length",
        max_length=384,
        truncation=True,
        return_attention_mask=True,
    )
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
    mask_reshp = (batch_size, 1, 1, attention_mask.shape[1])
    score_shape = (batch_size, 12, 384, 384)

    mask = (attention_mask == 0).view(mask_reshp).expand(score_shape)
    min_val = torch.zeros(score_shape)
    min_val_tensor = min_val.masked_fill(mask, torch.tensor(torch.finfo(torch.bfloat16).min))

    negative_val = torch.zeros(score_shape)
    negative_val_tensor = negative_val.masked_fill(mask, -1)
    torch_output = model(input_ids, attention_mask)

    tt_model_name = f"ttnn_{model_name}_optimized"

    # parameters = preprocess_model_parameters(
    #    model_name=tt_model_name,
    #    initialize_model=lambda: model,
    #    custom_preprocessor=ttnn_optimized_distilbert.custom_preprocessor,
    #    device=None,
    # )

    input_ids, position_ids, attention_mask = ttnn_optimized_distilbert.preprocess_inputs(
        input_ids, position_ids, attention_mask, device=mesh_device, mesh_mapper=inputs_mesh_mapper
    )

    min_val_tensor = ttnn.from_torch(
        min_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=inputs_mesh_mapper, device=mesh_device
    )
    negative_val_tensor = ttnn.from_torch(
        negative_val_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
    )

    tt_output = ttnn_optimized_distilbert.distilbert_for_question_answering(
        config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        parameters=parameters,
        device=mesh_device,
        min_val_tensor=min_val_tensor,
        negative_val_tensor=negative_val_tensor,
        mesh_mapper=weights_mesh_mapper,
    )

    tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)

    start_logits, end_logits = tt_output.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    assert_with_pcc(torch_output.start_logits[:4, :], start_logits, 0.99)
    assert_with_pcc(torch_output.end_logits[:4, :], end_logits, 0.99)
