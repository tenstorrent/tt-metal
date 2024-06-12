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


@pytest.mark.parametrize("model_name", ["distilbert-base-uncased-distilled-squad"])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("sequence_size", [768])
def test_distilbert_for_question_answering(device, model_name, batch_size, sequence_size, reset_seeds):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    HF_model = HF_DistilBertForQuestionAnswering.from_pretrained(model_name)

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

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: model,
        custom_preprocessor=ttnn_optimized_distilbert.custom_preprocessor,
        device=device,
    )

    input_ids, position_ids, attention_mask = ttnn_optimized_distilbert.preprocess_inputs(
        input_ids,
        position_ids,
        attention_mask,
        device=device,
    )

    min_val_tensor = ttnn.from_torch(min_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    negative_val_tensor = ttnn.from_torch(negative_val_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_output = ttnn_optimized_distilbert.distilbert_for_question_answering(
        config,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        parameters=parameters,
        device=device,
        min_val_tensor=min_val_tensor,
        negative_val_tensor=negative_val_tensor,
    )
    tt_output = ttnn.to_torch(tt_output)
    start_logits, end_logits = tt_output.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()

    assert_with_pcc(torch_output.start_logits, start_logits, 0.99)
    assert_with_pcc(torch_output.end_logits, end_logits, 0.99)
