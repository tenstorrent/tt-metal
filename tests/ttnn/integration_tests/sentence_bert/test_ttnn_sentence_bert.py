# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import transformers
import pytest
from models.experimental.functional_sentence_bert.reference.sentence_bert import BertEmbeddings

# from models.experimental.functional_sentence_bert.ttnn.ttnn_sentence_bert import ttnn_BertEmbeddings
from transformers import BertConfig


@pytest.mark.parametrize(
    "inputs",
    [
        [[2, 8], [2, 8], None, None, 0],
    ],
)
def test_ttnn_Bert_Embeddings(device, inputs):
    model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    transformers_model = transformers.AutoModel.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name)
    input_ids = torch.randint(0, 10, (inputs[0][0], inputs[0][1]), dtype=torch.long)
    token_type_ids = input_ids = torch.randint(0, config.vocab_size, (inputs[1][0], inputs[1][1]), dtype=torch.long)
    print("tensors are", input_ids.shape, input_ids.dtype, token_type_ids.shape, token_type_ids.dtype)
    reference_module = BertEmbeddings(config)
    reference_out = reference_module(input_ids=input_ids, token_type_ids=token_type_ids)
    # ttnn_module = ttnn_BertEmbeddings()
