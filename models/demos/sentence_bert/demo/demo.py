# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import transformers
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import ttnn
from models.demos.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner


@pytest.mark.parametrize(
    "inputs",
    [
        [
            [  # input sentences (turkish)
                "Yarın tatil yapacağım, ailemle beraber doğada vakit geçireceğiz, yürüyüşler yapıp, keşifler yapacağız, çok keyifli bir tatil olacak.",
                "Yarın tatilde olacağım, ailemle birlikte şehir dışına çıkacağız, doğal güzellikleri keşfedecek ve eğlenceli zaman geçireceğiz.",
                "Yarın tatil planım var, ailemle doğa yürüyüşlerine çıkıp, yeni yerler keşfedeceğiz, harika bir tatil olacak.",
                "Yarın tatil için yola çıkacağız, ailemle birlikte sakin bir yerlerde vakit geçirip, doğa aktiviteleri yapacağız.",
                "Yarın tatilde olacağım, ailemle birlikte doğal alanlarda gezi yapıp, yeni yerler keşfedeceğiz, eğlenceli bir tatil geçireceğiz.",
                "Yarın tatilde olacağım, ailemle birlikte şehir dışında birkaç gün geçirip, doğa ile iç içe olacağız.",
                "Yarın tatil için yola çıkıyoruz, ailemle birlikte doğada keşif yapıp, eğlenceli etkinliklere katılacağız.",
                "Yarın tatilde olacağım, ailemle doğada yürüyüş yapıp, yeni yerler keşfederek harika bir zaman geçireceğiz.",
            ],
        ]
    ],
)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("model_name, sequence_length", [("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", 384)])
def test_sentence_bert_demo_inference(device, inputs, model_name, sequence_length):
    transformers_model = transformers.AutoModel.from_pretrained(model_name).eval()
    config = transformers.BertConfig.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(
        inputs[0], padding="max_length", max_length=sequence_length, truncation=True, return_tensors="pt"
    )
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = encoded_input["token_type_ids"]
    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    Reference_sentence_embeddings = reference_module(
        input_ids,
        extended_attention_mask=extended_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    ).post_processed_output
    ttnn_module = SentenceBERTPerformantRunner(
        device=device,
        input_ids=input_ids,
        extended_mask=extended_mask,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
    )
    ttnn_module._capture_sentencebert_trace_2cqs()
    ttnn_out = ttnn_module.run(input_ids, token_type_ids, position_ids, extended_mask, attention_mask)
    ttnn_sentence_embeddings = ttnn.to_torch(ttnn_out, dtype=torch.float32)
    cosine_sim_matrix1 = cosine_similarity(Reference_sentence_embeddings.detach().squeeze().cpu().numpy())
    upper_triangle1 = np.triu(cosine_sim_matrix1, k=1)
    similarities1 = upper_triangle1[upper_triangle1 != 0]
    mean_similarity1 = similarities1.mean()
    cosine_sim_matrix2 = cosine_similarity(ttnn_sentence_embeddings.detach().squeeze().cpu().numpy())
    upper_triangle2 = np.triu(cosine_sim_matrix2, k=1)
    similarities2 = upper_triangle2[upper_triangle2 != 0]
    mean_similarity2 = similarities2.mean()
    logger.info(f"Mean Cosine Similarity for Reference Model: {mean_similarity1}")
    logger.info(f"Mean Cosine Similarity for TTNN Model:: {mean_similarity2}")
