# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import pytest
import torch
import transformers
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import ttnn
from models.demos.sentence_bert.common import load_torch_model
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
@pytest.mark.parametrize(
    "mesh_device",
    ((1, 8),),
    indirect=True,
)
@pytest.mark.parametrize("model_name, sequence_length", [("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", 384)])
def test_sentence_bert_demo_inference(
    mesh_device, inputs, model_name, sequence_length, model_location_generator, is_ci_env
):
    batch_size = len(inputs[0]) * mesh_device.get_num_devices()
    config = transformers.BertConfig.from_pretrained(model_name, local_files_only=is_ci_env)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, local_files_only=is_ci_env)
    encoded_input = tokenizer(
        inputs[0] * mesh_device.get_num_devices(),
        padding="max_length",
        max_length=sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = encoded_input["token_type_ids"]
    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix="", model_location_generator=model_location_generator
    )
    reference_sentence_embeddings = reference_module(
        input_ids,
        extended_attention_mask=extended_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    ).post_processed_output
    ttnn_module = SentenceBERTPerformantRunner(
        device=mesh_device,
        input_ids=input_ids,
        extended_mask=extended_mask,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        model_location_generator=model_location_generator,
    )
    ttnn_module._capture_sentencebert_trace_2cqs()
    t0 = time.time()
    ttnn_out = ttnn_module._execute_sentencebert_trace_2cqs_inference()
    t1 = time.time()
    ttnn_sentence_embeddings = ttnn.to_torch(
        ttnn_out, mesh_composer=ttnn_module.runner_infra.output_mesh_composer, dtype=torch.float32
    ).squeeze(dim=1)
    cosine_sim_matrix1 = cosine_similarity(reference_sentence_embeddings.detach().squeeze().cpu().numpy())
    upper_triangle1 = np.triu(cosine_sim_matrix1, k=1)
    similarities1 = upper_triangle1[upper_triangle1 != 0]
    mean_similarity1 = similarities1.mean()
    cosine_sim_matrix2 = cosine_similarity(ttnn_sentence_embeddings.detach().squeeze().cpu().numpy())
    upper_triangle2 = np.triu(cosine_sim_matrix2, k=1)
    similarities2 = upper_triangle2[upper_triangle2 != 0]
    mean_similarity2 = similarities2.mean()
    inference_time = t1 - t0
    sentence_per_sec = round(batch_size / inference_time)
    logger.info(
        f"ttnn_sentencebert_batch_size: {len(inputs[0])}, One inference iteration time (sec): {inference_time}, Sentence per sec: {sentence_per_sec}"
    )
    assert sentence_per_sec > 2984
    assert inference_time < 0.0214
    logger.info(f"Mean Cosine Similarity for Reference Model: {mean_similarity1}")
    logger.info(f"Mean Cosine Similarity for TTNN Model:: {mean_similarity2}")

    # Check if the cosine similarities are approximately close
    similarity_diff = abs(mean_similarity1 - mean_similarity2)
    tolerance = 0.02  # 2% tolerance
    assert (
        similarity_diff < tolerance
    ), f"Cosine similarities differ by {similarity_diff:.4f}, which exceeds tolerance of {tolerance}"
    logger.info(f"Cosine similarities are close (difference: {similarity_diff:.4f})")
