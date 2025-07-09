# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import io

import pandas as pd
import pytest
import requests
import torch
import torch.nn.functional as F
import transformers
from datasets import Dataset
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import ttnn
from models.demos.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.demos.sentence_bert.runner.performant_runner import SentenceBERTPerformantRunner


def load_sts_tr(split="test"):
    url = {
        "train": "https://raw.githubusercontent.com/emrecncelik/sts-benchmark-tr/main/sts-train-tr.csv",
        "dev": "https://raw.githubusercontent.com/emrecncelik/sts-benchmark-tr/main/sts-dev-tr.csv",
        "test": "https://raw.githubusercontent.com/emrecncelik/sts-benchmark-tr/main/sts-test-tr.csv",
    }[split]

    data = requests.get(url).content.decode("utf-8")
    df = pd.read_csv(io.StringIO(data))
    return Dataset.from_pandas(df)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_name, sequence_length,batch_size,num_samples",
    [("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr", 384, 8, 4)],
)
def test_sentence_bert_eval(device, model_name, sequence_length, batch_size, num_samples, start=0, end=7):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    transformers_model = transformers.AutoModel.from_pretrained(model_name).eval()
    config = transformers.BertConfig.from_pretrained(model_name)
    dataset = load_sts_tr("test")
    true_scores = []
    ref_pred_scores = []
    ttnn_pred_scores = []
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    ttnn_module = None
    for i in tqdm(range(num_samples), desc="Evaluating"):
        example = dataset[i]
        sen1, sen2, score = example["sentence1_tr"], example["sentence2_tr"], example["score"]
        sen1_list, sen2_list = [sen1] * (batch_size // 2), [sen2] * (batch_size // 2)
        encoded_input = tokenizer(
            sen1_list + sen2_list,
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
        reference_sentence_embeddings = reference_module(
            input_ids,
            extended_attention_mask=extended_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).post_processed_output
        if ttnn_module is None:
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
        sim1 = F.cosine_similarity(reference_sentence_embeddings[:1], reference_sentence_embeddings[-1:]).item()
        sim2 = F.cosine_similarity(ttnn_sentence_embeddings[:1], ttnn_sentence_embeddings[-1:]).item()
        ref_pred_score, ttnn_pred_score = (sim1 + 1) * 2.5, (sim2 + 1) * 2.5  # scale from [-1, 1] to [0, 5]
        true_scores.append(score)
        ref_pred_scores.append(ref_pred_score)
        ttnn_pred_scores.append(ttnn_pred_score)

    pearson1, spearman1 = pearsonr(true_scores, ref_pred_scores)[0], spearmanr(true_scores, ref_pred_scores)[0]
    pearson2, spearman2 = pearsonr(true_scores, ttnn_pred_scores)[0], spearmanr(true_scores, ttnn_pred_scores)[0]
    logger.info(
        f"Cosine Pearson correlation and Spearman correlation for reference model: {pearson1:.4f}, {spearman1:.4f}"
    )
    logger.info(f"Cosine Pearson correlation and Spearman correlation for ttnn model: {pearson2:.4f}, {spearman2:.4f}")
