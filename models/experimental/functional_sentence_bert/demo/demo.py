# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.experimental.functional_sentence_bert.ttnn.ttnn_sentence_bert import ttnn_BertModel, preprocess_inputs
from ttnn.model_preprocessing import preprocess_model_parameters
import transformers
import torch
import ttnn
import pytest
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@pytest.mark.parametrize(
    "inputs",
    [
        # [
        #     "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        #     [
        #         "Bu örnek bir cümle",
        #         "Her cümle vektöre çevriliyor",
        #         "Her cümle vektöre çevriliyor Bu örnek bir cümle Her cümle vektöre çevriliyor Bu örnek Her cümle vektöre çevriliyor Bu örnek Her cümle vektöre ",
        #     ],
        # ],
        [
            "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            [
                "Bugün hava çok güzel, güneş parlıyor ve insanlar dışarıda yürüyüş yaparak, doğanın tadını çıkarıyorlar, parklarda çocuklar oynuyor, insanlar kahve içiyor.",
                "Yarın tatilde olacağım, ailemle birlikte şehir dışında birkaç gün geçireceğiz, doğa yürüyüşleri yapacağız, yeni yerler keşfedeceğiz, eğlenceli bir tatil olacak.",
            ],
        ]
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_sentence_bert_demo_inference(device, inputs):
    transformers_model = transformers.AutoModel.from_pretrained(inputs[0]).eval()
    config = transformers.BertConfig.from_pretrained(inputs[0])
    tokenizer = transformers.AutoTokenizer.from_pretrained(inputs[0])
    encoded_input = tokenizer(inputs[1], padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = encoded_input["token_type_ids"]
    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.int64).unsqueeze(dim=0)
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module.load_state_dict(transformers_model.state_dict())
    reference_out = reference_module(
        input_ids, attention_mask=extended_mask, token_type_ids=token_type_ids, position_ids=position_ids
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        device=device,
    )
    ttnn_module = ttnn_BertModel(parameters=parameters, config=config)
    ttnn_input_ids, ttnn_token_type_ids, ttnn_position_ids, ttnn_attention_mask = preprocess_inputs(
        input_ids, token_type_ids, position_ids, extended_mask, device
    )
    ttnn_out = ttnn_module(ttnn_input_ids, ttnn_attention_mask, ttnn_token_type_ids, ttnn_position_ids, device=device)
    ttnn_out = ttnn.to_torch(ttnn_out[0])
    Reference_sentence_embeddings = mean_pooling(reference_out[0], attention_mask)
    ttnn_sentence_embeddings = mean_pooling(ttnn_out, attention_mask)
    ref_emb1 = Reference_sentence_embeddings[0].detach().squeeze().cpu().numpy()
    ref_emb2 = Reference_sentence_embeddings[1].detach().squeeze().cpu().numpy()
    cos_sim1 = cosine_similarity([ref_emb1], [ref_emb2])[0][0]
    pearson_corr1, _ = pearsonr(ref_emb1, ref_emb2)
    spearman_corr1, _ = spearmanr(ref_emb1, ref_emb2)
    ttnn_emb1 = ttnn_sentence_embeddings[0].squeeze().cpu().numpy()
    ttnn_emb2 = ttnn_sentence_embeddings[1].squeeze().cpu().numpy()
    cos_sim2 = cosine_similarity([ttnn_emb1], [ttnn_emb2])[0][0]
    pearson_corr2, _ = pearsonr(ttnn_emb1, ttnn_emb2)
    spearman_corr2, _ = spearmanr(ttnn_emb1, ttnn_emb2)
    logger.info(f"Cosine Similarity for Reference and ttnn Models: {cos_sim1} and {cos_sim2}")
    logger.info(f"Pearson Correlation for Reference and ttnn Models: {pearson_corr1} and {pearson_corr2}")
    logger.info(f"Spearman Correlation for Reference and ttnn Models: {spearman_corr1} and {spearman_corr2}")
