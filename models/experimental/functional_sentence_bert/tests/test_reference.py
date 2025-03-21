# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_sentence_bert.reference.sentence_bert import BertModel
import transformers
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import AutoTokenizer, AutoModel
import torch


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ["Bu örnek bir cümle", "Her cümle vektöre çevriliyor"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
model = BertModel.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
print("weigths are", model.state_dict().keys())
model2 = AutoModel.from_pretrained("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Compute token embeddings
with torch.no_grad():
    model_output1 = model(**encoded_input)
    model_output2 = model2(**encoded_input)

# Perform pooling. In this case, mean pooling.
sentence_embeddings1 = mean_pooling(model_output1, encoded_input["attention_mask"])
sentence_embeddings2 = mean_pooling(model_output2, encoded_input["attention_mask"])
print(
    "Sentence embeddings: shapes",
)
print(sentence_embeddings1, sentence_embeddings2)

assert_with_pcc(sentence_embeddings1, sentence_embeddings2, 1.0)
