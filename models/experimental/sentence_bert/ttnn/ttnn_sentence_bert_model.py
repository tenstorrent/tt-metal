# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.sentence_bert.ttnn.ttnn_sentencebert_embeddings import TtnnSentenceBertEmbeddings
from models.experimental.sentence_bert.ttnn.ttnn_sentencebert_encoder import TtnnSentenceBertEncoder
from models.experimental.sentence_bert.ttnn.ttnn_sentencebert_pooler import TtnnSentenceBertPooler


class TtnnSentenceBertModel:
    def __init__(self, parameters, config):
        self.embeddings = TtnnSentenceBertEmbeddings(parameters.embeddings, config)
        self.encoder = TtnnSentenceBertEncoder(parameters.encoder, config)
        self.pooler = TtnnSentenceBertPooler(parameters.pooler)

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        token_type_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        device=None,
    ):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, device=device)
        sequence_output = self.encoder(embedding_output, attention_mask, device=device)
        ttnn.deallocate(embedding_output)
        return (sequence_output,)
