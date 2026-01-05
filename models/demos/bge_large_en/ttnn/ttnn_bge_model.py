# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.bge_large_en.ttnn.ttnn_bge_embeddings import TtnnBGEEmbeddings
from models.demos.bge_large_en.ttnn.ttnn_bge_encoder import TtnnBGEEncoder
from models.demos.sentence_bert.ttnn.ttnn_sentencebert_pooler import TtnnSentenceBertPooler
from models.demos.wormhole.bge_large_en.ttnn.common import ttnn_mean_pooling


class TtnnBGEModel:
    def __init__(self, parameters, config):
        self.embeddings = TtnnBGEEmbeddings(parameters.embeddings, config)
        self.encoder = TtnnBGEEncoder(parameters.encoder, config)
        self.pooler = TtnnSentenceBertPooler(parameters.pooler)

    def __call__(
        self,
        input_ids: ttnn.Tensor,
        extended_attention_mask: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        token_type_ids: ttnn.Tensor,
        position_ids: ttnn.Tensor,
        apply_post_process=True,
        device=None,
    ):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids, device=device)
        sequence_output = self.encoder(embedding_output, extended_attention_mask, device=device)
        ttnn.deallocate(embedding_output)
        if apply_post_process:
            sequence_output = ttnn_mean_pooling(sequence_output, attention_mask)
        return (sequence_output,)
