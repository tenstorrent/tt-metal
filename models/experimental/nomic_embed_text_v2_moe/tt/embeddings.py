# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token ids to vectors, the TTNN form of reference.NomicBertEmbeddings.

Position is rotary-only and applied inside attention, so there is no position table here. The
token-type term is folded into the word table at load; see TtNomicBertEmbeddings.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device


class TtNomicBertEmbeddings(LightweightModule):
    """Word embedding lookup, with the token-type embedding folded into the table.

    type_vocab_size is 1, so token_type_ids can only ever select row 0 and the reference adds
    that one row to every token. Summing it into the word table on the host makes the device
    path a single lookup, and is exact for every legal input rather than only for the default
    all-zeros token_type_ids.

    The <pad> row is trained and non-zero, so the lookup must not special-case it: padding is
    excluded at the attention mask and at pooling instead.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix="embeddings."):
        super().__init__()
        self.tt_config = tt_config

        table = state_dict[f"{state_dict_prefix}word_embeddings.weight"]
        if config.type_vocab_size > 0:
            table = table + state_dict[f"{state_dict_prefix}token_type_embeddings.weight"][0]

        # ROW_MAJOR: ttnn.embedding reads the table row-wise and rejects a tiled one.
        self.word_embeddings = to_device(table, device, dtype=tt_config.weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def forward(self, input_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Look up one vector per token.

        Args:
            input_ids: (B, S) uint32 in ROW_MAJOR, from tt.common.prepare_token_ids.

        Returns:
            ttnn.Tensor: (B, 1, S, H), the batch-separated layout every block takes.
        """
        batch, seqlen = input_ids.shape
        embeddings = ttnn.embedding(
            input_ids,
            self.word_embeddings,
            layout=self.tt_config.layout,
            dtype=self.tt_config.activation_dtype,
        )
        return ttnn.reshape(embeddings, (batch, 1, seqlen, embeddings.shape[-1]))
