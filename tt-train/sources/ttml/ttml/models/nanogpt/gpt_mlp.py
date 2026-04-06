# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GPT-style MLP (feed-forward) layer for NanoGPT."""

from __future__ import annotations

import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, RunMode


class GPTMLP(AbstractModuleBase):
    """GPT-style MLP (feed-forward) layer."""

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout_prob = dropout

        self.fc1 = LinearLayer(
            embedding_dim,
            embedding_dim * 4,
            True,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )
        self.fc2 = LinearLayer(
            embedding_dim * 4,
            embedding_dim,
            True,
            weight_init=ttml.init.normal(0.0, 0.02),
            bias_init=ttml.init.zeros(),
        )

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        x = self.fc1(x)
        x = ttml.ops.unary.gelu(x)
        x = self.fc2(x)

        # Note: It's better to just use Dropout module here
        if self.get_run_mode() == RunMode.TRAIN and self.dropout_prob > 0.0:
            x = ttml.ops.dropout.dropout(x, self.dropout_prob)

        return x
