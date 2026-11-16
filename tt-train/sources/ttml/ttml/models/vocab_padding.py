# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

import ttnn
import ttml


class VocabPaddingMask:
    """Lowers the padded columns of logits sharded along a padded vocab by 1e4, which leaves them no probability.

    1e4 is what the samplers' logits mask takes off (``ttml.common.utils.build_logits_mask``).
    """

    def __init__(self, vocab_size: int, padded_vocab_size: int, axis_name: str = "tp") -> None:
        self._penalty = None
        if padded_vocab_size == vocab_size:
            return
        penalty = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
        penalty[..., vocab_size:] = -1e4
        mapper = ttml.mesh().axis_mapper(axis_name, tdim=3)
        self._penalty = ttml.autograd.Tensor.from_numpy(
            penalty, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper
        ).get_value()

    def __call__(self, logits: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        if self._penalty is None:
            return logits
        return ttml.ops.binary.add(logits, ttml.autograd.create_tensor(self._penalty, False))
