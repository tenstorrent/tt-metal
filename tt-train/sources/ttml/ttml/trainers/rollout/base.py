# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Rollout data type and producer interface consumed by ``GRPOTrainer``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RolloutBatch:
    """Describes multiple rollout samples (not a specific count).

    Fields:
        batch_id: Monotonic counter picked by the producer.
        weight_version: Which theta version produced this batch. The trainer
            can use it to detect / down-weight stale samples.
        prompts: B ragged prompt token IDs.
        completions: B ragged completion token IDs (one per prompt in
            single-generation mode; per prompt-completion pair otherwise).
        logprobs: [B, max_completion_length] float32. Per-generated-token
            log pi_old(a_t | s_t). Padding positions are don't-care; the
            trainer masks them out.
    """

    batch_id: int
    weight_version: int
    prompts: List[List[int]]
    completions: List[List[int]]
    logprobs: np.ndarray


class RolloutSampler(ABC):
    """Abstract base for producers of :class:`RolloutBatch`."""

    @abstractmethod
    def generate(self, prompts: List[List[int]]) -> RolloutBatch:
        """Generate completions for a batch of tokenised prompts and return them
        packaged (with per-token log pi_old and producer metadata) as a
        :class:`RolloutBatch`.
        """
