# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative-decoding drafter components.

* ``host``: numpy-only batch preparation (anchors, square masks, RoPE, loss weights).
* ``block_attention``: D1a square-padded anchor-block attention.
* ``model``: the ttml drafter, KL loss and training step.
* ``weights``: speculators <-> ttml state-dict mapping.
"""

from .block_attention import build_dflash_square_mask, dflash_square_attention
from .host import DFlashDrafterConfig, DFlashHostBatch, build_host_batch, sample_anchors
from .model import DFlashDeviceBatch, DFlashDrafter, DFlashTrainStep, dflash_kl_loss

__all__ = [
    "DFlashDeviceBatch",
    "DFlashDrafter",
    "DFlashDrafterConfig",
    "DFlashHostBatch",
    "DFlashTrainStep",
    "build_dflash_square_mask",
    "build_host_batch",
    "dflash_kl_loss",
    "dflash_square_attention",
    "sample_anchors",
]
