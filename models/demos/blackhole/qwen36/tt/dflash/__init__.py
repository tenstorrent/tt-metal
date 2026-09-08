# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN port of the Qwen3.6-27B DFlash drafter (``z-lab/Qwen3.6-27B-DFlash``).

A 1.73 B, 5-layer Qwen3-style drafter that proposes ``block_size - 1`` = 15 tokens per
step for the 27B target to verify. It ships no ``embed_tokens`` and no ``lm_head`` -- it
borrows the target's -- and is conditioned on the target's residual stream at
``target_layer_ids`` = ``[1, 16, 31, 46, 61]``, fused by ``fc`` (``[5120, 25600]``).

See ``reference/dflash/__init__.py`` for the checkpoint's full shape inventory and for why
the mask semantics here are per-layer asymmetric.
"""

from models.demos.blackhole.qwen36.tt.dflash.config import CONTEXT_BUCKETS, DFlashDrafterConfig

__all__ = ["DFlashDrafterConfig", "CONTEXT_BUCKETS"]
