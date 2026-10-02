# SPDX-License-Identifier: Apache-2.0
"""KV look-ahead allocation for Laguna DFlash serving.

A DFlash round verifies 16 rows (the known bonus token plus 15 drafts) at positions P..P+15, so their KV
writes need blocks the scheduler has not handed out yet when P sits in the last 15 slots of a 64-token
block. Stock vLLM reserves that room for its own DFlash (``num_lookahead_tokens = num_spec_tokens + 1``),
but Laguna's DFlash runs inside the model adapter with no ``speculative_config``, so the TT scheduler's
look-ahead stays 0. Without it the adapter must fall back to one eager target row per token for 15 of
every 64 positions.

With ``TT_LAGUNA_DFLASH=1`` this wrapper raises the TT scheduler's look-ahead to 16. With async scheduling
the adapter may run at host position + 1; the scheduler allocates through P + 1 + 16, which still covers
P + 1 + 15. ``applied_lookahead_tokens()`` reports what was applied in this process, and the adapter keeps
the one-row fallback unless it is at least the verify row count.
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

DFLASH_LOOKAHEAD_TOKENS = 16
PATCH_MARKER = "_laguna_dflash_lookahead_patch"
_APPLIED = {"tokens": 0}


def applied_lookahead_tokens() -> int:
    return int(_APPLIED["tokens"])


def _patch_scheduler(scheduler_class: type) -> bool:
    if scheduler_class.__dict__.get(PATCH_MARKER, False):
        return False
    original_init = scheduler_class.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if os.environ.get("TT_LAGUNA_DFLASH", "0") != "1":
            return
        current = int(getattr(self, "num_lookahead_tokens", 0) or 0)
        if current < DFLASH_LOOKAHEAD_TOKENS:
            self.num_lookahead_tokens = DFLASH_LOOKAHEAD_TOKENS
        _APPLIED["tokens"] = int(self.num_lookahead_tokens)
        logger.warning(
            "Laguna DFlash: scheduler KV look-ahead %d -> %d tokens", current, int(self.num_lookahead_tokens)
        )

    scheduler_class.__init__ = __init__
    setattr(scheduler_class, PATCH_MARKER, True)
    return True


def install_dflash_lookahead_patch() -> bool:
    from vllm_tt_plugin.scheduler import TTScheduler

    return _patch_scheduler(TTScheduler)


__all__ = [
    "DFLASH_LOOKAHEAD_TOKENS",
    "applied_lookahead_tokens",
    "install_dflash_lookahead_patch",
]
