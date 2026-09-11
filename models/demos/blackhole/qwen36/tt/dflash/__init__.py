# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn DFlash drafter for Qwen3.6-27B — block-diffusion speculative decoding, on device.

The drafter is a 5-layer Qwen3-style GQA model that borrows the target's embedding and LM head. It
reads the target's residual stream at layers ``[1, 16, 31, 46, 61]`` and fills a 16-slot block of
mask tokens in one forward; the target then verifies all 16 in one forward.

Validated against the host reference in ``tests/test_dflash_drafter_tp.py``; the host reference and
the speculative loop live in ``reference/dflash/``.
"""

from models.demos.blackhole.qwen36.tt.dflash.config import (
    DFlashDrafterConfig,
    load_drafter_state_dict,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.dflash.weights import load_drafter_weights

__all__ = [
    "DFlashDrafterConfig",
    "TtDFlashDrafter",
    "load_drafter_state_dict",
    "load_drafter_weights",
    "resolve_drafter_path",
    "resolve_target_path",
]
