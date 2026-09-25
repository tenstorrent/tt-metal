# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn DFlash for Qwen3.6-27B — block-diffusion speculative decoding, on device.

The drafter is a 5-layer Qwen3-style GQA model that borrows the target's embedding and LM head. It
reads the target's residual stream at layers ``[1, 16, 31, 46, 61]`` and fills a 16-slot block of
mask tokens in one forward; the target then verifies all 16 in one forward.

* :class:`TtDFlashDrafter` (``drafter.py``) is the ttnn drafter; :class:`TtDrafter`
  (``speculative_drafter.py``) adapts it to the speculative loop.
* :class:`TtTarget` (``target.py``) wraps :class:`~...tt.model.Qwen36Model` as the verifying target.

The drafter is validated against the host reference in ``tests/test_dflash_drafter_tp.py``. The host
reference and the backend-agnostic speculative loop (``dflash_generate``) live in
``reference/dflash/``.
"""

from models.demos.blackhole.qwen36.tt.dflash.config import (
    DFlashDrafterConfig,
    load_drafter_state_dict,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.dflash.speculative_drafter import TtDrafter
from models.demos.blackhole.qwen36.tt.dflash.target import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.weights import load_drafter_weights

__all__ = [
    "DFlashDrafterConfig",
    "TtDFlashDrafter",
    "TtDrafter",
    "TtTarget",
    "load_drafter_state_dict",
    "load_drafter_weights",
    "resolve_drafter_path",
    "resolve_target_path",
]
