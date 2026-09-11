# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host reference for **Qwen3.6-27B-DFlash** — block-diffusion speculative decoding.

DFlash (arXiv:2602.06036, https://github.com/z-lab/dflash) pairs the ``Qwen/Qwen3.6-27B`` target
with a 5-layer drafter, ``z-lab/Qwen3.6-27B-DFlash``. The drafter reads the target's residual
stream at layers ``[1, 16, 31, 46, 61]`` and, in a single forward, fills a 16-slot block of mask
tokens — 15 draft tokens in parallel, not autoregressively. The target then verifies all 16 in one
forward. The drafter has no embedding and no LM head of its own: it borrows the target's.

Nothing here touches ttnn. This is the golden implementation for a device port, and the way to
confirm the checkpoint pair actually works before any bring-up:

    export HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash
    pytest models/demos/blackhole/qwen36/tests/reference/test_dflash_host.py -sv

Modules:

* :mod:`.dflash` — the drafter itself, vendored from upstream (MIT, Z Lab).
* :mod:`.loader` — checkpoint resolution, host model construction, drafter/target cross-checks.
* :mod:`.generate` — the speculative loop, with the exact Gated DeltaNet rollback that a hybrid
  target needs and upstream's loop does not implement.
"""

from models.demos.blackhole.qwen36.reference.dflash.dflash import (
    DFlashDraftModel,
    Qwen3DFlashAttention,
    Qwen3DFlashDecoderLayer,
    extract_context_feature,
)
from models.demos.blackhole.qwen36.reference.dflash.generate import DFlashStats, dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    check_drafter_matches_target,
    load_drafter,
    load_target,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.reference.dflash.targets import HFTarget, SpeculativeTarget, TtTarget

__all__ = [
    "DFlashDraftModel",
    "DFlashDrafterConfig",
    "DFlashStats",
    "HFTarget",
    "SpeculativeTarget",
    "TtTarget",
    "Qwen3DFlashAttention",
    "Qwen3DFlashDecoderLayer",
    "check_drafter_matches_target",
    "dflash_generate",
    "extract_context_feature",
    "load_drafter",
    "load_target",
    "resolve_drafter_path",
    "resolve_target_path",
]
