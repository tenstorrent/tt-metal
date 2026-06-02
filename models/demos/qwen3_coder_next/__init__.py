# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Coder-Next demo package for TT-NN (P150a/Wormhole).

Lazy imports: only non-device-dependent modules are imported at package load.
Device-dependent modules (deltanet, attention, moe, moe_ep, model, generator)
must be imported directly after ttnn device initialization.
"""

from .tt.model_config import Qwen3CoderNextConfig

__all__ = [
    "Qwen3CoderNextConfig",
]
