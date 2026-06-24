# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet (linear attention) for Qwen3.5-9B, split into config/weights/state/prefill/decode.

The orchestrating layer lives in ``gated_deltanet.py``; this package re-exports it
(and ``GDNConfig``) as the public API.
"""

from models.demos.blackhole.qwen36.tt.gdn.config import GDNConfig
from models.demos.blackhole.qwen36.tt.gdn.gated_deltanet import Qwen35GatedDeltaNet

__all__ = ["Qwen35GatedDeltaNet", "GDNConfig"]
