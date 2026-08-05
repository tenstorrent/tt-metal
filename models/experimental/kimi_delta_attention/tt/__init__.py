# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Private TTNN implementation for Kimi Delta Attention."""

from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights

__all__ = ["KDAWeights", "KimiDeltaAttention"]
