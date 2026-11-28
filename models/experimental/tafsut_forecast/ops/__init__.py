# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tafsut model-local ``ttnn.generic_op`` ops (ported from chronos_forecast, ec79b81bd6e / 41c6856ef36; add_rms_norm from db063defe3b)."""

from models.experimental.tafsut_forecast.ops.add_rms_norm import add_rms_norm, rms_norm
from models.experimental.tafsut_forecast.ops.qkv_heads_rope import qkv_heads_rope

__all__ = ["add_rms_norm", "qkv_heads_rope", "rms_norm"]
