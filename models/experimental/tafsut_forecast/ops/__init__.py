# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tafsut model-local ``ttnn.generic_op`` ops (ported from chronos_forecast, ec79b81bd6e / 41c6856ef36)."""

from models.experimental.tafsut_forecast.ops.qkv_heads_rope import qkv_heads_rope

__all__ = ["qkv_heads_rope"]
