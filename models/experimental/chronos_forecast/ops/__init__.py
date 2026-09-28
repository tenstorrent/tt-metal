# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Chronos-2 model-local ``ttnn.generic_op`` ops. Kernel paths are relative to TT_METAL_HOME."""

from models.experimental.chronos_forecast.ops.eltwise_add import add
from models.experimental.chronos_forecast.ops.rotary_embedding import rotary_embedding

__all__ = ["add", "rotary_embedding"]
