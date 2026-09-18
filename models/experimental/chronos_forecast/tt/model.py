# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""TTNN Chronos implementation (stub)."""

from models.experimental.chronos_forecast.common.configs import ChronosModelConfig


class TtChronos:
    """Device Chronos model. Forward is not implemented yet."""

    def __init__(self, device, config: ChronosModelConfig | None = None):
        self.device = device
        self.config = config or ChronosModelConfig()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("TTNN Chronos forward is not implemented yet.")
