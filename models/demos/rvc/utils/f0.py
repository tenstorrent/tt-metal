# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class F0Method(str, Enum):
    RAPT = "rapt"
    DIO = "dio"
    HARVEST = "harvest"
    RMVPE = "rmvpe"

    @classmethod
    def from_str(cls, value: "F0Method | str") -> "F0Method":
        if isinstance(value, cls):
            return value
        return cls(value)
