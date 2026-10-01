# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A model's switches in one place: `models/demos/<model>/tt/settings.py` builds a ``Settings`` table and every module
of the model (and its bring-up hooks) reads switches only from it. Each switch has a default, its allowed values and
the reason for the default (owner decisions with their date). An environment variable ``<PREFIX><NAME>`` overrides
it for experiments and A/B reports; nothing else reads the environment. testing/settings_lint.py enforces this.

    from models.demos.common.bringup.core.model_settings import Setting, Settings
    S = Settings("XING_", {"KV_CACHE_DTYPE": Setting("bfp8", ("bfp8", "bf16"), "owner 2026-10-01: ...")})
    S.get("KV_CACHE_DTYPE")  # "bfp8", or $XING_KV_CACHE_DTYPE
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    default: object
    choices: tuple | None = None  # None: any value of the default's type
    why: str = ""


class Settings:
    def __init__(self, prefix: str, table: dict[str, Setting]):
        self.prefix, self.table = prefix, dict(table)

    def env_name(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def get(self, name: str):
        s = self.table[name]
        raw = os.environ.get(self.env_name(name))
        if raw is None:
            return s.default
        val = type(s.default)(raw) if not isinstance(s.default, bool) else raw.lower() in ("1", "true", "yes")
        if s.choices is not None and val not in s.choices:
            raise ValueError(f"{self.env_name(name)}={raw!r}: expected one of {s.choices}")
        return val

    def all(self) -> dict:
        """Every switch's current value (the profile and the dashboard record these)."""
        return {name: self.get(name) for name in self.table}
