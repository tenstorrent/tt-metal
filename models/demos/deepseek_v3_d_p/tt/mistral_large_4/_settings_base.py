# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A model's switches in one table. ``settings.py`` builds a ``Settings`` and every module of the model reads its
switches only from it. Each switch has a default, its allowed values and a short description; an environment
variable ``<PREFIX><NAME>`` overrides it. ``max_precision`` names the full-precision value of every compute-precision
switch.

    S = Settings("XING_", {"MATMUL_FIDELITY": Setting("HiFi4", ("HiFi4", "HiFi2"), "matmul math fidelity")})
    S.get("MATMUL_FIDELITY")  # "HiFi4", or $XING_MATMUL_FIDELITY
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
    def __init__(self, prefix: str, table: dict[str, Setting], max_precision: dict | None = None):
        """max_precision: {name: value} for every precision switch (fidelity, accumulation, dtypes)."""
        self.prefix, self.table = prefix, dict(table)
        self.max_precision = dict(max_precision or {})
        unknown = set(self.max_precision) - set(self.table)
        assert not unknown, f"max_precision names unknown settings: {sorted(unknown)}"

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

    def max_precision_env(self) -> dict[str, str]:
        """The environment overrides that put every precision switch at its maximum."""
        return {self.env_name(n): str(v) for n, v in self.max_precision.items()}

    def all(self) -> dict:
        """Every switch's current value."""
        return {name: self.get(name) for name in self.table}
