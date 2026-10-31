# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A model's switches in one place: `models/demos/<model>/tt/settings.py` builds a ``Settings`` table and every module
of the model (and its bring-up hooks) reads switches only from it. Each switch has a default (what every test and
the served model run, F137: there is no separate max-precision setting), its allowed values, the choices that only
emulate a reference format (``emulation``: never a default, agents' rule 17; testing/settings_lint.py) and the reason
for the default (owner decisions with their date). An environment variable ``<PREFIX><NAME>`` overrides
it for experiments and A/B reports; nothing else reads the environment. testing/settings_lint.py enforces this.
F222: ``test`` is the value a test process (pytest) gets instead of the default, only for a choice that gives the
default's numbers bit for bit (a host-side storage or speed choice, proven by a test), e.g. reading a table from disk
instead of building it in RAM at every test's start; the environment variable still overrides it.

    from models.demos.common.bringup.core.model_settings import Setting, Settings
    S = Settings("XING_", {"KV_CACHE_DTYPE": Setting("bfp8", ("bfp8", "bf16"), "owner 2026-10-01: ...")})
    S.get("KV_CACHE_DTYPE")  # "bfp8", or $XING_KV_CACHE_DTYPE
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    default: object
    choices: tuple | None = None  # None: any value of the default's type
    why: str = ""
    emulation: tuple = ()  # choices that round to a format the device neither computes nor stores in (rule 17)
    # F222: the value under pytest, only when bit-identical to the default (its why names the test that proves it)
    test: object = None


class Settings:
    def __init__(self, prefix: str, table: dict[str, Setting]):
        self.prefix, self.table = prefix, dict(table)
        # snapshot(): every switch's value read once (None = get() reads the environment on every call, as before)
        self.snap: dict | None = None

    def env_name(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def get(self, name: str):
        snap = self.snap
        if snap is not None:  # (a model's load froze the values: no environment read per call)
            return snap[name]
        return self._read(name)

    def snapshot(self) -> None:
        """Read every switch now and serve get() from these values (no os.environ read per call) until ``thaw()`` or
        the next ``snapshot()``. A model calls it at load time when its own switch asks for it (DeepSeek-V4.1 P.21
        HOST_GLUE: ~1500 get() calls a chunk); a process that changes the environment after it calls ``refresh()``."""
        self.snap = None
        self.snap = {name: self._read(name) for name in self.table}

    def thaw(self) -> None:
        """get() reads the environment on every call again (the default)."""
        self.snap = None

    def refresh(self) -> None:
        """Re-read a snapshot after the environment changed (in-process A/B tests); no-op when not frozen."""
        if self.snap is not None:
            self.snapshot()

    def _read(self, name: str):
        s = self.table[name]
        raw = os.environ.get(self.env_name(name))
        if raw is None:
            return s.test if s.test is not None and "pytest" in sys.modules else s.default
        val = type(s.default)(raw) if not isinstance(s.default, bool) else raw.lower() in ("1", "true", "yes")
        if s.choices is not None and val not in s.choices:
            raise ValueError(f"{self.env_name(name)}={raw!r}: expected one of {s.choices}")
        return val

    def all(self) -> dict:
        """Every switch's current value (the profile and the dashboard record these)."""
        return {name: self.get(name) for name in self.table}
