# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""V41_PREFILL_HOSTPROF=1 (measurement only, default off): host time per step of an island chunk, with NO syncs, so a step's
time is its host issue (plus any wait inside a blocking call). ``V41Prefill._chunk_islands`` logs one line per chunk. Used to
size the whole-chunk trace (tt-blaze DS41F-0163: an encoder chunk is ~0.30 s of host issue against ~0.30 s of device time)."""

from __future__ import annotations

import os
import time
from collections import defaultdict

ON = os.environ.get("V41_PREFILL_HOSTPROF", "0") == "1"

_acc: dict = defaultdict(float)
_cnt: dict = defaultdict(int)
_t = [0.0]
_pfx = [""]  # WS6: "d." while a decoder layer (20..39) runs, so its steps report apart from the encoder's


def set_prefix(p: str) -> None:
    _pfx[0] = p


def start() -> None:
    _acc.clear()
    _cnt.clear()
    _t[0] = time.perf_counter()


def mark(name: str) -> None:
    """The time since the previous mark (or ``start``) goes to ``name``."""
    now = time.perf_counter()
    name = _pfx[0] + name
    _acc[name] += now - _t[0]
    _cnt[name] += 1
    _t[0] = now


def report() -> str:
    total = sum(_acc.values())
    parts = " ".join(f"{k}={v * 1e3:.1f}/{_cnt[k]}" for k, v in sorted(_acc.items(), key=lambda kv: -kv[1]))
    return f"total {total * 1e3:.1f} ms: {parts}"
