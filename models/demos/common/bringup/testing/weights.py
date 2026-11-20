# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F281 (OWNER 2026-11-19, "iteration time"): the resident runner's device-weight registry. A model rebuilt in the
same process (its python changed, testing/resident.py) takes the device copies of the weights it loaded before
instead of uploading them again, when their source and layout are the same.

The registry sits under a model's weight-upload chokepoint for FILE-BACKED weights: ``load_tensor(path, device)``
(ttnn.load_tensor of a laid-out weight-cache file; DeepSeek-V4.1's routed experts, ~1.4 s a layer and nearly all of
the model's device bytes). Key: the file (real path, size, mtime) and the device (mesh id). A hit is a tensor the
previous build loaded from the same unchanged file, still allocated; each registered copy serves one load of a build
(two layers loading the same file get two copies, as without the registry). A changed file (a new layout or format
writes another file, or the same file again) misses and loads; the copies no build claimed are dropped at the end of
the next build (``end_build``).

Not registered (re-uploaded on a rebuild): tensors made by ``ttnn.from_torch`` from host data (the dense weights, the
load-time tables). At that chokepoint a weight cannot be told from a load-time buffer the model writes later (a
from_torch of zeros that becomes a running state); handing a written buffer to the new model would change its
numbers. Their host preparation (checkpoint read, rounding statistics, ~2 s a layer on DeepSeek-V4.1) is most of their
load time anyway; their upload is a fraction of it.

Outside a resident process (``active()`` False) ``load_tensor`` is ttnn.load_tensor: nothing kept, no behaviour
change.
"""

from __future__ import annotations

import os
from pathlib import Path

_STATE = {"on": False, "building": False}
_AVAIL: dict = {}  # key -> [tensor]: copies an earlier build loaded, not yet claimed by this build
_BUILT: dict = {}  # key -> [tensor]: copies this build holds (claimed or newly loaded)
STATS = {"reused": 0, "loaded": 0, "reused_bytes": 0, "loaded_bytes": 0, "dropped": 0}


def enable(on: bool = True) -> None:
    """The resident server turns the registry on for its process (testing/resident.py)."""
    _STATE["on"] = bool(on)


def active() -> bool:
    return _STATE["on"]


def begin_build() -> None:
    """A model build starts: every copy the earlier builds hold becomes available to it, the counters restart."""
    for k, ts in _BUILT.items():
        _AVAIL.setdefault(k, []).extend(ts)
    _BUILT.clear()
    _STATE["building"] = True
    for k in STATS:
        STATS[k] = 0


def end_build() -> dict:
    """A model build ended: the copies it did not claim are dropped (their weights changed or are not used now)."""
    n = sum(len(v) for v in _AVAIL.values())
    _AVAIL.clear()
    _STATE["building"] = False
    STATS["dropped"] = n
    return dict(STATS)


def clear() -> None:
    """Drop every registered copy (the process restarts, or its model is released for good)."""
    _AVAIL.clear()
    _BUILT.clear()
    _STATE["building"] = False


def held() -> int:
    return sum(len(v) for v in _BUILT.values()) + sum(len(v) for v in _AVAIL.values())


def _file_key(path, device) -> tuple:
    p = Path(path).resolve()
    st = p.stat()
    return ("file", str(p), int(st.st_size), int(st.st_mtime_ns), id(device))


def _alive(t) -> bool:
    try:
        return bool(t.is_allocated())
    except Exception:  # noqa: BLE001 -- a tensor kind without the query: as alive as its object
        return True


def load_tensor(path, device, load=None):
    """``ttnn.load_tensor(path, device=device)`` through the registry (a resident process) or as is (anywhere else).
    ``load(path, device)``: the loader (tests; default ttnn.load_tensor)."""
    if load is None:
        import ttnn

        def load(p, d):
            return ttnn.load_tensor(str(p), device=d)

    if not _STATE["on"]:
        return load(path, device)
    key = _file_key(path, device)
    size = int(key[2])
    pool = _AVAIL.get(key) or []
    while pool:
        t = pool.pop()
        if _alive(t):
            _BUILT.setdefault(key, []).append(t)
            STATS["reused"] += 1
            STATS["reused_bytes"] += size
            return t
    t = load(path, device)
    _BUILT.setdefault(key, []).append(t)
    STATS["loaded"] += 1
    STATS["loaded_bytes"] += size
    return t


def describe(stats: dict | None = None) -> str:
    s = stats or STATS
    gib = float(1 << 30)
    return (
        f"device weights: {s['reused']} file-backed tensors kept ({s['reused_bytes'] / gib:.1f} GiB of files), "
        f"{s['loaded']} loaded ({s['loaded_bytes'] / gib:.1f} GiB), {s.get('dropped', 0)} dropped; dense weights and "
        "load-time tables uploaded again (testing/weights.py)"
    )


def env_on() -> bool:
    """The resident server's process (BRINGUP_RESIDENT_DIR set by its supervisor)."""
    return bool(os.environ.get("BRINGUP_RESIDENT_DIR"))
