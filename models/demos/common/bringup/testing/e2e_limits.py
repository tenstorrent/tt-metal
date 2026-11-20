# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F128 (owner): every end-to-end accuracy limit (ladder per layer, state and index overlap, final hidden, top-5; the
contract's KV read-back) is calib x the reference's own drift from full precision on the same comparison
(reference/drift.py, F134). F135: the same convention as per step (testing/noise_floor.py): a float tensor is gated by
its rel L2 error, limit = calib x max(the drift's rel, one rounding's rel); a selection by its overlap, limit =
1 - calib x max(the drift's miss, one miss); a PCC (only where a read-back reports nothing else: the contract) at
1 - calib^2 x its miss (1 - pcc ~ rel^2 / 2). No fixed bar; a rung without its drift fails, never falls back."""

from __future__ import annotations

import json
import re
from pathlib import Path

from models.demos.common.bringup.core import defaults, metrics

CALIB = float(defaults.get("e2e.calib"))
STATE_KEY = re.compile(r"^(state|contract_state)_(.+)_L\d+$")


def load(golden_dir: Path, rung: str) -> dict:
    """The rung's drift (reference/drift.py: the official reference fed as the device is on that rung, F134)."""
    from models.demos.common.bringup.reference.drift import drift_path

    p = drift_path(golden_dir, rung)
    if not p.exists():
        raise FileNotFoundError(
            f"{p}: no end-to-end margin for rung {rung}; its golden's G task makes it (reference.generate_golden "
            f"--rung <the golden's rung> --set), or reference.drift --rung {rung} (F128, F134)"
        )
    return json.loads(p.read_text())


def _entry(drift: dict, key: str) -> dict:
    if key not in drift:
        src = {e.get("source", "reference") for e in drift.values()}
        raise KeyError(f"no drift for {key!r} on this rung (margin source {', '.join(sorted(src))}; F134, F136)")
    return drift[key]


def err_limit(drift: dict, key: str) -> float:
    """calib x the drift's rel (a float tensor) or its missed fraction (a selection: 1 - overlap, one miss at least)."""
    e = _entry(drift, key)
    if "overlap" in e:
        return CALIB * max(1.0 - e["overlap"], 1.0 - e["overlap_res"])
    return CALIB * max(e["rel"], _res(e))


def _res(e: dict) -> float:
    """F159: one rounding moves a value at most to 0, so its rel is <= 1; above 1 the target was exactly zero (an
    all-zero state: DeepSeek-V4.1 compress_partial gave 9e255 and a 1.8e256 limit): no rounding error, limit calib x rel.
    """
    return 0.0 if e.get("zero_target") or e["rel_res"] > 1.0 else e["rel_res"]


def pcc_limit(drift: dict, key: str) -> float:
    e = _entry(drift, key)
    res = 1.0 if e.get("zero_target") or e.get("rel_res", 0.0) > 1.0 else e["pcc_res"]  # F159: an all-zero target
    return 1.0 - CALIB * CALIB * max(1.0 - e["pcc"], 1.0 - res)


def overlap_limit(drift: dict, key: str) -> float:
    e = _entry(drift, key)
    return 1.0 - CALIB * max(1.0 - e["overlap"], 1.0 - e["overlap_res"])


def limit(drift: dict, key: str, metric: str) -> float:
    """The limit of ``metric`` (err_ / pcc_ / an overlap) on the drift entry ``key``."""
    if metric.startswith("err_"):
        return err_limit(drift, key)
    return pcc_limit(drift, key) if metric.startswith("pcc_") else overlap_limit(drift, key)


def check(drift: dict, key: str, metric: str, value: float, failed: list[str]) -> bool:
    """Records ``<metric>_limit`` (the gate's `<= limit` for an error, `>= limit` otherwise) and appends a failure when
    ``value`` is on the wrong side of it."""
    lim = limit(drift, key, metric)
    metrics.record(f"{metric}_limit", lim)
    err = metric.startswith("err_")
    if value > lim if err else value < lim:
        failed.append(
            f"{metric} {value:.5g} {'>' if err else '<'} limit {lim:.5g} (calib {CALIB} x the reference's drift)"
        )
        return False
    return True


def state_limit(drift: dict, name: str, kind: str = "contract_state") -> float:
    """The PCC limit of a state tensor's worst layer (the contract's KV read-back reports each state's worst layer
    as a PCC); kind state: the ladder's comparison."""
    keys = [k for k in drift if (m := STATE_KEY.match(k)) and m.group(1) == kind and m.group(2) == name]
    if not keys:
        raise KeyError(f"no {kind} drift for state {name!r} (F128)")
    return min(pcc_limit(drift, k) for k in keys)
