# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden files (made by ``tests/golden/make_goldens.py`` in the reference venv) and PCC helpers."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import torch

GOLDEN_DIR = Path(os.environ.get("TIMESFM3_GOLDENS", "/home/andy/bringup/goldens/timesfm3"))
MANIFEST = Path(__file__).resolve().parent / "golden" / "MANIFEST.json"


@functools.lru_cache(maxsize=None)
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


@functools.lru_cache(maxsize=4)
def load_golden(name: str) -> dict:
    """A golden by name, after checking its sha256 against ``MANIFEST.json``."""
    path = GOLDEN_DIR / f"{name}.pt"
    expected = manifest()["files"][name]["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise RuntimeError(f"{path}: sha256 {h.hexdigest()} != MANIFEST {expected}; regenerate the goldens")
    return torch.load(path, weights_only=True)


def pcc(expected: torch.Tensor, got: torch.Tensor) -> float:
    from tests.ttnn.utils_for_testing import comp_pcc

    return float(comp_pcc(expected.double(), got.double(), 0.0)[1])


def check_pcc(name: str, expected: torch.Tensor, got: torch.Tensor, floor: float, where=None) -> float:
    """Shape, finiteness (everywhere) and PCC (over ``where`` if given) against ``floor``."""
    expected, got = torch.as_tensor(expected), torch.as_tensor(got)
    assert tuple(got.shape) == tuple(expected.shape), f"{name}: shape {tuple(got.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(got).all(), f"{name}: non-finite output"
    if where is not None:
        expected, got = expected[where], got[where]
    value = pcc(expected, got)
    print(f"[PCC] {name}: {value:.6f} (floor {floor})", flush=True)
    assert value >= floor, f"{name}: PCC {value:.6f} < {floor}"
    return value


# ------------------------------------------------------------------------------------------ gate metrics
# PLAN section 9 (v1.1): ``pcc_normalized`` alone is dominated by per-row offsets, so a forecast whose
# quantiles all collapse to their mean can still score ~0.9997. The head output is (rows, F, 64 * 9) in
# normalized space; these metrics look at the quantile spread and at the worst rows.
NUM_QUANTILES = 9
# Rows whose reference head varies by less than this fraction of the whole head's std (e.g. an intermittent
# series forecast flat at zero, std ~0.002 vs 1.19) have an ill-conditioned per-row PCC: their absolute error is
# smaller than a typical row's, yet it is comparable to their own variation. ``*_active`` skips those rows.
ACTIVE_ROW_STD = 0.05
# Floors proposed at G1 from round 0 (None = reported, not enforced); the orchestrator sets them.
METRIC_FLOORS = {
    "accuracy": {
        "pcc_normalized": 0.999,
        "pcc_spread": 0.998,
        "pcc_row_p01": 0.5,
        "pcc_row_min": 0.2,
        "pcc_row_p01_active": 0.98,
        "pcc_row_min_active": 0.9,
    },
    # bfloat8_b FFN weights put a systematic error pattern on every row (round 0, PLAN section 9 metrics): the
    # per-row metrics go negative on flat rows. Reported, not enforced, until a per-role exception is chosen.
    "performance": {
        "pcc_normalized": 0.995,
        "pcc_spread": 0.99,
        "pcc_row_p01": None,
        "pcc_row_min": None,
        "pcc_row_p01_active": None,
        "pcc_row_min_active": None,
    },
}


def _pcc_rows(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Row-wise PCC of two (rows, n) float64 tensors; a constant row scores 1 if both are equal, else 0."""
    e = e - e.mean(-1, keepdim=True)
    g = g - g.mean(-1, keepdim=True)
    den = e.norm(dim=-1) * g.norm(dim=-1)
    num = (e * g).sum(-1)
    out = torch.where(den > 0, num / den.clamp_min(1e-300), torch.zeros_like(num))
    same = (e.abs().amax(-1) == 0) & (g.abs().amax(-1) == 0)
    return torch.where(same, torch.ones_like(out), out)


def head_metrics(expected: torch.Tensor, got: torch.Tensor) -> dict:
    """``pcc_normalized`` (all values), ``pcc_spread`` (the head minus its mean over the 9 quantiles),
    ``pcc_row_p01`` / ``pcc_row_min`` (per-row PCC, 1st percentile and minimum) of (rows, F, 576) heads."""
    e = torch.as_tensor(expected).double().reshape(expected.shape[0], -1, NUM_QUANTILES)
    g = torch.as_tensor(got).double().reshape(got.shape[0], -1, NUM_QUANTILES)
    flat = lambda t: t.reshape(1, -1)  # noqa: E731
    spread = lambda t: t - t.mean(-1, keepdim=True)  # noqa: E731
    er, gr = e.reshape(e.shape[0], -1), g.reshape(g.shape[0], -1)
    rows = _pcc_rows(er, gr)
    active = er.std(-1) >= ACTIVE_ROW_STD * er.std()
    act = rows[active] if active.any() else rows
    return {
        "pcc_normalized": float(_pcc_rows(flat(e), flat(g))[0]),
        "pcc_spread": float(_pcc_rows(flat(spread(e)), flat(spread(g)))[0]),
        "pcc_row_p01": float(torch.quantile(rows, 0.01)),
        "pcc_row_min": float(rows.min()),
        "pcc_row_p01_active": float(torch.quantile(act, 0.01)),
        "pcc_row_min_active": float(act.min()),
        "rows_inactive": int((~active).sum()),
    }


def collapse_quantiles(head: torch.Tensor) -> torch.Tensor:
    """The deliberately broken output: every quantile replaced by the mean over the 9 quantiles."""
    h = torch.as_tensor(head).reshape(head.shape[0], -1, NUM_QUANTILES)
    return h.mean(-1, keepdim=True).expand_as(h).reshape(head.shape).contiguous()


def check_head_metrics(name: str, expected, got, precision: str = "accuracy") -> dict:
    """Every gate metric of a head output against its floor (finite output required)."""
    got = torch.as_tensor(got)
    assert torch.isfinite(got).all(), f"{name}: non-finite head output"
    m = head_metrics(expected, got)
    floors = METRIC_FLOORS[precision]
    print(f"[METRICS] {name}: " + ", ".join(f"{k} {v:.6f} (floor {floors.get(k)})" for k, v in m.items()), flush=True)
    low = {k: v for k, v in m.items() if floors.get(k) is not None and v < floors[k]}
    assert not low, f"{name}: below floor: {low}"
    return m


def final_metrics(expected, got) -> dict:
    """``pcc_final`` (data-space quantiles) and ``pcc_final_spread`` (minus their mean over the quantile axis).
    Data-space levels dominate ``pcc_final``: a collapsed forecast still scores ~0.9996 on it."""
    e, g = torch.as_tensor(expected).double(), torch.as_tensor(got).double()
    spread = lambda t: (t - t.mean(-1, keepdim=True)).reshape(1, -1)  # noqa: E731
    return {
        "pcc_final": float(_pcc_rows(e.reshape(1, -1), g.reshape(1, -1))[0]),
        "pcc_final_spread": float(_pcc_rows(spread(e), spread(g))[0]),
    }
