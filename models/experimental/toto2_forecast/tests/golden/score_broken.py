# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: gate metrics of the measured device outputs and of the synthetic broken outputs (PLAN section 9 v1.9).

    python -m models.experimental.toto2_forecast.tests.golden.score_broken <saved outputs dir> <out json>

Broken outputs, built from the 2.5B-FT golden head output and quantiles (band = value - mean over the 9 quantiles):
    a  the bands of every inactive series collapsed to their mean
    b  about 1 % of the active series (at least one) at a 0.3x band
    c  every band scaled by 0.88x
    d  every band scaled by 1.12x
Each column is scored against both precisions' floors (``floors_for``, or ``PROPOSED_FLOORS`` with ``--proposed``). The measured columns are the
``BENCH_SAVE_DIR`` outputs of ``test_forward_perf.py`` (``2.5B-FT-<golden>-<precision>.pt``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from models.experimental.toto2_forecast.tests.golden.goldens import (
    active_rows,
    below_floors,
    floors_for,
    gate_metrics,
    load_golden,
)
from models.experimental.toto2_forecast.tests.golden import goldens


CKPT = "2.5B-FT"
CASES = ("paper_mix_1024x2048", "multivariate_mix_128x8x2048", "long_context_mix_32x8192", "single_sparse_1x2048")


def _scale_rows(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``t`` (9, B, V, ...): each series' band scaled by ``scale`` (B * V,)."""
    mean = t.mean(0, keepdim=True)
    s = scale.reshape(1, t.shape[1], t.shape[2], *([1] * (t.dim() - 3))).to(t.dtype)
    return mean + s * (t - mean)


def broken(head: torch.Tensor, q: torch.Tensor, kind: str):
    active = active_rows(head)
    n = active.numel()
    scale = torch.ones(n, dtype=torch.float64)
    if kind == "a":
        scale[~active] = 0.0
    elif kind == "b":
        idx = active.nonzero().flatten()
        k = max(1, round(0.01 * idx.numel()))
        scale[idx[torch.linspace(0, idx.numel() - 1, k).long()]] = 0.3
    else:
        scale[:] = {"c": 0.88, "d": 1.12}[kind]
    return _scale_rows(head, scale), _scale_rows(q, scale)


def main(argv=None) -> int:
    """``argv``: saved outputs dir, out json, and ``--proposed`` to score against ``PROPOSED_FLOORS``."""
    saved, out = Path(argv[0]), Path(argv[1])
    table_floors = goldens.PROPOSED_FLOORS if "--proposed" in argv else None

    def floors(p):
        return table_floors[p] if table_floors else floors_for(CKPT, p)

    table = {}
    for case in CASES:
        g = load_golden(CKPT, case)
        gh, gq = g["captures"]["head_out"], g["captures"]["q"]
        cols = {}
        for f in sorted(saved.glob(f"{CKPT}-{case}-*.pt")):
            d = torch.load(f, weights_only=False)
            cols[f.stem.split(f"{case}-", 1)[1]] = gate_metrics(d["head"], d["q"], gh, gq)
        for kind in "abcd":
            h, qq = broken(gh, gq, kind)
            cols[f"broken_{kind}"] = gate_metrics(h, qq, gh, gq)
        for col in cols.values():
            col["below"] = {p: below_floors(col, floors(p)) for p in ("accuracy", "performance")}
        table[case] = cols
    out.write_text(json.dumps(table, indent=1, default=str) + "\n")
    for case, cols in table.items():
        print(case)
        for name, col in cols.items():
            fails = {p: sorted(v) for p, v in col["below"].items()}
            print(f"  {name:24s} " + "  ".join(f"{p[:4]}: {','.join(f) or 'PASS'}" for p, f in fails.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
