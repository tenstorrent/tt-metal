# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests for the AGMM v3 rule engine (`utils/agmm_rules.py`).

The fixture (`agmm_rules_fixture.json`) pins the rules to the sweep campaign that produced
them: 159 swept (shape, projection) points on 4x8 Blackhole Galaxy, each row recording the
rules' pick plus its measured time and the swept optimum. Two properties are asserted:

  1. Behavior pin: `pick_v3` reproduces every fixture row exactly. Any edit to the rules that
     changes a pick fails here -- if the change is intentional, re-sweep (or re-score against
     `sweep_results_mm.csv`) and regenerate the fixture.
  2. Quality floor: >= 95% of rows are within 5% of the swept optimum (measured 153/159, with
     73 exact winners; blind validation was 86/89).

No device, ttnn, or torch required.
"""

import json
from pathlib import Path

import pytest

from models.tt_dit.utils.agmm_rules import pick_layout, pick_v3

FIXTURE = Path(__file__).parent / "agmm_rules_fixture.json"


def _rows():
    return json.loads(FIXTURE.read_text())


def test_pick_v3_reproduces_fixture():
    mismatches = []
    for row in _rows():
        got = pick_v3(
            row["M"],
            row["K"],
            row["N"],
            cluster_size=4,
            fuse_swiglu=row["fuse_swiglu"],
            use_addcmul=row["use_addcmul"],
        )
        want = {
            "core_grid": tuple(row["core_grid"]),
            "transposed": row["transposed"],
            "blocks": tuple(row["blocks"]),
            "subblock": tuple(row["subblock"]),
        }
        if got != want:
            mismatches.append((row["proj"], row["M"], want, got))
    assert not mismatches, f"{len(mismatches)} picks changed vs fixture; first: {mismatches[0]}"


def test_quality_floor_within_5pct():
    rows = _rows()
    ok = sum(1 for r in rows if r["pred_us"] <= 1.05 * r["best_us"])
    assert ok / len(rows) >= 0.95, f"only {ok}/{len(rows)} fixture rows within 5% of swept optimum"


def test_layout_basics():
    # M > N: the op auto-transposes; only the free-bottom-row grid is legal.
    assert pick_layout(8000, 5376, 5376) == ((12, 9), True)
    # Production H3 shapes: non-transposed full-width 12x9.
    assert pick_layout(3249, 5376, 5376) == ((12, 9), False)
    assert pick_layout(3249, 5376, 7168, fuse_swiglu=True) == ((12, 9), False)
    # Clean 10-row fits favor the narrow grid (score rule).
    assert pick_layout(960, 5376, 5376) == ((11, 10), False)
    # Tiny-M latency regime prefers full width despite the score.
    assert pick_layout(320, 5376, 5376) == ((12, 9), False)


def test_hard_constraints_hold():
    for row in _rows():
        m_blk, k_blk, n_blk = row["blocks"]
        sub_h, sub_w = row["subblock"]
        assert (row["K"] // 32 // 4) % k_blk == 0, "K_block must divide K_tiles_per_device"
        assert m_blk % sub_h == 0 and n_blk % sub_w == 0
        assert sub_h * sub_w <= 4, "fp32 dest halves DEST to 4 tiles"
        if row["fuse_swiglu"]:
            assert n_blk % 2 == 0, "fused SwiGLU requires even N_block"


def test_rules_decline_gracefully():
    # K not tile-aligned, or K_tiles not divisible by the ring size: caller keeps legacy path.
    assert pick_v3(3249, 5377, 5376, cluster_size=4) is None
    assert pick_v3(3249, 5376, 5376, cluster_size=5) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
