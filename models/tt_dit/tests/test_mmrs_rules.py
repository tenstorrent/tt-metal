# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only regression tests for the MMRS v2.3 rule engine (`utils/mmrs_rules.py`).

The fixture (`mmrs_rules_fixture.json`) pins the rules to the sweep campaign that produced
them: every measured 12x8 fused-MMRS shape across the five ff2 families (H3 3584x5376,
Wan2.2 3456x5120, LTX 4096x4096, Flux 2304x6144 / 3072x6144), each row recording the rules'
pick plus its measured time and the swept optimum. Two properties are asserted:

  1. Behavior pin: `pick_v23` reproduces every fixture row exactly. Any edit to the rules that
     changes a pick fails here -- if the change is intentional, re-sweep (or re-score against
     `sweep_results_mm.csv`) and regenerate the fixture.
  2. Quality floor: >= 95% of rows are within 5% of the swept optimum (measured 96/98; blind
     validation was 35/40 for v2 plus a dedicated pre-registered round for the v2.3 short-N
     branch).

No device, ttnn, or torch required.
"""

import json
from pathlib import Path

from models.tt_dit.utils.mmrs_rules import pick_subblock, pick_v23

FIXTURE = Path(__file__).parent / "mmrs_rules_fixture.json"


def _rows():
    return json.loads(FIXTURE.read_text())


def test_pick_v23_reproduces_fixture():
    mismatches = []
    for row in _rows():
        got = pick_v23(row["M"], row["K"], row["N"])
        want = {
            "mm_grid": (12, 8),
            "blocks": tuple(row["blocks"]),
            "subblock": tuple(row["subblock"]),
        }
        if got != want:
            mismatches.append((row["family"], row["M"], want, got))
    assert not mismatches, f"{len(mismatches)} picks changed vs fixture; first: {mismatches[0]}"


def test_quality_floor_within_5pct():
    rows = _rows()
    ok = sum(1 for r in rows if r["pred_us"] <= 1.05 * r["best_us"])
    assert ok / len(rows) >= 0.95, f"only {ok}/{len(rows)} fixture rows within 5% of swept optimum"


def test_rule_structure():
    # H3 / Wan / Flux (N_tiles > 128) stay on N_block = 8 with the family k.
    assert pick_v23(3424, 3584, 5376)["blocks"] == (6, 4, 8)
    assert pick_v23(9472, 3456, 5120)["blocks"] == (6, 4, 8)
    assert pick_v23(6921, 2304, 6144)["blocks"] == (6, 3, 8)  # 72 K-tiles -> k = 3
    # Short-N regime (LTX 4096x4096, N_tiles = 128): N_block 6 below the pcM boundary, wide
    # N_block 16 above it with K_block walked down 8 -> 4 so the CBs fit L1.
    assert pick_v23(4864, 4096, 4096)["blocks"] == (6, 8, 6)
    assert pick_v23(13593, 4096, 4096)["blocks"] == (6, 4, 16)
    # Small M drops M_block to keep balanced halves (even-snapped, floor 2).
    assert pick_v23(1216, 4096, 4096)["blocks"][0] == 4
    # Out of scope -> None: unaligned K, K_tiles with no divisor in [2, 16], foreign grid.
    assert pick_v23(3424, 3600, 5376) is None
    assert pick_v23(3424, 32 * 17, 5376) is None  # prime K_tiles > 16: no K_block divisor
    assert pick_v23(3424, 3584, 5376, full_grid=(8, 9)) is None


def test_hard_constraints_hold():
    for row in _rows():
        m_blk, k_blk, n_blk = row["blocks"]
        sub_h, sub_w = row["subblock"]
        assert (row["K"] // 32) % k_blk == 0, "K_block must divide K_tiles"
        assert m_blk % sub_h == 0 and n_blk % sub_w == 0
        assert sub_h * sub_w <= 4, "fp32 dest halves DEST to 4 tiles"
        assert m_blk % 2 == 0 and m_blk >= 2, "M_block is even-snapped with floor 2"


def test_pick_subblock():
    assert pick_subblock(6, 8) == (2, 2)
    assert pick_subblock(3, 8) in ((1, 4), (3, 1))  # h*w <= 4, dividing both
    h, w = pick_subblock(5, 7)
    assert h * w <= 4 and 5 % h == 0 and 7 % w == 0
