# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the host-side spec-step fixes (QWEN36_SPEC_HOSTFIX): new vs old code paths. No device.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_hostfix_host.py -q
"""

import pytest
import torch

from models.demos.blackhole.qwen36.tt.model import _spec_kv_write_conflicts_old, spec_kv_write_conflicts


# ---- R5: spec_kv_write_conflicts ------------------------------------------------------------------------
def _call(fn, *args):
    try:
        return ("ok", fn(*args))
    except IndexError as e:
        return ("err", str(e))


@pytest.mark.parametrize("seed", range(40))
def test_conflicts_random_match_old(seed):
    g = torch.Generator().manual_seed(seed)
    B, T, bs, nb = 8, 4, 64, int(torch.randint(2, 40, (1,), generator=g))
    # few distinct blocks and close positions force same-tile collisions; zeros give null blocks
    pt = torch.randint(0, 4, (B, nb), generator=g, dtype=torch.int32)
    pos = torch.randint(-3, nb * bs - T, (B,), generator=g).tolist()
    if seed % 2:
        pos = [int(torch.randint(0, 2 * bs, (1,), generator=g)) if p >= 0 else p for p in pos]
    for pt_in in (pt, pt.to(torch.int64)):
        assert _call(spec_kv_write_conflicts, pt_in, pos, T, bs) == _call(
            _spec_kv_write_conflicts_old, pt_in, pos, T, bs
        )


def test_conflicts_edge_cases(monkeypatch):
    bs, T = 64, 4
    pt = torch.zeros(8, 6, dtype=torch.int32)
    assert spec_kv_write_conflicts(pt, [-1] * 8, T, bs) == []  # all rows inactive
    assert spec_kv_write_conflicts(pt, [5] * 8, T, bs) == []  # all null blocks
    pt[:, 0] = 7  # all rows live on the same block and tile: every call collides
    pos = [3, 3, 40, 40, 70, -1, 0, 31]
    got = spec_kv_write_conflicts(pt, pos, T, bs)
    assert got == _spec_kv_write_conflicts_old(pt, pos, T, bs) and got
    # duplicate slots (identical rows and positions)
    assert spec_kv_write_conflicts(pt, [10, 10], T, bs) == _spec_kv_write_conflicts_old(pt, [10, 10], T, bs)
    # position past the table: same IndexError text; the first offender in (j, u) order is reported
    pos = [0, 6 * bs - 2, 6 * bs - 1, 0, 0, 0, 0, 0]
    a, b = _call(spec_kv_write_conflicts, pt, pos, T, bs), _call(_spec_kv_write_conflicts_old, pt, pos, T, bs)
    assert a == b and a[0] == "err"
    # list input, ragged rows, tensor positions, fewer positions than rows, empty positions
    rows = [[1, 2, 3], [1, 2], [4]]
    assert spec_kv_write_conflicts(rows, [0, 0, 0], T, bs) == _spec_kv_write_conflicts_old(rows, [0, 0, 0], T, bs)
    assert spec_kv_write_conflicts(pt, torch.tensor([0, 5, 9]), T, bs) == _spec_kv_write_conflicts_old(
        pt, [0, 5, 9], T, bs
    )
    assert spec_kv_write_conflicts(pt, [], T, bs) == []
    # more positions than rows: handled by the old code
    assert _call(spec_kv_write_conflicts, pt[:2], [0, 0, 0], T, bs)[0] == "err"
    assert _call(spec_kv_write_conflicts, pt[:2], [0, 0, 0], T, bs) == _call(
        _spec_kv_write_conflicts_old, pt[:2], [0, 0, 0], T, bs
    )


def test_conflicts_flag_off_is_old(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_HOSTFIX", "0")
    pt = torch.full((4, 3), 9, dtype=torch.int32)
    assert spec_kv_write_conflicts(pt, [0, 1, 2, 3], 4, 64) == _spec_kv_write_conflicts_old(pt, [0, 1, 2, 3], 4, 64)
