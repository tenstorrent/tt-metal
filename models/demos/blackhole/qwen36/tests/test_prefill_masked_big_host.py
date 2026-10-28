# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only test of the QWEN36_PREFILL_MASKED_BIG remainder split (tt/prefill_split.py: pure python, no ttnn).

Expected plans below are hand-computed from the default cost table (seconds, cost = A + B * depth, depth = start of the
replay): 2048: (0.238, 3.6e-6), 4096: (0.42, 4.0e-6), 8192: (0.78, 4.6e-6); a tail bucket <= 2048 scales A by bucket/2048.
Example r=2100, depth 131072: old = 0.238 + 3.6e-6*131072 + (0.238/16 + 3.6e-6*133120) = 1.204 s; one masked 4096 =
0.42 + 4.0e-6*131072 = 0.944 s -> m4096.

Run without the device conftest (this directory's conftest.py opens devices):
  cd <worktree> && PYTHONPATH=<worktree> tt-metal/python_env/bin/python -m pytest -q -p no:cacheprovider \
      --noconftest models/demos/blackhole/qwen36/tests/test_prefill_masked_big_host.py
"""

import importlib.util
import pathlib

import pytest

# Load the module by path: importing the package would pull in ttnn through tt/__init__ users.
_SPEC = importlib.util.spec_from_file_location(
    "prefill_split", pathlib.Path(__file__).resolve().parents[1] / "tt" / "prefill_split.py"
)
ps = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ps)

DEPTHS = (0, 32768, 131072)
EXPECTED = {
    100: ("old", "old", "old"),
    2048: ("old", "old", "old"),
    2100: ("old", "old", "m4096"),
    4096: ("m4096", "m4096", "m4096"),
    5000: ("old", "m8192", "m8192"),
    8191: ("m8192", "m8192", "m8192"),
}


@pytest.fixture(autouse=True)
def _default_table(monkeypatch):
    monkeypatch.delenv("QWEN36_PREFILL_COST_TABLE", raising=False)


@pytest.mark.parametrize("r", sorted(EXPECTED))
def test_plan_matches_hand_computed(r):
    for depth, want in zip(DEPTHS, EXPECTED[r]):
        plan, costs = ps.plan_remainder(r, depth, True)
        assert plan == want, (r, depth, plan, costs)
        assert costs[plan] == min(costs.values())


def test_hand_computed_costs():
    _, c = ps.plan_remainder(2100, 131072, True)
    assert c["old"] == pytest.approx(0.238 + 3.6e-6 * 131072 + 0.238 / 16 + 3.6e-6 * (131072 + 2048))
    assert c["m4096"] == pytest.approx(0.42 + 4.0e-6 * 131072)
    assert c["m8192"] == pytest.approx(0.78 + 4.6e-6 * 131072)
    _, c = ps.plan_remainder(8191, 0, True)  # k=3 chunks + a 2048 tail bucket
    assert c["old"] == pytest.approx(0.238 * 4 + 3.6e-6 * (0 + 2048 + 4096 + 6144))
    assert "m4096" not in c  # r > 4096


def test_flag_off_returns_old_split():
    for r in EXPECTED:
        for depth in DEPTHS:
            assert ps.plan_remainder(r, depth, False) == ("old", {})
    assert ps.old_split(2100) == (1, 52)
    assert ps.old_split(8191) == (3, 2047)
    assert ps.old_split(4096) == (2, 0)


def test_unavailable_bucket_is_not_chosen():
    plan, costs = ps.plan_remainder(4096, 131072, True, buckets=(8192,))
    assert "m4096" not in costs and plan in ("old", "m8192")
    plan, costs = ps.plan_remainder(5000, 131072, True, buckets=())
    assert plan == "old" and list(costs) == ["old"]


def test_tiny_remainder_and_zero_stay_old():
    assert ps.plan_remainder(2, 131072, True)[0] == "old"  # masked replay needs >= 3 real rows
    assert ps.plan_remainder(0, 131072, True)[0] == "old"


def test_cost_table_env_override(monkeypatch):
    monkeypatch.setenv("QWEN36_PREFILL_COST_TABLE", '{"4096": [0.0, 0.0]}')
    assert ps.cost_table()[4096] == (0.0, 0.0) and ps.cost_table()[2048] == (0.238, 3.6e-6)
    assert ps.plan_remainder(2100, 0, True)[0] == "m4096"
