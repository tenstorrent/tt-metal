# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pure CPU tests (no ttnn, no device) of tt/prefill_plan.py, the planner of QWEN36_PREFILL_INTRACE_SNAP (one segment
per spec fast-prefill call, GDN prefix-cache snapshots taken inside the replays) and QWEN36_PREFILL_MASKED_BIG (masked
4096 / 8192 remainder): segment_replays reproduces _fast_segment's replay list, assign_points maps every save point to
one replay (inside / at its end), plan_remainder's cost choice, snap_index_words' device words. The engine side is in
test_prefill_tail_planner_host.py.

Run without the device conftest (this directory's conftest.py opens devices):
  cd <worktree> && python -m pytest -q -p no:cacheprovider --noconftest \
      models/demos/blackhole/qwen36/tests/test_prefill_plan_host.py
"""
import importlib.util
import pathlib

import pytest

# prefill_plan is pure python: load it by path (the package import pulls in ttnn through tt/__init__ users).
_SPEC = importlib.util.spec_from_file_location(
    "prefill_plan_under_test", pathlib.Path(__file__).resolve().parents[1] / "tt" / "prefill_plan.py"
)
pp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pp)


# --------------------------------------------------------------------------- #
# prefill_plan (pure)
# --------------------------------------------------------------------------- #
def _fast_segment_reference(a, b, big, mb=None, chunk=2048):
    """_fast_segment's replay list written out the way the engine reads: masked big path, short path, chunks + tail."""
    n = b - a
    num_full, tail = divmod(n, chunk)
    if num_full == 0:
        return [("tail", a, n)]
    if mb:
        n_big = n // big
        return [("big", a + i * big, big) for i in range(n_big)] + [("mbig", a + n_big * big, b - a - n_big * big)]
    rows = num_full * chunk
    n_big = rows // big if big else 0
    out = [("big", a + i * big, big) for i in range(n_big)]
    out += [("c2k", a + n_big * big + j * chunk, chunk) for j in range((rows - n_big * big) // chunk)]
    if tail:
        out.append(("tail", a + rows, tail))
    return out


@pytest.mark.parametrize("big", [0, 8192])
def test_segment_replays_match_fast_segment(big):
    for a in (0, 128, 8192, 16384 + 640):
        for n in (1, 3, 100, 127, 128, 2047, 2048, 2049, 4096, 6143, 8191, 8192, 8193, 8492, 16384 + 3, 32768 + 61):
            got = [(r.kind, r.start, r.rows) for r in pp.segment_replays(a, a + n, big)]
            assert got == _fast_segment_reference(a, a + n, big), (a, n, big)
            # contiguous cover of [a, a + n)
            reps = pp.segment_replays(a, a + n, big)
            assert reps[0].start == a and sum(r.rows for r in reps) == n
            assert all(x.start + x.rows == y.start for x, y in zip(reps, reps[1:]))


def test_segment_replays_masked_big():
    reps = pp.segment_replays(0, 8192 + 7000, 8192, 8192)
    assert [(r.kind, r.start, r.rows, r.bucket) for r in reps] == [("big", 0, 8192, 8192), ("mbig", 8192, 7000, 8192)]
    reps = pp.segment_replays(256, 256 + 3000, 8192, 4096)
    assert [(r.kind, r.start, r.rows, r.bucket) for r in reps] == [("mbig", 256, 3000, 4096)]
    with pytest.raises(AssertionError):
        pp.segment_replays(0, 8192 + 5000, 8192, 4096)  # 5000 rows do not fit the 4096 bucket


def test_assign_points_inside_and_end():
    reps = pp.segment_replays(24576, 32768, 8192)  # one big chunk
    assert pp.assign_points(reps, [32640]) == [([32640], None)]
    reps = pp.segment_replays(16384, 32768 + 61, 8192)  # big, big, tail 61
    where = pp.assign_points(reps, [24576, 32768])
    assert where == [([], 24576), ([], 32768), ([], None)]
    # a resumed join: cap 8320, learn point 8448, end point 9088 inside one short replay [8320, 9236)
    reps = pp.segment_replays(8320, 9236, 8192)
    assert [r.kind for r in reps] == ["tail"]
    assert pp.assign_points(reps, [8448, 9088]) == [([8448, 9088], None)]
    with pytest.raises(ValueError):
        pp.assign_points(reps, [8320])  # the segment start is never a save point (p > s)


def test_plan_remainder_costs_and_override(monkeypatch):
    monkeypatch.delenv("QWEN36_PREFILL_COST_TABLE", raising=False)
    t = pp.cost_table()
    plan, costs = pp.plan_remainder(7000, 0)
    old = sum(t["c2048"][0] + t["c2048"][1] * j * 2048 for j in range(3)) + t["s1024"][0] + t["s1024"][1] * 6144
    assert costs["old"] == pytest.approx(old)
    assert costs["m8192"] == pytest.approx(t["m8192"][0]) and "m4096" not in costs
    assert plan == min(costs, key=costs.get)
    # deep in the prompt the masked 8192 replay wins a 7000-row remainder (2048 replays pay the depth slope 3x)
    assert pp.plan_remainder(7000, 64 * 1024)[0] == "m8192"
    # a tiny remainder never masks (3 real rows minimum), r == 0 is "old"
    assert pp.plan_remainder(2, 10**6)[0] == "old" and pp.plan_remainder(0, 0) == ("old", {"old": 0.0})
    # only the captured buckets are candidates
    assert "m8192" not in pp.plan_remainder(7000, 0, buckets=(4096,))[1]
    monkeypatch.setenv("QWEN36_PREFILL_COST_TABLE", '{"m4096": [0.0, 0.0]}')
    assert pp.plan_remainder(3000, 0)[0] == "m4096"


def test_snap_index_words():
    rec, conv = pp.snap_index_words([], 4)
    assert rec == [0x7FFFFFFF] * 8 and conv == [0] * 16
    rec, conv = pp.snap_index_words([128, 1920], 4)
    assert rec == [3, 59] + [0x7FFFFFFF] * 6
    assert conv[:6] == [125, 126, 127, 1917, 1918, 1919] and conv[6:] == [0] * 10
    with pytest.raises(AssertionError):
        pp.snap_index_words([128, 256, 384, 512, 640], 4)  # more points than slots
    with pytest.raises(AssertionError):
        pp.snap_index_words([200], 4)  # not on a 32-row chunk boundary / below 128


def test_plan_cost_one_segment_vs_split(monkeypatch):
    monkeypatch.delenv("QWEN36_PREFILL_COST_TABLE", raising=False)
    t = pp.cost_table()

    def c(key, depth):
        return t[key][0] + t[key][1] * depth

    # final 8K chunk of a 32K prompt: one big replay with the end point inside vs 3 x 2048 + tail + short128
    one = pp.plan_cost(24576, 32768, [32640], True)
    split = pp.plan_cost(24576, 32768, [32640], False)
    assert one == pytest.approx(c("c8192", 24576))
    assert split == pytest.approx(
        sum(c("c2048", 24576 + j * 2048) for j in range(3)) + c("s2048", 24576 + 6144) + c("s128", 32640)
    )
    assert one < split
    # a resumed join whose merge rounds 896 + 160 rows up to the 2048 bucket: the split is cheaper at this depth
    one = pp.plan_cost(8960, 10016, [9856], True)
    split = pp.plan_cost(8960, 10016, [9856], False)
    assert one == pytest.approx(c("s2048", 8960))
    assert split == pytest.approx(c("s1024", 8960) + c("s256", 9856))
    assert split < one
    # a 1-row tail of the whole call is an eager masked bucket; the split's last segment never is
    assert pp.plan_cost(0, 8193, [8064], True) == pytest.approx(c("c8192", 0) + c("eager", 8192))
    # the masked big choice applies per segment
    mb = pp.plan_cost(0, 8191, [6144], True, masked_choice=lambda n, a: 8192)
    assert mb == pytest.approx(c("m8192", 0))
