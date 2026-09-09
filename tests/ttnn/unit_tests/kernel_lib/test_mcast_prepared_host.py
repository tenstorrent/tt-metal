# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import subprocess
import pytest
import ttnn


def test_host_cpp_contract():
    repo = Path(__file__).resolve().parents[4]
    subprocess.run([str(repo / "build/test/ttnn/unit_tests_ttnn"), "--gtest_filter=Mcast*"], check=True)


@pytest.mark.parametrize("noc", [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1])
@pytest.mark.parametrize("kind", ["line", "rectangle"])
@pytest.mark.parametrize("ack", [None, 0, 1])
def test_mixed_prepared_layout(device, noc, kind, ack):
    receivers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1))])
    senders = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(3, 1), ttnn.CoreCoord(3, 1)),
        ]
    )
    cfg = ttnn.McastConfig(noc=noc, ack_count_override=ack)
    mc = (
        ttnn.Mcast1D(
            device, receivers, ttnn.Mcast1DShape.PerRow, ttnn.Mcast1DRotatingSenderConfig(sender_grid=senders), cfg
        )
        if kind == "line"
        else ttnn.Mcast2D(device, receivers, ttnn.Mcast2DRotatingSenderConfig(sender_grid=senders), cfg)
    )
    assert list(mc.compile_time_args()) == [
        1,
        1,
        0,
        1,
        0xFFFFFFFF if ack is None else ack,
        5 if noc == ttnn.NOC.NOC_1 else 1,
        2,
        4,
        0,
        0,
        1,
    ]
    coords = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 1)) for x in (1, 2, 3)]
    box = [coords[0].x, coords[0].y, coords[1].x, coords[1].y]
    if noc == ttnn.NOC.NOC_1:
        box = box[2:] + box[:2]
    sender_coords = [coords[0].x, coords[0].y, coords[2].x, coords[2].y]
    assert list(mc.runtime_args(ttnn.CoreCoord(1, 1))) == [1, 1 if ack is None else ack] + sender_coords + box + [
        1,
        2,
        3,
        3,
        0,
    ]
    assert list(mc.runtime_args(ttnn.CoreCoord(3, 1))) == [1, 2 if ack is None else ack] + sender_coords + box + [
        2,
        3,
        2,
        1,
        1,
    ]
    assert list(mc.runtime_args(ttnn.CoreCoord(2, 1))) == [0, 0] + sender_coords + [0] * 7 + [2, 0xFFFFFFFF]
    assert list(mc.runtime_args(ttnn.CoreCoord(0, 0))) == [0] * 14 + [0xFFFFFFFF]


@pytest.mark.parametrize("noc", [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1])
@pytest.mark.parametrize("inside", [True, False])
@pytest.mark.parametrize("kind", ["line", "rectangle"])
def test_mapped_counts_match_v16(device, noc, inside, kind):
    # On Blackhole this dense logical span crosses virtual non-worker columns 8 and 9.
    if device.compute_with_storage_grid_size().x < 9:
        pytest.skip("requires nine worker columns")
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(8, 1))])
    sender = ttnn.CoreCoord(0, 1) if inside else ttnn.CoreCoord(0, 0)
    cfg = ttnn.McastConfig(noc=noc)
    if kind == "line":
        # Per-column outside senders keep each line aligned, with a one-worker receiver span.
        senders = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(8, 0))])
        if inside:
            mc = ttnn.Mcast1D(device, grid, ttnn.Mcast1DShape.PerRow, ttnn.Mcast1DFixedSenderConfig(), cfg)
        else:
            mc = ttnn.Mcast1D(
                device, grid, ttnn.Mcast1DShape.PerColumn, ttnn.Mcast1DRotatingSenderConfig(sender_grid=senders), cfg
            )
    else:
        mc = ttnn.Mcast2D(device, grid, ttnn.Mcast2DFixedSenderConfig(sender), cfg)
    ct = list(mc.compile_time_args())
    rt = list(mc.runtime_args(sender))
    expected_logical = [(x, 1) for x in range(9)] if kind == "rectangle" or inside else [(0, 1)]
    expected = {
        (
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)).x,
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)).y,
        )
        for x, y in expected_logical
    }
    actual = set()
    remote_total = 0
    start = 2 + 2 * (ct[6] or 1)
    for i in range(rt[0]):
        sx, sy, ex, ey, remote, loopback, mode = rt[start + 7 * i : start + 7 * (i + 1)]
        rectangle = {(x, y) for x in range(min(sx, ex), max(sx, ex) + 1) for y in range(min(sy, ey), max(sy, ey) + 1)}
        remote_total += remote
        assert not actual.intersection(rectangle)
        actual.update(rectangle)
        assert (sx >= ex and sy >= ey) if noc == ttnn.NOC.NOC_1 else (sx <= ex and sy <= ey)
    assert actual == expected
    old_remote = len(expected) - int(inside)
    assert ct[8:10] == [old_remote, old_remote + 1]
    assert ct[4] == rt[1] == remote_total == old_remote
    if device.arch() == ttnn.device.Arch.BLACKHOLE and len(expected) == 9:
        assert rt[0] == 2  # The dense logical span crosses virtual columns 8 and 9.


def test_group_sender_lists(expect_error):
    receivers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))])
    fixed = ttnn.McastGroup(receivers, senders=[ttnn.CoreCoord(1, 0)])
    assert not fixed.rotating()
    rotating = ttnn.McastGroup(receivers, senders=[ttnn.CoreCoord(2, 0), ttnn.CoreCoord(0, 0)])
    assert rotating.rotating()
    assert [(c.x, c.y) for c in rotating.senders()] == [(2, 0), (0, 0)]
    with expect_error(RuntimeError, "sender schedule must not be empty"):
        ttnn.McastGroup(receivers, senders=[])
    with expect_error(RuntimeError, "duplicate sender"):
        ttnn.McastGroup(receivers, senders=[ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)])


@pytest.mark.parametrize("noc", [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1])
def test_prepared_group_api(device, noc, expect_error):
    def cores(values):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in values])

    first = ttnn.CoreCoord(0, 0)
    second = ttnn.CoreCoord(2, 2)
    raw = ttnn.McastGroup(cores([(0, 0), (2, 0), (4, 0)]), [first])
    with expect_error(RuntimeError, "prepared group"):
        raw.compile_time_args()
    with expect_error(RuntimeError, "prepared group"):
        raw.runtime_args(first)
    family = ttnn.McastFamily(device, [raw, ttnn.McastGroup(cores([(2, 2)]), [second])], ttnn.McastConfig(noc=noc))
    group = family.group(0)
    local = family.group(1)
    assert group.num_rectangles() == 3
    assert local.num_rectangles() == 1
    assert group.num_senders() == 1
    assert group.is_sender(first)
    assert group.num_receivers(first) == group.ack_count(first) == 2
    assert group.has_remote_receivers()
    assert not local.has_remote_receivers()
    assert group.participating_cores() == group.receiver_cores()
    assert group.sender_only_cores().num_cores() == 0
    for override in [None, False, True]:
        assert (
            group.compile_time_args(override) == local.compile_time_args(override) == family.compile_time_args(override)
        )
    for core in [first, ttnn.CoreCoord(2, 0), ttnn.CoreCoord(4, 0)]:
        assert group.runtime_args(core) == family.runtime_args(core)
    for core in [second, ttnn.CoreCoord(7, 7)]:
        with expect_error(RuntimeError, "not in this group"):
            group.runtime_args(core)
    with expect_error(RuntimeError, "out of range"):
        family.group(2)
    assert family.runtime_args(ttnn.CoreCoord(7, 7)) == [0] * (len(group.runtime_args(first)) - 1) + [0xFFFFFFFF]
    expected = group.runtime_args(first)
    del family
    import gc

    gc.collect()
    assert group.runtime_args(first) == expected  # The binding keeps the owning family alive.
