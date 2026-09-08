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
        2,
        1,
        0,
        1,
        0xFFFFFFFF if ack is None else ack,
        5 if noc == ttnn.NOC.NOC_1 else 1,
        2,
        4,
        0,
        0,
    ]
    coords = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 1)) for x in (1, 2, 3)]
    box = [coords[0].x, coords[0].y, coords[1].x, coords[1].y]
    if noc == ttnn.NOC.NOC_1:
        box = box[2:] + box[:2]
    prefix = box + [coords[0].x, coords[0].y, coords[2].x, coords[2].y]
    assert list(mc.runtime_args(ttnn.CoreCoord(1, 1))) == prefix + [1, 2, 1 if ack is None else ack, 3, 3, 0]
    assert list(mc.runtime_args(ttnn.CoreCoord(3, 1))) == prefix + [2, 3, 2 if ack is None else ack, 2, 1, 1]
    assert list(mc.runtime_args(ttnn.CoreCoord(2, 1))) == prefix + [0, 0, 0, 0, 2, 0xFFFFFFFF]
    outside = list(mc.runtime_args(ttnn.CoreCoord(0, 0)))
    assert len(outside) == 14 and outside[-6:] == [0, 0, 0, 0, 0, 0xFFFFFFFF]
    if kind == "line":
        assert outside == [0] * 13 + [0xFFFFFFFF]


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
    xlo, xhi = sorted([rt[0], rt[2]])
    ylo, yhi = sorted([rt[1], rt[3]])
    width = xhi - xlo + 1
    if device.arch() == ttnn.device.Arch.BLACKHOLE:
        width -= int(xlo <= 8 <= xhi) + int(xlo <= 9 <= xhi)
    virtual_sender = device.worker_core_from_logical_core(sender)
    old_inside = xlo <= virtual_sender.x <= xhi and ylo <= virtual_sender.y <= yhi
    old_remote = width * (yhi - ylo + 1) - int(old_inside)
    assert ct[8:10] == [old_remote, old_remote + 1]
    assert ct[4] == old_remote
    assert ct[7] == (1 if old_remote == 0 else 3 if old_inside else 2)
    assert (rt[0] >= rt[2] and rt[1] >= rt[3]) if noc == ttnn.NOC.NOC_1 else (rt[0] <= rt[2] and rt[1] <= rt[3])
