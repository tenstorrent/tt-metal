# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import subprocess
import pytest
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import attach_for_inspection


def inspect(family, device, noc=ttnn.NOC.NOC_0):
    size = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, size.y - 1))])
    return attach_for_inspection(family, cores, noc)


def test_host_cpp_contract():
    repo = Path(__file__).resolve().parents[4]
    subprocess.run(
        [str(repo / "build/test/ttnn/unit_tests_ttnn"), "--gtest_filter=Mcast*-McastHostFixture.SpecDevice*"],
        check=True,
    )


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
            device=device,
            receivers=receivers,
            shape=ttnn.Mcast1DShape.PerRow,
            sender_config=ttnn.Mcast1DRotatingSenderConfig(sender_grid=senders),
            config=cfg,
        )
        if kind == "line"
        else ttnn.Mcast2D(
            device=device,
            receivers=receivers,
            sender_config=ttnn.Mcast2DRotatingSenderConfig(sender_grid=senders),
            config=cfg,
        )
    )
    descriptor, kernel = inspect(mc, device, noc)
    assert list(kernel.compile_time_args) == [
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
    assert list(kernel.runtime_args[1][1]) == [1, 1 if ack is None else ack] + sender_coords + box + [
        1,
        2,
        3,
        3,
        0,
    ]
    assert list(kernel.runtime_args[3][1]) == [1, 2 if ack is None else ack] + sender_coords + box + [
        2,
        3,
        2,
        1,
        1,
    ]
    assert list(kernel.runtime_args[2][1]) == [0, 0] + sender_coords + [0] * 7 + [2, 0xFFFFFFFF]
    assert list(kernel.runtime_args[0][0]) == [0] * 14 + [0xFFFFFFFF]


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
    descriptor, kernel = inspect(mc, device, noc)
    ct = list(kernel.compile_time_args)
    rt = list(kernel.runtime_args[sender.x][sender.y])
    expected_logical = [(x, 1) for x in range(9)] if kind == "rectangle" or inside else [(0, 1)]
    expected = {
        (
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)).x,
            device.worker_core_from_logical_core(ttnn.CoreCoord(x, y)).y,
        )
        for x, y in expected_logical
    }
    actual = set()
    size = device.compute_with_storage_grid_size()
    workers = {
        (c.x, c.y)
        for y in range(size.y)
        for x in range(size.x)
        for c in [device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))]
    }
    remote_total = 0
    mapped_sender = device.worker_core_from_logical_core(sender)
    start = 2 + 2 * (ct[6] or 1)
    for i in range(rt[0]):
        sx, sy, ex, ey, remote, loopback, mode = rt[start + 7 * i : start + 7 * (i + 1)]
        rectangle = {(x, y) for x in range(min(sx, ex), max(sx, ex) + 1) for y in range(min(sy, ey), max(sy, ey) + 1)}
        rectangle &= workers
        assert remote == len(rectangle) - int((mapped_sender.x, mapped_sender.y) in rectangle)
        remote_total += remote
        assert not actual.intersection(rectangle)
        actual.update(rectangle)
        assert (sx >= ex and sy >= ey) if noc == ttnn.NOC.NOC_1 else (sx <= ex and sy <= ey)
    assert actual == expected
    old_remote = len(expected) - int(inside)
    assert ct[8:10] == [old_remote, old_remote + 1]
    assert ct[4] == rt[1] == remote_total == old_remote
    assert rt[0] == 1  # Non-worker gaps do not split a logical rectangle.


def test_group_sender_lists(device, expect_error):
    receivers = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))])
    family = ttnn.McastFamily(device)
    with expect_error(RuntimeError, "sender schedule must not be empty"):
        family.add_group(receivers, senders=[])
    with expect_error(RuntimeError, "duplicate sender"):
        family.add_group(receivers, senders=[ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0)])
    ordered = [ttnn.CoreCoord(2, 0), ttnn.CoreCoord(0, 0)]
    family.add_group(receivers, ordered)
    family.prepare_arguments()
    _, kernel = inspect(family, device)
    assert kernel.compile_time_args[6] == 2
    mapped = [device.worker_core_from_logical_core(c) for c in ordered]
    for phase, core in enumerate(ordered):
        rt = kernel.runtime_args[core.x][core.y]
        assert rt[2:6] == [mapped[0].x, mapped[0].y, mapped[1].x, mapped[1].y]
        assert rt[-1] == phase


@pytest.mark.parametrize("noc", [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1])
def test_family_lifecycle(device, noc, expect_error):
    def cores(values):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in values])

    first, second = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 2)
    cfg = ttnn.McastConfig(noc=noc, base_sem_id=4)
    family = ttnn.McastFamily(device, cfg)
    cfg.base_sem_id = 0
    queries = [
        lambda: inspect(family, device, noc),
        family.participating_cores,
        family.sender_only_cores,
    ]
    for query in queries:
        with expect_error(RuntimeError, "call prepare_arguments"):
            query()
    with expect_error(RuntimeError, "at least one group"):
        family.prepare_arguments()
    family.add_group(cores([(0, 0), (2, 0), (4, 0)]), [first])
    with expect_error(RuntimeError, "footprints overlap"):
        family.add_group(cores([(0, 0)]), [first])
    for query in queries:
        with expect_error(RuntimeError, "call prepare_arguments"):
            query()
    family.add_group(cores([(2, 2)]), [second])
    family.prepare_arguments()
    descriptor, kernel = inspect(family, device, noc)
    ct = kernel.compile_time_args
    assert ct[2:4] == [4, 5]
    assert [sem.id for sem in descriptor.semaphores] == [4, 5]
    assert kernel.runtime_args[first.x][first.y][:2] == [3, 2]
    assert kernel.runtime_args[second.x][second.y][:2] == [1, 0]
    assert kernel.runtime_args[first.x][first.y][-2] & 1 and kernel.runtime_args[second.x][second.y][-2] & 1
    participants = family.participating_cores()
    assert participants.num_cores() == 4
    assert all(participants.contains(ttnn.CoreCoord(x, y)) for x, y in [(0, 0), (2, 0), (4, 0), (2, 2)])
    assert family.sender_only_cores().num_cores() == 0
    assert kernel.runtime_args[7][7] == [0] * (len(kernel.runtime_args[first.x][first.y]) - 1) + [0xFFFFFFFF]
    family.prepare_arguments()
    assert inspect(family, device, noc)[1].compile_time_args == ct
    with expect_error(RuntimeError, "after prepare_arguments"):
        family.add_group(cores([(6, 6)]), [ttnn.CoreCoord(6, 6)])
    assert family.participating_cores().num_cores() == 4


def test_failed_family_prepare_arguments(device, expect_error):
    family = ttnn.McastFamily(device, ttnn.McastConfig(sem_ids=[0]))
    family.add_group(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))]), [ttnn.CoreCoord(0, 0)]
    )
    for _ in range(2):
        with expect_error(RuntimeError, "consumer_ready id"):
            family.prepare_arguments()
        with expect_error(RuntimeError, "call prepare_arguments"):
            inspect(family, device)


def test_family_retains_device(device):
    import gc
    import sys

    references = sys.getrefcount(device)
    family = ttnn.McastFamily(device)
    assert sys.getrefcount(device) == references + 1
    family.add_group(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]), [ttnn.CoreCoord(0, 0)]
    )
    family.prepare_arguments()
    assert sys.getrefcount(device) == references + 1
    del family
    gc.collect()
    assert sys.getrefcount(device) == references


@pytest.mark.parametrize("noc", [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1])
def test_chain_family_host_contract(device, noc, expect_error):
    def cores(values):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in values])

    sender = ttnn.CoreCoord(2, 0)
    receivers = cores([(0, 0), (2, 0), (0, 2)])
    chain = ttnn.McastFamily(
        device, ttnn.McastConfig(noc=noc, irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast)
    )
    chain.add_group(receivers, [sender])
    chain.prepare_arguments()
    multicast = ttnn.McastFamily(device, ttnn.McastConfig(noc=noc))
    multicast.add_group(receivers, [sender])
    multicast.prepare_arguments()
    chain_descriptor, chain_kernel = inspect(chain, device, noc)
    multicast_descriptor, multicast_kernel = inspect(multicast, device, noc)
    ct = list(chain_kernel.compile_time_args)
    # Chain extends the common header with its dedicated signal-source semaphore.
    assert len(ct) == 12 and len(multicast_kernel.compile_time_args) == 11
    assert ct[11] == 2
    assert [sem.id for sem in chain_descriptor.semaphores] == [0, 1, 2]
    assert ct[5] == (5 if noc == ttnn.NOC.NOC_1 else 1) | (1 << 3)
    assert ct[10] == 0 and multicast_kernel.compile_time_args[10] == 3
    assert ct[4] == 1  # One successor acknowledgment per hop; multicast reports the full fan-out.
    assert multicast_kernel.compile_time_args[4] == 2
    assert chain_kernel.runtime_args[sender.x][sender.y][:2] == [0, 1]
    assert multicast_kernel.runtime_args[sender.x][sender.y][:2] == [3, 2]
    assert chain_kernel.compile_time_args[10] == 0
    # Sender first, then remaining receivers in logical row-major order.
    order = [sender, ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2)]
    mapped = [device.worker_core_from_logical_core(c) for c in order]
    none = 0xFFFFFFFF
    for i, core in enumerate(order):
        rt = list(chain_kernel.runtime_args[core.x][core.y])
        assert len(rt) == 11
        predecessor = mapped[i - 1] if i else None
        successor = mapped[i + 1] if i + 1 < len(order) else None
        assert rt[:4] == [0, 1 if i == 0 else 0, mapped[0].x, mapped[0].y]
        assert rt[4:6] == ([predecessor.x, predecessor.y] if predecessor else [none, none])
        assert rt[6:8] == ([successor.x, successor.y] if successor else [none, none])
        assert rt[8] == 1  # The sender is one of the receivers.
        assert rt[9:] == [1 if i == 0 else 2, 0 if i == 0 else none]
    assert chain_kernel.runtime_args[7][7] == [0] * 10 + [none]
    for options, senders, group_ack, error in [
        ({"handshake": False}, [sender], None, "readiness handshakes"),
        ({"ack_count_override": 0}, [sender], None, "overrides are not supported"),
        ({}, [sender], 1, "overrides are not supported"),
        ({}, [sender, ttnn.CoreCoord(0, 0)], None, "one fixed sender"),
    ]:
        invalid = ttnn.McastFamily(
            device,
            ttnn.McastConfig(noc=noc, irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast, **options),
        )
        invalid.add_group(receivers, senders, group_ack)
        with expect_error(RuntimeError, error):
            invalid.prepare_arguments()


def test_shared_transfer_mode_policy():
    assert not hasattr(ttnn, "IrregularReceiverSetMode")
    assert not hasattr(ttnn._ttnn.mcast_host, "IrregularReceiverSetMode")
    assert ttnn.McastConfig().irregular_receiver_set_mode == ttnn.TransferMode.Multicast
    cfg = ttnn.McastConfig(irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast)
    assert cfg.irregular_receiver_set_mode == ttnn.TransferMode.ChainUnicast
    cfg.irregular_receiver_set_mode = ttnn.TransferMode.Multicast
    assert cfg.irregular_receiver_set_mode == ttnn.TransferMode.Multicast


def test_group_is_not_public():
    assert not hasattr(ttnn, "McastGroup")
    assert not hasattr(ttnn._ttnn.mcast_host, "McastGroup")
    assert not hasattr(ttnn.McastFamily, "group")


@pytest.mark.parametrize("helper_type", [ttnn.McastFamily, ttnn.Mcast1D, ttnn.Mcast2D])
def test_internal_accessors_are_not_public(helper_type):
    for name in [
        "compile_time_args",
        "runtime_args",
        "owned_semaphores",
        "next_base_sem_id",
        "num_semaphores",
        "rotating",
        "num_senders",
        "num_receivers",
        "has_remote_receivers",
        "ack_count_override",
        "ack_count",
        "num_rectangles",
        "is_sender",
        "receiver_cores",
        "rectangle_capacity",
        "sender_in_rect",
    ]:
        assert not hasattr(helper_type, name)


@pytest.mark.parametrize("ids", [[0, 1], [0, 1, 0xFFFFFFFF], [0, 1, 0], [0, 1, 1], [0, 0, 2]])
def test_chain_requires_distinct_adopted_source(device, expect_error, ids):
    family = ttnn.McastFamily(
        device, ttnn.McastConfig(sem_ids=ids, irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast)
    )
    family.add_group(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, 0)) for x in (0, 2)]),
        [ttnn.CoreCoord(0, 0)],
    )
    with expect_error(RuntimeError, "signal_source|distinct"):
        family.prepare_arguments()
