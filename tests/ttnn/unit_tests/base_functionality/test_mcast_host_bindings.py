# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python binding diagnostics and ownership; geometry contracts live in C++ gtests."""

import pytest
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import attach_for_inspection, core_set


def inspect(family, device, noc=ttnn.NOC.NOC_0):
    size = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, size.y - 1))])
    return attach_for_inspection(family, cores, noc)


@pytest.mark.parametrize("kind", ["line", "rectangle"])
def test_wrapper_keywords(device, kind):
    receivers = core_set([(1, 1), (2, 1)])
    senders = core_set([(1, 1), (3, 1)])
    config = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, ack_count_override=1)
    helper = (
        ttnn.Mcast1D(
            device=device,
            receivers=receivers,
            shape=ttnn.Mcast1DShape.PerRow,
            sender_config=ttnn.Mcast1DRotatingSenderConfig(sender_grid=senders),
            config=config,
        )
        if kind == "line"
        else ttnn.Mcast2D(
            device=device,
            receivers=receivers,
            sender_config=ttnn.Mcast2DRotatingSenderConfig(sender_grid=senders),
            config=config,
        )
    )
    _, kernel = inspect(helper, device, ttnn.NOC.NOC_1)
    assert kernel.runtime_args[1][1][1] == 1
    assert kernel.runtime_args[3][1][1] == 1


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
    # Python config is snapshotted by construction, and keyword values reach C++.
    descriptor, kernel = inspect(family, device, noc)
    assert [sem.id for sem in descriptor.semaphores] == [4, 5]
    family.prepare_arguments()
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


@pytest.mark.parametrize(
    "options,senders,group_ack,error",
    [
        ({"handshake": False}, [(2, 0)], None, "readiness handshakes"),
        ({"ack_count_override": 0}, [(2, 0)], None, "overrides are not supported"),
        ({}, [(2, 0)], 1, "overrides are not supported"),
        ({}, [(2, 0), (0, 0)], None, "one fixed sender"),
    ],
)
def test_chain_protocol_diagnostics(device, expect_error, options, senders, group_ack, error):
    family = ttnn.McastFamily(
        device, ttnn.McastConfig(irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast, **options)
    )
    family.add_group(core_set([(0, 0), (2, 0), (0, 2)]), [ttnn.CoreCoord(*c) for c in senders], group_ack)
    with expect_error(RuntimeError, error):
        family.prepare_arguments()
