# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Python binding diagnostics and ownership; geometry contracts live in C++ gtests."""

import gc
import sys

import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import (
    attach_for_inspection,
    core_set,
    inspect_mcast,
    inspect_mcast_ct,
)


def inspect(channel, device, noc=ttnn.NOC.NOC_0):
    size = device.compute_with_storage_grid_size()
    placement = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(size.x - 1, size.y - 1))])
    return attach_for_inspection(channel, placement, noc)


def test_mcast_keywords_and_sender_order(device):
    receivers = core_set([(0, 0), (1, 0), (0, 1), (1, 1)])
    senders = [[ttnn.CoreCoord(1, 0)], [ttnn.CoreCoord(0, 1)]]
    config = ttnn.McastConfig(noc=ttnn.NOC.NOC_1)
    channel = ttnn.Mcast(
        device=device,
        config=config,
        receivers=receivers,
        receiver_group_size=2,
        sender_config=ttnn.McastExplicitSenderConfig(senders),
        receiver_order=ttnn.McastCoreOrder.RowMajor,
    )
    try:
        channel.next_semaphore_id()
        assert False, "next_semaphore_id must require a descriptor attachment"
    except RuntimeError:
        pass
    senders.clear()
    descriptor, kernel = inspect(channel, device, ttnn.NOC.NOC_1)
    assert [semaphore.id for semaphore in descriptor.semaphores] == [0, 1]
    assert channel.next_semaphore_id() == 2
    assert inspect_mcast(kernel, ttnn.CoreCoord(1, 0))["roles"] & 1
    assert inspect_mcast(kernel, ttnn.CoreCoord(0, 1))["roles"] & 1


def test_owned_handshake_subset_and_external_sender(device):
    receivers = core_set([(0, 0), (1, 0), (2, 0)])
    config = ttnn.McastConfig(handshake_cores=core_set([(0, 0), (1, 0)]))
    channel = ttnn.Mcast(
        device,
        config,
        receivers,
        3,
        ttnn.McastExplicitSenderConfig([[ttnn.CoreCoord(3, 0)]]),
    )
    config.handshake_cores = core_set([])
    _, kernel = inspect(channel, device)
    assert inspect_mcast(kernel, ttnn.CoreCoord(3, 0))["ack"] == 2
    assert channel.sender_only_cores().contains(ttnn.CoreCoord(3, 0))


def test_rotating_and_sender_grid_configs(device):
    receivers = core_set([(0, 0), (1, 0), (0, 1), (1, 1)])
    rotating = ttnn.Mcast(device, ttnn.McastConfig(), receivers, 2, ttnn.McastRotatingSenderConfig())
    _, rotating_kernel = inspect(rotating, device)
    assert inspect_mcast_ct(rotating_kernel)["span"] == 2

    sender_grid = core_set([(3, 0), (3, 1), (4, 0), (4, 1)])
    external = ttnn.Mcast(
        device,
        ttnn.McastConfig(),
        receivers,
        2,
        ttnn.McastSenderGridConfig(sender_grid, sender_order=ttnn.McastCoreOrder.ColumnMajor),
    )
    _, external_kernel = inspect(external, device)
    assert inspect_mcast_ct(external_kernel)["span"] == 2
    assert external.sender_only_cores().num_cores() == 4


def test_chain_policy_and_diagnostics(device, expect_error):
    receivers = core_set([(0, 0), (2, 0), (0, 2)])
    channel = ttnn.Mcast(
        device,
        ttnn.McastConfig(irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast),
        receivers,
        3,
    )
    _, kernel = inspect(channel, device)
    assert (inspect_mcast_ct(kernel)["flags"] >> 3) & 3

    partial = ttnn.McastConfig(
        handshake_cores=core_set([(0, 0)]), irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast
    )
    with expect_error(RuntimeError, "all receivers"):
        ttnn.Mcast(device, partial, receivers, 3)


def test_device_lifetime_is_retained(device):
    references = sys.getrefcount(device)
    channel = ttnn.Mcast(device, ttnn.McastConfig(), core_set([(0, 0)]), 1)
    assert sys.getrefcount(device) == references + 1
    inspect(channel, device)
    del channel
    gc.collect()
    assert sys.getrefcount(device) == references


def test_current_host_api_is_exported():
    module = ttnn._ttnn.mcast_host
    for name in [
        "Mcast",
        "McastConfig",
        "McastDataReady",
        "TransferMode",
        "McastCoreOrder",
        "McastSenderPlacement",
        "McastFixedSenderConfig",
        "McastRotatingSenderConfig",
        "McastSenderGridConfig",
        "McastExplicitSenderConfig",
        "attach_absent",
    ]:
        assert getattr(ttnn, name) is getattr(module, name)


def test_current_config_bindings_round_trip():
    defaults = ttnn.McastConfig()
    assert defaults.noc.value == ttnn.NOC.NOC_0.value
    assert defaults.handshake
    assert defaults.handshake_cores is None
    assert defaults.data_ready == ttnn.McastDataReady.Flag
    assert defaults.irregular_receiver_set_mode == ttnn.TransferMode.Multicast

    handshake_cores = core_set([(0, 0), (1, 0)])
    config = ttnn.McastConfig(
        noc=ttnn.NOC.NOC_1,
        handshake_cores=handshake_cores,
        data_ready=ttnn.McastDataReady.Counter,
    )
    assert config.noc.value == ttnn.NOC.NOC_1.value
    assert config.handshake
    assert config.handshake_cores == handshake_cores
    assert config.data_ready == ttnn.McastDataReady.Counter
    assert config.irregular_receiver_set_mode == ttnn.TransferMode.Multicast

    chain = ttnn.McastConfig(irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast)
    assert chain.irregular_receiver_set_mode == ttnn.TransferMode.ChainUnicast


def test_current_sender_config_bindings_round_trip():
    fixed = ttnn.McastFixedSenderConfig(sender_index=2, placement=ttnn.McastSenderPlacement.Staggered)
    assert fixed.sender_index == 2
    assert fixed.placement == ttnn.McastSenderPlacement.Staggered

    rotating = ttnn.McastRotatingSenderConfig()
    assert isinstance(rotating, ttnn.McastRotatingSenderConfig)

    sender_cores = core_set([(2, 0), (2, 1)])
    sender_grid = ttnn.McastSenderGridConfig(sender_cores, sender_order=ttnn.McastCoreOrder.ColumnMajor)
    assert sender_grid.sender_cores == sender_cores
    assert sender_grid.sender_order == ttnn.McastCoreOrder.ColumnMajor

    senders = [[ttnn.CoreCoord(2, 0)], [ttnn.CoreCoord(2, 1)]]
    explicit = ttnn.McastExplicitSenderConfig(senders)
    assert explicit.senders_per_group == senders


def test_attach_absent_uses_current_named_offsets():
    kernel = ttnn.KernelDescriptor(
        kernel_source="inspection-only.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set([(0, 0)]),
        compile_time_args=[17],
        runtime_args=[(ttnn.CoreCoord(0, 0), [23])],
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0,
            noc=ttnn.NOC.NOC_0,
        ),
    )
    ttnn.attach_absent(kernel, "optional_channel")
    offsets = dict(kernel.named_compile_time_args)
    assert offsets["optional_channel_ct_offset"] == 1
    assert offsets["optional_channel_rt_offset"] == 0
    assert kernel.compile_time_args == [17, 0]
    assert kernel.runtime_args[0][0] == [23]


def test_shared_transfer_mode_policy():
    assert ttnn.McastConfig().irregular_receiver_set_mode == ttnn.TransferMode.Multicast
    config = ttnn.McastConfig(irregular_receiver_set_mode=ttnn.TransferMode.ChainUnicast)
    assert config.irregular_receiver_set_mode == ttnn.TransferMode.ChainUnicast
