# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Memory ordering across streams: a command queue's programs on one sub-device, or its host transfers.

Memory the allocator hands out again is ordered after its old tenant's accesses; an unordered access to the same tensor
is refused under "check" and ordered under "order". The correctness tests use a slow reader on one column of cores so
that, without ordering, the race window is wide (each corrupts 6/6 with ordering off).
"""

from contextlib import contextmanager

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

CQ0, CQ1 = ttnn.QueueId(0), ttnn.QueueId(1)


@contextmanager
def queue_ordering(mode):
    ttnn._ttnn.core.set_queue_ordering(mode)
    try:
        yield
    finally:
        ttnn._ttnn.core.set_queue_ordering("off")


@pytest.fixture
def two_streams(device):
    """Sub-device 0 is all but the last column of cores, sub-device 1 the last column; queue ordering checks are on."""
    grid = device.compute_with_storage_grid_size()
    rest = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 2, grid.y - 1))})
    column = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(grid.x - 1, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = device.create_sub_device_manager([ttnn.SubDevice([rest]), ttnn.SubDevice([column])], 0)
    device.load_sub_device_manager(manager)
    try:
        with queue_ordering("check"):
            yield device
    finally:
        ttnn.synchronize_device(device)
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


@contextmanager
def on(queue, sub_device=None):
    with ttnn.command_queue(queue), ttnn.sub_device(queue if sub_device is None else sub_device):
        yield


def order(device, after, before):
    """Queue `after` waits for everything queue `before` has enqueued so far on its sub-device."""
    event = ttnn.record_event(device, ttnn.QueueId(before), sub_device_ids=[ttnn.SubDeviceId(before)])
    ttnn.wait_for_event(ttnn.QueueId(after), event)


def tensor(device, torch_tensor):
    return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


# A matmul slow enough on one column that the next operations run while it still reads its input.
A, B = torch.randn(1, 1, 2048, 8192), torch.randn(1, 1, 8192, 2048)


def warm_up(device):
    """Compiles and runs the operations once: a program's first launch waits for its queue's other sub-devices."""
    a, b = tensor(device, A), tensor(device, B)
    for queue in (0, 1):
        with on(queue):
            ttnn.matmul(a, b)
            ttnn.multiply(a, 0.0)
        ttnn.synchronize_device(device)  # hands the sub-device to the other queue


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@skip_for_slow_dispatch()
def test_reading_another_queues_output_needs_ordering(two_streams, expect_error):
    device = two_streams
    x = tensor(device, torch.randn(1, 1, 256, 256))
    ttnn.synchronize_device(device)
    with on(1):
        produced = ttnn.exp(x)
    with on(0), expect_error(
        RuntimeError,
        "on command queue 0 on sub-device 0 reads .* that UnaryDeviceOperation on command queue 1 on sub-device 1 "
        "wrote",
    ):
        ttnn.exp(produced)
    order(device, after=0, before=1)
    with on(0):
        ttnn.exp(produced)


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@skip_for_slow_dispatch()
def test_one_queue_does_not_order_its_sub_devices(two_streams, expect_error):
    device = two_streams
    x = tensor(device, torch.randn(1, 1, 256, 256))
    ttnn.synchronize_device(device)
    with on(0, sub_device=1):
        produced = ttnn.exp(x)
    with on(0, sub_device=0), expect_error(
        RuntimeError, "on command queue 0 on sub-device 0 reads .* on command queue 0 on sub-device 1 wrote"
    ):
        ttnn.exp(produced)
    with on(0, sub_device=0):
        ttnn._ttnn.core.wait_for(produced)
        ttnn.exp(produced)


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@skip_for_slow_dispatch()
@pytest.mark.parametrize("consumer_queue", [0, 1])
def test_wait_for_orders_a_read_after_its_producer(two_streams, consumer_queue):
    device = two_streams
    warm_up(device)
    a, b = tensor(device, A), tensor(device, B)
    ttnn.synchronize_device(device)
    with on(1):
        produced = ttnn.matmul(a, b)
    with on(consumer_queue, sub_device=0):
        ttnn._ttnn.core.wait_for(produced)
        copied = ttnn.multiply(produced, 1.0)
    ttnn.synchronize_device(device)
    assert relative_error(ttnn.to_torch(copied), ttnn.to_torch(produced)) < 0.01


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@skip_for_slow_dispatch()
def test_order_mode_orders_a_read_after_its_producer(two_streams):
    device = two_streams
    warm_up(device)
    a, b = tensor(device, A), tensor(device, B)
    ttnn.synchronize_device(device)
    with queue_ordering("order"):
        with on(1):
            produced = ttnn.matmul(a, b)
        with on(0):
            copied = ttnn.multiply(produced, 1.0)
        ttnn.synchronize_device(device)
        assert relative_error(ttnn.to_torch(copied), ttnn.to_torch(produced)) < 0.01


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@skip_for_slow_dispatch()
@pytest.mark.parametrize("writer_queue", [0, 1])
def test_memory_handed_out_again_waits_for_its_old_readers(two_streams, writer_queue):
    """The host frees a tensor a slow matmul still reads; an operation's output lands on its address."""
    device = two_streams
    warm_up(device)
    a, b = tensor(device, A), tensor(device, B)
    zeros = tensor(device, torch.zeros_like(A))
    ttnn.synchronize_device(device)
    expected = ttnn.to_torch(a).float() @ ttnn.to_torch(b).float()
    with on(1, sub_device=1):
        product = ttnn.matmul(a, b)
    address = a.buffer_address()
    ttnn.deallocate(a)
    with on(writer_queue, sub_device=0):
        overwritten = ttnn.multiply(zeros, 0.0)
    assert overwritten.buffer_address() == address
    ttnn.synchronize_device(device)
    assert relative_error(ttnn.to_torch(product), expected) < 0.05


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@skip_for_slow_dispatch()
def test_a_host_write_into_memory_another_queue_reads_waits_for_it(two_streams):
    device = two_streams
    warm_up(device)
    a, b = tensor(device, A), tensor(device, B)
    host_zeros = ttnn.from_torch(torch.zeros_like(A), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn.synchronize_device(device)
    expected = ttnn.to_torch(a).float() @ ttnn.to_torch(b).float()
    with on(1):
        product = ttnn.matmul(a, b)
    address, spec = a.buffer_address(), a.spec
    ttnn.deallocate(a)
    overwritten = ttnn.allocate_tensor_on_device(spec, device)
    assert overwritten.buffer_address() == address
    ttnn.copy_host_to_device_tensor(host_zeros, overwritten, cq_id=CQ0)
    ttnn.synchronize_device(device)
    assert relative_error(ttnn.to_torch(product), expected) < 0.05


@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
@skip_for_slow_dispatch()
@pytest.mark.parametrize("barrier", [True, False])
@pytest.mark.parametrize("release", [True, False])
def test_an_fsdp_shaped_schedule(two_streams, expect_error, barrier, release):
    """A gather on CQ1 feeds compute on CQ0; the gathered weight is freed and the next gather may reuse its memory.

    The read of the gathered weight needs an edge; the reuse of its memory does not: the next gather waits for the
    compute that read it, which is what an explicit release event did by hand.
    """
    device = two_streams
    shard, activations = tensor(device, torch.randn(1, 1, 256, 256)), tensor(device, torch.randn(1, 1, 256, 256))
    ttnn.synchronize_device(device)

    def schedule():
        for _ in range(3):
            with on(1):
                gathered = ttnn.typecast(shard, ttnn.bfloat16)  # stands in for the all-gather
            if barrier:
                order(device, after=0, before=1)
            with on(0):
                ttnn.matmul(activations, gathered)
            if release:
                order(device, after=1, before=0)
            ttnn.deallocate(gathered)

    if barrier:
        schedule()
    else:
        with expect_error(RuntimeError, "MatmulDeviceOperation on command queue 0 on sub-device 0 reads"):
            schedule()
