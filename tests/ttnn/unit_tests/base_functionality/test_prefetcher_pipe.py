# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PrefetcherPipeSpace / PrefetcherPipe host surface from Python: reserve, carve, query, release.

Pushing data through a pipe from Python is covered by the prefetcher_pipe_copy op tests.
"""

import gc

import pytest
import ttnn


def _core(x, y):
    return ttnn.CoreCoord(x, y)


def _cores(*coords):
    return ttnn.CoreRangeSet({ttnn.CoreRange(_core(x, y), _core(x, y)) for x, y in coords})


def _rect(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(_core(x0, y0), _core(x1, y1))})


RING_SIZE = 1024


@pytest.mark.requires_grid_size((2, 1))
def test_create_space_and_carve_one_pipe(device):
    sender = _core(0, 0)
    receivers = _cores((1, 0))
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=_cores((0, 0)),
        receiver_domain=receivers,
        ring_size=RING_SIZE,
        max_receivers_per_pipe=1,
    )
    assert space.ring_size() == RING_SIZE
    assert space.max_receivers_per_pipe() == 1
    assert space.config_page_size() > 0
    assert space.sender_cores().num_cores() == 1
    assert space.receiver_domain().num_cores() == 1
    assert space.reservation_cores().num_cores() == 2
    assert space.unclaimed_cores().num_cores() == 2

    pipe = space.create_pipe(sender, receivers)
    # A pipe carved from a space shares the space's addresses by contract.
    assert pipe.buffer_address() == space.buffer_address()
    assert pipe.config_address() == space.config_address()
    assert pipe.ring_size() == RING_SIZE
    assert pipe.config_page_size() == space.config_page_size()
    assert pipe.sender_core() == sender
    assert pipe.sender_cores().num_cores() == 1
    assert pipe.receiver_cores().num_cores() == 1
    assert pipe.receiver_cores().contains(_core(1, 0))
    assert pipe.all_cores().num_cores() == 2
    assert space.unclaimed_cores().num_cores() == 0

    # Dropping the pipe returns its cores to the space; the same cores can then be re-carved.
    del pipe
    gc.collect()
    assert space.unclaimed_cores().num_cores() == 2
    pipe_again = space.create_pipe(sender, receivers)
    assert pipe_again.buffer_address() == space.buffer_address()


@pytest.mark.requires_grid_size((2, 1))
def test_pipe_keeps_space_alive(device):
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=_cores((0, 0)),
        receiver_domain=_cores((1, 0)),
        ring_size=RING_SIZE,
        max_receivers_per_pipe=1,
    )
    expected_addresses = (space.buffer_address(), space.config_address())
    pipe = space.create_pipe(_core(0, 0), _cores((1, 0)))
    del space
    gc.collect()
    # The pipe holds its space; its addresses are still valid after the space handle is gone.
    assert (pipe.buffer_address(), pipe.config_address()) == expected_addresses
    assert pipe.all_cores().num_cores() == 2


@pytest.mark.requires_grid_size((2, 2))
def test_create_pipes_batch_share_addresses(device):
    # Two disjoint 1:1 pipes carved from one space in one call: (0,0)->(1,0) and (0,1)->(1,1).
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=_cores((0, 0), (0, 1)),
        receiver_domain=_cores((1, 0), (1, 1)),
        ring_size=RING_SIZE,
        max_receivers_per_pipe=1,
    )
    pipes = space.create_pipes([(_core(0, 0), _cores((1, 0))), (_core(0, 1), _cores((1, 1)))])
    assert len(pipes) == 2
    assert {p.sender_core().x for p in pipes} == {0}
    assert sorted(p.sender_core().y for p in pipes) == [0, 1]
    for p in pipes:
        assert p.buffer_address() == space.buffer_address()
        assert p.config_address() == space.config_address()
    assert space.unclaimed_cores().num_cores() == 0


@pytest.mark.requires_grid_size((2, 1))
def test_space_config_rejects(device, expect_error):
    sender_cores = _cores((0, 0))
    receiver_domain = _cores((1, 0))
    with expect_error(RuntimeError, "ring_size must be > 0"):
        ttnn.experimental.create_prefetcher_pipe_space(
            device, sender_cores, receiver_domain, ring_size=0, max_receivers_per_pipe=1
        )
    with expect_error(RuntimeError, "must be a multiple of L1_ALIGNMENT"):
        ttnn.experimental.create_prefetcher_pipe_space(
            device, sender_cores, receiver_domain, ring_size=RING_SIZE + 8, max_receivers_per_pipe=1
        )
    with expect_error(RuntimeError, "max_receivers_per_pipe must be >= 1"):
        ttnn.experimental.create_prefetcher_pipe_space(
            device, sender_cores, receiver_domain, ring_size=RING_SIZE, max_receivers_per_pipe=0
        )
    with expect_error(RuntimeError, "exceeds the 1 cores in receiver_domain"):
        ttnn.experimental.create_prefetcher_pipe_space(
            device, sender_cores, receiver_domain, ring_size=RING_SIZE, max_receivers_per_pipe=2
        )
    with expect_error(RuntimeError, "receiver_domain is empty"):
        ttnn.experimental.create_prefetcher_pipe_space(
            device, sender_cores, ttnn.CoreRangeSet([]), ring_size=RING_SIZE, max_receivers_per_pipe=1
        )
    with expect_error(RuntimeError, "persistent-arena allocations require BufferType::L1"):
        ttnn.experimental.create_prefetcher_pipe_space(
            device,
            sender_cores,
            receiver_domain,
            ring_size=RING_SIZE,
            max_receivers_per_pipe=1,
            buffer_type=ttnn.BufferType.DRAM,
        )


@pytest.mark.requires_grid_size((3, 1))
def test_carve_rejects(device, expect_error):
    # Space over (0,0) sender and (1,0)-(2,0) receivers, one receiver per pipe.
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=_cores((0, 0)),
        receiver_domain=_rect(1, 0, 2, 0),
        ring_size=RING_SIZE,
        max_receivers_per_pipe=1,
    )
    with expect_error(RuntimeError, "is not one of the space's sender_cores"):
        space.create_pipe(_core(1, 0), _cores((2, 0)))
    with expect_error(RuntimeError, "requires at least one receiver"):
        space.create_pipe(_core(0, 0), ttnn.CoreRangeSet([]))
    with expect_error(RuntimeError, "exceed the space's max_receivers_per_pipe"):
        space.create_pipe(_core(0, 0), _rect(1, 0, 2, 0))
    with expect_error(RuntimeError, "are not all inside the space's receiver_domain"):
        space.create_pipe(_core(0, 0), _cores((0, 0)))

    live = space.create_pipe(_core(0, 0), _cores((1, 0)))
    # The sender is claimed by `live`; a second pipe cannot reuse it.
    with expect_error(RuntimeError, "is already claimed by a live pipe"):
        space.create_pipe(_core(0, 0), _cores((2, 0)))
    # A batch that names one core twice is rejected before anything is claimed.
    del live
    gc.collect()
    with expect_error(RuntimeError, "appears in more than one pipe of the batch"):
        space.create_pipes([(_core(0, 0), _cores((1, 0))), (_core(0, 0), _cores((2, 0)))])
    assert space.unclaimed_cores().num_cores() == 3
