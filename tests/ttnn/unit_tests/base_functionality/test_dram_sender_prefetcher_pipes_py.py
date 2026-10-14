# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tt_py_test_utils_common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.prefetcher_support import require_tensor_prefetcher


pytestmark = run_for_blackhole("DRAM-sender PrefetcherPipes require Blackhole")


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    require_tensor_prefetcher(device)


def _single_recv(x: int, y: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))})


def _create_pipes(device, bank_to_receivers, ring_size):
    receiver_domain = ttnn.CoreRangeSet({r for _, receivers in bank_to_receivers for r in receivers.ranges()})
    space = ttnn.experimental.create_prefetcher_pipe_space(
        device,
        sender_cores=ttnn.CoreRangeSet(set()),
        receiver_domain=receiver_domain,
        ring_size=ring_size,
        max_receivers_per_pipe=1,
        num_dram_senders=2 * len(bank_to_receivers),
    )
    return space, ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher(space, bank_to_receivers)


def test_create_dram_sender_prefetcher_pipes_single_bank(device):
    _space, pipes = _create_pipes(device, [(0, _single_recv(0, 0))], 1024)
    assert len(pipes) == 1
    assert pipes[0].sender_core_type() == "dram"
    assert pipes[0].ring_size() == 1024
    assert pipes[0].receiver_cores().num_cores() == 1


def test_create_dram_sender_prefetcher_pipes_multi_bank_disjoint(device):
    # Each bank gets a disjoint single-core receiver set, so each gets one pipe.
    _space, pipes = _create_pipes(device, [(b, _single_recv(b, 0)) for b in range(4)], 2048)
    assert len(pipes) == 4
    assert sorted(pipe.sender_core().x for pipe in pipes) == [0, 1, 2, 3]
    assert all(pipe.sender_core_type() == "dram" and pipe.ring_size() == 2048 for pipe in pipes)
    assert sum(pipe.receiver_cores().num_cores() for pipe in pipes) == 4
