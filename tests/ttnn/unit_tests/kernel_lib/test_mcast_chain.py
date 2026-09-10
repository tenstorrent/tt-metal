# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Chain payload/control forwarding through the public family API on real kernels."""

import pytest
from tests.ttnn.unit_tests.kernel_lib.test_mcast_family import _run

CASES = [
    ("two-core", [([(0, 0), (2, 0)], [(0, 0)])]),
    ("sender-middle", [([(0, 0), (2, 0), (0, 2)], [(2, 0)])]),
    ("sender-outside", [([(0, 0), (2, 0), (0, 2)], [(4, 0)])]),
    (
        "varied-geometry",
        [([(0, 0), (1, 0), (2, 0)], [(0, 0)]), ([(0, 2), (2, 2), (4, 2)], [(2, 2)]), ([(6, 0)], [(6, 0)])],
    ),
    ("concurrent", [([(0, 0), (2, 0), (0, 2)], [(2, 0)]), ([(4, 0), (6, 0), (4, 2)], [(4, 2)])]),
    ("mapped-gap", [([(x, 0) for x in range(9)], [(0, 0)])]),
    ("dense-chain", [([(0, 0), (1, 0), (2, 0)], [(0, 0)])]),
    ("self-only", [([(0, 0)], [(0, 0)])]),
]


def test_chain_smoke(device):
    _run(device, CASES[0][1], chain_link=True)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize("control", [False, True], ids=["payload", "control"])
@pytest.mark.parametrize("caller_managed", [False, True], ids=["guard", "caller-managed"])
def test_chain_family(device, case, noc, counter, control, caller_managed):
    _run(
        device,
        case[1],
        chain_link=True,
        noc=noc,
        counter=counter,
        control=control,
        caller_managed=caller_managed,
        rounds=8,
        min_rectangles=1 if case[0] == "mapped-gap" else None,
    )


@pytest.mark.parametrize("counter", [False, True])
def test_chain_adopted_semaphores(device, counter):
    _run(device, CASES[2][1], chain_link=True, counter=counter, adopted=True)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("mixed_events", [False, True])
def test_chain_large_payload_and_backpressure(device, noc, counter, mixed_events):
    _run(
        device,
        CASES[3][1],
        chain_link=True,
        noc=noc,
        counter=counter,
        large=True,
        mixed_events=mixed_events,
        delayed=True,
        rounds=12,
    )
