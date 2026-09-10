# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Any irregular group selects chain for the whole family; callers share one API."""

import pytest
from tests.ttnn.unit_tests.kernel_lib.test_mcast_family import _run

# "mixed" refers to receiver geometry, not transport: these families use chains throughout.
CASES = [
    ("dense-gap", [([(x, 0) for x in range(9)], [(2, 0)])]),
    ("irregular", [([(0, 0), (2, 0), (0, 2)], [(2, 0)])]),
    ("mixed", [([(0, 0), (1, 0), (2, 0)], [(1, 0)]), ([(0, 2), (2, 2), (4, 2)], [(2, 2)]), ([(6, 0)], [(6, 0)])]),
    ("mixed-outside", [([(0, 0), (1, 0)], [(2, 0)]), ([(0, 2), (2, 2), (4, 2)], [(6, 2)])]),
]


def test_policy_smoke(device):
    _run(device, CASES[2][1], chain_link=True)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("caller_managed", [False, True])
def test_policy(device, case, noc, counter, control, caller_managed):
    _run(
        device,
        case[1],
        chain_link=True,
        noc=noc,
        counter=counter,
        control=control,
        caller_managed=caller_managed,
        rounds=8,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_policy_backpressure(device, noc, counter):
    _run(
        device,
        CASES[2][1],
        chain_link=True,
        noc=noc,
        counter=counter,
        large=True,
        delayed=True,
        mixed_events=True,
        adopted=True,
        rounds=12,
    )
