# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Typed semaphore and value-view contracts, independent of the spec host adapter."""
import pytest

from tests.ttnn.unit_tests.kernel_lib.test_mcast_family import _run as run_family
from tests.ttnn.unit_tests.kernel_lib.test_mcast_prepared_device import _run as run_prepared


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize("control", [False, True], ids=["payload", "control"])
@pytest.mark.parametrize("chain", [False, True], ids=["multicast", "chain"])
def test_typed_family(device, noc, counter, control, chain):
    run_family(
        device,
        [([(0, 0), (2, 0), (2, 1)], [(0, 0)])],
        noc=noc,
        counter=counter,
        control=control,
        chain_link=chain,
        typed_bindings=True,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize("control", [False, True], ids=["payload", "control"])
@pytest.mark.parametrize("case", ["rotating", "no-handshake", "local"])
def test_typed_prepared(device, noc, counter, control, case):
    width = 1 if case == "local" else 2
    senders = [0, 2] if case == "rotating" else [0] if case == "local" else [2]
    run_prepared(
        device,
        width,
        senders,
        case == "rotating",
        noc,
        counter,
        control,
        True,
        False,
        handshake=case != "no-handshake",
        typed_bindings=True,
    )
