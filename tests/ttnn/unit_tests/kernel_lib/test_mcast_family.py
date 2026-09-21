# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact family membership, transport selection and protocol ordering."""

import pytest
import ttnn
from tests.ttnn.unit_tests.kernel_lib.mcast_test_utils import (
    attach_for_inspection,
    core_set,
    inspect_mcast_ct,
    run_family_case,
)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("chain_link", [False, True])
def test_non_worker_gap_preserves_worker_holes(device, noc, counter, chain_link):
    # Three logical row segments become four rectangles if the virtual NoC gap is treated
    # as missing receivers. Spectators in the partial rows must still remain untouched.
    receivers = [(x, 0) for x in range(2, 9)] + [(x, 1) for x in range(9)] + [(x, 2) for x in range(4)]
    run_family_case(device, [(receivers, [(2, 0)])], noc=noc, counter=counter, chain_link=chain_link, min_rectangles=3)


@pytest.mark.parametrize("noc", [0, 1])
def test_caller_managed_multicast_receiver(device, noc):
    run_family_case(device, [([(0, 0), (1, 0)], [(0, 0)])], noc=noc, receiver_caller_managed=True)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("caller_managed", [False, True])
def test_fixed_families(device, noc, counter, control, caller_managed):
    run_family_case(
        device,
        [
            ([(0, 0), (2, 0), (3, 0), (0, 1)], [(0, 0)]),
            ([(3, 2), (4, 2)], [(2, 2)]),
            ([(1, 3)], [(1, 3)]),
        ],
        noc=noc,
        counter=counter,
        control=control,
        caller_managed=caller_managed,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("control", [False, True])
def test_rotating_families(device, noc, counter, control):
    run_family_case(
        device,
        [
            ([(0, 0), (2, 0), (3, 0), (0, 1)], [(0, 0), (4, 0)]),
            ([(1, 2)], [(1, 2), (3, 2)]),
            ([(0, 3), (2, 3)], [(0, 3), (2, 3)]),
        ],
        noc=noc,
        counter=counter,
        control=control,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_family_no_handshake(device, noc, counter):
    run_family_case(device, [([(2, 0), (4, 0)], [(0, 0)])], noc=noc, counter=counter, handshake=False, rounds=1)


@pytest.mark.parametrize("noc", [0, 1])
def test_family_staircase_and_gap(device, noc):
    # Staircase with the sender in its local singleton, plus a dense logical row crossing BH's gap.
    run_family_case(
        device,
        [([(7, 0), (0, 1), (1, 1), (2, 1), (0, 2)], [(7, 0)]), ([(x, 3) for x in range(9)], [(0, 3)])],
        noc=noc,
        counter=True,
    )


@pytest.mark.parametrize("noc", [0, 1])
def test_family_three_rectangles(device, noc):
    # Three isolated destinations exercise the maximum supported owning argument array.
    run_family_case(device, [([(0, 0), (2, 0), (4, 0)], [(1, 0)])], noc=noc)


@pytest.mark.parametrize("counter", [False, True])
def test_family_zero_ack(device, counter):
    run_family_case(device, [([(2, 0), (4, 0)], [(0, 0)])], zero_ack=True, counter=counter, rounds=1)


@pytest.mark.parametrize("counter", [False, True])
def test_family_adopted_semaphores(device, counter):
    run_family_case(
        device, [([(0, 0), (2, 0), (3, 0)], [(0, 0)]), ([(0, 2), (2, 2)], [(0, 2)])], adopted=True, counter=counter
    )


# Policy's irregular case is identical to chain's sender-middle. Keep it once.
CHAIN_GEOMETRIES = [
    ("two-core", [([(0, 0), (2, 0)], [(0, 0)])]),
    ("sender-middle", [([(0, 0), (2, 0), (0, 2)], [(2, 0)])]),
    ("sender-outside", [([(0, 0), (2, 0), (0, 2)], [(4, 0)])]),
    (
        "varied-geometry",
        [([(0, 0), (1, 0), (2, 0)], [(0, 0)]), ([(0, 2), (2, 2), (4, 2)], [(2, 2)]), ([(6, 0)], [(6, 0)])],
    ),
    ("concurrent", [([(0, 0), (2, 0), (0, 2)], [(2, 0)]), ([(4, 0), (6, 0), (4, 2)], [(4, 2)])]),
    ("mapped-gap", [([(x, 0) for x in range(9)], [(0, 0)]), ([(0, 2), (2, 2)], [(0, 2)])]),
    ("dense-chain", [([(0, 0), (1, 0), (2, 0)], [(0, 0)]), ([(0, 2), (2, 2)], [(0, 2)])]),
    ("self-only", [([(0, 0)], [(0, 0)]), ([(0, 2), (2, 2)], [(0, 2)])]),
]
POLICY_GEOMETRIES = [
    ("dense-gap", [([(x, 0) for x in range(9)], [(2, 0)])]),
    ("irregular", [([(0, 0), (2, 0), (0, 2)], [(2, 0)])]),
    ("mixed", [([(0, 0), (1, 0), (2, 0)], [(1, 0)]), ([(0, 2), (2, 2), (4, 2)], [(2, 2)]), ([(6, 0)], [(6, 0)])]),
    ("mixed-outside", [([(0, 0), (1, 0)], [(2, 0)]), ([(0, 2), (2, 2), (4, 2)], [(6, 2)])]),
]
GEOMETRIES = CHAIN_GEOMETRIES + [case for case in POLICY_GEOMETRIES if case[0] != "irregular"]


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("name,groups", GEOMETRIES, ids=[case[0] for case in GEOMETRIES])
def test_family_geometry(device, noc, name, groups):
    run_family_case(
        device,
        groups,
        chain_link=True,
        noc=noc,
        rounds=8,
        min_rectangles=1 if name in ("mapped-gap", "dense-gap") else None,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize("control", [False, True], ids=["payload", "control"])
@pytest.mark.parametrize("caller_managed", [False, True], ids=["guard", "caller-managed"])
def test_chain_protocol(device, noc, counter, control, caller_managed):
    run_family_case(
        device,
        CHAIN_GEOMETRIES[3][1],
        chain_link=True,
        noc=noc,
        counter=counter,
        control=control,
        caller_managed=caller_managed,
        rounds=8,
    )


def test_chain_smoke(device):
    run_family_case(device, CHAIN_GEOMETRIES[0][1], chain_link=True)


@pytest.mark.parametrize("counter", [False, True])
def test_chain_adopted_semaphores(device, counter):
    run_family_case(device, CHAIN_GEOMETRIES[2][1], chain_link=True, counter=counter, adopted=True)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
@pytest.mark.parametrize("mixed_events", [False, True])
def test_chain_large_payload_and_backpressure(device, noc, counter, mixed_events):
    run_family_case(
        device,
        CHAIN_GEOMETRIES[3][1],
        chain_link=True,
        noc=noc,
        counter=counter,
        large=True,
        mixed_events=mixed_events,
        delayed=True,
        rounds=12,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
def test_chain_caller_managed_receiver(device, noc, counter):
    run_family_case(
        device,
        CHAIN_GEOMETRIES[3][1],
        chain_link=True,
        noc=noc,
        counter=counter,
        caller_managed=True,
        receiver_caller_managed=True,
        large=True,
        delayed=True,
        rounds=12,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True])
def test_policy_backpressure(device, noc, counter):
    run_family_case(
        device,
        POLICY_GEOMETRIES[2][1],
        chain_link=True,
        noc=noc,
        counter=counter,
        large=True,
        delayed=True,
        mixed_events=True,
        adopted=True,
        rounds=12,
    )


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("count", [61, 64])
def test_compressed_coordinate_lifetime(device, noc, column_major, count):
    size = device.compute_with_storage_grid_size()
    if size.x < 8 or size.y < 8:
        pytest.skip("requires an 8x8 worker grid")
    receivers = [(x, y) for y in range(8) for x in range(8)]
    senders = [(i // 8, i % 8) if column_major else (i % 8, i // 8) for i in range(count)]
    # Pass through all phases and wrap: the returned optional receiver must own
    # its expanded table after the temporary decoder/constructor has gone away.
    run_family_case(device, [(receivers, senders)], noc=noc, counter=True, rounds=count + 3)


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("counter", [False, True], ids=["flag", "counter"])
@pytest.mark.parametrize("column_major", [False, True], ids=["rows", "columns"])
@pytest.mark.parametrize("external_only", [False, True], ids=["mixed-senders", "external-senders"])
def test_compressed_external_sender_lifetime(device, noc, counter, column_major, external_only):
    size = device.compute_with_storage_grid_size()
    if size.x < 9 or size.y < 8:
        pytest.skip("requires a 9x8 worker grid including a spectator")

    def coord(along, across):
        return (across, along) if column_major else (along, across)

    # An incomplete final row/column must preserve its exact sender order. In
    # the mixed case, ACK counts and include/exclude-source mode vary by phase.
    senders = [coord(i % 8, i // 8) for i in range(31)]
    receivers = [coord(i, 6 if external_only else 0) for i in range(8)]
    config = ttnn.McastConfig(noc=ttnn.NOC.NOC_1 if noc else ttnn.NOC.NOC_0)
    family = ttnn.McastFamily(device, config)
    family.add_group(core_set(receivers), [ttnn.CoreCoord(*core) for core in senders])
    _, kernel = attach_for_inspection(family, core_set(receivers + senders), config.noc)
    metadata = inspect_mcast_ct(kernel)
    assert metadata["encoding"] != 0
    assert metadata["span"] == len(senders)
    assert 2 * (metadata["x_ranges"] + metadata["y_ranges"]) < 2 * len(senders)
    # Exercise every sender and wrap, with the decoded table owned by the pipe.
    run_family_case(device, [(receivers, senders)], noc=noc, counter=counter, rounds=len(senders) + 3)


def test_group_attention_external_sender_argument_counts(device):
    size = device.compute_with_storage_grid_size()
    if size.x * size.y < 32 or size.y >= 32:
        pytest.skip("requires a multi-column 32-sender grid")
    # Match group attention's column-major 10-Q-head receiver box, 32 rotating
    # senders, and full kernel placement (including inactive cores).
    senders = [(i // size.y, i % size.y) for i in range(32)]
    receivers = [(i // size.y, i % size.y) for i in range(10)]
    receiver_box = ttnn.CoreRangeSet([core_set(receivers).bounding_box()])
    family = ttnn.Mcast2D(
        device,
        receiver_box,
        ttnn.Mcast2DRotatingSenderConfig(
            sender_grid=core_set(senders), sender_order=ttnn.Mcast2DSenderOrder.ColumnMajor
        ),
        ttnn.McastConfig(),
    )
    _, kernel = attach_for_inspection(family, core_set([(x, y) for x in range(size.x) for y in range(size.y)]))
    metadata = inspect_mcast_ct(kernel)
    assert metadata["encoding"] != 0
    assert metadata["span"] == 32
    coordinate_words = 2 * (metadata["x_ranges"] + metadata["y_ranges"])
    expected_rt = 9 + coordinate_words  # Roles, phase, ACK, bounds, remote count, mode.
    for x in range(size.x):
        for y in range(size.y):
            assert len(kernel.runtime_args[x][y]) == expected_rt
    assert expected_rt < 73
    print(f"Group attention Q10 multicast RT: 73 -> {expected_rt} words/core")


def test_large_explicit_coordinate_lifetime(device):
    size = device.compute_with_storage_grid_size()
    if size.x < 8 or size.y < 8:
        pytest.skip("requires an 8x8 worker grid")
    receivers = [(x, y) for y in range(8) for x in range(8)]
    senders = receivers.copy()
    # The same 64-core geometry with a non-Cartesian traversal must retain pairs.
    # This also provides a matched explicit-storage artifact for the range audit.
    senders[0], senders[1] = senders[1], senders[0]
    run_family_case(device, [(receivers, senders)], noc=0, counter=True, rounds=67)
