# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-helper tests for tools/triage/galaxy_topology.py.

Does not require any hardware. Run from repo root:
    pytest tools/tests/triage/test_galaxy_topology.py -v
"""

import os
import sys

import pytest


_metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_triage_home = os.path.join(_metal_home, "tools", "triage")
sys.path.insert(0, _triage_home)


from galaxy_topology import device_to_cell  # noqa: E402


# Tray membership matches the user-verified diagram (matches tt-smi -glx_list_tray_to_device).
T1 = [0, 1, 2, 3, 4, 5, 6, 7]
T2 = [16, 17, 18, 19, 20, 21, 22, 23]
T3 = [8, 9, 10, 11, 12, 13, 14, 15]
T4 = [24, 25, 26, 27, 28, 29, 30, 31]


def _expected_8x4():
    """Return {(row, col): (tray_num, dev_id)} for the 8x4 layout from the diagram."""
    grid = {
        # rows 0..3 — top half: T1 (cols 0-1) | T3 (cols 2-3)
        # T1: idx 0..3 in col 0, 4..7 in col 1, sub-row = idx %4
        (0, 0): (1, 0),
        (0, 1): (1, 4),
        (0, 2): (3, 8),
        (0, 3): (3, 12),
        (1, 0): (1, 1),
        (1, 1): (1, 5),
        (1, 2): (3, 9),
        (1, 3): (3, 13),
        (2, 0): (1, 2),
        (2, 1): (1, 6),
        (2, 2): (3, 10),
        (2, 3): (3, 14),
        (3, 0): (1, 3),
        (3, 1): (1, 7),
        (3, 2): (3, 11),
        (3, 3): (3, 15),
        # rows 4..7 — bottom half: T2 (cols 0-1) | T4 (cols 2-3)
        (4, 0): (2, 16),
        (4, 1): (2, 20),
        (4, 2): (4, 24),
        (4, 3): (4, 28),
        (5, 0): (2, 17),
        (5, 1): (2, 21),
        (5, 2): (4, 25),
        (5, 3): (4, 29),
        (6, 0): (2, 18),
        (6, 1): (2, 22),
        (6, 2): (4, 26),
        (6, 3): (4, 30),
        (7, 0): (2, 19),
        (7, 1): (2, 23),
        (7, 2): (4, 27),
        (7, 3): (4, 31),
    }
    return grid


def _expected_4x8():
    """Return {(row, col): (tray_num, dev_id)} for the 4x8 layout from the diagram."""
    grid = {
        # rows 0-1: T1 (cols 0-3) | T2 (cols 4-7), within tray idx 0..3 in row 0, 4..7 in row 1
        (0, 0): (1, 0),
        (0, 1): (1, 1),
        (0, 2): (1, 2),
        (0, 3): (1, 3),
        (0, 4): (2, 16),
        (0, 5): (2, 17),
        (0, 6): (2, 18),
        (0, 7): (2, 19),
        (1, 0): (1, 4),
        (1, 1): (1, 5),
        (1, 2): (1, 6),
        (1, 3): (1, 7),
        (1, 4): (2, 20),
        (1, 5): (2, 21),
        (1, 6): (2, 22),
        (1, 7): (2, 23),
        # rows 2-3: T3 (cols 0-3) | T4 (cols 4-7)
        (2, 0): (3, 8),
        (2, 1): (3, 9),
        (2, 2): (3, 10),
        (2, 3): (3, 11),
        (2, 4): (4, 24),
        (2, 5): (4, 25),
        (2, 6): (4, 26),
        (2, 7): (4, 27),
        (3, 0): (3, 12),
        (3, 1): (3, 13),
        (3, 2): (3, 14),
        (3, 3): (3, 15),
        (3, 4): (4, 28),
        (3, 5): (4, 29),
        (3, 6): (4, 30),
        (3, 7): (4, 31),
    }
    return grid


_TRAYS = {1: T1, 2: T2, 3: T3, 4: T4}


@pytest.mark.parametrize(
    "expected_cell,tray_num,dev_id",
    [(cell, tray, dev) for cell, (tray, dev) in _expected_8x4().items()],
)
def test_device_to_cell_8x4_full(expected_cell, tray_num, dev_id):
    assert device_to_cell(dev_id, tray_num, _TRAYS[tray_num], (8, 4)) == expected_cell


@pytest.mark.parametrize(
    "expected_cell,tray_num,dev_id",
    [(cell, tray, dev) for cell, (tray, dev) in _expected_4x8().items()],
)
def test_device_to_cell_4x8_full(expected_cell, tray_num, dev_id):
    assert device_to_cell(dev_id, tray_num, _TRAYS[tray_num], (4, 8)) == expected_cell


@pytest.mark.parametrize("bad_shape", [(2, 16), (16, 2), (1, 32), (32, 1), (5, 7)])
def test_device_to_cell_unsupported_shape(bad_shape):
    with pytest.raises(ValueError, match="Unsupported Galaxy shape"):
        device_to_cell(0, 1, T1, bad_shape)
