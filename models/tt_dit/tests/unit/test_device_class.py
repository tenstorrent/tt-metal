# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.tt_dit.utils.device_class import GALAXY_BOARDS, DeviceClass, is_galaxy


@pytest.mark.parametrize(
    "name, expected",
    [
        ("p300", DeviceClass.P300),
        ("P300", DeviceClass.P300),
        ("p300x2", DeviceClass.P300X2),
        ("P300X2", DeviceClass.P300X2),
        ("t3k", DeviceClass.T3K),
        ("n150", DeviceClass.N150),
        ("p150x8", DeviceClass.P150X8),
    ],
)
def test_from_string_is_case_insensitive(name, expected):
    assert DeviceClass.from_string(name) is expected


def test_from_string_rejects_unknown(expect_error):
    # The error lists the valid names so the message is actionable.
    with expect_error(ValueError, "p300"):
        DeviceClass.from_string("does_not_exist")


def test_from_string_roundtrips_every_member():
    for member in DeviceClass:
        assert DeviceClass.from_string(member.name.lower()) is member
        assert DeviceClass.from_string(member.name) is member


def test_is_galaxy_matches_membership_set():
    for member in DeviceClass:
        assert is_galaxy(member) == (member in GALAXY_BOARDS)


def test_galaxy_boards_are_exactly_the_galaxy_members():
    expected = {
        DeviceClass.GALAXY,
        DeviceClass.GALAXY_T3K,
        DeviceClass.DUAL_GALAXY,
        DeviceClass.QUAD_GALAXY,
    }
    assert set(GALAXY_BOARDS) == expected


def test_non_galaxy_boards_are_not_galaxy():
    for member in (DeviceClass.N150, DeviceClass.N300, DeviceClass.T3K, DeviceClass.P150, DeviceClass.P300):
        assert not is_galaxy(member)
