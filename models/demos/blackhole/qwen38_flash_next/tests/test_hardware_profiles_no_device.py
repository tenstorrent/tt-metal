# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``resolve_hardware_profile``: a named public profile resolves on any host; a private table's lane keeps its host."""

from __future__ import annotations

import os
from dataclasses import replace
from unittest import mock

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools.hardware_profiles import (
    HARDWARE_PROFILES,
    HardwareProfileError,
    ResidentHardwareProfile,
    resolve_hardware_profile,
)

# A private table: two lanes of one host, one lane of another, on top of the public profiles.
HOST_ONE_B = ResidentHardwareProfile(
    host="host-one",
    partition="b",
    visible_devices="4,5,6,7",
    device_nodes=(4, 5, 6, 7),
    numa_node=None,
    ethernet_graph="line",
    route=None,
    route_nodes=None,
    system_mesh_local_shape=(4, 1),
)
HOST_ONE_A = replace(HOST_ONE_B, partition="a", visible_devices="0,1,2,3", device_nodes=(0, 1, 2, 3))
HOST_TWO_B = replace(HOST_ONE_B, host="host-two")
TABLE = {**HARDWARE_PROFILES, "host-one": HOST_ONE_B, "host-one-a": HOST_ONE_A, "host-two": HOST_TWO_B}


def _resolve(name: str | None, *, host: str, visible: str | None = None) -> ResidentHardwareProfile:
    environment = {key: value for key, value in os.environ.items() if key != "TT_VISIBLE_DEVICES"}
    if visible is not None:
        environment["TT_VISIBLE_DEVICES"] = visible
    with (
        mock.patch.object(hardware_profiles.socket, "gethostname", return_value=f"{host}.example"),
        mock.patch.dict(os.environ, environment, clear=True),
    ):
        return resolve_hardware_profile(name, table=TABLE)


@pytest.mark.parametrize("host", ("tt-quietbox", "p150-line", "host-one", "some-box"))
@pytest.mark.parametrize("name", sorted(HARDWARE_PROFILES))
def test_a_named_public_profile_resolves_on_any_host(name: str, host: str) -> None:
    # A QuietBox 2 shipped with the hostname of the QuietBox profile; the named profile must still resolve.
    assert _resolve(name, host=host) is HARDWARE_PROFILES[name]


def test_a_private_lane_resolves_on_its_host_and_on_a_host_the_table_does_not_know() -> None:
    assert _resolve("host-one", host="host-one") is HOST_ONE_B
    assert _resolve("host-one-a", host="host-one") is HOST_ONE_A
    assert _resolve("host-two", host="some-box") is HOST_TWO_B


def test_a_private_lane_of_another_known_host_refuses(expect_error) -> None:
    with expect_error(HardwareProfileError, match=r"hardware profile 'host-two' belongs to host-two, not host-one"):
        _resolve("host-two", host="host-one")
    with expect_error(HardwareProfileError, match=r"belongs to host-one, not tt-quietbox"):
        _resolve("host-one-a", host="tt-quietbox")


def test_an_unknown_name_lists_the_table(expect_error) -> None:
    with expect_error(
        HardwareProfileError,
        match=r"unknown hardware profile 'nowhere', expected one of \[.*'p150-line', 'tt-quietbox'",
    ):
        _resolve("nowhere", host="host-one")


def test_no_name_takes_the_default_lane_or_the_lane_the_visible_devices_select(expect_error) -> None:
    assert _resolve(None, host="host-one") is HOST_ONE_B
    assert _resolve(None, host="host-one", visible="0,1,2,3") is HOST_ONE_A
    assert _resolve(None, host="tt-quietbox") is HARDWARE_PROFILES["tt-quietbox"]
    with expect_error(HardwareProfileError, match=r"TT_VISIBLE_DEVICES '1,2' selects 0 of host-one's profiles"):
        _resolve(None, host="host-one", visible="1,2")
    with expect_error(HardwareProfileError, match=r"no hardware profile for host 'some-box'"):
        _resolve(None, host="some-box")
