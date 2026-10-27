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
    mesh_device_profile_name,
    resolve_hardware_profile,
)
from models.demos.blackhole.qwen38_flash_next.tp import _tp_size, mesh_device_shape

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
        match=r"unknown hardware profile 'nowhere', expected one of \[.*'p150-line', 'p150-x8', 'tt-quietbox'",
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


def test_the_p150_x8_profile_exports_a_descriptor_only_at_tp_8() -> None:
    profile = HARDWARE_PROFILES["p150-x8"]
    with mock.patch.dict(
        os.environ, {k: v for k, v in os.environ.items() if k != "TT_MESH_GRAPH_DESC_PATH"}, clear=True
    ):
        hardware_profiles.export_mesh_graph_descriptor(profile)
        exported = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = "/caller/set.textproto"
        hardware_profiles.export_mesh_graph_descriptor(profile)
        assert os.environ["TT_MESH_GRAPH_DESC_PATH"] == "/caller/set.textproto"
    if len(profile.device_nodes) == 8:
        assert profile.mesh_graph_descriptor is not None and exported is not None
        assert exported.endswith("p150_x8_1x8_line_mesh_graph_descriptor.textproto")
        assert os.path.isfile(exported)
    else:
        assert profile.mesh_graph_descriptor is None and exported is None


@pytest.mark.parametrize(
    "value, shape",
    [("P150x4", (1, 4)), ("P150x8", (1, 8)), ("P300x2", (1, 4)), ("(1, 8)", (1, 8)), (None, (1, 4)), ("", (1, 4))],
)
def test_mesh_device_shape(value, shape):
    assert mesh_device_shape(value) == shape


def test_mesh_device_shape_rejects_an_unknown_value(expect_error):
    with expect_error(ValueError, "MESH_DEVICE must be one of"):
        mesh_device_shape("nowhere")


@pytest.mark.parametrize("value, size", [("P150x8", 8), ("P300x2", 4), ("(1, 4)", 4), (None, 4)])
def test_tp_size_follows_mesh_device(monkeypatch, value, size):
    if value is None:
        monkeypatch.delenv("MESH_DEVICE", raising=False)
    else:
        monkeypatch.setenv("MESH_DEVICE", value)
    assert _tp_size() == size


@pytest.mark.parametrize("value", ["(1, 2)", "(2, 4)"])
def test_tp_size_rejects_a_mesh_other_than_1x4_or_1x8(monkeypatch, value, expect_error):
    monkeypatch.setenv("MESH_DEVICE", value)
    with expect_error(ValueError, "1x4 or 1x8"):
        _tp_size()


@pytest.fixture
def no_profile_overrides(monkeypatch):
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("QWEN38_HARDWARE_PROFILE_TABLE", raising=False)


@pytest.mark.parametrize(
    "environ, host, expected",
    [
        ({"MESH_DEVICE": "P150x8"}, "anyhost", "p150-x8"),
        ({"MESH_DEVICE": "(1, 8)"}, "anyhost", "p150-x8"),
        ({"MESH_DEVICE": "P300x2", "TT_VISIBLE_DEVICES": "4,5,6,7"}, "anyhost", "tt-quietbox-2-instance-1"),
        ({"MESH_DEVICE": "P300x2"}, "anyhost", "tt-quietbox-2"),
        ({"MESH_DEVICE": "P150x4"}, "anyhost", "p150-x8"),
        ({}, "tt-quietbox", "tt-quietbox"),
    ],
)
def test_mesh_device_profile_name(monkeypatch, no_profile_overrides, environ, host, expected):
    monkeypatch.setattr(hardware_profiles.socket, "gethostname", lambda: host)

    assert mesh_device_profile_name("p150-x8", environ) == expected


def test_mesh_device_profile_name_rejects_an_unknown_fallback(monkeypatch, no_profile_overrides, expect_error):
    monkeypatch.setattr(hardware_profiles.socket, "gethostname", lambda: "anyhost")

    with expect_error(HardwareProfileError, "unknown hardware profile"):
        mesh_device_profile_name("nowhere", {"MESH_DEVICE": "P150x4"})
