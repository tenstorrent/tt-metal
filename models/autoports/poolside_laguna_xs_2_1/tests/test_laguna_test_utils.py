# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free contract tests for Laguna test topology selection."""
from __future__ import annotations

import argparse

import pytest

from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    PROFILES,
    add_profile_args,
    assert_memory_margin,
    profile_from_args,
    resolve_profile,
    close_mesh,
    open_mesh,
)


@pytest.mark.parametrize(
    ("name", "devices", "fabric", "context", "topology"),
    [
        ("p150", 1, "DISABLED", 65_536, None),
        ("p150x2", 2, "FABRIC_1D_RING", 131_072, "ring"),
        ("p150x4", 4, "FABRIC_1D_RING", 131_072, "ring"),
    ],
)
def test_profile_contract(name, devices, fabric, context, topology):
    profile = PROFILES[name]
    assert profile.mesh_shape == (1, devices)
    assert profile.num_devices == devices
    assert profile.fabric_config == fabric
    assert profile.max_context == context
    assert profile.ccl_topology == topology


@pytest.mark.parametrize(("alias", "expected"), [("1,1", "p150"), ("1x2", "p150x2"), ("d4", "p150x4")])
def test_legacy_mesh_aliases(alias, expected):
    assert resolve_profile(environ={"TT_LAGUNA_MESH": alias}).name == expected


def test_explicit_profile_overrides_default_but_not_conflicting_legacy_mesh():
    assert resolve_profile("p150", environ={}).name == "p150"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_profile("p150", environ={"TT_LAGUNA_MESH": "1,4"})


@pytest.mark.parametrize(
    ("profile", "visible"), [("p150", "3"), ("p150x2", "0,1"), ("p150x4", "0,1,2,3")]
)
def test_visible_device_count_matches_profile(profile, visible):
    assert resolve_profile(profile, environ={"TT_VISIBLE_DEVICES": visible}).name == profile


def test_visible_device_mismatch_and_duplicates_fail_fast():
    with pytest.raises(ValueError, match="requires 2"):
        resolve_profile("p150x2", environ={"TT_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="distinct"):
        resolve_profile("p150x2", environ={"TT_VISIBLE_DEVICES": "0,0"})


def test_fabric_override_invariants():
    assert resolve_profile("p150x2", fabric_config="ring", environ={}).fabric_config == "FABRIC_1D_RING"
    with pytest.raises(ValueError, match="requires fabric DISABLED"):
        resolve_profile("p150", fabric_config="FABRIC_1D", environ={})
    with pytest.raises(ValueError, match="requires a 1D fabric"):
        resolve_profile("p150x2", fabric_config="DISABLED", environ={})


def test_ccl_topology_derives_matching_fabric_and_rejects_conflicts():
    linear = resolve_profile("p150x2", environ={"TT_LAGUNA_CCL_TOPOLOGY": "linear"})
    assert linear.fabric_config == "FABRIC_1D"
    assert linear.ccl_topology == "linear"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_profile(
            "p150x2",
            fabric_config="FABRIC_1D_RING",
            environ={"TT_LAGUNA_CCL_TOPOLOGY": "linear"},
        )


@pytest.mark.parametrize("links", ["0", "3", "bad"])
def test_invalid_ccl_link_count_fails_fast(links):
    with pytest.raises(ValueError, match="must be 1 or 2"):
        resolve_profile("p150x2", environ={"TT_LAGUNA_CCL_NUM_LINKS": links})


def test_cli_values_take_precedence_over_environment():
    parser = argparse.ArgumentParser()
    add_profile_args(parser)
    args = parser.parse_args(["--profile", "p150x2", "--fabric-config", "FABRIC_1D_RING"])
    spec = profile_from_args(args, environ={"LAGUNA_PROFILE": "p150x4"})
    assert spec.name == "p150x2"
    assert spec.fabric_config == "FABRIC_1D_RING"


def test_memory_margin_contract():
    passing = {
        "label": "warmup",
        "free_fraction": 0.10,
        "largest_contiguous_bytes_free_per_bank": 128 * 1024 * 1024,
    }
    assert_memory_margin(passing)
    with pytest.raises(AssertionError, match="DRAM margin failed"):
        assert_memory_margin({**passing, "free_fraction": 0.09})
    with pytest.raises(AssertionError, match="DRAM margin failed"):
        assert_memory_margin({**passing, "largest_contiguous_bytes_free_per_bank": 127 * 1024 * 1024})


class _FakeMesh:
    def __init__(self, devices):
        self.devices = devices

    def get_num_devices(self):
        return self.devices


class _FakeTtnn:
    class FabricConfig:
        DISABLED = "DISABLED"
        FABRIC_1D = "FABRIC_1D"
        FABRIC_1D_RING = "FABRIC_1D_RING"

    class MeshShape:
        def __init__(self, rows, columns):
            self.devices = rows * columns

    def __init__(self):
        self.fabric_calls = []
        self.closed = []

    def set_fabric_config(self, value):
        self.fabric_calls.append(value)

    def open_mesh_device(self, shape, trace_region_size):
        return _FakeMesh(shape.devices)

    def close_mesh_device(self, mesh):
        self.closed.append(mesh)


def test_single_device_open_and_close_never_touch_fabric(monkeypatch):
    fake = _FakeTtnn()
    monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)
    profile = resolve_profile("p150", environ={})
    mesh = open_mesh(fake, profile)
    close_mesh(fake, mesh)
    assert fake.fabric_calls == []
    assert fake.closed == [mesh]
    assert profile.mesh_graph_desc_path


def test_singleton_descriptor_is_rejected_for_multidevice_profile():
    singleton = resolve_profile("p150", environ={}).mesh_graph_desc_path
    with pytest.raises(ValueError, match="unset singleton"):
        resolve_profile("p150x2", environ={"TT_MESH_GRAPH_DESC_PATH": singleton})


def test_multidevice_open_and_close_configure_then_disable_fabric(monkeypatch):
    monkeypatch.delenv("TT_MESH_GRAPH_DESC_PATH", raising=False)
    monkeypatch.delenv("TT_LAGUNA_CCL_TOPOLOGY", raising=False)
    monkeypatch.delenv("TT_LAGUNA_CCL_NUM_LINKS", raising=False)
    fake = _FakeTtnn()
    mesh = open_mesh(fake, PROFILES["p150x2"])
    close_mesh(fake, mesh)
    assert fake.fabric_calls == ["FABRIC_1D_RING", "DISABLED"]
