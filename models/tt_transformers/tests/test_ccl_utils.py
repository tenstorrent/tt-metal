# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.common.modules import tt_ccl as common_tt_ccl
from models.tt_transformers.tt import ccl as tt_transformers_ccl


class FakeMeshDevice:
    def __init__(self, num_devices, fabric_links=3):
        self._num_devices = num_devices
        # What the fabric control plane would report for this mesh.
        self.fabric_links = fabric_links

    def get_num_devices(self):
        return self._num_devices


class FakeMeshDeviceWithoutLocalDevices(FakeMeshDevice):
    def get_device_ids(self):
        raise RuntimeError("debug assert from get_device_ids")


@pytest.fixture
def native_get_num_links(monkeypatch):
    """
    Fixture to monkeypatch the ttnn.get_num_links function to return the fabric links for the given mesh device.
    """
    calls = []

    def fake(mesh_device, cluster_axis=None):
        calls.append((mesh_device, cluster_axis))
        return mesh_device.fabric_links

    monkeypatch.setattr(ttnn, "get_num_links", fake)
    return calls


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_forwards_to_native_query(native_get_num_links, ccl_module):
    """
    Validates that the CCL get_num_links function forwards to the cpp get_num_links function.
    """
    mesh_device = FakeMeshDevice(num_devices=64)

    assert ccl_module.get_num_links(mesh_device, cluster_axis=0) == 3
    assert ccl_module.get_num_links(mesh_device, cluster_axis=1) == 3
    assert ccl_module.get_num_links(mesh_device) == 3
    assert native_get_num_links == [(mesh_device, 0), (mesh_device, 1), (mesh_device, None)]


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_follows_fabric_not_device_count(native_get_num_links, ccl_module):
    """
    Validates that the CCL get_num_links function follows the actual fabric links observed rather than the device count.
    """
    # Two p150a boards cabled with two QSFP-DD cables and a p300 both have 2 devices, but 4 and 2 links.
    two_p150a = FakeMeshDevice(num_devices=2, fabric_links=4)
    p300 = FakeMeshDevice(num_devices=2, fabric_links=2)

    assert ccl_module.get_num_links(two_p150a, cluster_axis=1) == 4
    assert ccl_module.get_num_links(p300, cluster_axis=1) == 2


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_single_device_has_no_links(native_get_num_links, ccl_module):
    """
    Validates that the CCL get_num_links function returns 0 for a single-device mesh.
    """
    mesh_device = FakeMeshDevice(num_devices=1)

    assert ccl_module.get_num_links(mesh_device) == 0
    assert ccl_module.get_num_links(mesh_device, cluster_axis=1) == 0
    assert native_get_num_links == []


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_rejects_invalid_cluster_axis(native_get_num_links, ccl_module):
    """
    Validates that the CCL get_num_links function rejects invalid cluster axis.
    """
    mesh_device = FakeMeshDevice(num_devices=8)

    with pytest.raises(ValueError, match="Unsupported cluster_axis: 2"):
        ccl_module.get_num_links(mesh_device, cluster_axis=2)
    assert native_get_num_links == []
