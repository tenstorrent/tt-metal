# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

import ttnn
from models.common.modules import tt_ccl as common_tt_ccl
from models.tt_transformers.tt import ccl as tt_transformers_ccl


class FakeMeshDevice:
    def __init__(self, global_num_devices, local_device_ids, dram_grid_x=8):
        self._global_num_devices = global_num_devices
        self._local_device_ids = local_device_ids
        self._dram_grid_size = SimpleNamespace(x=dram_grid_x)

    def get_num_devices(self):
        return self._global_num_devices

    def get_device_ids(self):
        return list(self._local_device_ids)

    def dram_grid_size(self):
        return self._dram_grid_size


class FakeMeshDeviceGetDeviceIdsRaises(FakeMeshDevice):
    def get_device_ids(self):
        raise RuntimeError("debug assert from get_device_ids")


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_uses_host_local_device_count_for_multihost_wormhole(monkeypatch, ccl_module):
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
    mesh_device = FakeMeshDevice(global_num_devices=64, local_device_ids=range(32))

    assert ccl_module.get_num_links(mesh_device, cluster_axis=0) == 4
    assert ccl_module.get_num_links(mesh_device, cluster_axis=1) == 4
    assert ccl_module.get_num_links(mesh_device) == 4


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_rejects_invalid_cluster_axis(monkeypatch, ccl_module):
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
    mesh_device = FakeMeshDevice(global_num_devices=8, local_device_ids=range(8))

    with pytest.raises(ValueError, match="Unsupported cluster_axis: 2"):
        ccl_module.get_num_links(mesh_device, cluster_axis=2)


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_requires_local_devices(monkeypatch, ccl_module):
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
    mesh_device = FakeMeshDevice(global_num_devices=64, local_device_ids=[])

    with pytest.raises(ValueError, match="requires at least one host-local device"):
        ccl_module.get_num_links(mesh_device)


@pytest.mark.parametrize("ccl_module", [tt_transformers_ccl, common_tt_ccl])
def test_get_num_links_normalizes_get_device_ids_failures(monkeypatch, ccl_module):
    monkeypatch.setattr(ttnn, "get_arch_name", lambda: "wormhole_b0")
    mesh_device = FakeMeshDeviceGetDeviceIdsRaises(global_num_devices=64, local_device_ids=[])

    with pytest.raises(ValueError, match="requires at least one host-local device"):
        ccl_module.get_num_links(mesh_device)
