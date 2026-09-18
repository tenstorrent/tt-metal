# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke tests for the per-card device pool in scripts/run_safe_pytest.sh.

  scripts/run_safe_pytest.sh tests/ttnn/unit_tests/test_device_pool_smoke.py::test_single_card
      -> runs on the lowest free card, which the process sees as device 0.
  scripts/run_safe_pytest.sh --mesh tests/ttnn/unit_tests/test_device_pool_smoke.py::test_mesh
      -> holds every card and opens them all as one mesh.

Both print the physical PCIe ids they landed on so a log reader can check the
assignment against the pool's "card(s)=" line.
"""

import pytest
import torch
import ttnn


def test_single_card(device):
    a = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(ttnn.add(a, a))
    ids = ttnn.get_device_ids()
    print(f"POOL_SMOKE single: visible={ids} pcie={[ttnn.GetPCIeDeviceID(i) for i in ids]}")
    assert float(out.sum()) == 2.0 * 32 * 32


def test_mesh(mesh_device):
    ids = mesh_device.get_device_ids()
    pcie = [ttnn.GetPCIeDeviceID(i) for i in ids]
    print(f"POOL_SMOKE mesh: shape={tuple(mesh_device.shape)} device_ids={ids} pcie={pcie}")
    assert len(ids) == mesh_device.get_num_devices() >= 2
    a = ttnn.from_torch(
        torch.ones(1, 1, 32, 32 * len(ids), dtype=torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    out = ttnn.to_torch(ttnn.add(a, a), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    assert out.shape[-1] == 32 * len(ids)
    assert float(out.sum()) == 2.0 * 32 * 32 * len(ids)
