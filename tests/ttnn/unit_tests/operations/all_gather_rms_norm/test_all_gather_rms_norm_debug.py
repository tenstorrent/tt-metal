# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""all_gather_rms_norm debug tests (implementer). DO NOT DELETE — documents the debugging process."""

import pytest
import torch
import ttnn


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True, ids=["torus_xy"])
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True, ids=["mesh1x4"])
def test_mesh_open_only(mesh_device):
    """Baseline: open the fabric mesh, move a tensor, no op. Tells whether teardown Ethernet timeouts are ours."""
    t = ttnn.from_torch(
        torch.ones(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    assert ttnn.to_torch(ttnn.get_device_tensors(t)[0]).sum() == 32 * 32
