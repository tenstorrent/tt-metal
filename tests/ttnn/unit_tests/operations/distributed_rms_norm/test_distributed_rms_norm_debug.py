# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic debug tests for distributed_rms_norm. DO NOT DELETE — they document the debugging process.

The mesh comes from the root conftest `mesh_device` fixture (fabric config via `device_params`), never opened here.
"""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_input_tensor, pytorch_distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = (14 if "blackhole" in ttnn.get_arch_name() else 7) * 1024 + 64
    return cfg


FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "fabric_router_config": _router_config()}]


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_all_ones_readback(mesh_device):
    """All-ones x, no gamma: inv = rsqrt(1 + eps) ~ 1, so out ~ 1 everywhere. Also checks that the input write
    reached the device (readback before the op) — the precompile-pass investigation (all-zero outputs)."""
    rows, cols = tuple(mesh_device.shape)
    shape = (1, 1, 32, 64)
    x = torch.ones((rows, cols, *shape), dtype=torch.bfloat16)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    back = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(tx)]
    print("DRMS_DEBUG input readback max:", [float(b.float().abs().max()) for b in back])
    out = distributed_rms_norm(tx, cluster_axis=1)
    devs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(out)]
    print("DRMS_DEBUG output mean per device:", [float(d.float().mean()) for d in devs])
    expected = pytorch_distributed_rms_norm(x, None, 1, 1e-6)
    for d in devs:
        assert torch.allclose(d.float(), expected[0, 0].float(), atol=0.05), f"max diff {(d.float() - 1).abs().max()}"
