# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage PCC + mesh-carve smoke tests for the BH-Galaxy host-bounce pipeline.

Run a single test:
    pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py::test_mesh_carve_smoke
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx import stages
from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.tt_bh_glx.transport import send_via_host


def test_mesh_carve_smoke():
    """Open the parent 8x4 mesh, carve 4/18/6 submeshes + per-chip 1x1s, host-bounce a probe tensor."""
    with open_galaxy_mesh(l1_small_size=24576) as h:
        assert h.parent.get_num_devices() == 32
        assert h.vision_submesh.get_num_devices() == stages.VISION_NUM_CHIPS
        assert h.prefill_submesh.get_num_devices() == stages.PREFILL_NUM_CHIPS
        assert h.denoise_submesh.get_num_devices() == stages.DENOISE_NUM_CHIPS
        assert len(h.vision_per_chip) == stages.VISION_NUM_CHIPS
        assert len(h.prefill_per_chip) == stages.PREFILL_NUM_CHIPS
        assert len(h.denoise_per_chip) == stages.DENOISE_NUM_CHIPS
        for sm in h.vision_per_chip + h.prefill_per_chip + h.denoise_per_chip:
            assert sm.get_num_devices() == 1

        # Host-bounce: tile-aligned probe tensor across the three stage boundaries.
        probe = torch.randn(1, 32, 32, dtype=torch.float32)
        x0 = ttnn.from_torch(
            probe,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=h.vision_per_chip[0],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x1 = send_via_host(x0, h.vision_per_chip[3])
        x2 = send_via_host(x1, h.prefill_per_chip[0])
        x3 = send_via_host(x2, h.denoise_per_chip[0])
        out = ttnn.to_torch(x3)
        assert out.shape == probe.shape
        # bf16 round-trip — loose tolerance suffices.
        assert torch.allclose(out.float(), probe, atol=1e-2)
