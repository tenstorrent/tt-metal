# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the 4-chip SigLIP vision stage under the device profiler and dump the
per-op device-kernel CSV. Run with TT_METAL_DEVICE_PROFILER=1:

    source models/experimental/pi0_5/local_env.sh
    TT_METAL_DEVICE_PROFILER=1 PI05_GLX_TRANSPORT=host \
      python_env/bin/python models/experimental/pi0_5/tests/perf/prof_siglip.py

Then the device CSV lands under generated/profiler/.logs/profile_log_device.csv.
"""

from __future__ import annotations

import os

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
from models.experimental.pi0_5.tt.tt_bh_glx.stage_vision import StageVision
from models.experimental.pi0_5.tt.tt_bh_glx.transport import send_via_host

CKPT = os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream")
BS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
ITERS = int(os.environ.get("PROF_ITERS", "3"))


class _Handles:
    def __init__(self, parent, vision_submesh, vision_per_chip):
        self.parent = parent
        self.vision_submesh = vision_submesh
        self.vision_per_chip = vision_per_chip
        self.prefill_per_chip = []
        self.denoise_per_chip = []


class _HostTransport:
    def send(self, src_tensor, dst_mesh, **kw):
        return send_via_host(src_tensor, dst_mesh)


def main():
    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    loader = Pi0_5WeightLoader(CKPT)
    torch.manual_seed(42)
    pixel_values = torch.randn(BS, 3, cfg.image_size, cfg.image_size)

    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 4), l1_small_size=24576)
    submeshes = []
    try:
        vision_submesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
        submeshes.append(vision_submesh)
        vision_per_chip = []
        for c in range(4):
            sm = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, c))
            vision_per_chip.append(sm)
            submeshes.append(sm)
        handles = _Handles(parent, vision_submesh, vision_per_chip)
        stage = StageVision(cfg, loader.categorized_weights, handles, transport=_HostTransport())

        # warmup
        out = stage.run(pixel_values)
        ttnn.synchronize_device(vision_per_chip[3])
        # profiled iters
        for _ in range(ITERS):
            out = stage.run(pixel_values)
            ttnn.synchronize_device(vision_per_chip[3])
        # dump device profiler for each chip
        for sm in vision_per_chip:
            ttnn.ReadDeviceProfiler(sm)
        print("PROFILED_OK", tuple(ttnn.to_torch(out).shape))
    finally:
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)


if __name__ == "__main__":
    main()
