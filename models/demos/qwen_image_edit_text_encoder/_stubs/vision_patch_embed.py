# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `Qwen2_5_VisionPatchEmbed` (`model.visual.patch_embed`).

Conv3d(3 -> 1280, kernel = stride = (2, 14, 14), no bias) over pre-flattened patches is exactly a
Linear(1176 -> 1280) with the conv weight flattened in (c, t, h, w) order. Column-parallel on a mesh
(output channels split across TP) followed by an all_gather on the last dim.
"""

from __future__ import annotations

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import (
    hifi4_config,
    mesh_shape,
    shard_mapper,
    split_linear,
    upload,
)


class TtVisionPatchEmbed:
    def __init__(self, device, torch_module):
        _, self.tp = mesh_shape(device)
        w = torch_module.proj.weight.detach().float()
        w = w.reshape(w.shape[0], -1).t().contiguous()  # [1176, 1280]
        self.weight = upload(device, w, mapper=shard_mapper(device, -1) if self.tp > 1 else None)
        self.compute_cfg = hifi4_config()

    def __call__(self, x, dtype=None, precise=False, **kwargs):
        if precise:  # float32 pixels carried as bf16 hi + lo -> exact products against the bf16 kernel
            out = split_linear(x, self.weight, compute_kernel_config=self.compute_cfg, limbs=getattr(self, "limbs", 2))
        else:
            out = ttnn.linear(x, self.weight, compute_kernel_config=self.compute_cfg, dtype=dtype)
        if self.tp > 1:
            out = ttnn.all_gather(out, dim=-1, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out


def build(device, torch_module=None):
    return TtVisionPatchEmbed(device, torch_module)


def vision_patch_embed(device, torch_module=None):
    return build(device, torch_module)
