# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision stage driver: 4-chip SigLIP pipeline (embed + 3x9 layers + tail).

Forward path:
    pixel_values (torch)
      → chip 0 (embed)               → (B, 256, 1152) ttnn
      → host bounce                  → chip 1
      → chip 1 (layers 0..8)         → hidden
      → host bounce                  → chip 2
      → chip 2 (layers 9..17)        → hidden
      → host bounce                  → chip 3
      → chip 3 (layers 18..26 + post_ln + mm_projector)
                                     → (B, 256, 2048) ttnn  (left on chip 3)
"""

from __future__ import annotations

from typing import Dict

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from . import stages
from .vision_slice import SigLIPEmbedSlice, SigLIPLayerSlice, SigLIPTailSlice


class StageVision:
    def __init__(self, config: SigLIPConfig, weights: Dict[str, dict], mesh_handles, transport=None):
        if len(mesh_handles.vision_per_chip) != 4:
            raise RuntimeError(f"vision stage requires 4 chips, got {len(mesh_handles.vision_per_chip)}")
        if config.num_hidden_layers != stages.SIGLIP_TOTAL_LAYERS:
            raise RuntimeError(
                f"SigLIP layer count {config.num_hidden_layers} != expected {stages.SIGLIP_TOTAL_LAYERS}"
            )

        vw = weights["vlm_vision"]
        pw = weights["vlm_projector"]
        chips = mesh_handles.vision_per_chip

        # 0..N1=9, N1..N2=18, N2..N3=27 — three equal 9-layer chunks.
        N1 = stages.SIGLIP_LAYERS_PER_CHIP
        N2 = 2 * stages.SIGLIP_LAYERS_PER_CHIP
        N3 = stages.SIGLIP_TOTAL_LAYERS

        self.embed_slice = SigLIPEmbedSlice(config, vw, chips[0])
        self.layer_slice_a = SigLIPLayerSlice(config, vw, chips[1], layer_range=(0, N1))
        self.layer_slice_b = SigLIPLayerSlice(config, vw, chips[2], layer_range=(N1, N2))
        self.tail_slice = SigLIPTailSlice(config, vw, pw, chips[3], layer_range=(N2, N3))

        self.chips = chips
        if transport is None:
            from .transport import SocketTransport

            transport = SocketTransport()
        self.transport = transport

    def run(self, pixel_values):
        """pixel_values: (B, 3, H, W) — torch tensor OR persistent ttnn.Tensor on chips[0].

        For trace-capture compatibility, callers should pass a persistent
        ttnn.Tensor (pre-allocated on chips[0], refreshed via
        copy_host_to_device_tensor on CQ 1 between calls). Torch tensor
        input is supported for eager testing; we upload it inline.
        """
        import os as _os

        _fold_host_prep = getattr(self.embed_slice.patch_embed, "_use_fold", False) and _os.environ.get(
            "PI0_SIGLIP_FOLD_HOST_PREP", ""
        ).lower() in ("1", "true", "yes", "on")
        if isinstance(pixel_values, torch.Tensor) and not _fold_host_prep:
            # Default: upload as TILE BCHW; the embed slice does the on-device
            # patch extraction. With fold host-prep active we pass the torch
            # tensor straight through so the slice can host-permute+reshape
            # into the fold fast-path layout before upload.
            pixel_values = ttnn.from_torch(
                pixel_values,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.chips[0],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        h0 = self.embed_slice.forward(pixel_values)
        h1 = self.transport.send(h0, self.chips[1])
        h1 = self.layer_slice_a.forward(h1)
        h2 = self.transport.send(h1, self.chips[2])
        h2 = self.layer_slice_b.forward(h2)
        h3 = self.transport.send(h2, self.chips[3])
        return self.tail_slice.forward(h3)
