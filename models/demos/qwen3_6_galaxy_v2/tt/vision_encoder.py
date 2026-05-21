# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-VISION-ENCODER: composite qwen3.6 vision encoder.

Wraps the full pipeline:
  pixel_values [N, 1536]
    → patch_embed   (CPU Conv3d, one-shot per request)
    → pos_embed     (CPU interpolated lookup, one-shot per request)
    → 27 × Qwen36VisionBlockTP  (BH GLX 8×4 mesh, TP=8 + DP=4-capable)
    → Qwen36VisionPatchMergerTP (replicated)
    → features [N // 4, 5120]   (ready for splice into text decoder)

The patch_embed and pos_embed are small one-shot ops (~ms) so we keep
them on CPU. The 27 blocks + merger run on device.

Construction loads weights ONCE from the HF reference vision model — the
Conv3d patch_embed weights and pos_embed lookup table go to CPU; the
block + merger weights go through the existing TT block constructors.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.vision_attention_tp import build_vision_rope_tensors
from models.demos.qwen3_6_galaxy_v2.tt.vision_block_tp import Qwen36VisionBlockTP
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.demos.qwen3_6_galaxy_v2.tt.vision_patch_merger import Qwen36VisionPatchMergerTP
from models.demos.qwen3_vl.reference.functional import qwen3_vl_fast_pos_embed_interpolation
from models.tt_dit.parallel.manager import CCLManager


class Qwen36VisionEncoder:
    """Composite vision encoder: HF patch_embed + pos_embed on CPU, then 27 TT blocks + merger.

    This is NOT a tt_dit `Module` — it owns CPU torch.nn modules AND a
    list of TT Module subclasses. Construction is one-shot per generator
    init; forward is one-shot per image at prefill.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        model_args: Qwen36VisionModelArgs,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.model_args = model_args
        self.vc = model_args.hf_config.vision_config

        # CPU pieces: patch_embed Conv3d + pos_embed lookup. Pulled from HF
        # reference so we get the exact qwen3.6 weights without rewriting them.
        ref = model_args.reference_vision_model()
        self._ref_patch_embed = ref.patch_embed
        self._ref_pos_embed = ref.pos_embed
        self._num_grid_per_side = ref.num_grid_per_side  # int(sqrt(num_position_embeddings))

        # Cache vision-encoder per-layer state-dicts so each block constructor
        # can pull its own subset.
        full_sd = ref.state_dict()
        # Strip "blocks." prefix and group per layer
        blocks_state: list[dict[str, torch.Tensor]] = [dict() for _ in range(self.vc.depth)]
        for k, v in full_sd.items():
            if not k.startswith("blocks."):
                continue
            rest = k[len("blocks.") :]
            layer_str, sub = rest.split(".", 1)
            blocks_state[int(layer_str)][sub] = v

        # 27 TP=8 TT blocks
        self.blocks: list[Qwen36VisionBlockTP] = []
        for layer_num in range(self.vc.depth):
            self.blocks.append(
                Qwen36VisionBlockTP(
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    state_dict=blocks_state[layer_num],
                    hidden_size=self.vc.hidden_size,
                    intermediate_size=self.vc.intermediate_size,
                    num_heads=self.vc.num_heads,
                    head_dim=self.vc.hidden_size // self.vc.num_heads,
                    norm_eps=1e-6,
                    tp_mesh_axis=0,
                    dtype=dtype,
                )
            )

        # PatchMerger
        merger_state = {k[len("merger.") :]: v for k, v in full_sd.items() if k.startswith("merger.")}
        self.merger = Qwen36VisionPatchMergerTP(
            mesh_device=mesh_device,
            state_dict=merger_state,
            hidden_size=self.vc.hidden_size,
            spatial_merge_size=self.vc.spatial_merge_size,
            out_hidden_size=self.vc.out_hidden_size,
            norm_eps=1e-6,
            dtype=dtype,
        )

        # Padded head_dim for RoPE — pulled from the first block's attention
        # (already computed there as ceil(72/32)*32 = 96).
        self.padded_head_dim = self.blocks[0].attn.padded_head_dim

    def _patch_embed_cpu(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run HF patch_embed (Conv3d) on CPU."""
        return self._ref_patch_embed(pixel_values)

    def _pos_embed_cpu(self, hidden: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Add interpolated positional embedding via the HF helper.

        The helper expects the FULL embedding module (not just the weight),
        because it does both `pos_embed.weight.dtype` and `pos_embed(idx)`.
        """
        pos = qwen3_vl_fast_pos_embed_interpolation(
            grid_thw=grid_thw,
            num_grid_per_side=self._num_grid_per_side,
            pos_embed=self._ref_pos_embed,
            spatial_merge_size=self.vc.spatial_merge_size,
        )
        return hidden + pos

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end vision encoder forward.

        Args:
            pixel_values: `[N, in_channels * temporal_patch_size * patch_size**2]`
                          = `[N, 1536]` for qwen3.6 (HF processor output, flat
                          across all images/frames in the request).
            grid_thw: `[num_images, 3]` (T, H, W) per image.

        Returns:
            torch.Tensor `[N // spatial_merge_unit, out_hidden_size=5120]`
            — ready to splice into the text decoder's embedding stream at
            vision_token positions.
        """
        # --- CPU pieces: patch_embed + pos_embed ---
        x = self._patch_embed_cpu(pixel_values)  # [N, hidden_size=1152]
        x = self._pos_embed_cpu(x, grid_thw)  # [N, 1152]
        seq_len = x.shape[0]
        H = self.vc.hidden_size

        # --- Vision 2D RoPE tables ---
        cos_tt, sin_tt = build_vision_rope_tensors(
            seq_len=seq_len,
            grid_thw=grid_thw,
            head_dim=self.vc.hidden_size // self.vc.num_heads,
            padded_head_dim=self.padded_head_dim,
            spatial_merge_size=self.vc.spatial_merge_size,
            mesh_device=self.mesh_device,
        )

        # --- Upload x to mesh (replicated across all 32 chips) ---
        x_tt = ttnn.from_torch(
            x.view(1, 1, seq_len, H),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        # --- 27 vision blocks ---
        for layer_num, block in enumerate(self.blocks):
            x_tt = block.forward(x_tt, cos_tt, sin_tt)

        # --- PatchMerger ---
        x_tt = self.merger.forward(x_tt)  # [B, 1, seq_len // 4, 5120] replicated

        # --- Pull chip 0's view back to torch ---
        x_torch = ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        # shape varies: [num_devices, 1, S/4, 5120] or [num_devices, S/4, 5120]; we slice chip 0
        if x_torch.dim() == 4:
            x_torch = x_torch[0, 0]
        else:
            x_torch = x_torch[0]
        return x_torch[: seq_len // (self.vc.spatial_merge_size**2), : self.vc.out_hidden_size]
