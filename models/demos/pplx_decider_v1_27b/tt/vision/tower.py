# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pplx-decider vision tower on one device: patch embed + pos embed -> 27 ViT blocks -> 2x2 merger.

Reproduces ``Qwen3_5Model.get_image_features`` for one image (``Qwen3_5VisionModel.forward``,
``modeling_qwen3_5.py`` 1084-1125): ``pixel_values [n, 1536]`` -> image features ``[n/4, 5120]`` BF16.

Two phases:
- ``prepare_inputs(pixel_values, grid_thw)`` (host, per image): pads the patch rows to the vision
  bucket (256 / 512 / 768 / 1024, ``config.VISION_BUCKETS``), computes everything that depends only on
  ``grid_thw`` with HF's own helpers and op order - the bilinear pos-embed interpolation
  (``get_vision_bilinear_indices_and_weights`` + the BF16 table), the (row, col) rotary cos/sin
  (``get_vision_position_ids`` + fp32 ``inv_freq``) - and uploads them with ``cu_window_seqlens``
  ``[0, n, S]`` (the padded-key mask). This is the analogue of ``PplxDeciderModel.upload_tokens``.
- ``forward(inputs)`` (device only): no torch, ``from_torch`` or ``to_torch`` (tests/vision audit).

Weights: the 333 snapshot ``visual.*`` tensors (shard 1 only), strict key check, BF16, ~1.0 GB on
device after the head-dim / intermediate padding.

Usage::

    tower = PplxVisionTower.from_snapshot(device)
    inputs = tower.prepare_inputs(pixel_values, image_grid_thw)    # host -> device, padded to the bucket
    features = tower(inputs)                                       # device [1, 1, n/4, 5120] BF16
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionOptimizations, VisionPrecisionPolicy
from models.demos.pplx_decider_v1_27b.tt.rope import rope_head_permutation
from models.demos.pplx_decider_v1_27b.tt.vision.block import VisionBlock, VisionBlockConfig
from models.demos.pplx_decider_v1_27b.tt.vision.config import PplxVisionArgs, vision_bucket_for
from models.demos.pplx_decider_v1_27b.tt.vision.inputs import host_tables
from models.demos.pplx_decider_v1_27b.tt.vision.merger import PatchMergerConfig, VisionPatchMerger
from models.demos.pplx_decider_v1_27b.tt.vision.patch_embed import PatchEmbedConfig, VisionPatchEmbed
from models.demos.pplx_decider_v1_27b.tt.vision.weights import VISION_PREFIX, VisionTowerWeights, build_vision_weights


@dataclass
class VisionTowerConfig:
    weights: VisionTowerWeights
    args: PplxVisionArgs
    optimizations: VisionOptimizations


@dataclass
class VisionInputs:
    """One image on device, padded to its bucket. ``num_patches`` real rows, ``bucket`` physical rows."""

    pixels: ttnn.Tensor  # [1, 1, S, 1536] BF16 TILE (zero rows past n)
    pos_embed: ttnn.Tensor  # [1, 1, S, 1152] BF16 TILE (zero rows past n)
    cos: ttnn.Tensor  # [1, 1, S, 96] BF16 TILE, rope-permuted, 1 on padded dims / rows
    sin: ttnn.Tensor  # [1, 1, S, 96] BF16 TILE, 0 on padded dims / rows
    cu_window_seqlens: ttnn.Tensor  # [3] int32 ROW_MAJOR [0, n, S]
    num_patches: int
    bucket: int
    grid_thw: tuple[int, int, int] = field(default=(1, 0, 0))

    @property
    def num_tokens(self) -> int:
        return self.num_patches // 4

    def deallocate(self) -> None:
        for t in (self.pixels, self.pos_embed, self.cos, self.sin, self.cu_window_seqlens):
            ttnn.deallocate(t)


class PplxVisionTower(LightweightModule):
    def __init__(self, config: VisionTowerConfig):
        super().__init__()
        self.config = config
        a, opts, w = config.args, config.optimizations, config.weights
        self.mesh_device = opts.mesh_device
        self.patch_embed = VisionPatchEmbed.from_config(PatchEmbedConfig(w.patch_embed, opts))
        self.blocks = [VisionBlock(VisionBlockConfig(bw, a, opts, i)) for i, bw in enumerate(w.blocks)]
        self.merger = VisionPatchMerger(PatchMergerConfig(w.merger, a, opts))
        # Host-side (setup-time) state for prepare_inputs.
        self.pos_table = w.pos_embed  # BF16 [2304, 1152], = the nn.Embedding weight
        self.rope_perm = rope_head_permutation(a.head_dim, a.rotary_dim)

    @classmethod
    def from_config(cls, config: VisionTowerConfig) -> "PplxVisionTower":
        return cls(config)

    @classmethod
    def from_snapshot(
        cls, mesh_device, reader=None, *, policy: VisionPrecisionPolicy | None = None, load: bool = True
    ) -> "PplxVisionTower":
        """Build from the snapshot ``visual.*`` tensors only (the 54 GB text checkpoint is never read)."""
        from models.demos.pplx_decider_v1_27b.reference.hf_reference import SnapshotReader

        reader = reader or SnapshotReader()
        from transformers import AutoConfig

        vision_config = AutoConfig.from_pretrained(reader.path, local_files_only=True).vision_config
        args = PplxVisionArgs.from_hf_config(vision_config)
        opts = VisionOptimizations.build(mesh_device, policy=policy)
        weights = build_vision_weights(reader.tensors_with_prefix(VISION_PREFIX), args, opts.policy)
        tower = cls(VisionTowerConfig(weights=weights, args=args, optimizations=opts))
        if load:
            tower.load_device_weights()
        return tower

    def load_device_weights(self) -> None:
        self.patch_embed.load_device_weights()
        for block in self.blocks:
            block.load_device_weights()
        self.merger.load_device_weights()

    # -- host input preparation (per image, before the measured forward) -------------------------
    def prepare_inputs(self, pixel_values: torch.Tensor, grid_thw, *, bucket: int | None = None) -> VisionInputs:
        """Host -> device for one image: pixel rows (HF casts them to BF16 first), pos embed, cos/sin, mask."""
        a = self.config.args
        grid = tuple(int(v) for v in torch.as_tensor(grid_thw).reshape(-1).tolist())
        if len(grid) != 3 or grid[0] != 1:
            raise ValueError(f"One still image per call (grid_thw (1, h, w)), got {grid}")
        n = grid[1] * grid[2]
        if tuple(pixel_values.shape) != (n, a.patch_dim):
            raise ValueError(f"pixel_values {tuple(pixel_values.shape)} != ({n}, {a.patch_dim}) for grid {grid}")
        s = bucket or vision_bucket_for(n)
        if s < n or s % 128:
            raise ValueError(f"bucket {s} cannot hold {n} patches")
        tables = host_tables(grid, a, self.pos_table)
        pad = s - n

        def rows(t, value=0.0):
            return torch.nn.functional.pad(t, (0, 0, 0, pad), value=value)

        def rope_table(t, pad_value):
            t = t[:, self.rope_perm]  # same head-dim permutation as the q/k weights
            t = torch.nn.functional.pad(t, (0, a.padded_head_dim - a.head_dim), value=pad_value)
            return rows(t, pad_value).to(torch.bfloat16)

        def upload(t):
            return ttnn.from_torch(
                t.reshape(1, 1, s, -1).contiguous(),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return VisionInputs(
            pixels=upload(rows(pixel_values.to(torch.bfloat16))),
            pos_embed=upload(rows(tables["pos_embed"].to(torch.bfloat16))),
            cos=upload(rope_table(tables["rotary_cos"], 1.0)),
            sin=upload(rope_table(tables["rotary_sin"], 0.0)),
            cu_window_seqlens=ttnn.from_torch(
                torch.tensor([0, n, s], dtype=torch.int32),
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            num_patches=n,
            bucket=s,
            grid_thw=grid,
        )

    def upload_hidden(self, hidden: torch.Tensor, bucket: int) -> ttnn.Tensor:
        """Test seam: a [n, 1152] hidden state (e.g. a golden block input), zero padded, as [1, 1, S, 1152]."""
        h = torch.nn.functional.pad(hidden.to(torch.bfloat16), (0, 0, 0, bucket - hidden.shape[0]))
        return ttnn.from_torch(
            h.reshape(1, 1, bucket, -1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # -- device forward ----------------------------------------------------------------------------
    def embed(self, inputs: VisionInputs) -> ttnn.Tensor:
        return self.patch_embed(inputs.pixels, inputs.pos_embed)

    def run_block(self, idx: int, x: ttnn.Tensor, inputs: VisionInputs) -> ttnn.Tensor:
        return self.blocks[idx](x, cos=inputs.cos, sin=inputs.sin, cu_window_seqlens=inputs.cu_window_seqlens)

    def forward_hidden(self, inputs: VisionInputs) -> ttnn.Tensor:
        """Patch embed -> 27 blocks: the padded last hidden state [1, 1, S, 1152]."""
        x = self.embed(inputs)
        for i in range(len(self.blocks)):
            nxt = self.run_block(i, x, inputs)
            ttnn.deallocate(x)
            x = nxt
        return x

    def forward(self, inputs: VisionInputs) -> ttnn.Tensor:
        """Image features [1, 1, n/4, 5120] BF16 TILE in DRAM (the rows ``masked_scatter`` places in the text)."""
        x = self.forward_hidden(inputs)
        merged = self.merger(x)
        ttnn.deallocate(x)
        if inputs.num_patches == inputs.bucket:
            return merged
        out = ttnn.slice(merged, [0, 0, 0, 0], [1, 1, inputs.num_tokens, merged.shape[-1]])
        ttnn.deallocate(merged)
        return out
