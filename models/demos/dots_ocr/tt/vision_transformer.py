# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full vision stack for Dots OCR: ``PatchEmbedTT`` → ``VisionBlockTT`` → post-trunk ``RMSNorm`` →
``PatchMergerTT``.

The TT path is intended to run fully in TTNN (QKV/proj/MLP + RoPE + SDPA + merger). No HF
``vision_tower`` forward is called; we only consume HF checkpoint tensors.
"""

from __future__ import annotations

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
from models.demos.dots_ocr.tt.vision_block import create_vision_block
from models.demos.dots_ocr.tt.vision_model_config import DotsVisionModelArgs
from models.demos.dots_ocr.tt.vision_patch_embed import create_patch_embed
from models.demos.dots_ocr.tt.vision_rmsnorm import RMSNorm


class VisionTransformerTT(LightweightModule):
    """
    Full TTNN Vision Transformer for Dots.mocr.

    This implements the complete vision tower:
    1. Patch embedding
    2. 42 transformer blocks with post-norm
    3. Final patch merging
    """

    def __init__(
        self,
        mesh_device,
        model_args: DotsVisionModelArgs,
        state_dict: dict,
        weight_cache_path=None,
        dtype=None,
    ):
        super().__init__()
        ttnn = get_ttnn()
        if dtype is None:
            dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16
        self.mesh_device = mesh_device
        self.model_args = model_args
        self.dtype = dtype
        self.num_layers = model_args.vision_config.num_hidden_layers  # 42 layers

        # Patch embedding
        self.patch_embed = create_patch_embed(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

        # 42 Vision blocks
        self.blocks = []
        for i in range(self.num_layers):
            block = create_vision_block(
                mesh_device=mesh_device,
                model_args=model_args,
                state_dict=state_dict,
                layer_num=i,
                weight_cache_path=weight_cache_path,
                dtype=dtype,
            )
            self.blocks.append(block)

        if "vision_tower.post_trunk_norm.weight" in state_dict:
            _trunk_nk = "post_trunk_norm"
        else:
            _trunk_nk = "norm"
        self.norm = RMSNorm(
            device=mesh_device,
            dim=model_args.vision_dim,
            state_dict=state_dict,
            state_dict_prefix="vision_tower.",
            weight_key=_trunk_nk,
            weight_dtype=dtype,
            eps=model_args.rms_norm_eps,
        )

        # Use existing PatchMerger (already implemented)
        from models.demos.dots_ocr.tt.patch_merger import PatchMerger as PatchMergerTT

        # Dots HF checkpoints have used both `vision_tower.merger.*` and `vision_tower.patch_merger.*`
        # naming schemes. Select the prefix that contains the actual FFN weights.
        def _has_merger_weights(prefix: str) -> bool:
            # Checkpoints may store patch-merger MLP weights as either:
            # - {prefix}.feed_forward.{0,2}.weight (our TT layout)
            # - {prefix}.mlp.{0,2}.weight         (HF layout)
            has_ffn = (
                f"{prefix}.feed_forward.0.weight" in state_dict and f"{prefix}.feed_forward.2.weight" in state_dict
            )
            has_mlp = f"{prefix}.mlp.0.weight" in state_dict and f"{prefix}.mlp.2.weight" in state_dict
            return has_ffn or has_mlp

        patch_merger_prefix = "vision_tower.patch_merger"
        alt = "vision_tower.merger"
        if not _has_merger_weights(patch_merger_prefix) and _has_merger_weights(alt):
            patch_merger_prefix = alt

        self.patch_merger = PatchMergerTT(
            mesh_device=mesh_device,
            hidden_size=model_args.vision_dim,
            out_hidden_size=model_args.vision_dim,  # Usually same size
            spatial_merge_size=model_args.spatial_merge_size,
            state_dict=state_dict,
            state_dict_prefix=patch_merger_prefix,
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None) -> torch.Tensor:
        """
        Full vision forward: patch embed → blocks → post-trunk RMSNorm → patch merger on TTNN.
        """
        if grid_thw is None:
            raise ValueError("grid_thw is required for Dots vision (TTNN path)")

        ttnn = get_ttnn()
        if ttnn is None or self.mesh_device is None:
            raise RuntimeError("VisionTransformerTT requires ttnn and mesh_device")

        x = self.patch_embed(pixel_values, grid_thw)
        if isinstance(x, torch.Tensor):
            x = ttnn.from_torch(
                x.unsqueeze(1) if x.dim() == 3 else x,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=getattr(ttnn, "DRAM_MEMORY_CONFIG", None),
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device)
                if hasattr(ttnn, "ReplicateTensorToMesh")
                else None,
            )
        # Normalize patch_embed output to [1,1,S,D]
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, x.shape[1], x.shape[2]))

        rot_mats, cu_t = self.build_rot_mats_and_cu(grid_thw=grid_thw, seq_len=int(x.shape[2]))

        for block in self.blocks:
            x = block(x, rot_mats=rot_mats, cu_seqlens=cu_t)

        x = self.norm(x)
        merged = self.patch_merger(x)
        composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        merged = ttnn.to_torch(merged, mesh_composer=composer).to(torch.bfloat16)
        # On multi-device meshes with replicated tensors, ConcatMeshToTensor will stack replicas
        # along dim=0. Slice off the first replica so the returned shape matches [B,1,S,D].
        try:
            num_devices = self.mesh_device.get_num_devices()
            if num_devices > 1 and merged.dim() >= 1 and merged.shape[0] % num_devices == 0:
                per = merged.shape[0] // num_devices
                merged = merged[:per]
        except Exception:
            pass
        if merged.dim() == 4:
            merged = merged.squeeze(0).squeeze(0)
        return merged

    def build_rot_mats_and_cu(self, *, grid_thw: torch.Tensor, seq_len: int):
        """
        Build TTNN RoPE cos/sin tables and cu_seqlens for varlen attention.

        Returns:
          rot_mats: (cos, sin) each [1,1,S,head_dim] in TILE layout (bf16)
          cu_t: uint32 [num_segments+1] row-major on device
        """
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("VisionTransformerTT requires ttnn")
        if grid_thw is None:
            raise ValueError("grid_thw is required")

        g = grid_thw.detach().cpu() if getattr(grid_thw, "is_cuda", False) else grid_thw
        if g.dim() != 2 or g.shape[1] != 3:
            raise ValueError(f"grid_thw must be [N,3], got {g.shape}")

        token_counts = [int(t) * int(h) * int(w) for t, h, w in g.tolist()]
        expected = int(sum(token_counts))
        if int(seq_len) != expected:
            raise ValueError(f"seq_len mismatch: got S={seq_len}, but grid_thw implies total={expected} tokens")

        mem = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
        sms = int(self.model_args.spatial_merge_size)

        # RoPE math is sensitive; do it in fp32 when available.
        rope_dtype = getattr(ttnn, "float32", None) or ttnn.bfloat16

        head_dim = int(self.model_args.vision_head_dim)
        if head_dim % 2 != 0:
            raise ValueError(f"vision_head_dim must be even for RoPE, got {head_dim}")
        # Match HF remote-code VisionRotaryEmbedding:
        # rotary_dim = head_dim//2, inv_freq length = rotary_dim//2 = head_dim//4
        rotary_dim = head_dim // 2
        if rotary_dim % 2 != 0:
            raise ValueError(f"vision rotary_dim must be even, got {rotary_dim} (head_dim={head_dim})")
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        inv = ttnn.from_torch(
            inv_freq.to(torch.float32).reshape(1, 1, 1, rotary_dim // 2),
            device=self.mesh_device,
            dtype=rope_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )

        hpos_segments = []
        wpos_segments = []
        cu = [0]
        running = 0
        for t, h, w in g.tolist():
            t = int(t)
            h = int(h)
            w = int(w)
            if h % sms != 0 or w % sms != 0:
                raise ValueError(f"grid {h}x{w} not divisible by spatial_merge_size={sms}")

            h_ids = ttnn.arange(
                0,
                h,
                dtype=rope_dtype,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
            )
            h_ids = ttnn.reshape(h_ids, (1, 1, h, 1))
            ones_w = ttnn.ones(
                (1, 1, 1, w), dtype=rope_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=mem
            )
            h_grid = ttnn.matmul(h_ids, ones_w, memory_config=mem)  # [1,1,h,w]
            h_grid = ttnn.reshape(h_grid, (h, w))
            h_grid = ttnn.reshape(h_grid, (h // sms, sms, w // sms, sms))
            h_grid = ttnn.permute(h_grid, (0, 2, 1, 3))
            hpos = ttnn.reshape(h_grid, (1, 1, h * w, 1))

            ones_h = ttnn.ones(
                (1, 1, h, 1), dtype=rope_dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, memory_config=mem
            )
            w_ids = ttnn.arange(
                0,
                w,
                dtype=rope_dtype,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
            )
            w_ids = ttnn.reshape(w_ids, (1, 1, 1, w))
            w_grid = ttnn.matmul(ones_h, w_ids, memory_config=mem)  # [1,1,h,w]
            w_grid = ttnn.reshape(w_grid, (h, w))
            w_grid = ttnn.reshape(w_grid, (h // sms, sms, w // sms, sms))
            w_grid = ttnn.permute(w_grid, (0, 2, 1, 3))
            wpos = ttnn.reshape(w_grid, (1, 1, h * w, 1))

            if t != 1:
                hpos = ttnn.concat([hpos] * t, dim=2)
                wpos = ttnn.concat([wpos] * t, dim=2)

            hpos_segments.append(hpos)
            wpos_segments.append(wpos)
            running += t * h * w
            cu.append(running)

        hpos_all = ttnn.concat(hpos_segments, dim=2) if len(hpos_segments) > 1 else hpos_segments[0]
        wpos_all = ttnn.concat(wpos_segments, dim=2) if len(wpos_segments) > 1 else wpos_segments[0]

        hpos_rm = ttnn.to_layout(ttnn.typecast(hpos_all, dtype=rope_dtype), ttnn.TILE_LAYOUT)
        wpos_rm = ttnn.to_layout(ttnn.typecast(wpos_all, dtype=rope_dtype), ttnn.TILE_LAYOUT)
        freqs_h = ttnn.matmul(hpos_rm, inv, memory_config=mem)
        freqs_w = ttnn.matmul(wpos_rm, inv, memory_config=mem)
        cos_h = ttnn.cos(freqs_h, memory_config=mem)
        sin_h = ttnn.sin(freqs_h, memory_config=mem)
        cos_w = ttnn.cos(freqs_w, memory_config=mem)
        sin_w = ttnn.sin(freqs_w, memory_config=mem)

        # HF builds freqs for (h,w) and flattens -> [S, head_dim//2], then repeats to [S, head_dim].
        cos_half = ttnn.concat([cos_h, cos_w], dim=-1)  # [1,1,S,head_dim//2]
        sin_half = ttnn.concat([sin_h, sin_w], dim=-1)  # [1,1,S,head_dim//2]
        cos_full = ttnn.concat([cos_half, cos_half], dim=-1)
        sin_full = ttnn.concat([sin_half, sin_half], dim=-1)
        cos = ttnn.typecast(cos_full, dtype=ttnn.bfloat16)
        sin = ttnn.typecast(sin_full, dtype=ttnn.bfloat16)
        rot_mats = (ttnn.to_layout(cos, ttnn.TILE_LAYOUT), ttnn.to_layout(sin, ttnn.TILE_LAYOUT))

        cu_t = ttnn.from_torch(
            torch.tensor(cu, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=mem,
            mesh_mapper=mapper,
        )
        return rot_mats, cu_t

    def to_host(self):
        """Ensure all tensors are moved to host for cleanup."""


# Main interface function to replace the hybrid vision_tower_forward
def vision_tower_forward_ttnn(
    vision_transformer: VisionTransformerTT, pixel_values: torch.Tensor, grid_thw: torch.Tensor
) -> torch.Tensor:
    """
    TTNN version of vision tower forward.

    This is the main entry point that should be called instead of
    the hybrid version when using full TTNN vision.
    """
    return vision_transformer.forward(pixel_values, grid_thw)


# Convenience function to create the full vision transformer
def create_dots_vision_transformer(
    mesh_device, model_args=None, state_dict=None, weight_cache_path=None, dtype=None, hf_model=None
):
    """Create full TTNN VisionTransformerTT for Dots OCR."""
    if model_args is None:
        model_args = DotsVisionModelArgs(mesh_device=mesh_device)

    if dtype is None:
        ttnn = get_ttnn()
        dtype = ttnn.bfloat16 if ttnn is not None else torch.bfloat16

    if state_dict is None:
        # Create dummy state dict for testing
        state_dict = {}

    return VisionTransformerTT(
        mesh_device=mesh_device,
        model_args=model_args,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        dtype=dtype,
    )
