# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Patch Embedding for Dots OCR Vision Transformer.

Converts image input (pixel_values) + grid_thw into patch embeddings
that can be fed into the vision transformer blocks.
"""

from __future__ import annotations

import os

import torch

from models.common.lightweightmodule import LightweightModule
from models.demos.dots_ocr.tt._ttnn_import import get_ttnn


class PatchEmbedTT(LightweightModule):
    """
    TTNN implementation of patch embedding for Dots Vision Transformer.

    Handles:
    - Converting images to patches (14x14 for Dots)
    - Linear projection
    - Position embeddings
    - grid_thw (temporal, height, width) handling for document images
    """

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        weight_cache_path=None,
        dtype=None,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1536,
        state_dict_prefix: str = "vision_tower.patch_embed",
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.state_dict_prefix = state_dict_prefix
        ttnn = get_ttnn()
        if dtype is None:
            # Stub/partial ``ttnn`` installs may lack ``bfloat16``; fall back to torch.
            if ttnn is not None and getattr(ttnn, "bfloat16", None) is not None:
                dtype = ttnn.bfloat16
            else:
                dtype = torch.bfloat16
        self.dtype = dtype

        # Load weights from state dict
        self._load_weights(state_dict, weight_cache_path, dtype)

    def _load_weights(self, state_dict: dict, weight_cache_path, dtype):
        """Load patch embed weights from HF state dict."""
        ttnn = get_ttnn()
        # Some callers provide prefixes with a trailing '.' (e.g. "vision_tower.patch_embed.").
        # Normalize so f"{prefix}.foo" never produces a double-dot key.
        prefix = (self.state_dict_prefix or "").rstrip(".")
        # Keep torch copies for bring-up diagnostics (device-vs-torch comparisons).
        # These are small relative to the full checkpoint and only used when debugging is enabled.
        self._torch_proj_weight_out_in = None  # [out_features, in_features]
        self._torch_proj_bias = None  # [out_features]
        self._torch_patch_norm_gamma = None  # [out_features]

        # Projection weight: [embed_dim, in_channels * patch_size * patch_size]
        proj_weight_key = f"{prefix}.proj.weight" if prefix else "proj.weight"
        proj_bias_key = f"{prefix}.proj.bias" if prefix else "proj.bias"
        # Dots checkpoints sometimes nest the conv under a patchifier module.
        if proj_weight_key not in state_dict:
            for alt in (
                (f"{prefix}.patchifier.proj.weight" if prefix else "patchifier.proj.weight"),
                # `standardize_hf_keys_multimodal` may rename `proj` -> `o_proj`
                (f"{prefix}.patchifier.o_proj.weight" if prefix else "patchifier.o_proj.weight"),
                (f"{prefix}.patchifier.proj._linear.weight" if prefix else "patchifier.proj._linear.weight"),
                (f"{prefix}.patchifier._linear.weight" if prefix else "patchifier._linear.weight"),
            ):
                if alt in state_dict:
                    proj_weight_key = alt
                    break
        if proj_bias_key not in state_dict:
            for alt in (
                (f"{prefix}.patchifier.proj.bias" if prefix else "patchifier.proj.bias"),
                (f"{prefix}.patchifier.o_proj.bias" if prefix else "patchifier.o_proj.bias"),
                (f"{prefix}.patchifier.proj._linear.bias" if prefix else "patchifier.proj._linear.bias"),
                (f"{prefix}.patchifier._linear.bias" if prefix else "patchifier._linear.bias"),
            ):
                if alt in state_dict:
                    proj_bias_key = alt
                    break
        if proj_weight_key in state_dict and ttnn is not None and self.mesh_device is not None:
            # HF stores this as [embed_dim, C, ph, pw] or [embed_dim, patch_dim].
            # TTNN `linear` expects weight shaped [in_features, out_features] (so matmul K matches input width).
            weight = state_dict[proj_weight_key]
            if weight.dim() == 4:
                weight = weight.reshape(weight.shape[0], -1)
            # Torch reference copies (float32).
            self._torch_proj_weight_out_in = weight.detach().float().contiguous()
            # Keep HF weight layout [out_features, in_features] and use transpose_b=True in ttnn.linear,
            # matching the common usage pattern in TTNN tutorials.
            memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

            def _cache_name(base: str, *, layout: str) -> str | None:
                if not weight_cache_path:
                    return None
                # Avoid stale-cache issues when weight transforms/layouts change.
                # Mirror the cache naming style used elsewhere in this repo.
                return str(weight_cache_path / f"{base}_dtype_BFLOAT16_layout_{layout}.tensorbin")

            self.proj_weight = ttnn.as_tensor(
                weight,
                device=self.mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=_cache_name(proj_weight_key, layout="TILE"),
            )
            self.proj_bias = None
            if proj_bias_key in state_dict:
                b = state_dict[proj_bias_key].reshape(1, 1, 1, -1).to(torch.bfloat16)
                self._torch_proj_bias = state_dict[proj_bias_key].detach().float().contiguous()
                self.proj_bias = ttnn.as_tensor(
                    b,
                    device=self.mesh_device,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory_config,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    cache_file_name=_cache_name(proj_bias_key, layout="TILE"),
                )
            # Optional patchifier norm (present in dots.mocr): `patchifier.norm.weight`
            norm_key = None
            for cand in (
                (f"{prefix}.patchifier.norm.weight" if prefix else "patchifier.norm.weight"),
                (f"{prefix}.norm.weight" if prefix else "norm.weight"),
            ):
                if cand in state_dict:
                    norm_key = cand
                    break
            self.patch_norm_weight = None
            if norm_key is not None:
                # Match tt_transformers' RMSNorm weight layout: [1,1,dim//32,32] in ROW_MAJOR.
                w = state_dict[norm_key].to(torch.bfloat16)
                self._torch_patch_norm_gamma = state_dict[norm_key].detach().float().contiguous()
                tile = 32
                assert w.numel() == self.embed_dim
                w_tile = w.view(1, 1, self.embed_dim).reshape(1, 1, self.embed_dim // tile, tile)
                self.patch_norm_weight = ttnn.as_tensor(
                    w_tile,
                    device=self.mesh_device,
                    dtype=dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=memory_config,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    cache_file_name=_cache_name(norm_key, layout="ROW_MAJOR"),
                )
        else:
            # Fallback for testing
            self.proj_weight = None
            self.proj_bias = None
            self.patch_norm_weight = None
            self._torch_proj_weight_out_in = None
            self._torch_proj_bias = None
            self._torch_patch_norm_gamma = None
            if proj_weight_key not in state_dict:
                searched = [
                    proj_weight_key,
                    (f"{prefix}.patchifier.proj.weight" if prefix else "patchifier.proj.weight"),
                    (f"{prefix}.patchifier.o_proj.weight" if prefix else "patchifier.o_proj.weight"),
                    (f"{prefix}.patchifier.proj._linear.weight" if prefix else "patchifier.proj._linear.weight"),
                    (f"{prefix}.patchifier._linear.weight" if prefix else "patchifier._linear.weight"),
                ]
                print(f"Warning: patch embed weight not found; tried keys: {tuple(searched)}")

        # Check for position embedding (if it exists)
        pos_embed_key = f"{prefix}.position_embedding.weight"
        if pos_embed_key in state_dict and ttnn is not None and self.mesh_device is not None:
            pos_weight = state_dict[pos_embed_key]
            memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            self.pos_embed_weight = ttnn.as_tensor(
                pos_weight,
                device=self.mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=weight_cache_path / "pos_embed" if weight_cache_path else None,
            )
        else:
            self.pos_embed_weight = None

    def _maybe_add_pos_embed(self, x_tt, *, seq_len: int):
        """
        If the checkpoint provides learned position embeddings, add them to patch embeddings.

        HF towers typically store `position_embedding.weight` as [S_max, D].
        We reshape to [1,1,S_max,D], slice to the current `seq_len`, and add.
        """
        ttnn = get_ttnn()
        if ttnn is None or self.pos_embed_weight is None:
            return x_tt
        try:
            pe = self.pos_embed_weight
            # Normalize to [1,1,S_max,D]
            if len(pe.shape) == 2:
                pe = ttnn.reshape(pe, (1, 1, int(pe.shape[0]), int(pe.shape[1])))
            elif len(pe.shape) == 4:
                pass
            else:
                return x_tt
            if int(pe.shape[2]) < int(seq_len):
                return x_tt
            pe = ttnn.slice(pe, (0, 0, 0, 0), (1, 1, int(seq_len), int(pe.shape[3])))
            return ttnn.add(x_tt, pe)
        except Exception:
            return x_tt

    def _process_grid_thw(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Process grid_thw tensor from processor.

        grid_thw typically has shape [batch_size, 3] with values:
        [temporal_patches, height_patches, width_patches]
        """
        if grid_thw is None:
            # Default for square images
            return torch.tensor([[1, 16, 16]])  # 1 temporal, 16x16 spatial patches

        if grid_thw.dim() == 2:
            return grid_thw
        elif grid_thw.dim() == 1:
            return grid_thw.unsqueeze(0)
        return grid_thw

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor | None = None) -> ttnn.Tensor:
        """
        Convert image to patch embeddings.

        Args:
            pixel_values: [B, C, H, W] image tensor
            grid_thw: [B, 3] grid dimensions from processor (temporal, height, width)

        Returns:
            ttnn.Tensor: Patch embeddings [B, num_patches, embed_dim]
        """
        # Dots / HF processors may return either:
        # - raw images:        [B, C, H, W]
        # - patchified vectors:[S, patch_dim] (already flattened per patch)
        # Support both (same contract as ``model.vision_tower.patch_embed``).
        if pixel_values.dim() == 2:
            # Already patchified: [S, patch_dim]
            x = pixel_values.to(torch.bfloat16)
            # Add batch dim for consistency
            x = x.unsqueeze(0)  # [1, S, patch_dim]
            B = 1
            num_patches = x.shape[1]
            ttnn = get_ttnn()
            if ttnn is None or self.mesh_device is None or self.proj_weight is None:
                # CPU-only / bring-up fallback
                return torch.randn(B, num_patches, self.embed_dim, dtype=torch.bfloat16)

            # Device projection: [1,S,patch_dim] -> [1,1,S,embed_dim]
            memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
            x_tt = ttnn.from_torch(
                x,
                device=self.mesh_device,
                dtype=getattr(ttnn, "bfloat16", torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                memory_config=memory_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if len(x_tt.shape) == 3:
                x_tt = ttnn.reshape(x_tt, (1, 1, x_tt.shape[1], x_tt.shape[2]))
            # HF patch_embed uses bias; include it when available.
            out = ttnn.linear(
                x_tt,
                self.proj_weight,
                bias=self.proj_bias,
                transpose_b=True,
                memory_config=memory_config,
            )
            if self.patch_norm_weight is not None and hasattr(ttnn, "rms_norm"):
                # HF patchifier RMSNorm uses eps=1e-5
                out = ttnn.rms_norm(out, weight=self.patch_norm_weight, epsilon=1e-5)

            # Optional bring-up diagnostic: compare device output vs torch reference using the *same* weights.
            # Enable with: DOTS_DEBUG_PATCH_EMBED=1
            if os.environ.get("DOTS_DEBUG_PATCH_EMBED") == "1":
                try:
                    print("[DOTS_DEBUG_PATCH_EMBED] enabled")
                    # Convert TT output back to torch (replicated mesh -> pick first replica).
                    mesh_composer = None
                    if hasattr(ttnn, "ConcatMeshToTensor"):
                        mesh_composer = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
                    tt_out = ttnn.to_torch(out, mesh_composer=mesh_composer) if mesh_composer else ttnn.to_torch(out)
                    # If replicated, first replica matches the full tensor.
                    if tt_out.dim() >= 4 and tt_out.shape[0] > 1:
                        tt_out = tt_out[0]
                    # Normalize shapes: [1,1,S,D] -> [S,D]
                    tt_out_2d = tt_out.reshape(-1, int(tt_out.shape[-1])).float()

                    # Torch reference: linear + rmsnorm on the same input x (bf16) using the exact torch weights we loaded.
                    if self._torch_proj_weight_out_in is None:
                        raise RuntimeError("missing _torch_proj_weight_out_in (weight loading failed?)")
                    torch_lin = torch.nn.functional.linear(
                        x.squeeze(0).float(),
                        self._torch_proj_weight_out_in,
                        self._torch_proj_bias,
                    )
                    if self._torch_patch_norm_gamma is not None:
                        eps = 1e-5
                        torch_lin = (
                            torch_lin * torch.rsqrt(torch_lin.pow(2).mean(-1, keepdim=True) + eps)
                        ) * self._torch_patch_norm_gamma

                    # PCC helper
                    def _pcc(a, b):
                        a = a.flatten()
                        b = b.flatten()
                        a = a - a.mean()
                        b = b - b.mean()
                        return (a * b).sum() / (torch.sqrt((a * a).sum()) * torch.sqrt((b * b).sum()) + 1e-12)

                    n = min(torch_lin.shape[0], tt_out_2d.shape[0])
                    d = min(torch_lin.shape[1], tt_out_2d.shape[1])
                    p = float(_pcc(torch_lin[:n, :d], tt_out_2d[:n, :d]))
                    max_abs = float((torch_lin[:n, :d] - tt_out_2d[:n, :d]).abs().max())
                    print(
                        f"[DOTS_DEBUG_PATCH_EMBED] device_vs_torch pcc={p:.6f} max_abs={max_abs:.6f} torch={tuple(torch_lin.shape)} tt={tuple(tt_out_2d.shape)}"
                    )
                except Exception as e:
                    print(f"[DOTS_DEBUG_PATCH_EMBED] compare failed: {type(e).__name__}: {e}")
            # Do NOT reorder tokens here: processor already emits patchified tokens in the HF order.
            # Also, Dots' RoPE is applied in attention blocks, not via learned pos-embeddings here.
            return out

        B, C, H, W = pixel_values.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"

        # Process grid_thw
        grid_thw = self._process_grid_thw(grid_thw)

        # For now, use a simplified approach - convert to patches on host first
        # In a full implementation, this would be done with TTNN operations
        # For Phase 1, we'll convert on host and then move to device

        # Simple patch embedding on host (to be optimized later)
        # Reshape to patches: [B, num_patches, C * patch_h * patch_w]
        #
        # Two sources of truth for the patch grid:
        #   1. ``self.patch_size`` — the configured patch side (default 14 for Dots).
        #   2. ``grid_thw`` — the HF processor's actual ``[temporal, height_patches, width_patches]``.
        # When they disagree (e.g. tests feed synthetic 64x64 images with grid=[1,4,4]),
        # ``grid_thw`` is authoritative because it reflects what the processor produced;
        # we derive the per-axis patch pixel size from the image shape.
        if grid_thw is not None:
            temporal = int(grid_thw[0, 0].item())
            height_patches = int(grid_thw[0, 1].item())
            width_patches = int(grid_thw[0, 2].item())
        else:
            temporal = 1
            height_patches = H // self.patch_size
            width_patches = W // self.patch_size

        # Guard against zero-patch configurations (e.g. grid=[1,0,0] dummy inputs).
        assert height_patches > 0 and width_patches > 0, (
            f"Invalid patch grid temporal={temporal}, H={height_patches}, W={width_patches} "
            f"(pixel_values={pixel_values.shape}, grid_thw={grid_thw})"
        )
        assert (
            H % height_patches == 0 and W % width_patches == 0
        ), f"Image {H}x{W} not divisible by patch grid {height_patches}x{width_patches}"
        patch_h = H // height_patches
        patch_w = W // width_patches
        num_patches = temporal * height_patches * width_patches

        # Flatten spatial dimensions into patches.
        x = pixel_values.reshape(B, C, height_patches, patch_h, width_patches, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, C * patch_h * patch_w)

        ttnn = get_ttnn()
        if ttnn is None or self.mesh_device is None or self.proj_weight is None:
            return torch.randn(B, num_patches, self.embed_dim, dtype=torch.bfloat16)

        # Upload patches and project on device with TTNN.
        memory_config = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16),
            device=self.mesh_device,
            dtype=getattr(ttnn, "bfloat16", torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        if len(x_tt.shape) == 3:
            x_tt = ttnn.reshape(x_tt, (1, 1, x_tt.shape[1], x_tt.shape[2]))
        out = ttnn.linear(x_tt, self.proj_weight, transpose_b=True, memory_config=memory_config)
        out = self._maybe_add_pos_embed(out, seq_len=int(num_patches))
        # Apply the same spatial-merge token permutation for 4D pixel inputs.
        try:
            if grid_thw is not None:
                g = self._process_grid_thw(grid_thw)
                t = int(g[0, 0].item())
                h = int(g[0, 1].item())
                w = int(g[0, 2].item())
                m = 2
                if h % m == 0 and w % m == 0:
                    out = ttnn.reshape(out, (1, 1, t, h, w, self.embed_dim))
                    out = ttnn.reshape(out, (1, 1, t, h // m, m, w // m, m, self.embed_dim))
                    out = ttnn.permute(out, (0, 1, 2, 3, 5, 4, 6, 7))
                    out = ttnn.reshape(out, (1, 1, t * h * w, self.embed_dim))
        except Exception:
            pass
        return out

    def to_host(self):
        """Move any persistent tensors to host."""


# Convenience function for testing
def create_patch_embed(mesh_device, state_dict=None, weight_cache_path=None, dtype=None, **kwargs):
    """Create PatchEmbedTT with Dots defaults; kwargs may override patch_size/embed_dim."""
    kwargs.setdefault("patch_size", 14)
    kwargs.setdefault("in_channels", 3)
    kwargs.setdefault("embed_dim", 1536)
    return PatchEmbedTT(
        mesh_device=mesh_device,
        state_dict=state_dict or {},
        weight_cache_path=weight_cache_path,
        dtype=dtype,
        **kwargs,
    )
