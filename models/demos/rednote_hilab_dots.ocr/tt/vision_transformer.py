# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN full vision tower for dots.ocr.

DotsVisionTransformer (modeling_dots_vision): ``patch_embed -> 42 x
DotsVisionBlock -> post_trunk_norm (RMSNorm eps=1e-5) -> DotsPatchMerger``.
Pure composition of the already brought-up sub-blocks TtVisionPatchEmbed,
TtVisionBlock, TtVisionRMSNorm and TtPatchMerger, mirroring reference_impl
models/demos/qwen25_vl/tt/model.py (DropInVisionTransformer: patch_embed ->
block loop -> merger over a single padded device tensor).

Input convention (same as the sub-blocks): the HF preprocessor's
PRE-FLATTENED patches ``[num_patches, C*P*P]`` are padded on host to a
multiple of 128 rows and shipped as ``[1, 1, padded_seq, C*P*P]``;
``cu_seqlens`` keeps the UNPADDED window boundaries so pad rows are never
attended to, and the caller slices the merger output back to
``seq // m^2`` rows (pad rows merge into trailing garbage rows only).
Rope tables and cu_seqlens are computed on host (ARCHITECTURE.md
hybrid_notes) and staged via prepare_rope / prepare_cu_seqlens.

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — all
weights ``ReplicateTensorToMesh`` on the 1x4 mesh, activations stay
replicated, no CCL; the handoff into the column-parallel decoder needs no
collective. On a single device the mesh_mapper degenerates gracefully.
"""

import importlib.util
import sys
from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule

# Dir name contains a dot -> not importable as a package; load siblings by path.
_TT_DIR = Path(__file__).resolve().parent


def _load_sibling(stem):
    name = f"dots_ocr_tt_{stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _TT_DIR / f"{stem}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


TtVisionPatchEmbed = _load_sibling("vision_patch_embed").TtVisionPatchEmbed
TtVisionBlock = _load_sibling("vision_block").TtVisionBlock
TtVisionRMSNorm = _load_sibling("vision_rmsnorm").TtVisionRMSNorm
TtPatchMerger = _load_sibling("patch_merger").TtPatchMerger


class TtVisionTransformer(LightweightModule):
    """dots.ocr vision tower: patch_embed -> 42x block -> post_trunk_norm -> merger.

    Args:
        mesh_device: ttnn mesh device handle (all weights replicated).
        state_dict: flat torch tensors with the HF ``vision_tower.`` prefix
            stripped, e.g. patch_embed.patchifier.proj.weight,
            blocks.0.norm1.weight, ..., post_trunk_norm.weight,
            merger.ln_q.weight.
        num_layers: vision blocks (dots.ocr: 42).
        num_heads: attention heads per block (dots.ocr: 12, head_dim 128).
        dtype: on-device weight/activation dtype.
        eps: RMSNorm epsilon (dots.ocr vision uses 1e-5; merger LayerNorm
            hard-codes its own 1e-6).
        spatial_merge_size: patch-merger spatial merge factor m (default 2).
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        num_layers=42,
        num_heads=12,
        dtype=ttnn.bfloat16,
        eps=1e-5,
        spatial_merge_size=2,
        tp_degree=1,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_layers = num_layers
        self.dtype = dtype
        self.tp_degree = tp_degree

        self.patch_embed = TtVisionPatchEmbed(
            mesh_device,
            {
                "proj.weight": state_dict["patch_embed.patchifier.proj.weight"],
                "proj.bias": state_dict["patch_embed.patchifier.proj.bias"],
                "norm.weight": state_dict["patch_embed.patchifier.norm.weight"],
            },
            dtype=dtype,
            eps=eps,
        )
        self.blocks = []
        for i in range(num_layers):
            prefix = f"blocks.{i}."
            block_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
            self.blocks.append(
                TtVisionBlock(mesh_device, block_sd, num_heads=num_heads, dtype=dtype, eps=eps, tp_degree=tp_degree)
            )
        self.post_trunk_norm = TtVisionRMSNorm(
            mesh_device, {"weight": state_dict["post_trunk_norm.weight"]}, dtype=dtype, eps=eps
        )
        merger_sd = {k[len("merger.") :]: v for k, v in state_dict.items() if k.startswith("merger.")}
        self.merger = TtPatchMerger(mesh_device, merger_sd, spatial_merge_size=spatial_merge_size, dtype=dtype)

    # Host-side input prep delegates to the first block's attention (rope
    # tables and cu_seqlens stay on host per ARCHITECTURE.md hybrid_notes).
    def prepare_rope(self, rotary_pos_emb, padded_seq):
        return self.blocks[0].prepare_rope(rotary_pos_emb, padded_seq)

    def prepare_cu_seqlens(self, cu_seqlens):
        return self.blocks[0].prepare_cu_seqlens(cu_seqlens)

    def forward(self, x_11SP: ttnn.Tensor, rot_mats, cu_seqlens: ttnn.Tensor) -> ttnn.Tensor:
        """x_11SP: [1, 1, padded_seq, C*P*P] TILE_LAYOUT, replicated.

        rot_mats: (cos, sin) each [1, 1, padded_seq, head_dim] from prepare_rope.
        cu_seqlens: uint32 device tensor of UNPADDED window boundaries.
        Returns: [padded_seq / m^2, hidden], replicated; rows past
        ``seq / m^2`` merge only padding and are sliced off by the caller.
        """
        embedded = self.patch_embed(x_11SP)
        # Residual-stream dtype follows the configured tower dtype. The
        # fp32 config carries the stream in fp32 (per-block sub-ops already
        # accumulate in fp32 via HiFi4 + fp32 dest acc; 42 bf16 residual
        # adds compound rounding below the 0.99 block-PCC bar). The bf16
        # production config (optimization REDO) keeps the stream bf16
        # end-to-end: tower PCC ~0.977 is accepted because the parity gate
        # is the E2E 5-sample WER check (TTNN==HF), which bf16 passes.
        if self.dtype == ttnn.float32 and embedded.dtype != ttnn.float32:
            h = ttnn.typecast(embedded, ttnn.float32)
            ttnn.deallocate(embedded)
        else:
            h = embedded
        for block in self.blocks:
            h_next = block.forward(h, rot_mats, cu_seqlens)
            ttnn.deallocate(h)
            h = h_next
        normed = self.post_trunk_norm(h)
        ttnn.deallocate(h)
        out = self.merger(normed)
        ttnn.deallocate(normed)
        return out
