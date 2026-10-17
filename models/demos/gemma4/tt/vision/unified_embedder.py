# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 "unified" (encoder-free) vision embedder, on device.

The 12B unified checkpoint has no SigLIP tower. Raw merged pixel patches are
projected straight into the language model's embedding space (HF
``Gemma4UnifiedVisionEmbedder`` + ``Gemma4UnifiedMultimodalEmbedder``):

    patches [P, 48*48*3] -> LayerNorm -> Linear(6912 -> mm_embed_dim) -> LayerNorm
        -> + factorized 2-D position embedding (table [mm_posemb_size, 2, mm_embed_dim])
        -> LayerNorm -> scale-free RMSNorm -> Linear(mm_embed_dim -> text_hidden, no bias)

Padding patches carry position (-1, -1) and are dropped before the soft tokens
are scattered into the ``image_token_id`` slots. ``forward`` returns the padded
projected tokens on device plus the host-side valid mask, the same contract the
SigLIP path (``VisionTower`` + ``Gemma4MultimodalEmbedder``) ends with, so the
generator's ``encode_vision`` serves both checkpoints.

The position-embedding lookup is a gather over a 1120 x 2 x 3840 table indexed
by per-patch (x, y); it is done on host in float32 (P x 3840 values, cheap)
and added on device.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.utils.general_utils import cast_host_for_ttnn, get_cache_file_name

# Two key layouts reach us for the same weights:
#   * the safetensors checkpoint:  model.vision_embedder.<p>  +  model.embed_vision.embedding_projection.weight
#   * an HF module's state_dict (the loader's AutoModel fallback, e.g. when given a
#     hub id): model.embed_vision.<p>  +  model.embed_vision.multimodal_embedder.embedding_projection.weight
_LAYOUTS = (
    ("model.vision_embedder.", "model.embed_vision.embedding_projection.weight"),
    ("model.embed_vision.", "model.embed_vision.multimodal_embedder.embedding_projection.weight"),
)
PREFIX, PROJECTION_KEY = _LAYOUTS[0]
PATCH_DENSE_KEY = PREFIX + "patch_dense.weight"
TILE = 32


def _layout(state_dict):
    for prefix, proj in _LAYOUTS:
        if prefix + "patch_dense.weight" in state_dict and proj in state_dict:
            return prefix, proj
    return None


def has_unified_vision_weights(state_dict) -> bool:
    """True when the checkpoint ships the encoder-free embedder (12B unified)."""
    return _layout(state_dict) is not None


class Gemma4UnifiedVisionEmbedder(LightweightModule):
    # Output is already in text-embedding space: the generator skips the projector.
    projects_to_text = True

    def __init__(
        self,
        mesh_device,
        state_dict,
        text_hidden_size: int,
        ln_eps: float = 1e-5,
        rms_eps: float = 1e-6,
        dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.ln_eps = ln_eps
        self.rms_eps = rms_eps
        self.text_hidden_size = text_hidden_size
        layout = _layout(state_dict)
        if layout is None:
            raise KeyError(
                f"'{PATCH_DENSE_KEY}' / '{PROJECTION_KEY}' not in checkpoint: not a unified Gemma-4 checkpoint"
            )
        PREFIX, PROJECTION_KEY = layout  # noqa: N806 - shadow the module defaults with this checkpoint's layout

        dense_w = state_dict[PREFIX + "patch_dense.weight"]  # [mm_embed, patch_dim]
        self.mm_embed_dim, self.patch_dim = int(dense_w.shape[0]), int(dense_w.shape[1])
        # pos table [mm_posemb_size, 2, mm_embed] stays on host (float32) for the gather.
        self.pos_table = state_dict[PREFIX + "pos_embedding"].float()
        proj_w = state_dict[PROJECTION_KEY]  # [text_hidden, mm_embed]
        if tuple(proj_w.shape) != (text_hidden_size, self.mm_embed_dim):
            raise ValueError(
                f"projection {tuple(proj_w.shape)} vs text_hidden={text_hidden_size}, mm_embed={self.mm_embed_dim}"
            )

        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None

        def dev(name, t, layout=ttnn.TILE_LAYOUT):
            cache_name = get_cache_file_name(weight_cache_path, name) if weight_cache_path is not None else None
            return ttnn.as_tensor(
                cast_host_for_ttnn(t.contiguous(), dtype),
                device=mesh_device,
                dtype=dtype,
                layout=layout,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
                cache_file_name=cache_name,
            )

        def vec(name, t):  # [1, 1, 1, N] rows for layer_norm weight / bias and the linear bias
            return dev(name, t.reshape(1, 1, 1, -1))

        self.ln1_w = vec("vision_embedder.patch_ln1.weight", state_dict[PREFIX + "patch_ln1.weight"])
        self.ln1_b = vec("vision_embedder.patch_ln1.bias", state_dict[PREFIX + "patch_ln1.bias"])
        self.dense_w = dev("vision_embedder.patch_dense.weight", dense_w.transpose(0, 1).unsqueeze(0).unsqueeze(0))
        self.dense_b = vec("vision_embedder.patch_dense.bias", state_dict[PREFIX + "patch_dense.bias"])
        self.ln2_w = vec("vision_embedder.patch_ln2.weight", state_dict[PREFIX + "patch_ln2.weight"])
        self.ln2_b = vec("vision_embedder.patch_ln2.bias", state_dict[PREFIX + "patch_ln2.bias"])
        self.ln3_w = vec("vision_embedder.pos_norm.weight", state_dict[PREFIX + "pos_norm.weight"])
        self.ln3_b = vec("vision_embedder.pos_norm.bias", state_dict[PREFIX + "pos_norm.bias"])
        self.proj_w = dev("embed_vision.embedding_projection.weight", proj_w.transpose(0, 1).unsqueeze(0).unsqueeze(0))

    def _replicate(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(self.mesh_device, "shape") else None

    def position_embeddings(self, image_position_ids: torch.Tensor) -> torch.Tensor:
        """Host gather: ``[P, 2]`` (x, y) -> ``[P, mm_embed]`` float32; padding rows are zero."""
        pos = image_position_ids.reshape(-1, 2).long()
        valid = (pos != -1).float().unsqueeze(-1)  # [P, 2, 1]
        clamped = pos.clamp(min=0)
        axes = torch.arange(2)
        return (self.pos_table[clamped, axes] * valid).sum(dim=-2)  # [P, mm_embed]

    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor):
        """Embed one image's patches.

        Args:
            pixel_values: ``[1, P, patch_dim]`` torch (HF ``Gemma4UnifiedProcessor``).
            image_position_ids: ``[1, P, 2]`` torch; (-1, -1) marks padding.

        Returns:
            (``ttnn [1, 1, P_padded, text_hidden]`` bfloat16, ``torch.bool [1, P_padded]`` valid mask)
        """
        p = int(pixel_values.shape[1])
        p_pad = ((p + TILE - 1) // TILE) * TILE
        x = torch.zeros(1, 1, p_pad, self.patch_dim, dtype=torch.float32)
        x[0, 0, :p] = pixel_values.reshape(p, self.patch_dim).float()
        pos = torch.zeros(1, 1, p_pad, self.mm_embed_dim, dtype=torch.float32)
        pos[0, 0, :p] = self.position_embeddings(image_position_ids)
        mask = torch.zeros(1, p_pad, dtype=torch.bool)
        mask[0, :p] = ~(image_position_ids.reshape(p, 2) == -1).all(dim=-1)

        common = dict(
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate(),
        )
        x_tt = ttnn.from_torch(x, **common)
        pos_tt = ttnn.from_torch(pos, **common)

        h = ttnn.layer_norm(x_tt, weight=self.ln1_w, bias=self.ln1_b, epsilon=self.ln_eps)
        ttnn.deallocate(x_tt)
        h = ttnn.linear(h, self.dense_w, bias=self.dense_b)
        h = ttnn.layer_norm(h, weight=self.ln2_w, bias=self.ln2_b, epsilon=self.ln_eps)
        h = ttnn.add(h, pos_tt)
        ttnn.deallocate(pos_tt)
        h = ttnn.layer_norm(h, weight=self.ln3_w, bias=self.ln3_b, epsilon=self.ln_eps)
        h = ttnn.rms_norm(h, epsilon=self.rms_eps)  # scale-free, as Gemma4UnifiedMultimodalEmbedder
        out = ttnn.linear(h, self.proj_w)  # [1, 1, P_padded, text_hidden]
        return out, mask
