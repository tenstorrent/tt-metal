# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 "unified" (encoder-free) audio embedder, on device.

The 12B unified checkpoint has no audio tower. ``Gemma4UnifiedAudioFeatureExtractor``
chunks raw 16 kHz audio into frames of ``audio_samples_per_token`` (640) samples, one
frame per soft token (40 ms), and the model projects each frame straight into the
language model's embedding space (HF ``Gemma4UnifiedMultimodalEmbedder`` over the
audio config):

    frames [T, 640] -> scale-free RMSNorm -> Linear(640 -> text_hidden, no bias)

Padding frames (``input_features_mask`` False) are dropped before the soft tokens are
scattered into the ``audio_token_id`` slots. ``forward`` returns the padded projected
tokens on device plus the host-side valid mask: the same contract as the image path
(``Gemma4UnifiedVisionEmbedder`` / ``VisionTower``), so the generator's scatter merge
serves every modality.
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.utils.general_utils import cast_host_for_ttnn, get_cache_file_name

# safetensors checkpoint layout and the HF module's state_dict layout (Gemma4UnifiedModel).
PROJECTION_KEYS = (
    "model.embed_audio.embedding_projection.weight",
    "embed_audio.embedding_projection.weight",
)
TILE = 32


def _projection_key(state_dict):
    for key in PROJECTION_KEYS:
        if key in state_dict:
            return key
    return None


def has_unified_audio_weights(state_dict) -> bool:
    """True when the checkpoint carries the encoder-free audio projection (12B unified)."""
    return _projection_key(state_dict) is not None


class Gemma4UnifiedAudioEmbedder(LightweightModule):
    # Output is already in text-embedding space: the generator skips any projector.
    projects_to_text = True

    def __init__(
        self,
        mesh_device,
        state_dict,
        text_hidden_size: int,
        rms_eps: float = 1e-6,
        dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.rms_eps = rms_eps
        self.text_hidden_size = text_hidden_size
        key = _projection_key(state_dict)
        if key is None:
            raise KeyError(f"none of {PROJECTION_KEYS} in checkpoint: not a unified Gemma-4 audio checkpoint")
        proj_w = state_dict[key]  # [text_hidden, audio_embed_dim]
        if int(proj_w.shape[0]) != text_hidden_size:
            raise ValueError(f"projection {tuple(proj_w.shape)} vs text_hidden={text_hidden_size}")
        self.audio_embed_dim = int(proj_w.shape[1])
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None
        cache_name = (
            get_cache_file_name(weight_cache_path, "embed_audio.embedding_projection.weight")
            if weight_cache_path is not None
            else None
        )
        self.proj_w = ttnn.as_tensor(
            cast_host_for_ttnn(proj_w.transpose(0, 1).unsqueeze(0).unsqueeze(0).contiguous(), dtype),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=cache_name,
        )

    def _replicate(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device) if hasattr(self.mesh_device, "shape") else None

    def forward(self, input_features: torch.Tensor, input_features_mask: torch.Tensor | None = None):
        """Embed one audio clip's frames.

        Args:
            input_features: ``[1, T, audio_embed_dim]`` torch raw-waveform frames
                (HF ``Gemma4UnifiedAudioFeatureExtractor``).
            input_features_mask: ``[1, T]`` torch bool, True for real frames; None = all real.

        Returns:
            (``ttnn [1, 1, T_padded, text_hidden]`` bfloat16, ``torch.bool [1, T_padded]`` valid mask)
        """
        t = int(input_features.shape[1])
        t_pad = ((t + TILE - 1) // TILE) * TILE
        x = torch.zeros(1, 1, t_pad, self.audio_embed_dim, dtype=torch.float32)
        x[0, 0, :t] = input_features.reshape(t, self.audio_embed_dim).float()
        mask = torch.zeros(1, t_pad, dtype=torch.bool)
        mask[0, :t] = True if input_features_mask is None else input_features_mask.reshape(t).bool()
        x_tt = ttnn.from_torch(
            x,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate(),
        )
        h = ttnn.rms_norm(x_tt, epsilon=self.rms_eps)  # scale-free, as Gemma4UnifiedMultimodalEmbedder
        ttnn.deallocate(x_tt)
        out = ttnn.linear(h, self.proj_w)  # [1, 1, T_padded, text_hidden]
        return out, mask
