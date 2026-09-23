# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Gemma-4 multimodal embedder (vision -> text projection), on device.

Mirrors HF ``Gemma4MultimodalEmbedder``: a scale-free RMSNorm followed by a
biasless linear projection from the vision hidden size to the text hidden size.
The pooled soft tokens from ``VisionTower`` (already on device) are projected
into the text embedding space so they can be scattered into the
``image_token_id`` slots of the token embedding stream.

Implemented entirely with ttnn ops (on-device RMSNorm + linear), matching the
rest of the Gemma-4 TT stack. ``forward`` consumes and returns ttnn tensors.
"""

from __future__ import annotations

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.utils.general_utils import cast_host_for_ttnn, get_cache_file_name

# HF state-dict key for the projection weight (bias-free Linear).
EMBED_VISION_PROJECTION_KEY = "model.embed_vision.embedding_projection.weight"


class Gemma4MultimodalEmbedder(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        vision_hidden_size: int,
        text_hidden_size: int,
        eps: float = 1e-6,
        dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.eps = eps
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size

        if EMBED_VISION_PROJECTION_KEY not in state_dict:
            raise KeyError(
                f"Vision projector weight '{EMBED_VISION_PROJECTION_KEY}' not found in checkpoint. "
                "The HF model must be a multimodal Gemma-4 checkpoint (with a vision tower)."
            )
        # Checkpoint stores Linear weight as [text_hidden, vision_hidden] (out_features, in_features).
        # ttnn.linear computes x @ weight with weight = [in_features, out_features], so transpose.
        proj = state_dict[EMBED_VISION_PROJECTION_KEY]  # [text_hidden, vision_hidden]
        if proj.shape[0] != text_hidden_size or proj.shape[1] != vision_hidden_size:
            raise ValueError(
                f"Projector weight {tuple(proj.shape)} incompatible with vision_hidden={vision_hidden_size}, "
                f"text_hidden={text_hidden_size}"
            )
        proj_tt = cast_host_for_ttnn(proj.transpose(0, 1).contiguous(), dtype)  # [vision_hidden, text_hidden]
        proj_tt = proj_tt.unsqueeze(0).unsqueeze(0)  # [1, 1, vision_hidden, text_hidden]

        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        cache_name = None
        if weight_cache_path is not None:
            cache_name = get_cache_file_name(weight_cache_path, "embed_vision.embedding_projection.weight")

        self.tt_weight = ttnn.as_tensor(
            proj_tt,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=cache_name,
        )

    @classmethod
    def from_state_dict(
        cls, mesh_device, state_dict, vision_args, text_hidden_size, dtype=ttnn.bfloat16, weight_cache_path=None
    ):
        vision_hidden_size = vision_args.hf_config.vision_config.hidden_size
        eps = getattr(vision_args.hf_config.vision_config, "rms_norm_eps", 1e-6)
        return cls(
            mesh_device=mesh_device,
            state_dict=state_dict,
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_hidden_size,
            eps=eps,
            dtype=dtype,
            weight_cache_path=weight_cache_path,
        )

    def forward(self, pooled_tt):
        """Project pooled vision soft tokens to text embedding space (on device).

        Args:
            pooled_tt: ttnn tensor ``[1, batch, output_length, vision_hidden]``
                (the pooled, standardized output from ``VisionTower``).

        Returns:
            ttnn tensor ``[1, batch, output_length, text_hidden]`` in the same
            scaled space as the text token embeddings (no extra scaling — HF
            scatters these directly into ``inputs_embeds``).
        """
        # Scale-free RMSNorm: x / sqrt(mean(x^2) + eps). ttnn.rms_norm supports
        # the no-weight (with_scale=False) variant used by Gemma4RMSNorm.
        normed = ttnn.rms_norm(pooled_tt, epsilon=self.eps)
        return ttnn.linear(normed, self.tt_weight)  # [1, batch, output_length, text_hidden]
