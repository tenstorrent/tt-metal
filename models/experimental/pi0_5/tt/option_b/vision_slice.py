# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 vision + embedding slice for Option B.

The first cut runs SigLIP on HOST (torch) — matches the upstream-openpi
default (PI0_SIGLIP_HF=1) used by Pi0_5PaliGemmaBackboneTTNN — and uploads
just the projected vision features + language token embeddings onto the
4x2 submesh. The on-device SigLIP path (Pi0_5SubmeshSigLIPSlice) will be
added once the per-stage TP=8 weight sharding lands; for now this gives a
functionally correct stage-0 forward that the pipeline orchestrator can
drive end-to-end.

Memory budget on stage 0's submesh (replicated):
  - VLM embed_tokens table: 527 MB at bf16 (rank 0 only; need sharding by
    vocab eventually). For the bring-up dry run we either upload it once
    replicated (fits the SRAM budget once we shard) or do host-side
    embedding lookup and upload the resulting [S, W] hidden. We default to
    HOST embed lookup so this slice stays under the L1 cap for testing.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.reference.torch_siglip_hf import HFSigLIPVisionTower
from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector as _TorchMMProjector

from .vlm_slice import _upload_replicated


class Pi0_5SubmeshVisionSlice:
    """Stage 0 — produces prefix hidden_states for stage 1 from images +
    language tokens.

    Args:
        config:   full PaliGemma config (we use siglip_config + vlm_config).
        weights:  full weights dict — vlm_vision, vlm_projector, vlm_language
                  required.
        submesh:  the stage-0 4x2 MeshDevice.
        embed_on_host: if True (default for bring-up), tokenized language
                  embeddings are looked up on the HOST via
                  weights['vlm_language']['model.embed_tokens.weight']. This
                  keeps the 527 MB embedding table off the submesh until we
                  shard it. Set False once vocab-sharding is in.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        submesh,
        embed_on_host: bool = True,
    ) -> None:
        self.config = config
        self.submesh = submesh
        self.embed_on_host = embed_on_host

        # SigLIP + projector run on host (torch).
        self._host_vision_tower = HFSigLIPVisionTower(config.siglip_config, weights["vlm_vision"])
        self._host_mm_projector = _TorchMMProjector(weights["vlm_projector"])

        # Language embedding: keep host copy; lookup is cheap (1 mm of size
        # [S, vocab] @ [vocab, W] never materialized — we index the table).
        lang = weights["vlm_language"]
        embed_torch = lang.get("model.embed_tokens.weight") or lang.get("lm_head.weight")
        if embed_torch is None:
            raise KeyError("vlm_language must contain 'model.embed_tokens.weight' or 'lm_head.weight'")
        if embed_on_host:
            self._host_embed_table = embed_torch
            self._device_embed_table: Optional["ttnn.Tensor"] = None
        else:
            # 527 MB table replicated across the submesh — won't fit per chip
            # at bf16 (180 MB cap). Will need vocab-sharding before we flip
            # this flag for real.
            self._host_embed_table = None
            # 527 MB table — way over the 180 MB L1 budget. Forced DRAM.
            self._device_embed_table = _upload_replicated(
                embed_torch.contiguous(),
                submesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    # ------------------------------------------------------------------ #
    # Image path                                                          #
    # ------------------------------------------------------------------ #

    def embed_images(self, pixel_values: torch.Tensor) -> "ttnn.Tensor":
        """pixel_values: torch [B, 3, H, W]. Returns [B, num_patches, vlm_W]
        on the submesh (replicated).
        """
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)
        with torch.no_grad():
            features = self._host_vision_tower.forward(pixel_values)  # [B, 256, 1152]
            projected = self._host_mm_projector.forward(features)  # [B, 256, vlm_W]
        return _upload_replicated(
            projected.to(torch.float32).contiguous(),
            self.submesh,
            dtype=ttnn.bfloat16,
        )

    # ------------------------------------------------------------------ #
    # Language path                                                       #
    # ------------------------------------------------------------------ #

    def embed_language_tokens(self, token_ids: torch.Tensor) -> "ttnn.Tensor":
        """token_ids: torch [B, S] int. Returns [B, S, vlm_W] on the submesh.

        Host lookup by default (avoids the 527 MB embedding table on chip).
        """
        if self.embed_on_host:
            embedded = torch.nn.functional.embedding(token_ids.to(torch.long), self._host_embed_table)  # [B, S, W]
            return _upload_replicated(
                embedded.to(torch.float32).contiguous(),
                self.submesh,
                dtype=ttnn.bfloat16,
            )
        # Device path: ttnn.embedding takes [B, S] token IDs and [vocab, W] table.
        if not isinstance(token_ids, ttnn.Tensor):
            token_ids = ttnn.from_torch(
                token_ids.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.submesh,
                mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.submesh),
            )
        return ttnn.embedding(token_ids, self._device_embed_table)

    # ------------------------------------------------------------------ #
    # Combined prefix builder                                             #
    # ------------------------------------------------------------------ #

    def build_prefix_hidden(
        self,
        pixel_values: torch.Tensor,
        language_token_ids: torch.Tensor,
    ) -> "ttnn.Tensor":
        """Convenience: image_embed || language_embed concatenated along seq
        dim. Returns [B, num_patches + S_lang, vlm_W] on the submesh.

        Note: in the real openpi prefix path there's also a robot-state slot
        appended after the language tokens (PrefixEmbedding handles that).
        We omit it here so the dry run can be exercised with image+text only;
        the orchestrator can concatenate state on the same submesh once we
        wire up the suffix module.
        """
        image_hidden = self.embed_images(pixel_values)
        lang_hidden = self.embed_language_tokens(language_token_ids)
        return ttnn.concat([image_hidden, lang_hidden], dim=1)
