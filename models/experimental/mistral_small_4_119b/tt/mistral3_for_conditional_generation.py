# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral-3 multi-modal generation orchestrator (phase-based).

Memory note
-----------
On T3K (8 chips × ~12 GB DRAM each) the full Mistral-Small-4-119B text stack
(36 decoder layers + KV caches + sharded LM head, with experts at bfloat4_b
and the shared MLP/gate at bfloat8_b) consumes ~99 % of every DRAM bank.
The Pixtral vision tower is replicated across all 8 chips and needs another
~64 MB/bank at bf16 — there is no room to keep both resident simultaneously.

The orchestrator therefore runs in three phases::

    Phase 1 (vision)        — build vision tower + projector, run forward,
                              pull the (small) image embeddings back to host,
                              free the vision/projector weights from device DRAM.
    Phase 2 (text load)     — build the text language model on device.
    Phase 3 (text inference) — upload image embeddings, splice them into the
                              text embedding sequence, prefill + decode.

Image embeddings are tiny (e.g. 25 tokens × 4096 × 2 B ≈ 200 KB), so caching
them on host between phases is cheap. Compute inside each phase is still
fully on device — the host crossings are weight management, not inference.

Multi-image
-----------
If you want to process several images in one session, call ``encode_image`` for
each *before* ``load_text``. Each call rebuilds the vision tower and frees it
again, then accumulates the host-side image embeddings in your own list.
"""

from __future__ import annotations

import gc
from typing import List

import torch

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_NUM_LAYERS,
    HIDDEN_SIZE,
    VISION_NUM_LAYERS,
)
from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation_helpers import (
    contiguous_runs,
    splice_embeddings,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_multimodal_projector import (
    TtMistral3MultiModalProjector,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.experimental.mistral_small_4_119b.tt.mistral4_vision_tower import TtPixtralVisionTower


class TtMistral3ForConditionalGeneration:
    """
    Mistral-Small-4-119B multi-modal generation orchestrator (phase-based).

    Construction is lightweight — only configuration is stored.
    Heavy device-side construction happens in ``encode_image`` (vision side)
    and ``load_text`` (language side), which are deliberately separate so that
    vision weights can be freed before the text stack is allocated.

    Args:
        mesh_device:        TTNN MeshDevice
        state_dict:         HF checkpoint filtered to:
                              vision_tower.*, multi_modal_projector.*,
                              language_model.model.*, language_model.lm_head.*
        text_config:        HF ``Mistral3Config.text_config``
        image_token_id:     token id marking image-embedding positions
                            (Mistral3 config: ``image_token_index = 10``)
        num_text_layers:    1..36, defaults to the full text stack
        num_vision_layers:  1..24, defaults to the full vision stack
        max_seq_len:        KV cache capacity (prefill + decode budget)
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        text_config,
        image_token_id: int,
        num_text_layers: int = EXPECTED_NUM_LAYERS,
        num_vision_layers: int = VISION_NUM_LAYERS,
        max_seq_len: int = 4096,
    ):
        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.text_config = text_config
        self.image_token_id = int(image_token_id)
        self.num_text_layers = num_text_layers
        self.num_vision_layers = num_vision_layers
        self.max_seq_len = max_seq_len

        self.text_model: TtMistral4TextModel | None = None

    # ── Phase 1: vision ──────────────────────────────────────────────────

    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Build the vision tower + projector, run image → embeddings, then free them.

        Args:
            pixel_values: torch ``[1, 3, H, W]`` bf16, H,W multiples of patch_size.
        Returns:
            torch ``[num_image_tokens, HIDDEN_SIZE]`` bf16 image embeddings on host.
            ``num_image_tokens = (H // patch_size // spatial_merge_size) ** 2``.

        Must be called before ``load_text`` — the vision weights and the full
        text stack don't fit in DRAM simultaneously.
        """
        if self.text_model is not None:
            raise RuntimeError(
                "encode_image() must be called BEFORE load_text(); there isn't "
                "enough DRAM to host both the text stack and the vision tower at once. "
                "Encode all images first, then call load_text()."
            )

        vision = TtPixtralVisionTower(
            mesh_device=self.mesh_device,
            state_dict=self.state_dict,
            num_layers=self.num_vision_layers,
        )
        projector = TtMistral3MultiModalProjector(
            mesh_device=self.mesh_device,
            state_dict=self.state_dict,
        )

        feats, h_p, w_p = vision.forward(pixel_values)
        img_embeds_tt = projector.forward(feats, h_p, w_p)
        ttnn.deallocate(feats)

        # Pull image embeddings to host. They're small (~tokens × 4096 × 2 B).
        img_embeds_host = ttnn.to_torch(
            img_embeds_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
        )[0, 0].to(
            torch.bfloat16
        )  # [num_image_tokens, HIDDEN_SIZE]
        ttnn.deallocate(img_embeds_tt)

        # Drop Python refs and force GC so the ttnn destructors run and free DRAM.
        del vision, projector
        gc.collect()

        return img_embeds_host

    # ── Phase 2: text load ───────────────────────────────────────────────

    def load_text(self) -> None:
        """Construct the text language model on device. Idempotent."""
        if self.text_model is not None:
            return
        self.text_model = TtMistral4TextModel(
            mesh_device=self.mesh_device,
            state_dict=self.state_dict,
            text_config=self.text_config,
            num_decoder_layers=self.num_text_layers,
            max_seq_len=self.max_seq_len,
        )

    def cache_rope_tables(self, cos_full: torch.Tensor, sin_full: torch.Tensor) -> None:
        """
        Upload precomputed RoPE ``(cos, sin)`` tables to device DRAM once.

        Must be called after ``load_text``; per-step decode then looks up cos/sin
        with an on-device ``ttnn.slice`` instead of a per-step PCIe upload.
        """
        assert self.text_model is not None, "load_text() must be called first"
        self.text_model.cache_rope_tables(cos_full, sin_full)

    # ── Phase 3: text inference ──────────────────────────────────────────

    def _upload_image_embeds(self, img_embeds_host: torch.Tensor) -> ttnn.Tensor:
        """Upload host-side image embeddings back to device as [1, 1, N, HIDDEN_SIZE]."""
        n = img_embeds_host.shape[0]
        return ttnn.as_tensor(
            img_embeds_host.reshape(1, 1, n, HIDDEN_SIZE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        img_embeds_host: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Build the full prefill embedding sequence on device.

        - Upload ``img_embeds_host`` back to device (small).
        - Embed ``input_ids`` via the language model's embedding table.
        - Splice the image embeddings in at every image_token_id position via
          slice + concat — fully on device.
        """
        assert self.text_model is not None, "load_text() must be called first"

        seq_len = input_ids.shape[-1]
        num_img = img_embeds_host.shape[0]

        mask = input_ids[0] == self.image_token_id
        img_positions: List[int] = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        assert len(img_positions) == num_img, (
            f"input_ids has {len(img_positions)} image_token_id={self.image_token_id} slots "
            f"but encode_image produced {num_img} image embeddings"
        )

        text_embeds = self.text_model.embed_tokens(input_ids)
        img_embeds = self._upload_image_embeds(img_embeds_host)

        runs = contiguous_runs(img_positions)
        final = splice_embeddings(text_embeds, img_embeds, runs, seq_len, HIDDEN_SIZE)
        ttnn.deallocate(text_embeds)
        ttnn.deallocate(img_embeds)
        return final

    def prefill_multimodal(
        self,
        img_embeds_host: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> int:
        """
        Run prefill for a multimodal prompt and return the greedy next-token id.

        Args:
            img_embeds_host: host tensor from ``encode_image``,
                             shape [num_image_tokens, HIDDEN_SIZE].
            input_ids:       torch [1, seq_len] long, with image-token slots.

        Requires ``cache_rope_tables`` to have been called.
        """
        assert self.text_model is not None, "load_text() must be called first"
        inputs_embeds = self._build_inputs_embeds(input_ids, img_embeds_host)
        return self.text_model.prefill_from_embeds_next_token(inputs_embeds)

    def prefill_multimodal_full_logits(
        self,
        img_embeds_host: torch.Tensor,
        input_ids: torch.Tensor,
        position_embeddings=None,
    ) -> torch.Tensor:
        """
        Same as ``prefill_multimodal`` but returns full ``[1, seq_len, vocab_size]``
        bf16 CPU logits — used by PCC tests to compare against an HF reference.

        If ``position_embeddings`` is provided as a ``(cos, sin)`` tuple, the RoPE
        tables are cached from it (equivalent to calling ``cache_rope_tables`` first).
        Otherwise ``cache_rope_tables`` must have been called already.
        """
        assert self.text_model is not None, "load_text() must be called first"
        if position_embeddings is not None:
            self.text_model.cache_rope_tables(*position_embeddings)
        inputs_embeds = self._build_inputs_embeds(input_ids, img_embeds_host)
        return self.text_model.prefill_from_embeds(inputs_embeds)

    def decode_next_token(self, input_id: torch.Tensor, current_pos: int) -> int:
        """Pass-through to the text model's decode (one token, greedy, on-device argmax)."""
        assert self.text_model is not None, "load_text() must be called first"
        return self.text_model.decode_next_token(input_id, current_pos)
