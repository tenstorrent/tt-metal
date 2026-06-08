# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Standalone driver for MoonViT-augmented prefill.

This module orchestrates the full multimodal prefill pipeline without
depending on DeepSeek-V3's `Generator` class. The text embedding and
post-embedding-LLM steps are passed in as callables so the driver is
testable in isolation (with stubs) and the same logic can be wired into
`Generator._prefill` later.

Pipeline:
    pixel_patches, grid_hws ─► MoonViT (device) ─► vision_tokens (host)
                                                          │
    tokens (host) ─► embed_text_fn (device) ─► text_embedded (device, row-sharded)
                                                          │
                                                          ▼
                                            splice_vision_via_host
                                                          │
                                                          ▼
                                       forward_from_embeddings_fn (device) ─► logits

The text-embedding callable should be e.g.::

    lambda tt_tokens: Embedding2D.forward_prefill(tt_tokens, cfg["embedding"])

The post-embedding LLM callable should be e.g.::

    lambda x_embedded: RowBatchedModel.forward_prefill_from_embeddings(
        x_embedded, user_id=..., cfg=..., rope_tensors=..., page_tables=...,
    )

Wrap whichever set of kwargs makes sense at the call site.

Per the plan (Phase 2), the splice runs through a host roundtrip
(Gemma3 pattern). Device-side splice is a deferred optimization.

Production wiring sketch (for Generator._prefill multimodal branch)::

    # In Generator.__init__ (or lazy on first multimodal call):
    self._mm_driver = build_driver(
        model_args=moonvit_model_args,  # MoonViTModelArgs instance
        mesh_device=self.mesh_device,
        dtype=ttnn.bfloat16,
    )

    # In Generator._prefill, when pixel_patches is provided:
    def embed_text(tokens_tt: ttnn.Tensor) -> ttnn.Tensor:
        return Embedding2D.forward_prefill(tokens_tt, self.model_run_config_prefill["embedding"])

    def llm_from_embeddings(x_embedded: ttnn.Tensor):
        return RowBatchedModel.forward_prefill_from_embeddings(
            x_embedded,
            user_id=user_id,
            cfg=self.model_run_config_prefill,
            rope_tensors=rope_tensors,
            page_tables=page_tables_to_use,
            prompt_len=prompt_len,
        )

    logits_tt = self._mm_driver.run(
        tokens=tokens,
        pixel_patches=pixel_patches,
        grid_hws=grid_hws,
        embed_text_fn=embed_text,
        forward_from_embeddings_fn=llm_from_embeddings,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

import ttnn
from models.demos.deepseek_v3.tt.moonvit.model import MoonViT
from models.demos.deepseek_v3.tt.moonvit.prefill_splice import splice_vision_via_host

# Type aliases for clarity at call sites.
EmbedTextFn = Callable[[ttnn.Tensor], ttnn.Tensor]
ForwardFromEmbeddingsFn = Callable[[ttnn.Tensor], object]


def _is_mesh_device(device) -> bool:
    return type(device).__name__ == "MeshDevice"


@dataclass
class MultimodalPrefillDriver:
    """End-to-end multimodal prefill orchestrator.

    Holds a single MoonViT instance plus the configuration the splice
    needs (mesh device, image token id, dtype). Construct once at
    Generator init time, then call `.run(...)` per prefill request.
    """

    moonvit: MoonViT
    mesh_device: object
    image_token_id: int
    dtype: object = ttnn.bfloat16

    def encode_image(
        self,
        pixel_patches: torch.Tensor,
        grid_hws: torch.Tensor,
    ) -> torch.Tensor:
        """Run MoonViT on host-side pixel patches; return vision tokens on host.

        Args:
            pixel_patches: (L_patches, 3, 14, 14) — output of HF
                `KimiVLImageProcessor.patchify` (or equivalent).
            grid_hws: (num_images, 2) — per-image (H, W) patch grid.

        Returns:
            host torch tensor of shape (L_new, text_hidden) where
            L_new = sum((H_i // kh) * (W_i // kw)).
        """
        vision_tt = self.moonvit(pixel_patches, grid_hws)
        is_mesh = _is_mesh_device(self.mesh_device)
        vision_pt = ttnn.to_torch(
            vision_tt,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0) if is_mesh else None,
        )
        if is_mesh and vision_pt.shape[0] != 1:
            vision_pt = vision_pt[:1]
        # Squeeze leading 1, 1 dims; collapse to (L_new, text_hidden).
        return vision_pt.view(-1, vision_pt.shape[-1])

    def stage_tokens(self, tokens: torch.Tensor) -> ttnn.Tensor:
        """Push token ids to device with the layout `embed_text_fn` expects.

        Mirrors what `Generator._prefill` does for text-only prefill —
        replicated uint32 tensor in row-major layout.
        """
        if tokens.ndim == 1:
            tokens = tokens.view(1, 1, -1)
        elif tokens.ndim == 2:
            tokens = tokens.view(1, *tokens.shape)
        if tokens.ndim != 3:
            raise ValueError(f"tokens must be 1D, 2D, or 3D; got shape {tuple(tokens.shape)}")
        is_mesh = _is_mesh_device(self.mesh_device)
        return ttnn.from_torch(
            tokens.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
        )

    def run(
        self,
        *,
        tokens: torch.Tensor,
        pixel_patches: torch.Tensor,
        grid_hws: torch.Tensor,
        embed_text_fn: EmbedTextFn,
        forward_from_embeddings_fn: ForwardFromEmbeddingsFn,
        precomputed_vision_tokens: Optional[torch.Tensor] = None,
    ) -> object:
        """Run the full multimodal prefill pipeline.

        Args:
            tokens: host tensor of shape (seq_len,), (1, seq_len), or
                (1, 1, seq_len). Positions where ``tokens == self.image_token_id``
                are the splice targets. The total count of such positions
                MUST equal the vision-token count produced by MoonViT for
                the given ``grid_hws``.
            pixel_patches: (L_patches, 3, 14, 14) per-patch pixel data.
                Ignored if ``precomputed_vision_tokens`` is provided.
            grid_hws: (num_images, 2) per-image patch grid shape.
            embed_text_fn: callable mapping a device uint32 token tensor
                to a device-embedded tensor in the layout
                ``RowBatchedModel.forward_prefill_from_embeddings`` expects.
                In production this is
                ``lambda t: Embedding2D.forward_prefill(t, cfg["embedding"])``.
            forward_from_embeddings_fn: callable mapping the spliced device
                tensor to the LLM's prefill output (logits, optionally with
                hidden states). In production this is
                ``lambda x: RowBatchedModel.forward_prefill_from_embeddings(x, ...)``.
            precomputed_vision_tokens: optional override for the MoonViT
                step. Useful when the same image is reused across multiple
                prefills (avoid re-encoding) or for tests with synthetic
                vision tokens.

        Returns:
            whatever ``forward_from_embeddings_fn`` returns (typically a
            logits ttnn.Tensor or a (logits, hidden) tuple).
        """
        # 1. Vision encode.
        if precomputed_vision_tokens is not None:
            vision_tokens = precomputed_vision_tokens
        else:
            vision_tokens = self.encode_image(pixel_patches, grid_hws)

        # 2. Text token → device → embed.
        tokens_tt = self.stage_tokens(tokens)
        text_embedded = embed_text_fn(tokens_tt)
        ttnn.deallocate(tokens_tt)

        # 3. Host-roundtrip splice.
        # The splice helper needs the original tokens in shape (1, 1, seq_len)
        # to locate image positions.
        if tokens.ndim == 1:
            tokens_for_splice = tokens.view(1, 1, -1)
        elif tokens.ndim == 2:
            tokens_for_splice = tokens.view(1, *tokens.shape)
        else:
            tokens_for_splice = tokens

        fused = splice_vision_via_host(
            mesh_device=self.mesh_device,
            text_embedded_tt=text_embedded,
            tokens=tokens_for_splice,
            vision_tokens=vision_tokens,
            image_token_id=self.image_token_id,
            dtype=self.dtype,
        )
        ttnn.deallocate(text_embedded)

        # 4. LLM forward from spliced embeddings.
        return forward_from_embeddings_fn(fused)


def build_driver(
    model_args,
    mesh_device,
    dtype=ttnn.bfloat16,
    *,
    with_projector: bool = True,
) -> MultimodalPrefillDriver:
    """Construct a MultimodalPrefillDriver from MoonViTModelArgs.

    Loads MoonViT lazily via the model_args reference factories (touches
    the HF cache). The driver caches the MoonViT instance for reuse.

    For testing without HF weights, instantiate `MultimodalPrefillDriver`
    directly with a pre-built or mocked MoonViT.
    """
    moonvit = MoonViT.from_torch(
        model_args=model_args,
        mesh_device=mesh_device,
        with_projector=with_projector,
        dtype=dtype,
    )
    return MultimodalPrefillDriver(
        moonvit=moonvit,
        mesh_device=mesh_device,
        image_token_id=model_args.media_placeholder_token_id,
        dtype=dtype,
    )
