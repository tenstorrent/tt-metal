# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-PIPELINE: end-to-end vision-side pipeline for qwen3.6 multimodal prefill.

Composes:
  user prompt + PIL image(s)
    → Qwen36MMPreprocessor (HF Qwen3VLProcessor wrapper)
        → input_ids, pixel_values, image_grid_thw, position_ids_3d
    → Qwen36VisionEncoder.forward(pixel_values, image_grid_thw)
        → vision_features [N_image_tokens, 5120]
    → splice_vision_into_embeddings(text_embeddings, vision_features, input_ids)
        → fused_embeddings [B, S, 5120]   ready for text decoder

The text decoder is NOT part of this pipeline — it's invoked separately
because it requires loading the 64-layer decoder weights and constructing
the existing Qwen36Generator. The pipeline produces the decoder INPUTS
(fused embeddings + 3D position_ids).

Splice rule (matches HF qwen3_vl):
  For each batch row, find positions where input_ids == image_token_id.
  Replace text_embeddings at those positions with vision_features (in
  visit order). Other positions keep their text embedding.
"""

from __future__ import annotations

import torch
from PIL.Image import Image

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_preprocessor import Qwen36MMInputs, Qwen36MMPreprocessor
from models.demos.qwen3_6_galaxy_v2.tt.vision_encoder import Qwen36VisionEncoder
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.tt_dit.parallel.manager import CCLManager


def splice_vision_into_embeddings(
    text_embeddings: torch.Tensor,
    vision_features: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    image_token_id: int = 248056,
) -> torch.Tensor:
    """Replace text_embeddings at image_token_id positions with vision_features.

    Args:
        text_embeddings: `[B, S, H]` (typically H=5120 for qwen3.6 text decoder).
        vision_features: `[N_image_tokens_total, H]` flat across all images
            in the batch, in the same order they appear in input_ids.
        input_ids: `[B, S]`.

    Returns:
        `[B, S, H]` fused embeddings.
    """
    B, S, H = text_embeddings.shape
    assert input_ids.shape == (B, S), f"input_ids shape mismatch: {input_ids.shape} vs {(B, S)}"
    assert (
        vision_features.shape[1] == H
    ), f"vision_features last-dim {vision_features.shape[1]} != text_embeddings H {H}"

    fused = text_embeddings.clone()
    image_mask = input_ids == image_token_id  # [B, S]
    n_image_positions_total = int(image_mask.sum().item())
    assert n_image_positions_total == vision_features.shape[0], (
        f"#image_pad positions ({n_image_positions_total}) != #vision_features rows " f"({vision_features.shape[0]})"
    )
    # Place vision_features at the True positions in row-major order across the batch.
    fused[image_mask] = vision_features.to(dtype=fused.dtype)
    return fused


class Qwen36MMPipeline:
    """End-to-end vision pipeline producing decoder-ready inputs.

    Owns:
      - preprocessor (HF processor wrapper, CPU)
      - vision_encoder (composite TT vision encoder)
      - text_embed_lookup (HF nn.Embedding from the text decoder weights)

    The text decoder itself is NOT owned by this pipeline. The pipeline's
    `prepare_decoder_inputs()` method returns the spliced embeddings and 3D
    position_ids that the text decoder's prefill should consume.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        model_args: Qwen36VisionModelArgs,
        text_embed_weight: torch.Tensor | None = None,
        *,
        hf_model_path: str | None = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.model_args = model_args

        self.preprocessor = Qwen36MMPreprocessor(
            hf_model_path or model_args.CKPT_DIR,
            image_token_id=self.model_args.hf_config.image_token_id,
            spatial_merge_size=self.model_args.hf_config.vision_config.spatial_merge_size,
        )
        self.vision_encoder = Qwen36VisionEncoder(
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            model_args=model_args,
            dtype=dtype,
        )

        # Text-token embedding table (CPU). If not provided, the caller must
        # supply it via `prepare_decoder_inputs(text_embed_lookup=...)` at call time.
        # The qwen3.6 text decoder's embed_tokens.weight is shape
        # [vocab_size=248320, hidden_size=5120].
        self.text_embed_weight = text_embed_weight

    def prepare_decoder_inputs(
        self,
        prompt: str,
        images: list[Image] | None = None,
        *,
        text_embed_weight: torch.Tensor | None = None,
    ) -> tuple[Qwen36MMInputs, torch.Tensor]:
        """Run the full vision-side pipeline.

        Returns:
          (inputs, fused_embeddings) where
            - inputs: `Qwen36MMInputs` from preprocessor (input_ids, pixel_values,
              image_grid_thw, position_ids_3d, etc.) — all metadata needed
              alongside the embeddings.
            - fused_embeddings: `[B, S, hidden_size=5120]` torch tensor with
              vision_features spliced in at image_token_id positions.
        """
        # 1. Preprocess
        inputs = self.preprocessor(prompt, images=images)

        # 2. Run vision encoder (only if there are images)
        if inputs.pixel_values is not None and inputs.image_grid_thw is not None:
            vision_features = self.vision_encoder.forward(inputs.pixel_values, inputs.image_grid_thw)
        else:
            # Text-only: skip vision pipeline
            vision_features = None

        # 3. Build text embeddings (CPU lookup via the provided weight table)
        embed_weight = text_embed_weight if text_embed_weight is not None else self.text_embed_weight
        if embed_weight is None:
            raise ValueError(
                "text_embed_weight not provided. Pass it at construction time or via "
                "`prepare_decoder_inputs(..., text_embed_weight=...)`."
            )
        text_embeddings = torch.nn.functional.embedding(inputs.input_ids, embed_weight)
        # shape: [B, S, hidden_size]

        # 4. Splice vision features in at image_pad positions
        if vision_features is not None:
            fused_embeddings = splice_vision_into_embeddings(
                text_embeddings,
                vision_features,
                inputs.input_ids,
                image_token_id=self.preprocessor.image_token_id,
            )
        else:
            fused_embeddings = text_embeddings

        return inputs, fused_embeddings
