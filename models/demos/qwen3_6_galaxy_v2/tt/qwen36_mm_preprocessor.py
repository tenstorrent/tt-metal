# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-PREPROC: CPU helper wrapping HF `Qwen3VLProcessor` for multimodal input.

Takes a user prompt + optional PIL image(s) and produces everything the TT
multimodal forward needs:
  - input_ids:        [B, S]
  - pixel_values:     [N_total_patches, in_channels * temporal_patch_size * patch_size**2]
                       (flat across all images, with grid_thw delimiting them)
  - image_grid_thw:   [num_images, 3]
  - attention_mask:   [B, S]
  - position_ids_3d:  [3, B, S]  (already built via `get_rope_index`)

qwen3.6 uses `processor_class: Qwen3VLProcessor` per the preprocessor_config.
HF transformers 4.57.1 ships `Qwen3VLProcessor` even though the
`Qwen3_5ForConditionalGeneration` model class doesn't exist — the processor
is independent of the model class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL.Image import Image
from transformers import AutoProcessor

from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import get_rope_index


@dataclass
class Qwen36MMInputs:
    """Container for all multimodal-prefill inputs."""

    input_ids: torch.Tensor  # [B, S], dtype=torch.long
    attention_mask: torch.Tensor  # [B, S], dtype=torch.long
    pixel_values: torch.Tensor | None  # [N_patches, patch_feat_dim=1536] or None
    image_grid_thw: torch.Tensor | None  # [num_images, 3] or None
    position_ids_3d: torch.Tensor  # [3, B, S], dtype=torch.long
    mrope_deltas: torch.Tensor  # [B, 1], dtype=torch.long


class Qwen36MMPreprocessor:
    """CPU-side multimodal preprocessor for qwen3.6.

    Loads the HF Qwen3VLProcessor and wraps it into a simple call that
    produces TT-ready torch tensors.
    """

    def __init__(
        self,
        hf_model_path: str,
        *,
        image_token_id: int = 248056,
        spatial_merge_size: int = 2,
    ) -> None:
        self.image_token_id = image_token_id
        self.spatial_merge_size = spatial_merge_size
        # AutoProcessor uses the processor_class declared in preprocessor_config.json
        # (Qwen3VLProcessor for qwen3.6). Doesn't depend on the model class.
        self.processor = AutoProcessor.from_pretrained(hf_model_path, trust_remote_code=False)

    def __call__(
        self,
        prompt: str,
        images: list[Image] | None = None,
    ) -> Qwen36MMInputs:
        """Process a text prompt + optional images into TT-ready tensors.

        For now, `prompt` should be the raw text (no chat-template wrapping).
        The caller is responsible for applying the chat template if desired.
        Image tokens in the prompt are passed through verbatim — the HF
        processor expands them into `n_image_tokens` `<|image_pad|>` placeholders
        per image based on each image's grid_thw.
        """
        # HF processor: accepts text + images, produces input_ids, attention_mask,
        # pixel_values, image_grid_thw, etc. The `text` is the prompt; `images` is
        # the list of PIL images (or None for text-only).
        proc_out = self.processor(text=prompt, images=images, return_tensors="pt", padding=True)

        input_ids = proc_out["input_ids"]
        attention_mask = proc_out["attention_mask"]
        pixel_values = proc_out.get("pixel_values")
        image_grid_thw = proc_out.get("image_grid_thw")

        # HF flattens pixel_values to [batch, num_patches, 1536]; we collapse
        # the batch dim away (qwen3.6 has no batch in the patch tokens — they're
        # all delimited by image_grid_thw). For text-only inputs pixel_values is None.
        if pixel_values is not None and pixel_values.ndim == 3:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])

        # Build 3D position_ids
        position_ids_3d, mrope_deltas = get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            image_token_id=self.image_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )

        return Qwen36MMInputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids_3d=position_ids_3d,
            mrope_deltas=mrope_deltas,
        )

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Convenience: wrap a list of chat-format messages into the prompt string.

        Mirrors the HF processor's apply_chat_template. Use this when the caller
        has structured messages with {role, content} and wants the chat template
        expansion (including the <|im_start|>...<|im_end|> wrappers).
        """
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
