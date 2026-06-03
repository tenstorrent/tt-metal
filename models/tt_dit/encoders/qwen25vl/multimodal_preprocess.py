# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side multimodal batching for Qwen2.5-VL style inputs (replaces ``Qwen2VLProcessor``).

``encode_with_images`` only needs ``input_ids``, ``attention_mask``, ``pixel_values``, and
``image_grid_thw``. This module reproduces the ``Qwen2VLProcessor.__call__`` logic for the
image + text path using:

- The pipeline/encoder ``Qwen2Tokenizer`` (same vocabulary as the TT text encoder).
- ``AutoImageProcessor.from_pretrained(...)`` for the vision front-end (same tensors as the
  bundled HF processor; avoids the monolithic ``Qwen2VLProcessor`` wrapper).

Tokenizer string handling remains on host; vision tensors are produced here then uploaded
inside ``Qwen25VlTokenizerEncoderPair.encode_with_images``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import AutoImageProcessor, BatchFeature, PreTrainedTokenizerBase
from transformers.image_utils import ImageInput

DEFAULT_VL_IMAGE_PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass
class Qwen25VlMultimodalPreprocessor:
    """Prepare ``BatchFeature`` matching ``Qwen2VLProcessor`` for image-conditioned VL encode."""

    tokenizer: PreTrainedTokenizerBase
    image_processor: Any
    image_token: str = "<|image_pad|>"

    @classmethod
    def from_hub(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        *,
        image_processor_hub: str = DEFAULT_VL_IMAGE_PROCESSOR_ID,
    ) -> Qwen25VlMultimodalPreprocessor:
        image_processor = AutoImageProcessor.from_pretrained(image_processor_hub)
        image_token = "<|image_pad|>"
        if hasattr(tokenizer, "image_token"):
            image_token = tokenizer.image_token  # type: ignore[assignment]
        return cls(tokenizer=tokenizer, image_processor=image_processor, image_token=image_token)

    @property
    def merge_size(self) -> int:
        return int(self.image_processor.merge_size)

    def __call__(
        self,
        *,
        text: Sequence[str],
        images: Sequence[ImageInput],
        padding: bool | str = True,
        return_tensors: str | None = "pt",
        **images_kwargs: Any,
    ) -> BatchFeature:
        """Mirror ``Qwen2VLProcessor.__call__`` for ``images`` + ``text`` (no video path)."""
        image_inputs: dict = {}
        if images is not None:
            image_inputs = self.image_processor(
                images=list(images),
                return_tensors=return_tensors,
                **images_kwargs,
            )
            image_grid_thw = image_inputs["image_grid_thw"]
            if isinstance(image_grid_thw, torch.Tensor):
                grid_rows = [image_grid_thw[i] for i in range(image_grid_thw.shape[0])]
            else:
                grid_rows = [torch.as_tensor(row) for row in image_grid_thw]

        text_work = list(text).copy()
        merge_length = self.merge_size**2
        idx = 0
        if images is not None:
            for i in range(len(text_work)):
                while self.image_token in text_work[i]:
                    g = grid_rows[idx]
                    num_image_tokens = int(g.prod().item()) // merge_length
                    text_work[i] = text_work[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    idx += 1
                text_work[i] = text_work[i].replace("<|placeholder|>", self.image_token)

        tok_out = self.tokenizer(
            text_work,
            padding=padding,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )
        if not isinstance(tok_out, dict):
            tok_out = dict(tok_out)

        return BatchFeature(data={**tok_out, **image_inputs}, tensor_type=return_tensors)
