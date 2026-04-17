# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .embeddings import build_inputs_embeds
from .hf_utils import HFLoadSpec, load_processor_and_model
from .rope import Qwen2RopeHelper
from .vision import vision_tower_forward


@dataclass
class DotsOCRInputs:
    """
    Canonical inputs for our modular pipeline.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None


class DotsOCRReference:
    """
    Thin wrapper around HF Dots OCR model that exposes modular entry points.

    This is used as the correctness oracle for TTNN modules via PCC.
    """

    def __init__(self, spec: HFLoadSpec):
        self.processor, self.model = load_processor_and_model(spec)
        self.config = self.model.config

    @property
    def image_token_id(self) -> int:
        return int(getattr(self.config, "image_token_id"))

    def preprocess_image_and_prompt(self, image, prompt: str) -> DotsOCRInputs:
        """
        Uses the HF processor to create token ids + pixel_values + image_grid_thw.

        Multimodal checkpoints (Dots, Qwen2-VL) accept ``images=``; plain CausalLM
        tokenizers only accept ``text=`` — branch accordingly for text-only tests.
        """
        try:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        except TypeError:
            if image is not None:
                raise
            inputs = self.processor(text=prompt, return_tensors="pt")
        return DotsOCRInputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )

    @torch.no_grad()
    def forward(self, inputs: DotsOCRInputs, max_new_tokens: int = 32) -> Dict[str, Any]:
        """
        End-to-end reference forward using HF generate().
        """
        gen = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            max_new_tokens=max_new_tokens,
        )
        return {"generated_ids": gen}

    @torch.no_grad()
    def vision_forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Vision token rows from `DotsVisionTransformer` (`model.vision_tower`).

        Args use processor keys: `pixel_values`, `image_grid_thw` (passed as `grid_thw` to the tower).
        """
        return vision_tower_forward(self.model, pixel_values, image_grid_thw)

    @torch.no_grad()
    def build_inputs_embeds(self, inputs: DotsOCRInputs) -> torch.Tensor:
        """
        Fused `inputs_embeds` [B, S, D] via HF `prepare_inputs_embeds` (vision + text).
        """
        return build_inputs_embeds(
            self.model,
            inputs.input_ids,
            inputs.pixel_values,
            inputs.image_grid_thw,
        )

    @torch.no_grad()
    def get_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Get raw logits from the model for PCC comparison (text-only).

        This is used to validate that TTNN prefill produces the same logits
        as the HF reference model.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.logits

    def get_rope_helper(self) -> Qwen2RopeHelper:
        """Get RoPE helper configured from model parameters."""
        config = self.model.config
        head_dim = getattr(
            config, "head_dim", getattr(config, "hidden_size", 4096) // getattr(config, "num_attention_heads", 32)
        )
        theta = getattr(config, "rope_theta", 10000.0)
        max_seq_len = getattr(config, "max_position_embeddings", 8192)

        return Qwen2RopeHelper(head_dim=head_dim, max_seq_len=max_seq_len, theta=theta)
