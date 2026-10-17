# SPDX-FileCopyrightText: © 2026 Black Forest Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/black-forest-labs/flux-3-image-klein (src/flux3/text_encoder.py) at e53f4ea,
# licensed under Apache-2.0. Modified by Tenstorrent as the PyTorch reference for the tt-metal port.

"""Qwen3-VL-4B as text encoder """

import math

import torch
from torch import Tensor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

OUTPUT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
MAX_LENGTH = 4096
PAD_MULTIPLE = 80


class TextEncoder:
    def __init__(self, path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        self.processor.tokenizer.padding_side = "right"
        self.device = device

    @torch.inference_mode()
    def encode(self, prompt: str) -> tuple[Tensor, Tensor]:
        """prompt -> tokens (L, 20480) bf16 and ids (L, 4) = (0, 0, 0, position); L padded to a multiple of 80."""
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        tokenizer = self.processor.tokenizer
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)["input_ids"].shape[1]
        length = min(math.ceil(real / PAD_MULTIPLE) * PAD_MULTIPLE, MAX_LENGTH)
        inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        mask = inputs["attention_mask"].to(self.device)
        positions = (mask.long().cumsum(-1) - 1).masked_fill(mask == 0, 1)
        output = self.model(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=mask,
            position_ids=positions[None].expand(4, -1, -1),
            output_hidden_states=True,
            use_cache=False,
        )
        stacked = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)  # (1, 8, L, 2560)
        tokens = stacked.permute(0, 2, 1, 3).reshape(length, -1).to(torch.bfloat16)
        positions = torch.arange(length, device=self.device)
        ids = torch.stack([torch.zeros_like(positions)] * 3 + [positions], dim=-1)
        return tokens, ids
