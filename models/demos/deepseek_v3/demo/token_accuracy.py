# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger


class TokenAccuracy:
    """
    Minimal teacher-forcing helper for DeepSeek-V3 demo.

    Expects a reference .pt/.refpt file with:
      - reference_tokens: LongTensor [1, T] (prompt + continuation)
      - top5_tokens: LongTensor [T, 5] (optional)

    Splits reference_tokens at T//2 + 1 into input prompt tokens and ground-truth continuation.
    """

    def __init__(self, reference_file: str | Path, prompt_len: int | None = None):
        self._path = str(reference_file)
        data = torch.load(self._path)

        self.reference_tokens = data["reference_tokens"].long()
        if self.reference_tokens.dim() == 1:
            self.reference_tokens = self.reference_tokens.unsqueeze(0)

        self.top5_tokens = data.get("top5_tokens", None)
        if self.top5_tokens is not None:
            self.top5_tokens = self.top5_tokens.long()

        T = self.reference_tokens.shape[-1]
        # Determine prompt length. Default matches simple_text_demo: T//2 + 1
        if prompt_len is None:
            split_point = (T // 2) + 1
        else:
            # Clamp to valid range [1, T-1]
            split_point = max(1, min(int(prompt_len), T - 1))
        self.input_prompt = self.reference_tokens[0, :split_point]
        self.gt_tokens = self.reference_tokens[0, split_point:]

        if self.top5_tokens is not None:
            # Align top5 tokens to decode steps (starting from last prompt token index)
            self.top5_tokens = self.top5_tokens[split_point - 1 :, ...]

        self._gt_pos = -1
        self._pred_tokens: list[int] = []
        self._max_index = len(self.gt_tokens) - 1

        logger.info(
            f"Loaded reference file: {self._path} (prompt_len={len(self.input_prompt)}, gt_len={len(self.gt_tokens)})"
        )

    def prepare_ref_tokens(self, tokenizer) -> str:
        """Decode the prompt token ids to a string prompt."""
        return tokenizer.decode(self.input_prompt.tolist())

    def num_gt_tokens(self) -> int:
        return len(self.gt_tokens)

    def collect_predicted_tokens(self, token_id: int) -> int:
        """Record model’s prediction for the current step and return the ground-truth token to force next step.

        Returns the GT token id at this step as an int.
        """
        self._pred_tokens.append(int(token_id))
        self._gt_pos += 1
        idx = min(self._gt_pos, self._max_index)
        return int(self.gt_tokens[idx].item())

    def compute_accuracy(self) -> dict[str, float | None]:
        """Compute top-1 and optional top-5 accuracy for the collected predictions.

        Returns a dict with keys 'top1' and 'top5' (top5 is None if unavailable).
        """
        matching = min(len(self.gt_tokens), len(self._pred_tokens))
        if matching == 0:
            return {"top1": 0.0, "top5": None}

        gt = self.gt_tokens[:matching]
        pred = torch.tensor(self._pred_tokens[:matching], dtype=torch.long)
        top1 = float((gt == pred).sum().item()) / matching

        top5_val = None
        if self.top5_tokens is not None and self.top5_tokens.numel() > 0:
            t5 = self.top5_tokens[:matching]
            # check if pred in each row of top5
            in_top5 = [(int(pred[i].item()) in set(t5[i].tolist())) for i in range(matching)]
            top5_val = float(sum(in_top5)) / matching

        return {"top1": top1, "top5": top5_val}
