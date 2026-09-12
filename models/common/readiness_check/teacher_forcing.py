# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Teacher-forcing accuracy harness driven by a `readiness_v1` reference file.

Usage:

    acc = TokenAccuracy("llama31_8b.refpt")
    prompt_ids = acc.get_prompt_token_ids(user_idx=0)

    # Prefill the TT model with prompt_ids, then decode one token at a time:
    for _ in range(acc.num_gt_tokens(user_idx=0)):
        tt_pred = run_tt_decode(...)            # int token id
        forced_next = acc.collect_predicted_tokens(tt_pred, user_idx=0)
        feed_token_to_tt(forced_next)            # teacher forcing

    print(acc.compute_accuracy(user_idx=0))
    # {'top1': ..., 'top5': ..., 'top100': ..., 'matches_top1': ..., ...}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch

from models.common.readiness_check.schema import Reference, load_reference


class TokenAccuracy:
    """
    Per-user teacher-forcing helper backed by a `readiness_v1` reference file.

    Each `entry` in the reference corresponds to one prompt and is exposed as
    a separate "user." For multi-user vLLM-style decode, callers pass
    `user_idx` to route per-user state.
    """

    def __init__(self, reference: str | Path | Reference) -> None:
        if isinstance(reference, Reference):
            self.reference = reference
        else:
            self.reference = load_reference(reference)

        self.entry_count = len(self.reference.entries)
        self.k = self.reference.k

        # Flattened per-user views.
        self.prompt_1d: List[torch.Tensor] = [
            e.prompt_tokens[0].to(torch.long).contiguous() for e in self.reference.entries
        ]
        self.gt_gen_1d: List[torch.Tensor] = [
            e.generated_tokens[0].to(torch.long).contiguous() for e in self.reference.entries
        ]
        self.topk: List[torch.Tensor] = [e.topk_tokens.to(torch.long).contiguous() for e in self.reference.entries]
        self.tf_prompt_len: List[int] = [int(e.tf_prompt_len) for e in self.reference.entries]

        meta = self.reference.token_ids_meta or {}
        self.eos_id: Optional[int] = int(meta["eos_id"]) if meta.get("eos_id") is not None else None

        # Per-user runtime state.
        self.pred_tokens: List[List[int]] = [[] for _ in range(self.entry_count)]
        self.cursor: List[int] = [0] * self.entry_count

    # Bounds + reset.

    @property
    def num_entries(self) -> int:
        return self.entry_count

    def _check_idx(self, user_idx: int) -> int:
        idx = int(user_idx)
        if idx < 0 or idx >= self.entry_count:
            raise IndexError(f"user_idx={idx} out of range for {self.entry_count} entry(ies)")
        return idx

    def reset(self) -> None:
        for buf in self.pred_tokens:
            buf.clear()
        for i in range(self.entry_count):
            self.cursor[i] = 0

    # Accessors.

    def get_prompt_token_ids(self, user_idx: int = 0) -> List[int]:
        return self.prompt_1d[self._check_idx(user_idx)].tolist()

    def num_gt_tokens(self, user_idx: int = 0) -> int:
        return int(self.gt_gen_1d[self._check_idx(user_idx)].numel())

    def num_pred_tokens(self, user_idx: int = 0) -> int:
        return len(self.pred_tokens[self._check_idx(user_idx)])

    def get_predicted_tokens(self, user_idx: int = 0) -> List[int]:
        return list(self.pred_tokens[self._check_idx(user_idx)])

    # Core teacher-forcing interface.

    def collect_predicted_tokens(self, tt_pred_token: int, *, user_idx: int = 0) -> int:
        """
        Record TT's predicted token for the next generated position of
        `user_idx` and return the ground-truth token to force into the
        next TT decode step.

        Once the ground-truth sequence is exhausted, returns `eos_id` if
        known, else the last ground-truth token.
        """
        idx = self._check_idx(user_idx)
        gt_gen = self.gt_gen_1d[idx]
        cursor = self.cursor[idx]

        self.pred_tokens[idx].append(int(tt_pred_token))

        if cursor >= int(gt_gen.numel()):
            if self.eos_id is not None:
                return int(self.eos_id)
            return int(gt_gen[-1].item())

        forced = int(gt_gen[cursor].item())
        self.cursor[idx] = cursor + 1
        return forced

    def compute_accuracy(self, user_idx: int = 0) -> Dict[str, float]:
        """
        Top-1 / top-5 / top-K hit rate for `user_idx`:

          For generated step i:
            - top-1 = topk_tokens[i, 0]
            - top-5 = topk_tokens[i, 0:5]
            - top-K = topk_tokens[i, :]

        Returns a dict with float ratios and integer counts:
            {top1, top5, top100, matches_top1, matches_top5, matches_top100, total}
        (`top100` is named for the K=100 default; if the reference has a
        different K, it is still keyed `top100` for consistency. `k` is
        also returned so callers can verify.)
        """
        idx = self._check_idx(user_idx)
        preds = self.pred_tokens[idx]
        gt_gen = self.gt_gen_1d[idx]
        topk = self.topk[idx]

        total = min(len(preds), int(gt_gen.numel()))
        if total == 0:
            return {
                "top1": 0.0,
                "top5": 0.0,
                "top100": 0.0,
                "matches_top1": 0,
                "matches_top5": 0,
                "matches_top100": 0,
                "total": 0,
                "k": self.k,
            }

        matches_top1 = 0
        matches_top5 = 0
        matches_topk = 0
        k_cols = min(5, topk.shape[1])

        for i in range(total):
            row = topk[i]
            tt_pred = int(preds[i])
            if tt_pred == int(row[0].item()):
                matches_top1 += 1
            if tt_pred in row[:k_cols].tolist():
                matches_top5 += 1
            if tt_pred in row.tolist():
                matches_topk += 1

        return {
            "top1": matches_top1 / total,
            "top5": matches_top5 / total,
            "top100": matches_topk / total,
            "matches_top1": matches_top1,
            "matches_top5": matches_top5,
            "matches_top100": matches_topk,
            "total": total,
            "k": self.k,
        }
