# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class AccuracyResult:
    top1: float
    top5: float
    matches_top1: int
    matches_top5: int
    total: int


class TokenAccuracy:
    """
    Teacher forcing helper backed by the HF-generated reference .refpt file.

    Supports two payload formats:

    **Single-entry (legacy)**::

        {
            "prompt_tokens":    Tensor [1, P],
            "generated_tokens": Tensor [1, G],
            "top5_tokens":      Tensor [L, 5],   # L = P + G
            "tf_prompt_len":    int,
        }

    **Multi-entry** (``format_version == "multi_prompt_v1"``)::

        {
            "format_version": "multi_prompt_v1",
            "entries": [
                {
                    "prompt_tokens":    Tensor [1, P_i],
                    "generated_tokens": Tensor [1, G_i],
                    "top5_tokens":      Tensor [L_i, 5],
                    "tf_prompt_len":    int,
                    ...
                },
                ...
            ],
            # backward-compat top-level keys copied from entries[0]
            "prompt_tokens": ...,
            ...
        }

    All per-user methods accept a *user_idx* keyword (default ``0``) so that
    existing single-entry callers keep working unchanged.
    """

    def __init__(self, reference_file: str | Path, prompt_len: Optional[int] = None) -> None:
        self.reference_file = Path(reference_file)
        payload = torch.load(self.reference_file, weights_only=False)

        # ---- Detect single vs multi-entry ----
        raw_entries = payload.get("entries")
        if isinstance(raw_entries, list) and raw_entries:
            self._entries = raw_entries
        else:
            self._entries = [payload]

        self._num_entries = len(self._entries)

        # ---- Per-entry validated data ----
        self._prompt_1d_list: List[torch.Tensor] = []
        self._gt_gen_1d_list: List[torch.Tensor] = []
        self._reference_1d_list: List[torch.Tensor] = []
        self._top5_tokens_list: List[torch.Tensor] = []
        self._tf_prompt_len_list: List[int] = []

        for idx, entry in enumerate(self._entries):
            required = ("prompt_tokens", "generated_tokens", "top5_tokens", "tf_prompt_len")
            missing = [k for k in required if k not in entry]
            if missing:
                raise KeyError(f"Entry {idx} missing keys: {missing}. File={self.reference_file}")

            pt = entry["prompt_tokens"]
            gt = entry["generated_tokens"]
            t5 = entry["top5_tokens"]
            tfl = int(entry["tf_prompt_len"])

            if not isinstance(pt, torch.Tensor) or pt.dim() != 2 or pt.shape[0] != 1:
                raise ValueError(
                    f"Entry {idx}: prompt_tokens must be [1, P], got {type(pt)} {getattr(pt, 'shape', None)}"
                )
            if not isinstance(gt, torch.Tensor) or gt.dim() != 2 or gt.shape[0] != 1:
                raise ValueError(
                    f"Entry {idx}: generated_tokens must be [1, G], got {type(gt)} {getattr(gt, 'shape', None)}"
                )
            if not isinstance(t5, torch.Tensor) or t5.dim() != 2 or t5.shape[1] != 5:
                raise ValueError(
                    f"Entry {idx}: top5_tokens must be [L, 5], got {type(t5)} {getattr(t5, 'shape', None)}"
                )

            p1d = pt[0].to(torch.long).contiguous()
            g1d = gt[0].to(torch.long).contiguous()

            self._prompt_1d_list.append(p1d)
            self._gt_gen_1d_list.append(g1d)
            self._reference_1d_list.append(torch.cat([p1d, g1d], dim=0))
            self._top5_tokens_list.append(t5)
            self._tf_prompt_len_list.append(tfl)

        # ---- Backward-compat aliases for entry 0 ----
        e0 = self._entries[0]
        self.prompt_tokens = e0["prompt_tokens"]
        self.generated_tokens = e0["generated_tokens"]
        self.top5_tokens = self._top5_tokens_list[0]
        self.tf_prompt_len = self._tf_prompt_len_list[0]
        self._prompt_1d = self._prompt_1d_list[0]
        self._gt_gen_1d = self._gt_gen_1d_list[0]
        self._reference_1d = self._reference_1d_list[0]

        if prompt_len is not None and int(prompt_len) != self.tf_prompt_len:
            raise ValueError(f"prompt_len arg ({prompt_len}) != tf_prompt_len in file ({self.tf_prompt_len})")

        # ---- Per-user state ----
        self._pred_tokens_list: List[List[int]] = [[] for _ in range(self._num_entries)]
        self._cursor_list: List[int] = [0] * self._num_entries
        # Legacy alias (same list object as _pred_tokens_list[0])
        self._pred_tokens = self._pred_tokens_list[0]
        self._cursor = 0

        # ---- EOS / token metadata ----
        meta = payload.get("token_ids_meta", {}) if isinstance(payload.get("token_ids_meta", {}), dict) else {}
        self.eos_id: Optional[int] = int(meta["eos_id"]) if "eos_id" in meta and meta["eos_id"] is not None else None

        # ---- TopK candidates (single-entry only) ----
        self.topk_candidate_k = 0
        self.topk_candidate_generated_prefix_len = 0
        self.topk_candidate_token_ids: Optional[torch.Tensor] = None
        self.topk_candidate_probs: Optional[torch.Tensor] = None
        topk_candidates = payload.get("topk_candidates")
        if topk_candidates is not None:
            if not isinstance(topk_candidates, dict):
                raise ValueError(f"topk_candidates must be a dict when present. File={self.reference_file}")
            token_ids = topk_candidates.get("token_ids")
            probs = topk_candidates.get("probs")
            k = int(topk_candidates.get("k", 0))
            generated_prefix_len = int(topk_candidates.get("generated_prefix_len", 0))
            if (
                not isinstance(token_ids, torch.Tensor)
                or token_ids.dim() != 2
                or not isinstance(probs, torch.Tensor)
                or probs.dim() != 2
                or token_ids.shape != probs.shape
            ):
                raise ValueError(
                    f"topk_candidates must contain matching 2D token_ids/probs tensors, got "
                    f"{getattr(token_ids, 'shape', None)} and {getattr(probs, 'shape', None)}. "
                    f"File={self.reference_file}"
                )
            if token_ids.shape[1] != k:
                raise ValueError(
                    f"topk_candidates k={k} does not match stored width {token_ids.shape[1]}. "
                    f"File={self.reference_file}"
                )
            if token_ids.shape[0] != generated_prefix_len:
                raise ValueError(
                    f"topk_candidates generated_prefix_len={generated_prefix_len} does not match stored rows "
                    f"{token_ids.shape[0]}. File={self.reference_file}"
                )
            self.topk_candidate_k = k
            self.topk_candidate_generated_prefix_len = generated_prefix_len
            self.topk_candidate_token_ids = token_ids.to(torch.long).contiguous()
            self.topk_candidate_probs = probs.to(torch.float32).contiguous()

    # ---- Properties ----

    @property
    def num_entries(self) -> int:
        return self._num_entries

    # ---- Reset ----

    def reset(self) -> None:
        """Reset internal cursor/prediction buffer for all users."""
        for lst in self._pred_tokens_list:
            lst.clear()
        for i in range(self._num_entries):
            self._cursor_list[i] = 0
        self._cursor = 0

    # ---- Per-user accessors ----

    def get_prompt_token_ids(self, user_idx: int = 0) -> List[int]:
        idx = min(user_idx, self._num_entries - 1)
        return self._prompt_1d_list[idx].tolist()

    def num_gt_tokens(self, user_idx: int = 0) -> int:
        idx = min(user_idx, self._num_entries - 1)
        return int(self._gt_gen_1d_list[idx].numel())

    def num_pred_tokens(self, user_idx: int = 0) -> int:
        idx = min(user_idx, self._num_entries - 1)
        return len(self._pred_tokens_list[idx])

    def get_predicted_tokens(self, user_idx: int = 0) -> List[int]:
        idx = min(user_idx, self._num_entries - 1)
        return list(self._pred_tokens_list[idx])

    # ---- Garbage-token helpers (single-entry / user 0 only) ----

    def _resolve_pred_tokens(self, pred_tokens: Optional[List[int]] = None) -> List[int]:
        if pred_tokens is None:
            return self._pred_tokens_list[0]
        return [int(tok) for tok in pred_tokens]

    def has_garbage_check(self) -> bool:
        return self.topk_candidate_token_ids is not None and self.topk_candidate_probs is not None

    def num_garbage_check_tokens(self, pred_tokens: Optional[List[int]] = None) -> int:
        if not self.has_garbage_check():
            return 0
        return min(
            len(self._resolve_pred_tokens(pred_tokens)),
            self.num_gt_tokens(0),
            self.topk_candidate_generated_prefix_len,
        )

    def get_garbage_token_details(
        self,
        pred_tokens: Optional[List[int]] = None,
        *,
        context_window: int = 16,
        tail_width: int = 5,
    ) -> List[Dict[str, Any]]:
        if not self.has_garbage_check():
            return []

        pred_tokens_resolved = self._resolve_pred_tokens(pred_tokens)
        total_checked = self.num_garbage_check_tokens(pred_tokens_resolved)
        details: List[Dict[str, Any]] = []
        assert self.topk_candidate_token_ids is not None
        assert self.topk_candidate_probs is not None

        for step in range(total_checked):
            tt_pred = int(pred_tokens_resolved[step])
            candidate_ids_row = self.topk_candidate_token_ids[step]
            candidate_probs_row = self.topk_candidate_probs[step]
            candidate_ids = candidate_ids_row.tolist()
            if tt_pred in candidate_ids:
                continue

            pos = self.tf_prompt_len + step
            context_start = max(0, pos - context_window)
            details.append(
                {
                    "generated_step": step,
                    "position": pos,
                    "predicted_id": tt_pred,
                    "true_id": int(self._gt_gen_1d_list[0][step].item()),
                    "context_token_ids": self._reference_1d_list[0][context_start:pos].tolist(),
                    "top5_ids": self._top5_tokens_list[0][pos].tolist(),
                    "topk_k": self.topk_candidate_k,
                    "topk_tail_prob": float(candidate_probs_row[-1].item()),
                    "topk_head_ids": candidate_ids[: min(5, len(candidate_ids))],
                    "topk_tail_ids": candidate_ids[max(0, len(candidate_ids) - tail_width) :],
                }
            )

        return details

    @staticmethod
    def _sanitize_decoded(text: str) -> str:
        return repr(text)[1:-1]

    def format_garbage_token_details(
        self,
        tokenizer,
        pred_tokens: Optional[List[int]] = None,
        *,
        context_window: int = 16,
        tail_width: int = 5,
    ) -> List[str]:
        lines = []
        for detail in self.get_garbage_token_details(
            pred_tokens,
            context_window=context_window,
            tail_width=tail_width,
        ):
            context_text = self._sanitize_decoded(
                tokenizer.decode(detail["context_token_ids"], skip_special_tokens=False)
            )
            predicted_text = self._sanitize_decoded(
                tokenizer.decode([detail["predicted_id"]], skip_special_tokens=False)
            )
            true_text = self._sanitize_decoded(tokenizer.decode([detail["true_id"]], skip_special_tokens=False))
            top5_text = ", ".join(
                self._sanitize_decoded(tokenizer.decode([tok], skip_special_tokens=False)) for tok in detail["top5_ids"]
            )
            tail_text = ", ".join(
                self._sanitize_decoded(tokenizer.decode([tok], skip_special_tokens=False))
                for tok in detail["topk_tail_ids"]
            )
            lines.append(
                f"step {detail['generated_step']} pos {detail['position']}: "
                f"context='{context_text}' "
                f"predicted='{predicted_text}' (id={detail['predicted_id']}) "
                f"not in teacher top-{detail['topk_k']}; "
                f"true='{true_text}' (id={detail['true_id']}); "
                f"top5=[{top5_text}]; "
                f"tail_prob={detail['topk_tail_prob']:.3e}; "
                f"tail=[{tail_text}]"
            )
        return lines

    # ---- Core teacher-forcing interface ----

    def collect_predicted_tokens(self, tt_pred_token: int, *, user_idx: int = 0) -> int:
        """
        Record TT's predicted token for the *next* generated position of *user_idx*,
        and return the ground-truth token to force into TT decode.
        """
        idx = min(user_idx, self._num_entries - 1)
        gt_gen = self._gt_gen_1d_list[idx]
        cursor = self._cursor_list[idx]

        if cursor >= int(gt_gen.numel()):
            self._pred_tokens_list[idx].append(int(tt_pred_token))
            if self.eos_id is not None:
                return int(self.eos_id)
            return int(gt_gen[-1].item())

        self._pred_tokens_list[idx].append(int(tt_pred_token))
        forced = int(gt_gen[cursor].item())
        self._cursor_list[idx] = cursor + 1
        return forced

    def compute_accuracy(self, user_idx: int = 0) -> Dict[str, float]:
        """
        Accuracy vs top5_tokens for user *user_idx*:
          For generated step i:
            - sequence position = tf_prompt_len + i
            - top-1 token = top5_tokens[pos][0]
            - top-5 tokens = top5_tokens[pos][:]
        """
        idx = min(user_idx, self._num_entries - 1)
        preds = self._pred_tokens_list[idx]
        gt_gen = self._gt_gen_1d_list[idx]
        t5 = self._top5_tokens_list[idx]
        tfl = self._tf_prompt_len_list[idx]

        total = min(len(preds), int(gt_gen.numel()))
        if total == 0:
            return {"top1": 0.0, "top5": 0.0, "matches_top1": 0, "matches_top5": 0, "total": 0}

        matches_top1 = 0
        matches_top5 = 0

        for i in range(total):
            pos = tfl + i
            hf_top5 = t5[pos].tolist()
            hf_top1 = hf_top5[0]
            tt_pred = int(preds[i])

            if tt_pred == hf_top1:
                matches_top1 += 1
            if tt_pred in hf_top5:
                matches_top5 += 1

        return {
            "top1": matches_top1 / total,
            "top5": matches_top5 / total,
            "matches_top1": matches_top1,
            "matches_top5": matches_top5,
            "total": total,
        }
