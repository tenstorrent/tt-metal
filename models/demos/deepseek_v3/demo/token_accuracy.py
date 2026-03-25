# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
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

    Expected payload keys (your current refpt format):
      - prompt_tokens: Tensor [1, prompt_len]
      - generated_tokens: Tensor [1, gen_len]  (ground-truth continuation)
      - top5_tokens: Tensor [L, 5] where L = prompt_len + gen_len
      - tf_prompt_len: int
    """

    def __init__(self, reference_file: str | Path, prompt_len: Optional[int] = None) -> None:
        self.reference_file = Path(reference_file)
        payload = torch.load(self.reference_file, weights_only=False)

        # --- Validate payload shape invariants (keeps teacher forcing alignment unambiguous) ---
        required = ("prompt_tokens", "generated_tokens", "top5_tokens", "tf_prompt_len")
        missing = [k for k in required if k not in payload]
        if missing:
            raise KeyError(f"Reference file missing keys: {missing}. File={self.reference_file}")

        self.prompt_tokens = payload["prompt_tokens"]  # [1, P]
        self.generated_tokens = payload["generated_tokens"]  # [1, G]
        self.top5_tokens = payload["top5_tokens"]  # [P+G, 5]
        self.tf_prompt_len = int(payload["tf_prompt_len"])

        if (
            not isinstance(self.prompt_tokens, torch.Tensor)
            or self.prompt_tokens.dim() != 2
            or self.prompt_tokens.shape[0] != 1
        ):
            raise ValueError(
                f"prompt_tokens must be a Tensor of shape [1, P], got {type(self.prompt_tokens)} {getattr(self.prompt_tokens,'shape',None)}"
            )
        if (
            not isinstance(self.generated_tokens, torch.Tensor)
            or self.generated_tokens.dim() != 2
            or self.generated_tokens.shape[0] != 1
        ):
            raise ValueError(
                f"generated_tokens must be a Tensor of shape [1, G], got {type(self.generated_tokens)} {getattr(self.generated_tokens,'shape',None)}"
            )
        if (
            not isinstance(self.top5_tokens, torch.Tensor)
            or self.top5_tokens.dim() != 2
            or self.top5_tokens.shape[1] != 5
        ):
            raise ValueError(
                f"top5_tokens must be a Tensor of shape [L, 5], got {type(self.top5_tokens)} {getattr(self.top5_tokens,'shape',None)}"
            )

        file_prompt_len = int(self.prompt_tokens.shape[1])
        file_gen_len = int(self.generated_tokens.shape[1])
        expected_L = file_prompt_len + file_gen_len
        if int(self.top5_tokens.shape[0]) != expected_L:
            raise ValueError(
                f"top5_tokens length mismatch: expected L={expected_L} (=prompt {file_prompt_len} + gen {file_gen_len}), got {int(self.top5_tokens.shape[0])}"
            )
        if self.tf_prompt_len != file_prompt_len:
            raise ValueError(
                f"tf_prompt_len in file ({self.tf_prompt_len}) != prompt_tokens length ({file_prompt_len}). File={self.reference_file}"
            )

        if prompt_len is not None and int(prompt_len) != self.tf_prompt_len:
            raise ValueError(f"prompt_len arg ({prompt_len}) != tf_prompt_len in file ({self.tf_prompt_len})")

        # Flatten to 1D for convenience
        self._prompt_1d = self.prompt_tokens[0].to(torch.long).contiguous()
        self._gt_gen_1d = self.generated_tokens[0].to(torch.long).contiguous()
        self._reference_1d = torch.cat([self._prompt_1d, self._gt_gen_1d], dim=0)

        # Collected TT predictions (one per generated token position)
        self._pred_tokens: List[int] = []
        self._cursor = 0  # how many GT tokens have been consumed/forced

        # Optional token id metadata for nicer debugging / fallbacks
        meta = payload.get("token_ids_meta", {}) if isinstance(payload.get("token_ids_meta", {}), dict) else {}
        self.eos_id: Optional[int] = int(meta["eos_id"]) if "eos_id" in meta and meta["eos_id"] is not None else None

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

    def reset(self) -> None:
        """Reset internal cursor/prediction buffer so the same instance can be reused safely."""
        self._pred_tokens.clear()
        self._cursor = 0

    def get_prompt_token_ids(self) -> List[int]:
        return self._prompt_1d.tolist()

    def num_gt_tokens(self) -> int:
        return int(self._gt_gen_1d.numel())

    def num_pred_tokens(self) -> int:
        return len(self._pred_tokens)

    def _resolve_pred_tokens(self, pred_tokens: Optional[List[int]] = None) -> List[int]:
        if pred_tokens is None:
            return self._pred_tokens
        return [int(tok) for tok in pred_tokens]

    def has_garbage_check(self) -> bool:
        return self.topk_candidate_token_ids is not None and self.topk_candidate_probs is not None

    def num_garbage_check_tokens(self, pred_tokens: Optional[List[int]] = None) -> int:
        if not self.has_garbage_check():
            return 0
        return min(
            len(self._resolve_pred_tokens(pred_tokens)),
            self.num_gt_tokens(),
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
                    "true_id": int(self._gt_gen_1d[step].item()),
                    "context_token_ids": self._reference_1d[context_start:pos].tolist(),
                    "top5_ids": self.top5_tokens[pos].tolist(),
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

    def collect_predicted_tokens(self, tt_pred_token: int) -> int:
        """
        Record TT's predicted token for the *next* generated position,
        and return the ground-truth token to force into TT decode.
        """
        if self._cursor >= self.num_gt_tokens():
            # If TT generates beyond GT length, just keep returning EOS-ish
            self._pred_tokens.append(int(tt_pred_token))
            if self.eos_id is not None:
                return int(self.eos_id)
            return int(self._gt_gen_1d[-1].item())

        self._pred_tokens.append(int(tt_pred_token))
        forced = int(self._gt_gen_1d[self._cursor].item())
        self._cursor += 1
        return forced

    def compute_accuracy(self) -> Dict[str, float]:
        """
        Accuracy vs HF baseline top5_tokens:
          For generated step i:
            - sequence position = tf_prompt_len + i
            - HF top-1 token = top5_tokens[pos][0]
            - HF top-5 tokens = top5_tokens[pos][:]
        """
        total = min(len(self._pred_tokens), self.num_gt_tokens())
        if total == 0:
            return {"top1": 0.0, "top5": 0.0, "matches_top1": 0, "matches_top5": 0, "total": 0}

        matches_top1 = 0
        matches_top5 = 0

        for i in range(total):
            pos = self.tf_prompt_len + i
            hf_top5 = self.top5_tokens[pos].tolist()
            hf_top1 = hf_top5[0]
            tt_pred = int(self._pred_tokens[i])

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
