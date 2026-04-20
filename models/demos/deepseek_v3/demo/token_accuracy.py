# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import lzma
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def decompress_lzma_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Decompress a ``multi_prompt_v1_lzma_v1`` payload into multi-prompt entries.

    Reconstructed entry schema:
      - prompt_tokens:    Tensor [1, P]
      - generated_tokens: Tensor [1, G]
      - top5_tokens:      Tensor [P + G, 5]
      - tf_prompt_len:    int
    """
    if "tensor_lzma" not in payload or "entry_layout" not in payload:
        raise KeyError("LZMA payload must contain 'tensor_lzma' and 'entry_layout'.")

    raw = lzma.decompress(payload["tensor_lzma"])
    expected_raw_bytes = payload.get("tensor_raw_bytes_uncompressed")
    if isinstance(expected_raw_bytes, int) and expected_raw_bytes != len(raw):
        raise ValueError(f"LZMA payload raw byte mismatch: metadata={expected_raw_bytes}, actual={len(raw)}")

    entries: List[Dict[str, Any]] = []
    offset = 0
    layout = payload["entry_layout"]
    if not isinstance(layout, list):
        raise ValueError("LZMA payload 'entry_layout' must be a list.")

    for idx, meta in enumerate(layout):
        if not isinstance(meta, dict):
            raise ValueError(f"entry_layout[{idx}] must be a dict, got {type(meta)}")
        P = int(meta["P"])
        G = int(meta["G"])
        tf_prompt_len = int(meta.get("tf_prompt_len", P))
        if P < 0 or G < 0:
            raise ValueError(f"entry_layout[{idx}] has negative lengths: P={P}, G={G}")

        p_bytes = P * 4
        g_bytes = G * 4
        t_bytes = G * 5 * 4
        end_needed = offset + p_bytes + g_bytes + t_bytes
        if end_needed > len(raw):
            raise ValueError(f"LZMA payload ended early at entry {idx}: need {end_needed} bytes, have {len(raw)}")

        prompt_ids = list(struct.unpack(f"<{P}i", raw[offset : offset + p_bytes]))
        offset += p_bytes
        generated_ids = list(struct.unpack(f"<{G}i", raw[offset : offset + g_bytes]))
        offset += g_bytes
        top5_flat = list(struct.unpack(f"<{G * 5}i", raw[offset : offset + t_bytes]))
        offset += t_bytes

        prompt_t = torch.tensor([prompt_ids], dtype=torch.long)
        gen_t = torch.tensor([generated_ids], dtype=torch.long)
        top5_full = torch.zeros(P + G, 5, dtype=torch.long)
        top5_full[P:] = torch.tensor(top5_flat, dtype=torch.long).view(G, 5)

        entries.append(
            {
                "prompt_tokens": prompt_t,
                "generated_tokens": gen_t,
                "top5_tokens": top5_full,
                "tf_prompt_len": tf_prompt_len,
            }
        )

    if offset != len(raw):
        raise ValueError(f"LZMA payload has trailing bytes: consumed={offset}, total={len(raw)}")

    return entries


class TokenAccuracy:
    """
    Teacher forcing helper backed by the HF-generated reference .refpt file.

    Supports two payload formats:

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
        }

    **LZMA-compressed multi-entry** (``format_version == "multi_prompt_v1_lzma_v1"``)::

        {
            "format_version": "multi_prompt_v1_lzma_v1",
            "tensor_lzma": bytes,
            "entry_layout": [{"P": int, "G": int, "tf_prompt_len": int}, ...],
            ...
        }

    All teacher-forcing methods are per-user and accept a *user_idx* keyword
    (default ``0``).
    """

    def __init__(self, reference_file: str | Path) -> None:
        self.reference_file = Path(reference_file)
        payload = torch.load(self.reference_file, weights_only=False)

        # Detect payload format.
        fmt = payload.get("format_version", "")
        if fmt == "multi_prompt_v1_lzma_v1":
            entries = decompress_lzma_payload(payload)
        else:
            entries = payload.get("entries")

        if not isinstance(entries, list) or not entries:
            raise ValueError(
                "Teacher-forcing reference must contain a non-empty multi-prompt 'entries' list. "
                f"Got format_version={fmt!r} file={self.reference_file}"
            )

        self.entries = entries

        self.entry_count = len(self.entries)

        # Validate per-entry tensors.
        self.prompt_1d_list: List[torch.Tensor] = []
        self.gt_gen_1d_list: List[torch.Tensor] = []
        self.reference_1d_list: List[torch.Tensor] = []
        self.top5_tokens_list: List[torch.Tensor] = []
        self.tf_prompt_len_list: List[int] = []

        for idx, entry in enumerate(self.entries):
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

            self.prompt_1d_list.append(p1d)
            self.gt_gen_1d_list.append(g1d)
            self.reference_1d_list.append(torch.cat([p1d, g1d], dim=0))
            self.top5_tokens_list.append(t5)
            self.tf_prompt_len_list.append(tfl)

        # Per-user runtime state.
        self.pred_tokens_list: List[List[int]] = [[] for entry_idx in range(self.entry_count)]
        self.cursor_list: List[int] = [0] * self.entry_count

        # Token metadata.
        meta = payload.get("token_ids_meta", {}) if isinstance(payload.get("token_ids_meta", {}), dict) else {}
        self.eos_id: Optional[int] = int(meta["eos_id"]) if "eos_id" in meta and meta["eos_id"] is not None else None

        # Optional top-k metadata for entry 0.
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

    # Public properties.

    @property
    def num_entries(self) -> int:
        return self.entry_count

    def checked_user_idx(self, user_idx: int) -> int:
        idx = int(user_idx)
        if idx < 0 or idx >= self.entry_count:
            raise IndexError(
                f"user_idx={idx} is out of range for {self.entry_count} entry(ies) " f"in {self.reference_file}"
            )
        return idx

    # Reset state.

    def reset(self) -> None:
        """Reset internal cursor/prediction buffer for all users."""
        for lst in self.pred_tokens_list:
            lst.clear()
        for i in range(self.entry_count):
            self.cursor_list[i] = 0

    # Per-user accessors.

    def get_prompt_token_ids(self, user_idx: int = 0) -> List[int]:
        idx = self.checked_user_idx(user_idx)
        return self.prompt_1d_list[idx].tolist()

    def num_gt_tokens(self, user_idx: int = 0) -> int:
        idx = self.checked_user_idx(user_idx)
        return int(self.gt_gen_1d_list[idx].numel())

    def num_pred_tokens(self, user_idx: int = 0) -> int:
        idx = self.checked_user_idx(user_idx)
        return len(self.pred_tokens_list[idx])

    def get_predicted_tokens(self, user_idx: int = 0) -> List[int]:
        idx = self.checked_user_idx(user_idx)
        return list(self.pred_tokens_list[idx])

    # Garbage-token helpers for entry 0.

    def resolve_pred_tokens(self, pred_tokens: Optional[List[int]] = None) -> List[int]:
        if pred_tokens is None:
            return self.pred_tokens_list[0]
        return [int(tok) for tok in pred_tokens]

    def has_garbage_check(self) -> bool:
        return self.topk_candidate_token_ids is not None and self.topk_candidate_probs is not None

    def num_garbage_check_tokens(self, pred_tokens: Optional[List[int]] = None) -> int:
        if not self.has_garbage_check():
            return 0
        return min(
            len(self.resolve_pred_tokens(pred_tokens)),
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

        pred_tokens_resolved = self.resolve_pred_tokens(pred_tokens)
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

            pos = self.tf_prompt_len_list[0] + step
            context_start = max(0, pos - context_window)
            details.append(
                {
                    "generated_step": step,
                    "position": pos,
                    "predicted_id": tt_pred,
                    "true_id": int(self.gt_gen_1d_list[0][step].item()),
                    "context_token_ids": self.reference_1d_list[0][context_start:pos].tolist(),
                    "top5_ids": self.top5_tokens_list[0][pos].tolist(),
                    "topk_k": self.topk_candidate_k,
                    "topk_tail_prob": float(candidate_probs_row[-1].item()),
                    "topk_head_ids": candidate_ids[: min(5, len(candidate_ids))],
                    "topk_tail_ids": candidate_ids[max(0, len(candidate_ids) - tail_width) :],
                }
            )

        return details

    @staticmethod
    def sanitize_decoded(text: str) -> str:
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
            context_text = self.sanitize_decoded(
                tokenizer.decode(detail["context_token_ids"], skip_special_tokens=False)
            )
            predicted_text = self.sanitize_decoded(
                tokenizer.decode([detail["predicted_id"]], skip_special_tokens=False)
            )
            true_text = self.sanitize_decoded(tokenizer.decode([detail["true_id"]], skip_special_tokens=False))
            top5_text = ", ".join(
                self.sanitize_decoded(tokenizer.decode([tok], skip_special_tokens=False)) for tok in detail["top5_ids"]
            )
            tail_text = ", ".join(
                self.sanitize_decoded(tokenizer.decode([tok], skip_special_tokens=False))
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

    # Core teacher-forcing interface.

    def collect_predicted_tokens(self, tt_pred_token: int, *, user_idx: int = 0) -> int:
        """
        Record TT's predicted token for the *next* generated position of *user_idx*,
        and return the ground-truth token to force into TT decode.
        """
        idx = self.checked_user_idx(user_idx)
        gt_gen = self.gt_gen_1d_list[idx]
        cursor = self.cursor_list[idx]

        if cursor >= int(gt_gen.numel()):
            self.pred_tokens_list[idx].append(int(tt_pred_token))
            if self.eos_id is not None:
                return int(self.eos_id)
            return int(gt_gen[-1].item())

        self.pred_tokens_list[idx].append(int(tt_pred_token))
        forced = int(gt_gen[cursor].item())
        self.cursor_list[idx] = cursor + 1
        return forced

    def compute_accuracy(self, user_idx: int = 0) -> Dict[str, float]:
        """
        Accuracy vs top5_tokens for user *user_idx*:
          For generated step i:
            - sequence position = tf_prompt_len + i
            - top-1 token = top5_tokens[pos][0]
            - top-5 tokens = top5_tokens[pos][:]
        """
        idx = self.checked_user_idx(user_idx)
        preds = self.pred_tokens_list[idx]
        gt_gen = self.gt_gen_1d_list[idx]
        t5 = self.top5_tokens_list[idx]
        tfl = self.tf_prompt_len_list[idx]

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
