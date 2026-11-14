# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference file schema for the model-readiness check.

A reference file pins, for one or more prompts, the HuggingFace teacher's
top-K next-token predictions at every generated position. The TT model
under test is then run with teacher forcing against this file and scored
on top-1 / top-5 / top-K hit rate.

On-disk format (single torch.save'd dict):

    {
        "format_version": "readiness_v1",
        "k": int,
        "hf_model_id": str,
        "token_ids_meta": {"bos_id": int|None, "eos_id": int, "pad_id": int|None},
        "entries": [
            {
                "prompt_text": str,
                "prompt_tokens":    Tensor [1, P]   int64,
                "generated_tokens": Tensor [1, G]   int64,
                "topk_tokens":      Tensor [G, K]   int32,
                "tf_prompt_len":    int,            # == P, stored for clarity
            },
            ...
        ],
    }

`topk_tokens[i]` is the teacher's top-K prediction for the token at
position `tf_prompt_len + i`, conditioned on positions `[0 .. tf_prompt_len + i - 1]`.
Top-1 is column 0; top-5 is columns 0..4; top-K is the full row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

FORMAT_VERSION = "readiness_v1"


@dataclass
class ReferenceEntry:
    prompt_text: str
    prompt_tokens: torch.Tensor  # [1, P] int64
    generated_tokens: torch.Tensor  # [1, G] int64
    topk_tokens: torch.Tensor  # [G, K] int32
    tf_prompt_len: int

    def __post_init__(self) -> None:
        if self.prompt_tokens.dim() != 2 or self.prompt_tokens.shape[0] != 1:
            raise ValueError(f"prompt_tokens must be [1, P], got {tuple(self.prompt_tokens.shape)}")
        if self.generated_tokens.dim() != 2 or self.generated_tokens.shape[0] != 1:
            raise ValueError(f"generated_tokens must be [1, G], got {tuple(self.generated_tokens.shape)}")
        if self.topk_tokens.dim() != 2:
            raise ValueError(f"topk_tokens must be [G, K], got {tuple(self.topk_tokens.shape)}")
        if self.topk_tokens.shape[0] != self.generated_tokens.shape[1]:
            raise ValueError(
                f"topk_tokens rows ({self.topk_tokens.shape[0]}) must equal "
                f"generated_tokens length ({self.generated_tokens.shape[1]})"
            )
        if self.tf_prompt_len != self.prompt_tokens.shape[1]:
            raise ValueError(
                f"tf_prompt_len ({self.tf_prompt_len}) must equal prompt length " f"({self.prompt_tokens.shape[1]})"
            )

    @property
    def num_generated(self) -> int:
        return int(self.generated_tokens.shape[1])

    @property
    def k(self) -> int:
        return int(self.topk_tokens.shape[1])


@dataclass
class Reference:
    k: int
    hf_model_id: str
    entries: List[ReferenceEntry]
    token_ids_meta: Dict[str, Optional[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("Reference must contain at least one entry")
        for idx, entry in enumerate(self.entries):
            if entry.k != self.k:
                raise ValueError(f"Entry {idx} has k={entry.k}, expected {self.k} from Reference.k")


def save_reference(reference: Reference, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "k": reference.k,
        "hf_model_id": reference.hf_model_id,
        "token_ids_meta": reference.token_ids_meta,
        "entries": [
            {
                "prompt_text": e.prompt_text,
                "prompt_tokens": e.prompt_tokens.to(torch.int64).cpu().contiguous(),
                "generated_tokens": e.generated_tokens.to(torch.int64).cpu().contiguous(),
                "topk_tokens": e.topk_tokens.to(torch.int32).cpu().contiguous(),
                "tf_prompt_len": int(e.tf_prompt_len),
            }
            for e in reference.entries
        ],
    }
    torch.save(payload, path)
    return path


def load_reference(path: Path | str) -> Reference:
    path = Path(path)
    payload = torch.load(path, weights_only=False)
    fmt = payload.get("format_version")
    if fmt != FORMAT_VERSION:
        raise ValueError(f"Unsupported reference format_version={fmt!r} in {path}; " f"expected {FORMAT_VERSION!r}")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError(f"Reference {path} missing non-empty 'entries' list")
    entries = [
        ReferenceEntry(
            prompt_text=str(e["prompt_text"]),
            prompt_tokens=e["prompt_tokens"],
            generated_tokens=e["generated_tokens"],
            topk_tokens=e["topk_tokens"],
            tf_prompt_len=int(e["tf_prompt_len"]),
        )
        for e in raw_entries
    ]
    return Reference(
        k=int(payload["k"]),
        hf_model_id=str(payload["hf_model_id"]),
        entries=entries,
        token_ids_meta=dict(payload.get("token_ids_meta") or {}),
    )
