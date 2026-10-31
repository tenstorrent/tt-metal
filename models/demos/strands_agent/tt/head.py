# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import load_file


@dataclass
class StrandsDeciderConfig:
    pointer_dim: int = 768
    temperature: float = 1.0
    temperature_by_kind: Dict[str, float] = field(default_factory=dict)
    ordinal_smoothing: float = 0.0
    max_length: int = 4096
    num_slots: int = 24
    head_type: str = "pointer"


def load_config(ckpt_dir) -> StrandsDeciderConfig:
    with open(Path(ckpt_dir) / "strands_decider_config.json") as f:
        raw = json.load(f)
    names = StrandsDeciderConfig.__dataclass_fields__
    return StrandsDeciderConfig(**{k: v for k, v in raw.items() if k in names})


class PointerHead(torch.nn.Module):
    """Option k is scored from its own last-token hidden state against the <answer> position (fp32)."""

    def __init__(self, hidden_size: int, pointer_dim: int = 768):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_size)
        self.q = torch.nn.Linear(hidden_size, pointer_dim)
        self.k = torch.nn.Linear(hidden_size, pointer_dim)
        self.scale = pointer_dim**-0.5

    def forward(self, query: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """query [B, d], options [B, K, d] -> logits [B, K]."""
        q = self.q(self.norm(query)).unsqueeze(-1)
        k = self.k(self.norm(options))
        return (k @ q).squeeze(-1) * self.scale

    def forward_flat(self, rows: list[torch.Tensor]) -> list[torch.Tensor]:
        """Logits of many questions in one pass: rows[i] is [1 + K_i, d] (query row, then option rows)."""
        if len(rows) == 1:  # nothing to batch
            return [self(rows[0][:1], rows[0][None, 1:])[0]]
        sizes = [r.shape[0] for r in rows]
        allr = self.norm(torch.cat(rows))
        starts = torch.tensor([0, *sizes[:-1]]).cumsum(0)
        is_query = torch.zeros(allr.shape[0], dtype=torch.bool)
        is_query[starts] = True
        q = self.q(allr[starts])  # [N, pointer_dim]
        k = self.k(allr[~is_query])  # [sum K_i, pointer_dim]
        owner = torch.repeat_interleave(torch.arange(len(rows)), torch.tensor(sizes) - 1)
        logits = (k * q[owner]).sum(-1) * self.scale
        return list(logits.split([n - 1 for n in sizes]))


def load_head(ckpt_dir, hidden: int) -> PointerHead:
    ckpt_dir = Path(ckpt_dir)
    head = PointerHead(hidden, load_config(ckpt_dir).pointer_dim).to(torch.float32)
    head.load_state_dict(load_file(str(ckpt_dir / "head.safetensors")), strict=True)
    return head.eval()
