# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Latent + text-embedding dataset over the precomputed cache."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentEmbedDataset(Dataset):
    def __init__(self, cache_dir: str, indices: list[int]):
        self.samples_dir = Path(cache_dir) / "samples"
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        data = torch.load(self.samples_dir / f"sample_{idx:04d}.pt", weights_only=False)
        return {"latent": data["latent"], "caption": data["caption"], "idx": idx}


def make_collate_fn(embeds: dict[str, torch.Tensor], p_drop: float, seed: int = 0):
    rng = random.Random(seed)

    def collate(batch: list[dict]) -> dict:
        latents, text_embeds, caps, idxs = [], [], [], []
        for item in batch:
            cap = "" if rng.random() < p_drop else item["caption"]
            if cap not in embeds:
                raise KeyError(f"missing precomputed embed for caption {cap!r}")
            latents.append(item["latent"])
            text_embeds.append(embeds[cap])
            caps.append(cap)
            idxs.append(item["idx"])
        return {
            "latent": torch.stack(latents, 0),
            "text_embed": torch.stack(text_embeds, 0),
            "captions": caps,
            "idx": idxs,
        }

    return collate
