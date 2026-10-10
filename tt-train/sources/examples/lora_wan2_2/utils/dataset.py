# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
from pathlib import Path

import ml_dtypes
import numpy as np


class LatentEmbedDataset:
    def __init__(self, cache_dir: str, indices: list[int]):
        cache = Path(cache_dir)
        self.samples_dir = cache / "samples"
        self.indices = list(indices)
        meta = json.loads((cache / "metadata.json").read_text())
        self.captions = {m["idx"]: m["caption"] for m in meta}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        latent = np.load(self.samples_dir / f"sample_{idx:04d}.npy")
        return {"latent": latent, "caption": self.captions[idx], "idx": idx}


class TextEmbeds:
    def __init__(self, cache_dir: str):
        cache = Path(cache_dir)
        self.table = np.load(cache / "embeds.npy").view(ml_dtypes.bfloat16)
        self.index = json.loads((cache / "embeds_index.json").read_text())

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, caption: str) -> bool:
        return caption in self.index

    def __getitem__(self, caption: str) -> np.ndarray:
        return self.table[self.index[caption]]


def make_collate_fn(embeds: TextEmbeds, p_drop: float, seed: int = 0):
    rng = random.Random(seed)

    def collate(examples: list[dict]) -> dict:
        latents, text_embeds, caps, idxs = [], [], [], []
        for item in examples:
            cap = "" if rng.random() < p_drop else item["caption"]
            if cap not in embeds:
                raise KeyError(f"missing precomputed embed for caption {cap!r}")
            latents.append(item["latent"])
            text_embeds.append(embeds[cap])
            caps.append(cap)
            idxs.append(item["idx"])
        return {
            "latent": np.stack(latents, 0).astype(np.float32),
            "text_embed": np.stack(text_embeds, 0).astype(np.float32),
            "captions": caps,
            "idx": idxs,
        }

    return collate
