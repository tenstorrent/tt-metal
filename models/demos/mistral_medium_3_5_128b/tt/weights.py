# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Weight sources the device model is built from, streamed one layer at a time.

Every source yields host tensors in ReferenceModel naming (``layers.{i}.self_attn.q_proj.weight`` with the
``layers.{i}.`` prefix stripped, ``embed_tokens.weight``, ``norm.weight``, ``lm_head.weight``) in HF
layout; the modules do their own sharding / permutation. An empty dict / None from a source means "load
this module from the tilized tensor cache".

* ``StateDictWeights``  in-memory dict (random weights for the PCC tests);
* ``CheckpointWeights`` the real HF checkpoint: per-layer safetensors walk + per-tensor fp8 dequant
  (``reference/checkpoint.py``), read a few layers ahead;
* ``CacheOnlyWeights``  nothing from the checkpoint: every module loads its tilized cache file.

The tilized cache (``ttnn.as_tensor(cache_file_name=...)``) lives outside the repo, under
``$MISTRAL_TT_CACHE`` or ttnn's model cache dir, keyed by checkpoint + mesh; ``COMPLETE`` marks a cache
that a full build finished writing, which is what permits cache-only loading.
"""

import hashlib
import os
from pathlib import Path

from ..reference.checkpoint import CheckpointReader

CACHE_COMPLETE_MARKER = "COMPLETE"
# Bump when a module changes how it lays a weight out on device (sharding, permutation, padding), so an
# old tilized cache is never loaded into the new layout.
CACHE_FORMAT_VERSION = 1


class StateDictWeights:
    def __init__(self, state_dict: dict):
        self.sd = state_dict

    def iter_layers(self, indices):
        for i in indices:
            prefix = f"layers.{i}."
            yield i, {k[len(prefix) :]: v for k, v in self.sd.items() if k.startswith(prefix)}

    def embedding(self):
        return self.sd.get("embed_tokens.weight")

    def final_norm(self):
        return self.sd.get("norm.weight")

    def lm_head(self):
        return self.sd.get("lm_head.weight")


class CheckpointWeights:
    def __init__(self, checkpoint_dir, prefetch: int = 3):
        self.reader = CheckpointReader(checkpoint_dir)
        self.prefetch = prefetch

    def iter_layers(self, indices):
        yield from self.reader.iter_layers(indices, prefetch=self.prefetch)

    def embedding(self):
        return self.reader.embedding()

    def final_norm(self):
        return self.reader.final_norm()

    def lm_head(self):
        return self.reader.lm_head()


class CacheOnlyWeights:
    def iter_layers(self, indices):
        for i in indices:
            yield i, {}

    def embedding(self):
        return None

    def final_norm(self):
        return None

    def lm_head(self):
        return None


def weight_cache_dir(checkpoint_dir, mesh_shape) -> Path:
    """Tilized-cache directory for this checkpoint on this mesh (outside the repo)."""
    root = os.environ.get("MISTRAL_TT_CACHE")
    if not root:
        import ttnn

        root = Path(ttnn.CONFIG.model_cache_path) / "mistral_medium_3_5_128b"
    ckpt = Path(checkpoint_dir)
    digest = hashlib.sha1((ckpt / "model.safetensors.index.json").read_bytes()).hexdigest()[:10]
    return Path(root) / f"{ckpt.name}_{digest}_mesh{mesh_shape[0]}x{mesh_shape[1]}_v{CACHE_FORMAT_VERSION}"


def cache_is_complete(cache_dir, tag: str) -> bool:
    return (Path(cache_dir) / f"{CACHE_COMPLETE_MARKER}_{tag}").is_file()


def mark_cache_complete(cache_dir, tag: str) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / f"{CACHE_COMPLETE_MARKER}_{tag}").write_text("ok\n")


def select_weights(checkpoint_dir, cache_dir, tag: str):
    """Cache-only when a finished build marked the cache complete (unless MISTRAL_FORCE_LOAD_WEIGHTS=1),
    else the checkpoint (which also (re)populates the cache)."""
    if os.environ.get("MISTRAL_FORCE_LOAD_WEIGHTS") != "1" and cache_is_complete(cache_dir, tag):
        return CacheOnlyWeights(), True
    return CheckpointWeights(checkpoint_dir), False
