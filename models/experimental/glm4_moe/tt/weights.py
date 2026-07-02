# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict


def hf_cache_root() -> Path:
    # huggingface_hub uses HF_HOME if set; otherwise defaults to ~/.cache/huggingface
    return Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))


def model_repo_cache_dir(model_id: str) -> Path:
    # HF cache encodes model ids as "models--org--name"
    return hf_cache_root() / "hub" / f"models--{model_id.replace('/', '--')}"


def snapshot_has_safetensors_weights(snapshot_dir: Path) -> bool:
    if (snapshot_dir / "model.safetensors").is_file():
        return True
    return any(snapshot_dir.glob("model-*.safetensors"))


def resolve_best_effort_snapshot_dir(model_id: str, *, hint_dir: Optional[Path] = None) -> Path:
    """
    Resolve a local snapshot directory for a HuggingFace model id without network access.

    Why not just use huggingface_hub.snapshot_download(..., local_files_only=True)?
    - The HF cache can contain multiple snapshots; the most recent metadata snapshot
      is not guaranteed to include weight shards.
    - For large models, weights are sometimes downloaded at an earlier revision and
      remain present locally even if a newer snapshot is fetched later.
    """
    if hint_dir is not None:
        hint_dir = Path(hint_dir)
        if (hint_dir / "model.safetensors.index.json").is_file() and snapshot_has_safetensors_weights(hint_dir):
            return hint_dir

    repo_dir = model_repo_cache_dir(model_id)
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(
            f"HF cache snapshots dir not found: {snapshots_dir}. "
            f"Is '{model_id}' downloaded under {hf_cache_root()}?"
        )

    candidates = []
    for p in snapshots_dir.iterdir():
        if not p.is_dir():
            continue
        if not (p / "model.safetensors.index.json").is_file():
            continue
        if not snapshot_has_safetensors_weights(p):
            continue
        shard_count = len(list(p.glob("model-*.safetensors")))
        candidates.append((shard_count, p))

    if not candidates:
        raise FileNotFoundError(
            f"No local snapshot found for '{model_id}' with weights. "
            f"Found snapshots under {snapshots_dir}, but none contained safetensors shards."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]
    logger.info(
        "Resolved snapshot dir via local HF cache scan: model_id={} snapshot_dir={} shard_count={}",
        model_id,
        str(best),
        candidates[0][0],
    )
    return best


def iter_index_shard_files(snapshot_dir: Path) -> Iterable[str]:
    snapshot_dir = Path(snapshot_dir)
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    idx = json.loads(index_path.read_text())
    return sorted(set(idx["weight_map"].values()))


def find_missing_shards(snapshot_dir: Path) -> list[str]:
    snapshot_dir = Path(snapshot_dir)
    missing = []
    for filename in iter_index_shard_files(snapshot_dir):
        if not (snapshot_dir / filename).exists():
            missing.append(filename)
    return missing


def load_glm_lazy_state_dict(snapshot_dir: Path, *, num_layers: Optional[int] = None) -> LazyStateDict:
    """
    Return a lazy state dict view over the GLM snapshot.

    If num_layers is provided, filters out weights for layers >= num_layers.
    """
    state = LazyStateDict(Path(snapshot_dir))
    if num_layers is not None:
        state = state.view_with_prefix("", num_layers=num_layers)
    return state
