# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Download only the HF checkpoint shards that hold chosen decoder layers.

Imports neither ttnn nor pytest, so scripts can call it before their device and path guards run.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path

from loguru import logger

ALLOW_PATTERNS = (
    "config.json",
    "*.safetensors.index.json",
    "generation_config.json",
    "tokenizer*",
    "tiktoken*",  # Kimi ships its BBPE tokenizer as tiktoken.model
    "configuration_deepseek.py",
    "modeling_deepseek.py",
    "*.py",  # custom model code for trust_remote_code=True
)


def required_shards(weight_map: dict, layers: Iterable[int], embed: bool = True) -> list[str]:
    """Shard files holding any tensor of `layers`, plus the embedding when `embed` is set."""
    shards = set()
    if embed:
        shards |= {shard for key, shard in weight_map.items() if "embed_tokens" in key}
    for n in layers:
        shards |= {shard for key, shard in weight_map.items() if f"model.layers.{n}." in key}
    return sorted(shards)


def shard_patterns(shards: Iterable[str]) -> list[str]:
    """allow_patterns for shards named like ``model-00001-of-000163.safetensors``."""
    return [f"*-{shard.split('-')[1]}-of-*.safetensors" for shard in sorted(shards)]


def download_layers(
    repo_id: str,
    cache_dir: Path,
    layers: Iterable[int],
    embed: bool = True,
    on_plan: Callable[[Path, list[str]], None] | None = None,
) -> Path:
    """Download config, tokenizer, custom code and the shards for `layers` (+ embedding) into the HF
    hub cache at `cache_dir`; return the snapshot dir.

    `on_plan(snapshot_dir, shards)` runs after the index is fetched and before any shard is; it may
    raise to stop the download.
    """
    from huggingface_hub import snapshot_download

    layers = list(layers)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step 1/2: downloading config and index files of {repo_id} into {cache_dir}")
    index_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        allow_patterns=list(ALLOW_PATTERNS),
        ignore_patterns=["*.safetensors"],
    )
    logger.info(f"✓ Configuration downloaded to: {index_dir}")

    with open(Path(index_dir) / "model.safetensors.index.json", "r") as f:
        weight_map = json.load(f).get("weight_map", {})
    shards = required_shards(weight_map, layers, embed)

    contiguous = bool(layers) and layers == list(range(layers[0], layers[-1] + 1))
    span = f"{layers[0]}..{layers[-1]}" if contiguous else str(layers)
    logger.info(f"Step 2/2: downloading weight shards for layers {span}{' + embeddings' if embed else ''}")
    logger.info(f"Required shards: {len(shards)} files ({', '.join(shards)})")
    if on_plan is not None:
        on_plan(Path(index_dir), shards)

    model_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        allow_patterns=list(ALLOW_PATTERNS) + shard_patterns(shards),
    )
    logger.success("✓ Model weights downloaded successfully!")
    logger.info(f"Model location: {model_dir}")
    return Path(model_dir)


def flat_config_dir(snapshot: Path, dest_root: Path, repo_id: str) -> Path:
    """Copy the non-weight files of `snapshot` as real files into a dot-free dir under `dest_root`.

    Hub snapshots hold symlinks into blobs/; trust_remote_code follows them and then looks for the
    remote code's relative imports by name inside blobs/, where they do not exist. It also cannot
    import a module from a dir whose name contains a '.'.
    """
    import shutil

    dest = Path(dest_root) / repo_id.replace("/", "--").replace(".", "_")
    shutil.copytree(snapshot, dest, symlinks=False, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.safetensors"))
    return dest


def resolve_hf_snapshot_dir(path: Path) -> Path:
    """If `path` is an HF hub-cache repo root (``models--org--name/`` with a ``snapshots/`` subdir),
    return the active snapshot dir (the ``refs/main`` commit, else the newest snapshot that has the
    safetensors index) so callers see the real config.json + shards. Otherwise return `path` as-is.

    Lets ``*_HF_MODEL`` point at either the hub root (``.../hub/models--zai-org--GLM-5.3``) or a plain
    checkout dir. The hash snapshot dir also sidesteps the trust_remote_code dot-in-path import issue.
    """
    if (path / "model.safetensors.index.json").exists():
        return path
    snaps = path / "snapshots"
    if snaps.is_dir():
        ref = path / "refs" / "main"
        if ref.is_file():
            cand = snaps / ref.read_text().strip()
            if (cand / "model.safetensors.index.json").exists():
                return cand
        cands = [d for d in snaps.iterdir() if d.is_dir() and (d / "model.safetensors.index.json").exists()]
        if cands:
            return max(cands, key=lambda d: d.stat().st_mtime)
    return path
