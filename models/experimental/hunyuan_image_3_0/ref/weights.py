# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Safetensors weight loading helpers."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

# HuggingFace model repos (``hf download <repo>`` → ~/.cache/huggingface/hub/…).
HF_REPO_BASE = "tencent/HunyuanImage-3.0"
HF_REPO_INSTRUCT = "tencent/HunyuanImage-3.0-Instruct"
HF_REPO_INSTRUCT_DISTIL = "tencent/HunyuanImage-3.0-Instruct-Distil"

ENV_BASE = "HUNYUAN_MODEL_DIR"
ENV_INSTRUCT = "HUNYUAN_INSTRUCT_MODEL_DIR"
ENV_INSTRUCT_DISTIL = "HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR"

_WEIGHT_INDEX = "model.safetensors.index.json"


def _hub_cache_dir() -> Path:
    if cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(cache)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_snapshots_dir(repo_id: str) -> Path:
    return _hub_cache_dir() / f"models--{repo_id.replace('/', '--')}" / "snapshots"


def _weight_shard_names(model_dir: Path) -> list[str]:
    """Unique safetensor shard filenames listed in ``model.safetensors.index.json``."""
    index_path = model_dir / _WEIGHT_INDEX
    if not index_path.is_file():
        return []
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    return sorted(set(weight_map.values()))


def _shard_exists(model_dir: Path, shard: str) -> bool:
    """True when the shard file is present and its blob target resolves (HF uses symlinks)."""
    path = model_dir / shard
    if not path.is_file():
        return False
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return resolved.is_file() and resolved.stat().st_size > 0


def missing_weight_shards(model_dir: Path) -> list[str]:
    """Shard filenames from the index that are absent or broken under ``model_dir``."""
    return [s for s in _weight_shard_names(model_dir) if not _shard_exists(model_dir, s)]


def is_checkpoint_complete(model_dir: Path) -> bool:
    """True when the index exists and every referenced safetensor shard is on disk."""
    if not (model_dir / _WEIGHT_INDEX).is_file():
        return False
    return len(missing_weight_shards(model_dir)) == 0


def find_hf_snapshot(repo_id: str) -> Path | None:
    """Return the newest complete hub snapshot (index + all safetensor shards)."""
    snaps = _repo_snapshots_dir(repo_id)
    if not snaps.is_dir():
        return None
    candidates = [snap for snap in snaps.iterdir() if snap.is_dir() and is_checkpoint_complete(snap)]
    if not candidates:
        # Fall back to newest snapshot with an index so ensure_checkpoint can resume it.
        partial = [snap for snap in snaps.iterdir() if snap.is_dir() and (snap / _WEIGHT_INDEX).is_file()]
        if not partial:
            return None
        return max(partial, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def has_weights(model_dir: Path) -> bool:
    return is_checkpoint_complete(model_dir)


def _hf_download(repo_id: str) -> None:
    """Download ``repo_id`` into the HF hub cache (``snapshot_download`` / ``hf download``)."""
    if os.environ.get("HY_SKIP_WEIGHT_DOWNLOAD", "0") == "1":
        raise FileNotFoundError(f"Checkpoint for {repo_id!r} not found and HY_SKIP_WEIGHT_DOWNLOAD=1")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None  # type: ignore[misc, assignment]

    if snapshot_download is not None:
        print(f"[weights] downloading {repo_id} via huggingface_hub → {_hub_cache_dir()} ...", flush=True)
        snapshot_download(repo_id)
        return

    hf = shutil.which("hf") or shutil.which("huggingface-cli")
    if hf is None:
        raise RuntimeError(f"Install huggingface_hub or put `hf` on PATH to download {repo_id!r}")

    print(f"[weights] downloading {repo_id} → {_hub_cache_dir()} ...", flush=True)
    proc = subprocess.run([hf, "download", repo_id])
    if proc.returncode == 0:
        return
    # ``hf download`` often prints "Download complete" then exits 1 (click Exit quirk).
    snap = find_hf_snapshot(repo_id)
    if snap is not None and is_checkpoint_complete(snap):
        print(
            f"[weights] hf exited {proc.returncode} but checkpoint is complete at {snap}",
            flush=True,
        )
        return
    raise subprocess.CalledProcessError(proc.returncode, proc.args)


def resolve_checkpoint(*, env_var: str, repo_id: str) -> Path:
    """Resolve checkpoint dir: env override, then newest HF hub snapshot."""
    if override := os.environ.get(env_var):
        return Path(override)
    if snap := find_hf_snapshot(repo_id):
        return snap
    raise FileNotFoundError(f"No checkpoint for {repo_id!r}. Set {env_var} or run: hf download {repo_id}")


def resolve_base_model_dir() -> Path:
    """Resolve base VAE/DiT weights: ``HUNYUAN_MODEL_DIR``, then HF base, then instruct fallbacks."""
    if override := os.environ.get(ENV_BASE):
        return Path(override)
    if snap := find_hf_snapshot(HF_REPO_BASE):
        return snap
    for env_var, repo_id in (
        (ENV_INSTRUCT_DISTIL, HF_REPO_INSTRUCT_DISTIL),
        (ENV_INSTRUCT, HF_REPO_INSTRUCT),
    ):
        if override := os.environ.get(env_var):
            return Path(override)
        if snap := find_hf_snapshot(repo_id):
            return snap
    raise FileNotFoundError(
        f"No checkpoint for {HF_REPO_BASE!r}. Set {ENV_BASE} (or {ENV_INSTRUCT_MODEL_DIR} / "
        f"{ENV_INSTRUCT_DISTIL_MODEL_DIR}) or run: hf download {HF_REPO_BASE}"
    )


def ensure_checkpoint(*, env_var: str, repo_id: str) -> Path:
    """Like ``resolve_checkpoint``, downloading from HuggingFace when missing or incomplete."""
    path: Path | None = None
    try:
        path = resolve_checkpoint(env_var=env_var, repo_id=repo_id)
        if is_checkpoint_complete(path):
            return path
        missing = missing_weight_shards(path)
        print(
            f"[weights] incomplete checkpoint at {path}: "
            f"missing {len(missing)} shard(s) (e.g. {missing[:3]}). Resuming download ...",
            flush=True,
        )
    except FileNotFoundError:
        path = None

    _hf_download(repo_id)
    path = resolve_checkpoint(env_var=env_var, repo_id=repo_id)
    missing = missing_weight_shards(path)
    if missing:
        raise RuntimeError(
            f"Download finished but {len(missing)} weight shard(s) still missing under {path} "
            f"(e.g. {missing[:5]}). Re-run: hf download {repo_id}"
        )
    print(f"[weights] using {path}", flush=True)
    return path


def ensure_base_weights() -> Path:
    return ensure_checkpoint(env_var=ENV_BASE, repo_id=HF_REPO_BASE)


def ensure_instruct_weights() -> Path:
    return ensure_checkpoint(env_var=ENV_INSTRUCT, repo_id=HF_REPO_INSTRUCT)


def ensure_instruct_distil_weights() -> Path:
    return ensure_checkpoint(env_var=ENV_INSTRUCT_DISTIL, repo_id=HF_REPO_INSTRUCT_DISTIL)


def has_distil_weights(model_dir: Path | None = None) -> bool:
    if model_dir is not None:
        return has_weights(model_dir)
    try:
        return has_weights(resolve_checkpoint(env_var=ENV_INSTRUCT_DISTIL, repo_id=HF_REPO_INSTRUCT_DISTIL))
    except FileNotFoundError:
        return False


def __getattr__(name: str):
    """Lazy paths: env override, HF hub snapshot, or placeholder (for skip-if-missing tests)."""
    _mapping = {
        "MODEL_DIR": resolve_base_model_dir,
        "INSTRUCT_MODEL_DIR": lambda: resolve_checkpoint(env_var=ENV_INSTRUCT, repo_id=HF_REPO_INSTRUCT),
        "INSTRUCT_DISTIL_MODEL_DIR": lambda: resolve_checkpoint(
            env_var=ENV_INSTRUCT_DISTIL, repo_id=HF_REPO_INSTRUCT_DISTIL
        ),
    }
    if name in _mapping:
        return _mapping[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_tensors(model_dir: Path, keys: list[str]) -> dict[str, Tensor]:
    index_path = model_dir / _WEIGHT_INDEX
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    shard_to_keys: dict[str, list[str]] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"{key} not found in {index_path}")
        shard_to_keys.setdefault(weight_map[key], []).append(key)

    tensors: dict[str, Tensor] = {}
    for shard_file, shard_keys in shard_to_keys.items():
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing weight shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in shard_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def load_prefixed_state_dict(model_dir: Path, prefix: str, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
    index_path = model_dir / _WEIGHT_INDEX
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    keys = [k for k in weight_map if k.startswith(prefix)]
    if not keys:
        raise RuntimeError(f"No keys with prefix {prefix!r} in {index_path}")

    tensors = load_tensors(model_dir, keys)
    strip = len(prefix)
    return {k[strip:]: v.to(dtype) for k, v in tensors.items()}
