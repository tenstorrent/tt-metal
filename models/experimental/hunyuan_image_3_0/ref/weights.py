# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Safetensors weight loading helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor

from models.experimental.hunyuan_image_3_0.ref.safe_paths import safe_join

# HuggingFace model repos (``hf download <repo>`` → ~/.cache/huggingface/hub/…).
HF_REPO_BASE = "tencent/HunyuanImage-3.0"
HF_REPO_INSTRUCT = "tencent/HunyuanImage-3.0-Instruct"
HF_REPO_INSTRUCT_DISTIL = "tencent/HunyuanImage-3.0-Instruct-Distil"

# Only these three repos may be handed to `hf download` / snapshot_download.
_ALLOWED_REPOS = frozenset({HF_REPO_BASE, HF_REPO_INSTRUCT, HF_REPO_INSTRUCT_DISTIL})

ENV_BASE = "HUNYUAN_MODEL_DIR"
ENV_INSTRUCT = "HUNYUAN_INSTRUCT_MODEL_DIR"
ENV_INSTRUCT_DISTIL = "HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR"

_WEIGHT_INDEX = "model.safetensors.index.json"


def _checked_repo_id(repo_id: str) -> str:
    """Reject any repo id outside the hardcoded HunyuanImage-3.0 safelist."""
    if repo_id not in _ALLOWED_REPOS:
        raise ValueError(f"refusing to download unknown repo {repo_id!r}; expected one of {sorted(_ALLOWED_REPOS)}")
    return repo_id


def _index_path(model_dir: Path) -> Path:
    """``model_dir/model.safetensors.index.json``, pinned inside ``model_dir``."""
    return safe_join(model_dir, _WEIGHT_INDEX)


def _read_weight_index(model_dir: Path) -> tuple[Path, dict[str, str]]:
    """Read the checkpoint's weight index → ``(index_path, weight_map)``.

    The only place in this module that opens the index. The join and the containment
    check are written out here rather than delegated, so the path reaching ``open`` is
    visibly constrained to the absolute checkpoint directory at the call site.
    """
    base_dir = os.path.abspath(str(model_dir))
    index_path = os.path.abspath(os.path.join(base_dir, _WEIGHT_INDEX))
    if not index_path.startswith(base_dir + os.sep):
        raise ValueError(f"refusing path {index_path!r}: outside checkpoint directory {base_dir!r}")
    with open(index_path) as f:
        return Path(index_path), json.load(f)["weight_map"]


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
    if not _index_path(model_dir).is_file():
        return []
    _, weight_map = _read_weight_index(model_dir)
    return sorted(set(weight_map.values()))


def _shard_exists(model_dir: Path, shard: str) -> bool:
    """True when the shard file is present and its blob target resolves (HF uses symlinks)."""
    # ``shard`` comes out of the checkpoint's index JSON — pin it under model_dir.
    try:
        path = safe_join(model_dir, shard)
    except ValueError:
        return False
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
    if not _index_path(model_dir).is_file():
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
        partial = [snap for snap in snaps.iterdir() if snap.is_dir() and _index_path(snap).is_file()]
        if not partial:
            return None
        return max(partial, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def has_weights(model_dir: Path) -> bool:
    return is_checkpoint_complete(model_dir)


def _downloads_disabled() -> bool:
    """True when weight downloads must not be attempted (offline CI / explicit skip).

    The test pipelines run with ``HF_HUB_OFFLINE=1`` and pre-staged weights, so a
    missing/incomplete checkpoint should fail fast with a clear message rather than
    a doomed ``snapshot_download`` that only dies with an ``OfflineModeIsEnabled``
    traceback. Weights are downloaded once, out of band, into ``HUNYUAN_MODEL_DIR``.
    """
    if os.environ.get("HY_SKIP_WEIGHT_DOWNLOAD", "0") == "1":
        return True
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        val = os.environ.get(var, "").strip().lower()
        if val and val not in ("0", "false", "no"):
            return True
    return False


def _hf_download(repo_id: str, local_dir: Path | None = None) -> None:
    """Download ``repo_id`` with ``huggingface_hub.snapshot_download``.

    When ``local_dir`` is set (e.g. ``HUNYUAN_MODEL_DIR``), files land there so CI
    paths stay stable across runs. Otherwise the HF hub cache under ``HF_HOME`` is used.

    In-process only — no ``hf download`` subprocess. The ``hf`` / ``huggingface-cli``
    executables are console-script entry points of ``huggingface_hub`` itself, so a
    CLI fallback could never run in an environment where this import fails; it was
    unreachable code. ``repo_id`` is still safelisted so only the three known
    HunyuanImage-3.0 repos can be fetched.
    """
    repo_id = _checked_repo_id(repo_id)
    if os.environ.get("HY_SKIP_WEIGHT_DOWNLOAD", "0") == "1":
        where = f" (expected at {local_dir})" if local_dir is not None else ""
        raise FileNotFoundError(f"Checkpoint for {repo_id!r} not found{where} and HY_SKIP_WEIGHT_DOWNLOAD=1")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            f"huggingface_hub is required to download {repo_id!r}; install it "
            f"(pip install huggingface_hub) or pre-stage the checkpoint and set the model-dir env var"
        ) from exc

    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[weights] downloading {repo_id} via huggingface_hub → {local_dir} ...", flush=True)
        snapshot_download(repo_id, local_dir=str(local_dir))
    else:
        print(f"[weights] downloading {repo_id} via huggingface_hub → {_hub_cache_dir()} ...", flush=True)
        snapshot_download(repo_id)


def resolve_checkpoint(*, env_var: str, repo_id: str) -> Path:
    """Lookup-only: env override, then newest HF hub snapshot (may be incomplete).

    Prefer ``resolve_base_model_dir`` / ``resolve_instruct_model_dir`` /
    ``resolve_instruct_distil_model_dir`` (or ``ensure_*``) when missing weights
    should auto-download into the HF hub cache.
    """
    if override := os.environ.get(env_var):
        return Path(override)
    if snap := find_hf_snapshot(repo_id):
        return snap
    raise FileNotFoundError(f"No checkpoint for {repo_id!r}. Set {env_var} or run: hf download {repo_id}")


def _resolve_complete_or_ensure(*, env_var: str, repo_id: str, ensure_fn) -> Path:
    """Return a complete checkpoint, downloading to hub cache (or ``env_var``) if needed."""
    if override := os.environ.get(env_var):
        path = Path(override)
        if is_checkpoint_complete(path):
            return path
        return ensure_fn()
    if snap := find_hf_snapshot(repo_id):
        if is_checkpoint_complete(snap):
            return snap
    return ensure_fn()


def resolve_base_model_dir() -> Path:
    """Resolve base VAE/DiT weights: ``HUNYUAN_MODEL_DIR``, then HF base, then instruct fallbacks.

    If nothing local is found, downloads the base repo into the HF hub cache
    (``~/.cache/huggingface/hub``, or ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE``).
    Set ``HUNYUAN_MODEL_DIR`` to download/reuse a fixed directory instead.
    """
    if override := os.environ.get(ENV_BASE):
        path = Path(override)
        if is_checkpoint_complete(path):
            return path
        return ensure_base_weights()
    if snap := find_hf_snapshot(HF_REPO_BASE):
        if is_checkpoint_complete(snap):
            return snap
    for env_var, repo_id in (
        (ENV_INSTRUCT_DISTIL, HF_REPO_INSTRUCT_DISTIL),
        (ENV_INSTRUCT, HF_REPO_INSTRUCT),
    ):
        if override := os.environ.get(env_var):
            path = Path(override)
            if is_checkpoint_complete(path):
                return path
        if snap := find_hf_snapshot(repo_id):
            if is_checkpoint_complete(snap):
                return snap
    return ensure_base_weights()


def try_find_instruct_model_dir() -> Path | None:
    """Return Instruct checkpoint if a local index exists; never download.

    Prefer this over ``INSTRUCT_MODEL_DIR`` / ``resolve_instruct_model_dir`` when a
    caller only wants Instruct *if already staged* (e.g. I2I helpers that fall back
    to base weights). Offline CI that only stages base must not raise here.
    """
    if override := os.environ.get(ENV_INSTRUCT):
        path = Path(override)
        if _index_path(path).is_file():
            return path
        return None
    snap = find_hf_snapshot(HF_REPO_INSTRUCT)
    if snap is not None and _index_path(snap).is_file():
        return snap
    return None


def resolve_instruct_model_dir() -> Path:
    """Resolve Instruct weights; download to HF hub cache when missing/incomplete."""
    return _resolve_complete_or_ensure(
        env_var=ENV_INSTRUCT, repo_id=HF_REPO_INSTRUCT, ensure_fn=ensure_instruct_weights
    )


def resolve_instruct_distil_model_dir() -> Path:
    """Resolve Instruct-Distil weights; download to HF hub cache when missing/incomplete."""
    return _resolve_complete_or_ensure(
        env_var=ENV_INSTRUCT_DISTIL,
        repo_id=HF_REPO_INSTRUCT_DISTIL,
        ensure_fn=ensure_instruct_distil_weights,
    )


def ensure_checkpoint(*, env_var: str, repo_id: str) -> Path:
    """Like ``resolve_checkpoint``, downloading from HuggingFace when missing or incomplete.

    If ``env_var`` (e.g. ``HUNYUAN_MODEL_DIR``) is set, the snapshot is written there so
    the first CI run can populate the shared path and later runs reuse it.
    """
    try:
        path = resolve_checkpoint(env_var=env_var, repo_id=repo_id)
        if is_checkpoint_complete(path):
            print(f"[weights] using {path}", flush=True)
            return path
        if not _index_path(path).is_file():
            reason = f"incomplete checkpoint at {path}: missing {_WEIGHT_INDEX}"
        else:
            missing = missing_weight_shards(path)
            reason = f"incomplete checkpoint at {path}: missing {len(missing)} shard(s) (e.g. {missing[:3]})"
    except FileNotFoundError:
        path = None
        reason = f"no local checkpoint for {repo_id!r}"

    # Weights are missing/incomplete. When downloads are disabled (offline CI or
    # HY_SKIP_WEIGHT_DOWNLOAD=1), fail fast at the start with an actionable message
    # instead of attempting a snapshot_download that can only die with a confusing
    # OfflineModeIsEnabled traceback. Weights must be pre-staged (see _downloads_disabled).
    if _downloads_disabled():
        target = os.environ.get(env_var) or "the HF hub cache"
        raise FileNotFoundError(
            f"[weights] {reason}; downloads are disabled (HF_HUB_OFFLINE / "
            f"HY_SKIP_WEIGHT_DOWNLOAD). Stage the weights before running, e.g.: "
            f"hf download {repo_id} --local-dir {target}"
        )

    print(f"[weights] {reason}. Downloading ...", flush=True)

    # Prefer the env override path (CI / HUNYUAN_*_MODEL_DIR); else HF hub cache.
    local_dir = Path(os.environ[env_var]) if os.environ.get(env_var) else None
    _hf_download(repo_id, local_dir=local_dir)

    path = resolve_checkpoint(env_var=env_var, repo_id=repo_id)
    if is_checkpoint_complete(path):
        print(f"[weights] using {path}", flush=True)
        return path

    # Env override still empty but hub cache got the snapshot (e.g. read-only local_dir).
    if local_dir is not None:
        snap = find_hf_snapshot(repo_id)
        if snap is not None and is_checkpoint_complete(snap):
            print(f"[weights] env path still incomplete; using hub snapshot {snap}", flush=True)
            return snap

    missing = missing_weight_shards(path)
    index_ok = _index_path(path).is_file()
    hint = f" --local-dir {local_dir}" if local_dir is not None else ""
    raise RuntimeError(
        f"Download finished but checkpoint still incomplete under {path}: "
        f"index={'ok' if index_ok else 'MISSING'}, "
        f"missing {len(missing)} shard(s) (e.g. {missing[:5]}). "
        f"Re-run: hf download {repo_id}{hint}"
    )


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
    """Lazy paths: env / hub snapshot, downloading into the HF hub cache when missing.

    Import-time ``MODEL_DIR`` / ``INSTRUCT_*`` (e.g. from ``ref.vae.decoder``) auto-download
    unless ``HY_SKIP_WEIGHT_DOWNLOAD=1``.
    """
    _mapping = {
        "MODEL_DIR": resolve_base_model_dir,
        "INSTRUCT_MODEL_DIR": resolve_instruct_model_dir,
        "INSTRUCT_DISTIL_MODEL_DIR": resolve_instruct_distil_model_dir,
    }
    if name in _mapping:
        return _mapping[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def load_tensors(model_dir: Path, keys: list[str]) -> dict[str, Tensor]:
    index_path, weight_map = _read_weight_index(model_dir)

    shard_to_keys: dict[str, list[str]] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"{key} not found in {index_path}")
        shard_to_keys.setdefault(weight_map[key], []).append(key)

    tensors: dict[str, Tensor] = {}
    for shard_file, shard_keys in shard_to_keys.items():
        # Shard names come from the index JSON; pin them under model_dir.
        shard_path = safe_join(model_dir, shard_file)
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing weight shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in shard_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def load_prefixed_state_dict(model_dir: Path, prefix: str, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
    index_path, weight_map = _read_weight_index(model_dir)

    keys = [k for k in weight_map if k.startswith(prefix)]
    if not keys:
        raise RuntimeError(f"No keys with prefix {prefix!r} in {index_path}")

    tensors = load_tensors(model_dir, keys)
    strip = len(prefix)
    return {k[strip:]: v.to(dtype) for k, v in tensors.items()}
