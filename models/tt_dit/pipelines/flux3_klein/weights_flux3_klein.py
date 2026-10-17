# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Resolve a FLUX 3 Image Klein checkpoint directory, optionally fetching it from HuggingFace.

Three variants share one resolver: ``distilled`` (``black-forest-labs/flux-3-image-klein``, the flow-map DiT,
16 steps, no CFG), ``base`` (``...-base``, 50 steps, CFG 3.5) and ``pup`` (``...-pup``, the Apache-2.0 prompt
upsampler). Resolution order matches ``pipelines/minimax_h3/weights_minimax_h3.py``: an explicit path first
(``FLUX3_KLEIN_MODEL_PATH``), then the HuggingFace cache (``HF_HUB_CACHE``, else ``HF_HOME/hub``, else
``~/.cache/huggingface/hub``), then a download -- and the download only happens when it has been opted into
with ``TT_DIT_ALLOW_HF_DOWNLOAD=1``.

The explicit path may be the snapshot itself or, as upstream's ``WEIGHTS_DIR`` layout, a root holding one
directory per repository name (``flux-3-image-klein/``, ``flux-3-image-klein-base/``, ``flux-3-image-klein-pup/``)
or per variant name.

A DiT snapshot holds both the BFL layout the port reads (``model.safetensors``, ``ae.safetensors``) and the
diffusers partitions (``transformer/``, ``vae/``, ``text_encoder/``); the port reads ``model.safetensors`` for the
DiT, ``vae/`` for the Flux2 VAE decoder and ``text_encoder/`` for Qwen3-VL. A download therefore skips the
diffusers ``transformer/`` duplicate and the fp8 / nvfp4 ``variants/``.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

FLUX3_KLEIN_REPO_IDS = {
    "distilled": "black-forest-labs/flux-3-image-klein",
    "base": "black-forest-labs/flux-3-image-klein-base",
    "pup": "black-forest-labs/flux-3-image-klein-pup",
}
FLUX3_KLEIN_VARIANTS = tuple(FLUX3_KLEIN_REPO_IDS)

MODEL_PATH_ENV = "FLUX3_KLEIN_MODEL_PATH"
ALLOW_DOWNLOAD_ENV = "TT_DIT_ALLOW_HF_DOWNLOAD"

# What a complete snapshot of each variant must hold (files or directories, relative to the snapshot).
_DIT_REQUIRED = ("model.safetensors", "ae.safetensors", "text_encoder", "vae")
REQUIRED = {
    "distilled": _DIT_REQUIRED,
    "base": _DIT_REQUIRED,
    "pup": ("model.safetensors", "config.json", "tokenizer.json"),
}
# Not fetched: the diffusers duplicate of the DiT and the quantized variants, neither read by the port.
_IGNORE_PATTERNS = {
    "distilled": ("transformer/*", "variants/*", "*.jpg", "assets/*"),
    "base": ("transformer/*", "variants/*", "*.jpg", "assets/*"),
    "pup": (),
}
# Approximate on-disk cost of what is fetched, for the pre-download log line.
_DOWNLOAD_GB = {"distilled": 29.0, "base": 29.0, "pup": 19.0}


class WeightsNotFoundError(FileNotFoundError):
    """Raised when the checkpoint is not on disk and downloads are disabled."""


def _check_variant(variant: str) -> str:
    if variant not in FLUX3_KLEIN_REPO_IDS:
        raise ValueError(f"unknown FLUX 3 Klein variant {variant!r}; expected one of {FLUX3_KLEIN_VARIANTS}")
    return variant


def repo_id(variant: str) -> str:
    return FLUX3_KLEIN_REPO_IDS[_check_variant(variant)]


def repo_name(variant: str) -> str:
    """The repository name without the organization, as upstream names its weight directories."""
    return repo_id(variant).split("/", 1)[1]


def hf_cache_root() -> Path:
    """The ``hub`` directory of the HuggingFace cache: ``HF_HUB_CACHE``, else ``HF_HOME/hub``, else the default."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


def _missing(directory: Path, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not (directory / name).exists()]


def _is_complete(directory: Path, required: tuple[str, ...]) -> bool:
    return directory.is_dir() and not _missing(directory, required)


def cached_snapshot(variant: str, required: tuple[str, ...] | None = None) -> Path | None:
    """A cached snapshot of ``variant`` holding every required entry: the revision ``refs/main`` points at is
    preferred, then the most recently modified one. Incomplete snapshots are passed over."""
    required = REQUIRED[_check_variant(variant)] if required is None else tuple(required)
    repo_dir = hf_cache_root() / f"models--{repo_id(variant).replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted((s for s in snapshots.iterdir() if s.is_dir()), key=lambda s: s.stat().st_mtime, reverse=True)
    main_ref = repo_dir / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text().strip()
        candidates.sort(key=lambda s: s.name != revision)  # stable: refs/main first, then by mtime
    for snapshot in candidates:
        if _is_complete(snapshot, required):
            return snapshot
    return None


def _download(variant: str) -> Path:
    from huggingface_hub import snapshot_download

    logger.info(
        f"Downloading FLUX 3 Klein '{variant}' (~{_DOWNLOAD_GB[variant]:.0f} GB) from HuggingFace: "
        f"{repo_id(variant)} -> {hf_cache_root()}"
    )
    return Path(snapshot_download(repo_id=repo_id(variant), ignore_patterns=list(_IGNORE_PATTERNS[variant]) or None))


def _explicit_dir(explicit: Path, variant: str, required: tuple[str, ...]) -> Path:
    """The snapshot inside an explicit path: the path itself, or its ``<repo name>/`` or ``<variant>/`` child."""
    if not explicit.is_dir():
        raise WeightsNotFoundError(f"{MODEL_PATH_ENV} points at {explicit}, which is not a directory")
    candidates = [explicit, explicit / repo_name(variant), explicit / variant]
    for candidate in candidates:
        if _is_complete(candidate, required):
            return candidate
    missing = _missing(explicit, required)
    raise WeightsNotFoundError(
        f"FLUX 3 Klein '{variant}' checkpoint at {explicit} is missing {missing} "
        f"(also looked in {[c.name for c in candidates[1:]]})"
    )


def resolve_weights_dir(
    variant: str = "distilled",
    *,
    required: tuple[str, ...] | None = None,
    weights_dir: str | os.PathLike | None = None,
    allow_download: bool | None = None,
) -> Path:
    """The snapshot directory of ``variant``.

    ``weights_dir`` (or ``$FLUX3_KLEIN_MODEL_PATH``) wins when it is set; otherwise the HuggingFace cache is
    searched and then, with ``$TT_DIT_ALLOW_HF_DOWNLOAD=1``, fetched. Raises ``WeightsNotFoundError`` when
    nothing resolves and downloads are off. ``required`` narrows the completeness check (a test that only reads
    ``model.safetensors`` may pass ``("model.safetensors",)``).
    """
    variant = _check_variant(variant)
    required = REQUIRED[variant] if required is None else tuple(required)
    if allow_download is None:
        allow_download = os.environ.get(ALLOW_DOWNLOAD_ENV) == "1"

    explicit = weights_dir or os.environ.get(MODEL_PATH_ENV)
    if explicit:
        return _explicit_dir(Path(explicit), variant, required)

    cached = cached_snapshot(variant, required)
    if cached is not None:
        return cached

    if not allow_download:
        raise WeightsNotFoundError(
            f"FLUX 3 Klein '{variant}' weights not found (needed: {list(required)}).\n"
            f"Point {MODEL_PATH_ENV} at the snapshot (or a directory holding {repo_name(variant)}/), or re-run "
            f"with {ALLOW_DOWNLOAD_ENV}=1 to fetch {repo_id(variant)} into {hf_cache_root()}."
        )

    snapshot = _download(variant)
    missing = _missing(snapshot, required)
    if missing:
        raise WeightsNotFoundError(f"downloaded FLUX 3 Klein '{variant}' snapshot at {snapshot} is missing {missing}")
    return snapshot


def dit_checkpoint(weights_dir: str | os.PathLike) -> Path:
    """The BFL-layout DiT file of a resolved DiT snapshot."""
    return Path(weights_dir) / "model.safetensors"


def autoencoder_checkpoint(weights_dir: str | os.PathLike) -> Path:
    """The BFL-layout autoencoder file of a resolved DiT snapshot (the diffusers twin is ``vae/``)."""
    return Path(weights_dir) / "ae.safetensors"
