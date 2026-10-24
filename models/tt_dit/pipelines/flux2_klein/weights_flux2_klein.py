# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Resolve a FLUX.2 Klein diffusers snapshot directory, optionally fetching it from HuggingFace.

Four checkpoints share one resolver, keyed by ``(variant, size)``: the distilled models ``FLUX.2-klein-4B`` /
``FLUX.2-klein-9B`` (4 steps, no CFG; ``model_index.json`` ``is_distilled: true``) and the base models
``FLUX.2-klein-base-4B`` / ``FLUX.2-klein-base-9B`` (50 steps, CFG 4.0), the repository ids of the BFL ``flux2``
``util.py:16-70`` model table. Resolution order mirrors ``pipelines/flux3_klein/weights_flux3_klein.py``: an explicit
path first (the ``weights_dir`` argument, then ``FLUX2_KLEIN_MODEL_PATH``), then the HuggingFace cache
(``HF_HUB_CACHE``, else ``HF_HOME/hub``, else ``~/.cache/huggingface/hub``), then a download -- and the download only
happens when it has been opted into with ``TT_DIT_ALLOW_HF_DOWNLOAD=1``.

The explicit path may be the snapshot itself or a root holding one directory per repository name
(``FLUX.2-klein-4B/``, ``FLUX.2-klein-base-9B/``, ...) or per ``<variant>-<size>`` name (``distilled-4b/``).

A snapshot is the diffusers layout the port reads (``model_index.json``, ``transformer/``, ``text_encoder/``,
``tokenizer/``, ``vae/``, ``scheduler/``); the BFL-layout twin ``flux-2-klein-*.safetensors`` that the same
repository ships is never read, so a download skips it along with the sample images. A snapshot missing any
required entry (a cache that holds only ``LICENSE.md`` / ``README.md``) is passed over, never returned.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

FLUX2_KLEIN_REPO_IDS: dict[tuple[str, str], str] = {
    ("distilled", "4b"): "black-forest-labs/FLUX.2-klein-4B",
    ("distilled", "9b"): "black-forest-labs/FLUX.2-klein-9B",
    ("base", "4b"): "black-forest-labs/FLUX.2-klein-base-4B",
    ("base", "9b"): "black-forest-labs/FLUX.2-klein-base-9B",
}
FLUX2_KLEIN_VARIANTS = ("distilled", "base")
FLUX2_KLEIN_MODEL_SIZES = ("4b", "9b")

MODEL_PATH_ENV = "FLUX2_KLEIN_MODEL_PATH"
ALLOW_DOWNLOAD_ENV = "TT_DIT_ALLOW_HF_DOWNLOAD"

# What a complete snapshot must hold (files or directories, relative to the snapshot).
REQUIRED = (
    "model_index.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder",
    "tokenizer",
    "vae",
    "scheduler",
)
# Not fetched: the BFL-layout duplicate of the DiT and the sample images, neither read by the port.
_IGNORE_PATTERNS = ("flux-2-klein-*.safetensors", "*.jpg", "assets/*")
# Approximate on-disk cost of what is fetched (DiT + text encoder + VAE, bf16), for the pre-download log line.
_DOWNLOAD_GB = {"4b": 16.0, "9b": 35.0}

_DIT_CHECKPOINT = "transformer/diffusion_pytorch_model.safetensors"
_VAE_CHECKPOINT = "vae/diffusion_pytorch_model.safetensors"


class WeightsNotFoundError(FileNotFoundError):
    """Raised when the checkpoint is not on disk and downloads are disabled."""


def _check_model(variant: str, size: str) -> tuple[str, str]:
    if variant not in FLUX2_KLEIN_VARIANTS:
        raise ValueError(f"unknown FLUX.2 Klein variant {variant!r}; expected one of {FLUX2_KLEIN_VARIANTS}")
    if size not in FLUX2_KLEIN_MODEL_SIZES:
        raise ValueError(f"unknown FLUX.2 Klein size {size!r}; expected one of {FLUX2_KLEIN_MODEL_SIZES}")
    return variant, size


def repo_id(variant: str, size: str) -> str:
    return FLUX2_KLEIN_REPO_IDS[_check_model(variant, size)]


def repo_name(variant: str, size: str) -> str:
    """The repository name without the organization (``FLUX.2-klein-4B``), as a weights root names its children."""
    return repo_id(variant, size).split("/", 1)[1]


def model_name(variant: str, size: str) -> str:
    """``<variant>-<size>`` (``distilled-4b``): the second child name a weights root may use."""
    return f"{variant}-{size}"


def hf_cache_root() -> Path:
    """The ``hub`` directory of the HuggingFace cache: ``HF_HUB_CACHE``, else ``HF_HOME/hub``, else the default."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "hub"


def _missing(directory: Path, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not (directory / name).exists()]


def _is_complete(directory: Path, required: tuple[str, ...]) -> bool:
    return directory.is_dir() and not _missing(directory, required)


def cached_snapshot(variant: str, size: str, required: tuple[str, ...] | None = None) -> Path | None:
    """A cached snapshot of ``(variant, size)`` holding every required entry: the revision ``refs/main`` points at
    is preferred, then the most recently modified one. Incomplete snapshots are passed over."""
    variant, size = _check_model(variant, size)
    required = REQUIRED if required is None else tuple(required)
    repo_dir = hf_cache_root() / f"models--{repo_id(variant, size).replace('/', '--')}"
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


def _download(variant: str, size: str) -> Path:
    from huggingface_hub import snapshot_download

    logger.info(
        f"Downloading FLUX.2 Klein '{variant}' {size} (~{_DOWNLOAD_GB[size]:.0f} GB) from HuggingFace: "
        f"{repo_id(variant, size)} -> {hf_cache_root()}"
    )
    return Path(snapshot_download(repo_id=repo_id(variant, size), ignore_patterns=list(_IGNORE_PATTERNS)))


def _explicit_dir(explicit: Path, variant: str, size: str, required: tuple[str, ...]) -> Path:
    """The snapshot inside an explicit path: the path itself, or its ``<repo name>/`` or ``<variant>-<size>/``
    child."""
    if not explicit.is_dir():
        raise WeightsNotFoundError(f"{MODEL_PATH_ENV} points at {explicit}, which is not a directory")
    candidates = [explicit, explicit / repo_name(variant, size), explicit / model_name(variant, size)]
    for candidate in candidates:
        if _is_complete(candidate, required):
            return candidate
    missing = _missing(explicit, required)
    raise WeightsNotFoundError(
        f"FLUX.2 Klein '{variant}' {size} checkpoint at {explicit} is missing {missing} "
        f"(also looked in {[c.name for c in candidates[1:]]})"
    )


def model_markers(variant: str, size: str) -> set[str]:
    """Path components that identify ``(variant, size)``: ``models--<org>--<repo>`` (the HuggingFace cache), the bare
    repository name (a ``<root>/<repo>/`` layout) and ``<variant>-<size>`` (``_explicit_dir``'s other child)."""
    return {
        f"models--{repo_id(variant, size).replace('/', '--')}",
        repo_name(variant, size),
        model_name(variant, size),
    }


def names_other_variant(path: Path, variant: str, size: str) -> bool:
    """True when ``path`` identifiably belongs to a DIFFERENT model: one of its components is another
    ``(variant, size)``'s marker and none is a marker of this one. A path that names neither (a renamed copy, a
    smoke checkpoint) is trusted as given."""
    parts = set(Path(path).resolve().parts)
    if parts & model_markers(variant, size):
        return False
    return any(parts & model_markers(*other) for other in FLUX2_KLEIN_REPO_IDS if other != (variant, size))


def resolve_weights_dir(
    variant: str = "distilled",
    size: str = "4b",
    *,
    required: tuple[str, ...] | None = None,
    weights_dir: str | os.PathLike | None = None,
    allow_download: bool | None = None,
) -> Path:
    """The snapshot directory of ``(variant, size)``.

    ``weights_dir`` wins when it is set, then ``$FLUX2_KLEIN_MODEL_PATH`` unless its path identifiably names a
    different model (``names_other_variant``); otherwise the HuggingFace cache is searched and then, with
    ``$TT_DIT_ALLOW_HF_DOWNLOAD=1``, fetched. Raises ``WeightsNotFoundError`` when nothing resolves and downloads
    are off. ``required`` narrows the completeness check (a test that only reads the VAE may pass ``("vae",)``).
    """
    variant, size = _check_model(variant, size)
    required = REQUIRED if required is None else tuple(required)
    if allow_download is None:
        allow_download = os.environ.get(ALLOW_DOWNLOAD_ENV) == "1"

    if weights_dir:
        return _explicit_dir(Path(weights_dir), variant, size, required)
    env_dir = os.environ.get(MODEL_PATH_ENV)
    if env_dir and names_other_variant(Path(env_dir), variant, size):
        # One $FLUX2_KLEIN_MODEL_PATH serves one model; asking for another falls back to the cache so a base
        # pipeline under a distilled export does not silently load distilled weights (and vice versa).
        logger.info(f"{MODEL_PATH_ENV}={env_dir} names another model; resolving '{variant}' {size} from the cache")
    elif env_dir:
        return _explicit_dir(Path(env_dir), variant, size, required)

    cached = cached_snapshot(variant, size, required)
    if cached is not None:
        return cached

    if not allow_download:
        raise WeightsNotFoundError(
            f"FLUX.2 Klein '{variant}' {size} weights not found (needed: {list(required)}).\n"
            f"Point {MODEL_PATH_ENV} at the snapshot (or a directory holding {repo_name(variant, size)}/), or "
            f"re-run with {ALLOW_DOWNLOAD_ENV}=1 to fetch {repo_id(variant, size)} into {hf_cache_root()}."
        )

    snapshot = _download(variant, size)
    missing = _missing(snapshot, required)
    if missing:
        raise WeightsNotFoundError(
            f"downloaded FLUX.2 Klein '{variant}' {size} snapshot at {snapshot} is missing {missing}"
        )
    return snapshot


def dit_checkpoint(weights_dir: str | os.PathLike) -> Path:
    """The diffusers DiT file of a resolved snapshot (``transformer/diffusion_pytorch_model.safetensors``)."""
    return Path(weights_dir) / _DIT_CHECKPOINT


def vae_checkpoint(weights_dir: str | os.PathLike) -> Path:
    """The diffusers autoencoder file of a resolved snapshot (``vae/diffusion_pytorch_model.safetensors``)."""
    return Path(weights_dir) / _VAE_CHECKPOINT


__all__ = [
    "ALLOW_DOWNLOAD_ENV",
    "FLUX2_KLEIN_MODEL_SIZES",
    "FLUX2_KLEIN_REPO_IDS",
    "FLUX2_KLEIN_VARIANTS",
    "MODEL_PATH_ENV",
    "REQUIRED",
    "WeightsNotFoundError",
    "cached_snapshot",
    "dit_checkpoint",
    "hf_cache_root",
    "model_markers",
    "model_name",
    "names_other_variant",
    "repo_id",
    "repo_name",
    "resolve_weights_dir",
    "vae_checkpoint",
]
