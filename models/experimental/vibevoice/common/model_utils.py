# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resolve VibeVoice-1.5B checkpoint paths and download weights when missing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from models.experimental.vibevoice.common.config import (
    DEFAULT_MODEL_PATH,
    HF_REPO_ID,
    MODEL_PATH_ENV_VAR,
)

PathLike = Union[str, Path]


def is_model_dir(path: PathLike) -> bool:
    """Return True when *path* looks like a complete VibeVoice checkpoint."""
    model_dir = Path(path)
    if not model_dir.is_dir():
        return False
    has_config = (model_dir / "config.json").is_file()
    has_weights = (model_dir / "model.safetensors").is_file() or (model_dir / "model.safetensors.index.json").is_file()
    return has_config and has_weights


def _resolve_base_path(
    model_path: Optional[PathLike] = None,
    *,
    model_location_generator=None,
) -> Path:
    if model_path is not None:
        return Path(model_path)

    env_path = os.environ.get(MODEL_PATH_ENV_VAR)
    if env_path:
        return Path(env_path)

    if model_location_generator is not None and "TT_GH_CI_INFRA" in os.environ:
        ci_path = model_location_generator(
            HF_REPO_ID,
            model_subdir="",
            download_if_ci_v2=True,
        )
        return Path(ci_path)

    return DEFAULT_MODEL_PATH


def download_model_weights(
    local_dir: PathLike,
    *,
    repo_id: str = HF_REPO_ID,
) -> Path:
    """Download the full Hugging Face snapshot into *local_dir*."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download VibeVoice weights. " "Install it with: pip install huggingface_hub"
        ) from exc

    destination = Path(local_dir)
    destination.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {repo_id} weights to {destination}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
    )
    logger.info(f"Finished downloading {repo_id} to {destination}")
    return destination


def ensure_model_weights(
    model_path: Optional[PathLike] = None,
    *,
    model_location_generator=None,
    download: bool = True,
    repo_id: str = HF_REPO_ID,
) -> Path:
    """Return a local checkpoint directory, downloading weights when needed.

    Raises ``FileNotFoundError`` or ``RuntimeError`` when weights are missing
    and *download* is False or the download fails.
    """
    resolved = _resolve_base_path(model_path, model_location_generator=model_location_generator)

    if is_model_dir(resolved):
        return resolved.resolve()

    if not download:
        raise FileNotFoundError(
            f"VibeVoice weights not found at {resolved}. " f"Set {MODEL_PATH_ENV_VAR} or pass an explicit model_path."
        )

    try:
        download_model_weights(resolved, repo_id=repo_id)
    except Exception as exc:
        raise RuntimeError(f"Failed to download VibeVoice weights from {repo_id}: {exc}") from exc

    if not is_model_dir(resolved):
        raise RuntimeError(
            f"Download completed but VibeVoice weights are still missing at {resolved}. "
            "Check network access and Hugging Face credentials."
        )

    return resolved.resolve()


def get_model_path(
    model_path: Optional[PathLike] = None,
    *,
    model_location_generator=None,
    download: bool = False,
) -> str:
    """Return the resolved checkpoint path as a string.

    When *download* is False, returns the configured path without verifying
    that weights exist (legacy ``MODEL_PATH`` behavior).
    """
    if download:
        return str(
            ensure_model_weights(
                model_path,
                model_location_generator=model_location_generator,
                download=True,
            )
        )
    return str(_resolve_base_path(model_path, model_location_generator=model_location_generator))
