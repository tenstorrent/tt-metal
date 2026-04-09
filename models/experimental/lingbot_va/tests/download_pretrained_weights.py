# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lingbot-VA checkpoint layout check, Hugging Face download, and CLI.

Import :func:`ensure_lingbot_va_checkpoints`, :func:`ensure_checkpoint_path_for_run`, :func:`resolve_demo_checkpoint_arg`, or :func:`setup_checkpoint_root_for_tests`. Run as script to fetch weights:

``python3 models/experimental/lingbot_va/tests/download_pretrained_weights.py``
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "robbyant/lingbot-va-posttrain-robotwin"
REL_CHECKPOINT_ROOT = "models/experimental/lingbot_va/reference/checkpoints"
REQUIRED_SUBDIRS = ("vae", "tokenizer", "text_encoder", "transformer")


def tt_metal_home() -> Path:
    return Path(os.environ.get("TT_METAL_HOME", os.getcwd())).resolve()


def default_checkpoint_root() -> Path:
    return tt_metal_home() / REL_CHECKPOINT_ROOT


def resolve_demo_checkpoint_arg(checkpoint_cli: str | None = None) -> Path:
    """``--checkpoint`` value, else ``LINGBOT_VA_CHECKPOINT``, else :func:`default_checkpoint_root`."""
    raw = (checkpoint_cli or os.environ.get("LINGBOT_VA_CHECKPOINT", "")).strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default_checkpoint_root()


def checkpoint_repo_id() -> str:
    return os.environ.get("LINGBOT_VA_CHECKPOINT_REPO", DEFAULT_REPO_ID).strip() or DEFAULT_REPO_ID


def skip_checkpoint_download() -> bool:
    return os.environ.get("LINGBOT_VA_SKIP_CHECKPOINT_DOWNLOAD", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def checkpoints_layout_complete(root: Path) -> bool:
    root = root.resolve()
    if not root.is_dir():
        return False
    for name in REQUIRED_SUBDIRS:
        d = root / name
        try:
            if not d.is_dir() or not any(d.iterdir()):
                return False
        except OSError:
            return False
    return True


def ensure_lingbot_va_checkpoints(
    root: Path | None = None,
    *,
    allow_download: bool | None = None,
) -> Path:
    """Ensure ``vae/``, ``tokenizer/``, ``text_encoder/``, ``transformer/`` exist under ``root``."""
    root = (root or default_checkpoint_root()).resolve()

    if checkpoints_layout_complete(root):
        return root

    if allow_download is None:
        allow_download = not skip_checkpoint_download()
    if not allow_download:
        raise FileNotFoundError(
            f"Lingbot-VA checkpoints incomplete under {root}. "
            "Run this module as a script or unset LINGBOT_VA_SKIP_CHECKPOINT_DOWNLOAD."
        )

    root.mkdir(parents=True, exist_ok=True)

    repo = checkpoint_repo_id()
    t0 = time.perf_counter()
    logger.warning("Lingbot-VA: downloading from %s -> %s", repo, root)
    snapshot_download(
        repo_id=repo,
        local_dir=str(root),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.warning("Lingbot-VA: download done in %.1f s", time.perf_counter() - t0)

    if not checkpoints_layout_complete(root):
        raise RuntimeError(f"Lingbot-VA: layout still incomplete after download: {root}")
    return root


def ensure_checkpoint_path_for_run(checkpoint_path: Path | None = None) -> Path:
    """Ensure weights at ``checkpoint_path`` (default: default root). Used by ``demo.py`` and tests.

    If ``LINGBOT_VA_SKIP_CHECKPOINT_DOWNLOAD`` is set, only verifies layout (no download); raises
    :class:`FileNotFoundError` if incomplete. Otherwise may download via Hugging Face.
    """
    root = (checkpoint_path or default_checkpoint_root()).resolve()
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if skip_checkpoint_download():
        if checkpoints_layout_complete(root):
            return root
        raise FileNotFoundError(
            f"Lingbot-VA checkpoints incomplete under {root}. "
            "Run: python3 models/experimental/lingbot_va/tests/download_pretrained_weights.py "
            "or unset LINGBOT_VA_SKIP_CHECKPOINT_DOWNLOAD."
        )
    ensure_lingbot_va_checkpoints(root, allow_download=True)
    return root


def setup_checkpoint_root_for_tests() -> None:
    """Call once at module import from ``tests/pcc`` or ``tests/perf`` (not counted in per-test timeout)."""
    if skip_checkpoint_download():
        return
    ensure_checkpoint_path_for_run(default_checkpoint_root())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_lingbot_va_checkpoints(default_checkpoint_root(), allow_download=True)
