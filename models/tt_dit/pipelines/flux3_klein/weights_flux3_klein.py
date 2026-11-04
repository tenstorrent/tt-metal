# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

REPO_IDS = {
    "distilled": "black-forest-labs/flux-3-image-klein",
    "base": "black-forest-labs/flux-3-image-klein-base",
}
# the BFL checkpoint (model.safetensors) and the diffusers text_encoder / vae are used; the diffusers transformer is not
IGNORE_PATTERNS = ["transformer/*", "variants/*", "assets/*", "*.jpg"]


def resolve_weights_dir(model: str) -> Path:
    try:
        return Path(snapshot_download(REPO_IDS[model], local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(REPO_IDS[model], ignore_patterns=IGNORE_PATTERNS))


def checkpoint_id(path: Path) -> str:
    """Names a weights file's content for the conversion cache: its Hugging Face blob hash, else its path, size and
    modification time."""
    if path.is_symlink():
        return Path(os.path.realpath(path)).name[:16]
    stat = path.stat()
    return hashlib.sha256(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
