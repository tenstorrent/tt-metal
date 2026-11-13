# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
import re
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


def checkpoint_id(*paths: Path) -> str:
    """Names the content of a checkpoint's weights files for the conversion cache: per file, the Hugging Face blob hash
    it resolves to, else the resolved file's path, size and modification time."""

    def file_id(path: Path) -> str:
        real = Path(os.path.realpath(path))
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", real.name):
            return real.name[:16]
        stat = real.stat()
        return hashlib.sha256(f"{real}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]

    ids = [file_id(path) for path in paths]
    return ids[0] if len(ids) == 1 else hashlib.sha256(":".join(ids).encode()).hexdigest()[:16]
