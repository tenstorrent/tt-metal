# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

REPO_IDS = {
    "distilled": "black-forest-labs/flux-3-image-klein",
    "base": "black-forest-labs/flux-3-image-klein-base",
}
IGNORE_PATTERNS = ["transformer/*", "variants/*", "assets/*", "*.jpg"]


def snapshot(repo_id: str, ignore_patterns: list[str] | None = None) -> Path:
    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(repo_id, ignore_patterns=ignore_patterns))


def resolve_weights_dir(model: str) -> Path:
    return snapshot(REPO_IDS[model], IGNORE_PATTERNS)
