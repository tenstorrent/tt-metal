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
# the BFL checkpoint (model.safetensors) and the diffusers text_encoder / vae are used; the diffusers transformer is not
IGNORE_PATTERNS = ["transformer/*", "variants/*", "assets/*", "*.jpg"]


def resolve_weights_dir(model: str) -> Path:
    try:
        return Path(snapshot_download(REPO_IDS[model], local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(REPO_IDS[model], ignore_patterns=IGNORE_PATTERNS))
