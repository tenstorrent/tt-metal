# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pinned Toto 2.0 checkpoints (HF repo, revision) and their local snapshots in the HF cache."""

from __future__ import annotations

from pathlib import Path

CHECKPOINTS = {
    "4m": ("Datadog/Toto-2.0-4m", "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9"),
    "22m": ("Datadog/Toto-2.0-22m", "685e4ae3e2be8d8998025e53dd98e7fdcb296a89"),
    "313m": ("Datadog/Toto-2.0-313m", "a7bab288f5e95f8606f8306f86659357e1c001ef"),
    "2.5B-FT": ("Datadog/Toto-2.0-2.5B-FT", "b4f030fb09ba0fe3f61dd11cf730539bda568eb1"),
}


def snapshot_dir(name: str) -> Path:
    """Local snapshot directory (``config.json`` + ``model.safetensors``) of a pinned checkpoint; never downloads."""
    from huggingface_hub import hf_hub_download

    repo, revision = CHECKPOINTS[name]
    # The shared cache may hold only config.json and model.safetensors of a repo.
    hf_hub_download(repo, "config.json", revision=revision, local_files_only=True)
    return Path(hf_hub_download(repo, "model.safetensors", revision=revision, local_files_only=True)).parent
