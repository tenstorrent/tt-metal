# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Pins and paths shared by the t0-gamma port. Pure Python: imported by the reference venv too."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ID = "theforecastingcompany/t0-gamma"
REVISION = "e08aee083778f1f32da4974c0a34276776a35c52"
SAFETENSORS_SHA256 = "1603e255629be22ea45518b502f16c55d06821c7090a1a851673e779e9c245a6"
REFERENCE_REPO = "https://github.com/theforecastingcompany/tfc-t0"
REFERENCE_TAG = "v0.6.0"
REFERENCE_COMMIT = "4871c643c37d9bd8e7ade2500b9c3a5ba8519903"

# Goldens live outside git; MANIFEST.json (tests/golden/) carries their sha256.
GOLDEN_DIR = Path(os.environ.get("T0_GOLDEN_DIR", "/home/andy/bringup/goldens/t0-gamma"))
MANIFEST_PATH = Path(__file__).resolve().parent / "tests" / "golden" / "MANIFEST.json"


def checkpoint_dir() -> Path:
    """The pinned snapshot in the shared HF cache (``config.json`` + ``model.safetensors``); never copied."""
    hub = Path(
        os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    path = hub / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"{path}/model.safetensors missing; download it with "
            f"`ttq run --model t0-gamma --kind download -- hf download {REPO_ID} --revision {REVISION} "
            f"--include model.safetensors config.json README.md`"
        )
    return path
