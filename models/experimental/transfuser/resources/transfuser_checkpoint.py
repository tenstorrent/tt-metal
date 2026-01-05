# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def ensure_transfuser_checkpoint_2022(
    *,
    ckpt_root: str | Path = "models/experimental/transfuser/resources/model_ckpt",
    cleanup_zip: bool = True,
    prefer_name_contains: str = "model_seed1_39",
) -> str:
    """
    Downloads + extracts models_2022.zip into ckpt_root and returns the FULL PATH
    to a checkpoint file (.pth/.pt), not the directory.

    prefer_name_contains: try to pick a checkpoint whose filename contains this substring
                          (set to "" to disable preference).
    """

    url = "https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/models_2022.zip"
    zip_name = "models_2022.zip"

    ckpt_root = Path(ckpt_root).expanduser().resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    zip_path = ckpt_root / zip_name

    def find_ckpt_file() -> Path | None:
        files = [p for p in ckpt_root.rglob("*") if p.is_file() and p.suffix in {".pth", ".pt"}]
        if not files:
            return None
        if prefer_name_contains:
            preferred = [p for p in files if prefer_name_contains in p.name]
            if preferred:
                # pick shortest path among preferred (usually the “main” file)
                return sorted(preferred, key=lambda p: (len(str(p)), str(p)))[0]
        return sorted(files, key=lambda p: (len(str(p)), str(p)))[0]

    # If already extracted, just return the file
    existing = find_ckpt_file()
    if existing is not None:
        return str(existing)

    # Allow CI to block downloads
    if os.environ.get("TRANSFUSER_SKIP_CKPT_DOWNLOAD", "0") == "1":
        raise FileNotFoundError(f"No .pth/.pt found under {ckpt_root}, and TRANSFUSER_SKIP_CKPT_DOWNLOAD=1")

    # Download (resume supported)
    if not zip_path.exists():
        subprocess.run(["wget", "-c", url, "-O", str(zip_path)], check=True)

    # Extract zip into ckpt_root
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(ckpt_root)], check=True)

    if cleanup_zip:
        try:
            zip_path.unlink()
        except Exception:
            pass

    extracted = find_ckpt_file()
    if extracted is None:
        raise FileNotFoundError(f"Extraction finished but no .pth/.pt found under: {ckpt_root}")

    return str(extracted)
