# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _git_toplevel(cwd: Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd, capture_output=True, text=True, check=True, timeout=60,
        )
        root = Path(proc.stdout.strip())
        return root if root.is_dir() else None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def get_published_versions(conf_file: Path) -> list[str]:
    override = os.environ.get("DOCS_PUBLISHED_VERSIONS", "").strip()
    if override:
        return [x.strip() for x in override.split(",") if x.strip()]
    return ["latest"]
