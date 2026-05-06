#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Download `facebook/seamless-m4t-v2-large` weights from the Hugging Face Hub.

Also exposes helpers to locate, verify, and optionally download the Transformers checkpoint
(`ensure_seamless_m4t_v2_large_weights`, etc.) for tests and demos.

Requires: `pip install huggingface_hub` (or `transformers` with hub support).

CLI example:
  python models/experimental/seamless_m4t_v2_large/scripts/download_weights.py
  python .../download_weights.py --destination /tmp/seamless-m4t-v2-large
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Allow importing `models.*` when run as a script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_REPO_ID = "facebook/seamless-m4t-v2-large"


def get_seamless_m4t_v2_large_weights_dir() -> Path:
    """Directory used for the Transformers checkpoint (config + safetensors shards)."""
    env = os.environ.get("TT_METAL_HOME")
    if env:
        return Path(env) / "models/experimental/seamless_m4t_v2_large/weights/seamless-m4t-v2-large"
    return Path(__file__).resolve().parent.parent / "weights" / "seamless-m4t-v2-large"


def _is_safetensors_checkpoint_complete(directory: Path) -> bool:
    if not (directory / "config.json").is_file():
        return False
    single = directory / "model.safetensors"
    if single.is_file():
        return True
    index_path = directory / "model.safetensors.index.json"
    if not index_path.is_file():
        return False
    data = json.loads(index_path.read_text())
    shards = {v for v in data.get("weight_map", {}).values()}
    return bool(shards) and all((directory / name).is_file() for name in shards)


def ensure_seamless_m4t_v2_large_weights(
    *,
    directory: Optional[Path] = None,
    repo_id: str = DEFAULT_REPO_ID,
    token: Optional[str] = None,
) -> Path:
    """
    Return ``directory`` if it contains a complete Transformers sharded checkpoint; otherwise
    ``snapshot_download`` from the Hub and re-check.

    Raises:
        RuntimeError: if the checkpoint is still incomplete after download.
        ImportError: if ``huggingface_hub`` is not installed.
    """
    d = Path(directory) if directory is not None else get_seamless_m4t_v2_large_weights_dir()
    d.mkdir(parents=True, exist_ok=True)

    if _is_safetensors_checkpoint_complete(d):
        return d

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Install huggingface_hub to download weights: pip install huggingface_hub") from e

    auth = token or os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=repo_id, local_dir=str(d), token=auth)

    if not _is_safetensors_checkpoint_complete(d):
        raise RuntimeError(
            f"Checkpoint under {d} is incomplete after download (expected config.json and "
            "model.safetensors or all shards listed in model.safetensors.index.json)."
        )
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO_ID,
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--destination",
        default=None,
        help="Output directory (default: TT_METAL_HOME/models/experimental/seamless_m4t_v2_large/weights/seamless-m4t-v2-large)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face API token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    dest = Path(args.destination).expanduser() if args.destination else None
    token = args.token or os.environ.get("HF_TOKEN")
    path = ensure_seamless_m4t_v2_large_weights(directory=dest, repo_id=args.repo, token=token)
    print(f"Checkpoint ready at {path}")


if __name__ == "__main__":
    main()
