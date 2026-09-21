# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fetch only the Diffusers layout of the fixed evaluation checkpoint."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

MODEL = "black-forest-labs/FLUX.2-dev"
REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"
PATTERNS = ["model_index.json", "LICENSE.md", "transformer/*", "text_encoder/*", "tokenizer/*", "vae/*", "scheduler/*"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    print(f"Downloading {MODEL}@{REVISION}; Diffusers components only", flush=True)
    path = snapshot_download(MODEL, revision=REVISION, cache_dir=args.cache_dir, allow_patterns=PATTERNS, max_workers=4)
    print(json.dumps({"model": MODEL, "revision": REVISION, "checkpoint": path}), flush=True)


if __name__ == "__main__":
    main()
