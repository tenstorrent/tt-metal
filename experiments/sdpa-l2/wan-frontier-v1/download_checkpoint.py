# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Download a pinned public checkpoint with host curl, without container DNS.

No credentials or host networking configuration are changed. TLS verification
stays enabled. Completed files are verified against the repository's LFS hash.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

REPO = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
CURL = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "5", "--retry-all-errors"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-snapshot", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    metadata = json.loads(
        subprocess.check_output(CURL + [f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"])
    )
    assert metadata["sha"] == REVISION
    files = [
        f
        for f in metadata["siblings"]
        if f["rfilename"].endswith((".json", ".safetensors")) or f["rfilename"].startswith("tokenizer/")
    ]
    assert len(files) == 41
    print("TOTAL_BYTES", sum(f["size"] for f in files), "FILES", len(files), flush=True)

    def fetch(info):
        name = info["rfilename"]
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = args.existing_snapshot / name
        if not target.exists() and existing.is_file() and existing.stat().st_size == info["size"]:
            os.link(existing.resolve(), target)
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            subprocess.run(
                CURL
                + [
                    "--continue-at",
                    "-",
                    "--output",
                    str(partial),
                    f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}",
                ],
                check=True,
            )
            assert partial.stat().st_size == info["size"], name
            partial.rename(target)
        assert target.stat().st_size == info["size"], name
        if info.get("lfs"):
            digest = hashlib.sha256()
            with target.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == info["lfs"]["sha256"], name
        print("VERIFIED", name, info["size"], flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, files))
    (args.output / "download-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("WAN_SNAPSHOT_VERIFIED", args.output, "SECONDS", time.monotonic() - started, flush=True)


if __name__ == "__main__":
    main()
