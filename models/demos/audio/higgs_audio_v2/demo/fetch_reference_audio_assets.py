# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

from models.demos.audio.higgs_audio_v2.demo._prompts import (  # noqa: E402
    DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    resolve_child_path_under_root,
    resolve_path_under_root,
    resolve_reference_audio_assets_root,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, assets_root: Path):
    destination = resolve_path_under_root(destination, assets_root, "destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response, destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def fetch_reference_audio_assets(
    reference_audio_manifest_path: str | Path = DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    assets_root: str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    manifest_path = Path(reference_audio_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_assets_root = resolve_reference_audio_assets_root(manifest_path, assets_root)
    resolved_assets_root.mkdir(parents=True, exist_ok=True)

    fetched_paths = []
    for clip in manifest["clips"]:
        destination = resolve_child_path_under_root(resolved_assets_root, clip["audio_path"], "audio path")
        if destination.exists() and not force:
            actual_sha = _sha256(destination)
            if actual_sha != clip["sha256"]:
                raise ValueError(
                    f"Existing asset `{destination}` failed sha256 verification: {actual_sha} != {clip['sha256']}. "
                    "Re-run with `--force` to replace it."
                )
            fetched_paths.append(destination)
            continue

        _download(clip["source_url"], destination, resolved_assets_root)
        actual_sha = _sha256(destination)
        if actual_sha != clip["sha256"]:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded asset `{destination}` failed sha256 verification: {actual_sha} != {clip['sha256']}"
            )
        fetched_paths.append(destination)

    return fetched_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-audio-manifest", default=str(DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH))
    parser.add_argument("--assets-root")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    fetched_paths = fetch_reference_audio_assets(
        reference_audio_manifest_path=args.reference_audio_manifest,
        assets_root=args.assets_root,
        force=args.force,
    )
    for path in fetched_paths:
        print(path)


if __name__ == "__main__":
    main()
