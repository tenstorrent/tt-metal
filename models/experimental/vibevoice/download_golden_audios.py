#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Download golden reference audio from https://microsoft.github.io/VibeVoice/

Converts MP3 demo clips to 24 kHz mono WAV under resources/golden/ for direct
comparison with TTNN outputs (no HF re-generation required during development).

Usage (from tt-metal root):
    python models/experimental/vibevoice/download_golden_audios.py
    python models/experimental/vibevoice/download_golden_audios.py --force
    python models/experimental/vibevoice/download_golden_audios.py --list
"""
from __future__ import annotations

import argparse
import sys

from models.experimental.vibevoice.common.golden_audio_utils import (
    GOLDEN_DEMOS,
    GOLDEN_DIR,
    download_golden_audios,
    write_manifest,
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download VibeVoice website demo audio for TT golden comparison.",
    )
    ap.add_argument(
        "--out_dir",
        default=str(GOLDEN_DIR),
        help=f"Output directory for WAV files (default: {GOLDEN_DIR})",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-convert even if WAV already exists",
    )
    ap.add_argument(
        "--no-transcripts",
        action="store_true",
        help="Skip downloading *_gt_timestamp.json transcript sidecars",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List catalog entries and exit",
    )
    args = ap.parse_args()

    if args.list:
        print(f"Source: https://microsoft.github.io/VibeVoice/\n")
        for entry in GOLDEN_DEMOS:
            text = entry.text_file or "(transcript JSON only)"
            print(f"  {entry.wav_filename:<24}  [{entry.website_section}]  text={text}")
            print(f"    {entry.website_title}")
        return 0

    try:
        paths = download_golden_audios(
            args.out_dir,
            download_transcripts=not args.no_transcripts,
            force=args.force,
        )
    except RuntimeError as exc:
        print(f"[download_golden_audios] ERROR: {exc}", file=sys.stderr)
        return 1

    manifest = write_manifest(manifest_path=__import__("pathlib").Path(args.out_dir) / "manifest.json")
    print(f"[download_golden_audios] Wrote {len(paths)} WAV file(s) under {args.out_dir}")
    print(f"[download_golden_audios] Manifest: {manifest}")
    print("[download_golden_audios] Compare TT output with:")
    for entry in GOLDEN_DEMOS:
        if entry.text_file:
            print(f"  --text_file resources/text/{entry.text_file}  →  golden/{entry.wav_filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
