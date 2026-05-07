#!/usr/bin/env python3
"""
Download Mistral-Small-4-119B-2603 weights from HuggingFace Hub.

Caching / reuse
---------------
- ``snapshot_download(..., local_dir=SAVE_DIR)`` writes a **full copy** under ``SAVE_DIR``.
- Demos load with ``from_pretrained(SAVE_DIR)``, so **weights are not downloaded again**
  on every run; they are read from disk. Re-running this script only refreshes missing
  or updated shards (may do light Hub metadata checks unless ``--local-files-only``).
- Hugging Face also keeps its own hub cache under ``~/.cache/huggingface/hub``; that is
  separate from ``SAVE_DIR`` but can speed up the **first** download if files were
  cached from another project.

Strict offline after a full download
-------------------------------------
- Re-run download with ``--local-files-only`` to verify the folder is complete without
  contacting the Hub (fails if anything is missing).
- For demos, set ``MISTRAL_LOAD_LOCAL_ONLY=1`` so ``from_pretrained`` uses
  ``local_files_only=True`` (no Hub calls; requires a complete snapshot).

Note: the first **quantized** load (``--quant 4``/``8``) can still take a long time
because bitsandbytes builds quantized weights in memory—that is not a re-download.

Python deps: this demo expects Transformers with ``mistral4`` (5.5+). If imports fail, from the
tt-metal repo root run ``./python_env/bin/python3 -m pip install -r models/demos/mistral_small_4_119B/requirements.txt``
(using ``python -m pip`` avoids Debian's system ``pip`` and PEP 668 "externally-managed-environment" errors).

Usage:
    python download_model.py [--save-dir DIR] [--token HF_TOKEN]
    python download_model.py --save-dir ~/models/mistral_small_4
    python download_model.py --save-dir DIR --local-files-only   # offline verify

Do not use the literal string ``/path/to/...`` from generic docs — that is not a
writable directory on Linux (and will raise permission errors under ``/path``).
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from huggingface_hub import login, snapshot_download

REPO_ID = "mistralai/Mistral-Small-4-119B-2603"

# Same layout as paths.DEFAULT_MODEL_DIR (inline so this script runs without PYTHONPATH).
_DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "models" / "mistral_small_4"


def _reject_doc_placeholder_save_dir(save_path: Path) -> None:
    """Fail fast if the user pasted a generic ``/path/to/...`` example path."""
    posix = save_path.expanduser().as_posix()
    if "/path/to" in posix:
        print(
            "ERROR: --save-dir looks like a documentation placeholder (`/path/to/...`).\n"
            "  That path is not meant to be used literally — you cannot create `/path` without root.\n"
            "  Use a directory under your home or project, for example:\n"
            f"    --save-dir {_DEFAULT_SAVE_DIR}\n"
            "    --save-dir ~/models/mistral_small_4\n"
            "    --save-dir $PWD/models/mistral_small_4",
            file=sys.stderr,
        )
        sys.exit(2)


def _mkdir_save_dir(save_path: Path) -> None:
    try:
        save_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        print(
            f"ERROR: cannot create or use --save-dir ({save_path}): {exc}\n"
            "  Pick a path on a disk where you have write permission and enough free space (~240 GB).\n"
            "  Examples:\n"
            f"    --save-dir {_DEFAULT_SAVE_DIR}\n"
            "    --save-dir ~/models/mistral_small_4",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        default=str(_DEFAULT_SAVE_DIR),
        help=f"Local directory for weights (default: {_DEFAULT_SAVE_DIR}; needs ~240 GB BF16)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", None),
        help="HuggingFace access token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*.bin", "original/*", "consolidated*"],
        help="File patterns to skip (default: skip legacy .bin shards and Mistral-native consolidated shards)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact the Hub; only succeed if SAVE_DIR already has a complete snapshot",
    )
    args = parser.parse_args()

    save_path = Path(args.save_dir).expanduser()
    _reject_doc_placeholder_save_dir(save_path)
    _mkdir_save_dir(save_path)

    if args.token:
        login(token=args.token)
    else:
        print("No HF_TOKEN provided. Model may be gated — ensure you are already logged in.")
        print("  Run: huggingface-cli login")

    print(f"Downloading {REPO_ID}")
    print(f"  → destination : {save_path.resolve()}")
    print(f"  → free space needed : ~240 GB (BF16) or ~60 GB (4-bit mode)")
    print()

    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(save_path),
        ignore_patterns=args.ignore_patterns,
        local_files_only=args.local_files_only,
    )

    # Transformers expects preprocessor_config.json; the Hub snapshot nests vision config
    # under processor_config.json only.
    _pl = Path(__file__).resolve().parent / "processor_layout.py"
    if _pl.is_file():
        spec = importlib.util.spec_from_file_location("mistral_small_4_processor_layout", _pl)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.ensure_preprocessor_config_json(str(save_path))

    print(f"\nDownload complete: {path}")


if __name__ == "__main__":
    main()
