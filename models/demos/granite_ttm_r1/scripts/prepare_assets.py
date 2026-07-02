#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Download and prepare benchmark datasets for Granite TTM-R1 accuracy tests.

Usage:
    python scripts/prepare_assets.py --datasets etthi
    python scripts/prepare_assets.py --datasets etthi etth2 ettm1 ettm2

Datasets are saved to models/demos/granite_ttm_r1/data/.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Raw CSV URLs for each dataset (ETT variants are publicly hosted on GitHub).
# Weather and Electricity are no longer available at the old Autoformer URLs;
# use ETT variants for multi-dataset zero-shot evaluation.
DATASET_URLS: dict[str, str] = {
    "etthi": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "etth2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ettm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ettm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

OUTPUT_NAMES: dict[str, str] = {
    "etthi": "etthi.csv",
    "etth2": "etth2.csv",
    "ettm1": "ettm1.csv",
    "ettm2": "ettm2.csv",
}


def download(name: str, force: bool = False) -> Path:
    if name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_URLS)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / OUTPUT_NAMES[name]

    if dest.exists() and not force:
        print(f"[skip] {dest} already exists (use --force to re-download)")
        return dest

    url = DATASET_URLS[name]
    print(f"[download] {name} → {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"[ok] saved {dest.stat().st_size:,} bytes")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["etthi"],
        choices=list(DATASET_URLS),
        help="Which datasets to download (default: etthi)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()

    for name in args.datasets:
        download(name, force=args.force)


if __name__ == "__main__":
    main()
