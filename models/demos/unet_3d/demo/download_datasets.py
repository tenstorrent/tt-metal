# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
import tarfile
import urllib.parse
import urllib.request
import zipfile
from enum import Enum
from pathlib import Path

from models.demos.unet_3d.demo.utils import configure_logging

logger = configure_logging()


class DatasetType(str, Enum):
    VALIDATION = "validation"
    TEST = "test"


class DatasetName(str, Enum):
    CELL_BOUNDARY = "cell_boundary"
    CONFOCAL_BOUNDARY = "confocal_boundary"


DATASETS = {
    (DatasetType.VALIDATION, DatasetName.CELL_BOUNDARY): [
        "https://osf.io/download/ucv5s/",
        "https://osf.io/download/t9zy5/",
    ],
    (DatasetType.TEST, DatasetName.CELL_BOUNDARY): [
        "https://osf.io/download/63kje/",
        "https://osf.io/download/qhg7s/",
        "https://osf.io/download/vt5n4/",
        "https://osf.io/download/w6nz4/",
    ],
    (DatasetType.VALIDATION, DatasetName.CONFOCAL_BOUNDARY): [
        "https://osf.io/download/4296h/",
        "https://osf.io/download/wqekp/",
    ],
}

DEFAULT_DATASET_KEY = (DatasetType.VALIDATION, DatasetName.CONFOCAL_BOUNDARY)


def _safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            member_path = output_dir / member.filename
            if not member_path.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"Unsafe zip path: {member.filename}")
        zf.extractall(output_dir)


def _safe_extract_tar(tar_path: Path, output_dir: Path) -> None:
    with tarfile.open(tar_path) as tf:
        for member in tf.getmembers():
            member_path = output_dir / member.name
            if not member_path.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError(f"Unsafe tar path: {member.name}")
        tf.extractall(output_dir)


def _filename_from_headers(url: str, headers: object) -> str | None:
    content_disp = headers.get("Content-Disposition")
    if content_disp:
        parts = content_disp.split("filename*=")
        if len(parts) > 1:
            value = parts[-1].strip()
            if value.lower().startswith("utf-8''"):
                return urllib.parse.unquote(value[7:])
        parts = content_disp.split("filename=")
        if len(parts) > 1:
            return parts[-1].strip().strip("\"'")
    parsed = urllib.parse.urlparse(url)
    name = Path(parsed.path).name
    return name if name else None


def download_file(url: str, output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        filename = _filename_from_headers(url, response.headers) or "dataset"
        output_path = output_dir / filename
        if output_path.exists() and not force:
            return output_path
        with open(output_path, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    return output_path


def maybe_extract(archive_path: Path, output_dir: Path) -> bool:
    suffixes = "".join(archive_path.suffixes)
    if suffixes in {".tar.gz", ".tgz", ".tar"}:
        _safe_extract_tar(archive_path, output_dir)
        return True
    if archive_path.suffix == ".zip":
        _safe_extract_zip(archive_path, output_dir)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UNet3D demo data into the local data directory.")
    parser.add_argument(
        "--name",
        choices=[name.value for name in DatasetName],
        help="Dataset name to download (default: first dataset in the list).",
    )
    parser.add_argument(
        "--type",
        choices=[dtype.value for dtype in DatasetType],
        help="Dataset type to download (validation or test).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the archive already exists.",
    )
    args = parser.parse_args()

    selected_key = DEFAULT_DATASET_KEY
    if args.name or args.type:
        candidates = []
        for key in DATASETS.keys():
            dtype, dname = key
            if args.name and dname.value != args.name:
                continue
            if args.type and dtype.value != args.type:
                continue
            candidates.append(key)
        if not candidates:
            raise ValueError("No dataset matches the requested name/type.")
        selected_key = candidates[0]

    selected = DATASETS[selected_key]

    dtype, dname = selected_key
    output_dir = Path(__file__).resolve().parents[1] / "data" / "datasets" / dname.value / dtype.value
    for url in selected:
        archive_path = download_file(url, output_dir, force=args.force)
        extracted = maybe_extract(archive_path, output_dir)
        if extracted:
            archive_path.unlink()
        if extracted:
            logger.info(f"Extracted data to: {output_dir}")
        else:
            logger.info(f"Downloaded data to: {archive_path}")


if __name__ == "__main__":
    main()
