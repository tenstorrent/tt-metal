# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the real distilled adapter; CI stages it before entering offline device tests."""

import argparse
import os
import shutil
from pathlib import Path

REPO_ID = "Lightricks/LTX-2.3"
FILENAME = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
REVISION = "5948be4ced3a4493d1f836df64378ff136ddb770"


def resolve_lora(*, download_dir=None):
    from huggingface_hub import constants, hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    def stage(path):
        if download_dir is None:
            return str(path)
        target = Path(download_dir) / FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != target.resolve():
            shutil.copyfile(path, target)
        return str(target.resolve())

    explicit = os.environ.get("LORA_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"LORA_PATH does not exist: {path}")
        return stage(path.resolve())
    local = Path.home() / ".cache/ltx-checkpoints" / FILENAME
    if local.is_file():
        return stage(local)
    try:
        return stage(hf_hub_download(REPO_ID, FILENAME, revision=REVISION, local_files_only=True))
    except LocalEntryNotFoundError:
        if download_dir is None:
            if not constants.HF_HUB_OFFLINE:
                return hf_hub_download(REPO_ID, FILENAME, revision=REVISION)
            raise RuntimeError(
                "Distilled LoRA is missing from the offline cache. Stage it with "
                "HF_HUB_OFFLINE=0 python -m models.tt_dit.utils.ltx_lora_asset --download-dir DIR "
                "and set LORA_PATH to the returned file."
            ) from None
    # Never write into the shared, read-only /mnt/models cache. Only this explicit
    # preflight may fetch; the subsequent test remains offline.
    return hf_hub_download(REPO_ID, FILENAME, revision=REVISION, local_dir=download_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-dir", required=True)
    args = parser.parse_args()
    print(resolve_lora(download_dir=args.download_dir))
