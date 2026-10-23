# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

import pytest

from models.experimental.moss_image.tests.assets import Capture

_verified_weights = set()


@pytest.fixture
def moss_assets(request):
    variant = getattr(request, "param", "Base")
    root = Path(os.environ.get("MOSS_IMAGE_ASSET_ROOT", "/mnt/MLPerf/tt_dnn-models/OpenMOSS-Team/MOSS-Image"))
    pins = json.loads((Path(__file__).parent / "capture_pins.json").read_text())
    return _load_assets(root, variant, pins[variant])


def _load_assets(root, variant, pin):
    directory = root / pin["capture"]
    checkpoint = root / "checkpoints" / variant
    if not (directory / "manifest.json").is_file():
        pytest.skip(f"Missing pinned MOSS CUDA reference at {directory}; see tests/README.md asset provisioning")
    # Validate an available reference even if the checkpoint is absent. Corrupt
    # or incompatible assets must fail rather than being reported as skips.
    capture = Capture(directory, pin["manifest_sha256"], variant)
    if not checkpoint.is_dir():
        pytest.skip(f"Missing MOSS {variant} checkpoint at {checkpoint}; see tests/README.md asset provisioning")
    return checkpoint, capture


def verify_weights(checkpoint, capture, components):
    from models.experimental.moss_image.tt.checkpoint import verify_checkpoint_manifest

    # Hash each immutable asset set once per pytest process. Recheck the
    # directory listing and inode/size/mtime/ctime on every case so replacing,
    # editing or adding an index/config invalidates this optimization. This
    # avoids rereading ~25 GiB before every independent DiT block test.
    selected = set(components or ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"))
    signature = []
    for component in sorted(selected):
        for path in sorted((checkpoint / component).iterdir()):
            if path.is_file():
                stat = path.stat()
                signature.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    key = (
        str(checkpoint.resolve()),
        tuple(sorted(selected)),
        json.dumps(capture.manifest["checkpoint"], sort_keys=True),
        tuple(signature),
    )
    if key in _verified_weights:
        return
    verify_checkpoint_manifest(checkpoint, capture.manifest["checkpoint"], components=components)
    _verified_weights.add(key)
