# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

import pytest

from models.experimental.moss_image.tests.assets import Capture


@pytest.fixture
def moss_assets(request):
    variant = getattr(request, "param", "Base")
    root = Path(os.environ.get("MOSS_IMAGE_ASSET_ROOT", "/mnt/MLPerf/tt_dnn-models/OpenMOSS-Team/MOSS-Image"))
    pins = json.loads((Path(__file__).parent / "capture_pins.json").read_text())
    directory = root / pins[variant]["capture"]
    checkpoint = root / "checkpoints" / variant
    if not (directory / "manifest.json").is_file() or not checkpoint.is_dir():
        message = f"Missing pinned MOSS assets at {root}; see model README asset provisioning"
        if os.environ.get("MOSS_IMAGE_ALLOW_MISSING_ASSETS") == "1" and not os.environ.get("CI"):
            pytest.skip(message)
        pytest.fail(message)
    return checkpoint, Capture(directory, pins[variant]["manifest_sha256"], variant)


def verify_weights(checkpoint, capture, components):
    from models.experimental.moss_image.tt.checkpoint import verify_checkpoint_manifest

    verify_checkpoint_manifest(checkpoint, capture.manifest["checkpoint"], components=components)
