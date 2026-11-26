# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shim of the reference workspace's ``tests.common`` for this tree: the stack backend imports the tile-cache
location (unused here: ``optimized_decoder.FdTileCacheSource`` is replaced by a source over our HF checkpoint,
``models/demos/glm53_flash/tests/test_mk_backend_probe.py``) and the HF text config."""

from __future__ import annotations

CACHE_DIR = ""  # the reference's tile cache (not on this box)
MANIFEST = ""


def hf_config():
    from models.autoports.zai_org_glm_5_3_flash.tt.config import load_hf_text_config
    from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir

    return load_hf_text_config(str(default_snapshot_dir()))
