# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.demos.deepseek_v4_flash.fp4 import EXPERT_FP4_BLOCK_SIZE, EXPERT_WEIGHT_ABI


TT_MANIFEST_FILENAME = "tt_manifest.json"
TT_MANIFEST_SCHEMA_VERSION = 1
MODEL_NAME = "deepseek_v4_flash"


def load_tt_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / TT_MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_tt_manifest(manifest)
    return manifest


def validate_tt_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest object, got {type(manifest).__name__}")
    schema_version = manifest.get("schema_version")
    if schema_version != TT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported DeepSeek V4 Flash manifest schema_version {schema_version!r}")
    if manifest.get("model_name") != MODEL_NAME:
        raise ValueError(f"Expected model_name {MODEL_NAME!r}, got {manifest.get('model_name')!r}")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Manifest missing config object")
    for field in ("hidden_size", "num_hidden_layers", "compress_rope_theta", "compress_ratios"):
        if field not in config:
            raise ValueError(f"Manifest config missing required field '{field}'")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Manifest missing artifacts object")
    for field in ("non_expert_safetensors", "expert_safetensors", "metadata_safetensors"):
        if field not in artifacts:
            raise ValueError(f"Manifest artifacts missing required field '{field}'")

    expert_format = manifest.get("expert_format")
    if not isinstance(expert_format, dict):
        raise ValueError("Manifest missing expert_format object")
    if expert_format.get("abi") != EXPERT_WEIGHT_ABI:
        raise ValueError(f"Unsupported expert ABI {expert_format.get('abi')!r}")
    if expert_format.get("block_size") != EXPERT_FP4_BLOCK_SIZE:
        raise ValueError(f"Unsupported expert block_size {expert_format.get('block_size')!r}")


def write_tt_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    validate_tt_manifest(manifest)
    output_path = Path(output_dir) / TT_MANIFEST_FILENAME
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
