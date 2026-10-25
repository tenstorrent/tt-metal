# SPDX-License-Identifier: Apache-2.0
"""Resolve the local checkpoint prepared by the serving launcher."""
import os
from pathlib import Path


def serving_checkpoint(hf_config):
    candidates = [(key, os.environ.get(key)) for key in
                  ("QWEN38_CHECKPOINT", "HF_MODEL", "MODEL_WEIGHTS_DIR")]
    candidates += [("hf_config.name_or_path", getattr(hf_config, "name_or_path", None)),
                   ("hf_config._name_or_path", getattr(hf_config, "_name_or_path", None)),
                   ("legacy default", ".cache/qwen38-full")]
    source, value = next((source, value) for source, value in candidates if value)
    path = Path(value).expanduser().resolve()
    missing = [name for name in ("config.json", "model.safetensors.index.json")
               if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{source}={value!r} must name a prepared local checkpoint; "
            f"missing {missing} under {path}. Set QWEN38_CHECKPOINT to the local weights directory.")
    return path
