# SPDX-License-Identifier: Apache-2.0
"""Pinned checkpoint access; all host weight work is setup-only."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

from safetensors import safe_open

MODEL_ID = "Aleph-Alpha/Kolibri-1-BF16"
REVISION = "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"
CONTEXT = 1048576
HF_SNAPSHOT = (
    Path(os.environ.get("HF_HOME", "/mnt/models/huggingface"))
    / "hub/models--Aleph-Alpha--Kolibri-1-BF16/snapshots"
    / REVISION
)


def _mounted_snapshot():
    """Serving harnesses (TTI) mount the checkpoint and export HF_MODEL /
    MODEL_WEIGHTS_DIR as a local snapshot directory. Prefer it when it holds
    the safetensors index, so the container neither re-downloads the 156 GB
    nor needs hub access."""
    for name in ("MODEL_WEIGHTS_DIR", "HF_MODEL"):
        value = os.environ.get(name)
        if (
            value
            and (Path(value) / "model.safetensors.index.json").is_file()
            and (Path(value) / "config.json").is_file()
        ):
            return Path(value)
    return None


SNAPSHOT = Path(os.environ.get("KOLIBRI_CHECKPOINT_DIR") or _mounted_snapshot() or HF_SNAPSHOT)
if SNAPSHOT != HF_SNAPSHOT and (SNAPSHOT / "verified_checkpoint.json").is_file():
    # Local verified copies made by the bring-up carry a manifest; TTI mounts do not.
    manifest = json.loads((SNAPSHOT / "verified_checkpoint.json").read_text())
    if manifest["revision"] != REVISION or manifest["model_id"] != MODEL_ID or not manifest["all_weights_sha256_match"]:
        raise ValueError("Local checkpoint copy is not verified against the pinned HF snapshot")


def config():
    values = json.loads((SNAPSHOT / "config.json").read_text())
    values["max_position_embeddings"] = CONTEXT
    return SimpleNamespace(**values)


def load_weights(prefix):
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    names = [k for k in index if k.startswith(prefix)]
    result = {}
    for shard in sorted({index[k] for k in names}):
        with safe_open(SNAPSHOT / shard, framework="pt", device="cpu") as f:
            result.update({k: f.get_tensor(k) for k in names if index[k] == shard})
    return result
