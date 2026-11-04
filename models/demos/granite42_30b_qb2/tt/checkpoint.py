# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the pinned cache snapshot or TTI's explicitly supplied checkpoint."""

import json
import os
from pathlib import Path

MODEL_ID = "ibm-granite/granite-4.2-30b"
REVISION = "9e668ce1c538387ef24d3644e9b0606647762636"


def resolve_checkpoint():
    source = os.environ.get("HF_MODEL", MODEL_ID)
    if source == MODEL_ID:
        from huggingface_hub import snapshot_download

        checkpoint = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    else:
        checkpoint = Path(source).expanduser().resolve()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Granite checkpoint is missing {checkpoint / name}")
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "granite" or config.get("architectures") != ["GraniteForCausalLM"]:
        raise ValueError(f"Expected a GraniteForCausalLM checkpoint at {checkpoint}")
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (checkpoint / shard).is_file():
            raise FileNotFoundError(f"Granite checkpoint is missing {checkpoint / shard}")
    return checkpoint
