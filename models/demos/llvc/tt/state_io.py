# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint / config loading for the LLVC reference model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.demos.llvc.reference.llvc_reference import Net, build_reference_model
from models.demos.llvc.tt.config import LLVCConfig, llvc_config_from_json


def load_llvc_checkpoint(model: Net, checkpoint_path: str | Path) -> Net:
    """Load an official KoeAI LLVC checkpoint into the reference ``Net``.

    KoeAI checkpoints store the generator under the ``"model"`` key.
    """
    ckpt_file = Path(checkpoint_path).resolve(strict=True)
    ckpt = torch.load(str(ckpt_file), map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        # Positional-encoding / dropout buffers may be absent; anything else is a real mismatch.
        real_missing = [k for k in missing if "pos_enc.pe" not in k]
        if real_missing:
            raise RuntimeError(f"Missing checkpoint keys: {real_missing[:8]} ...")
    model.eval()
    return model


def load_llvc_config_and_model(config_path: str | Path, checkpoint_path: str | Path) -> tuple[LLVCConfig, Net]:
    """Build ``(LLVCConfig, reference Net)`` from KoeAI ``config.json`` + checkpoint."""
    config_file = Path(config_path).resolve(strict=True)
    with config_file.open() as f:
        raw = json.load(f)
    model_params = raw["model_params"]
    config = llvc_config_from_json(model_params, dtype="bfloat16")
    config.sample_rate = int(raw.get("data", {}).get("sr", 16000))

    reference = build_reference_model(model_params)
    load_llvc_checkpoint(reference, checkpoint_path)
    return config, reference
