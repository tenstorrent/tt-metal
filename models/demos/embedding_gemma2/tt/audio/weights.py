# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Setup-time loading of the audio tensors (`audio_tower.*`, `embed_audio.*`).

`Gemma4ClippableLinear` stores `<name>.linear.weight` plus four scalar buffers
`<name>.{input_min,input_max,output_min,output_max}` (bf16). The loader renames
`<name>.linear.weight` -> `<name>.weight` and keeps the scalars; `clip_bounds` turns them into the
python floats the TT modules fold into `ttnn.clamp` at load time.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.demos.embedding_gemma2.tt.audio.config import ClipBounds

AUDIO_PREFIX = "audio_tower."
EMBED_PREFIX = "embed_audio."
_CLIP_KEYS = ("input_min", "input_max", "output_min", "output_max")


def load_audio_config_dicts(snapshot: Path) -> tuple[dict, int, int]:
    """(`audio_config` dict, text hidden size, mel bins) from config.json / preprocessor_config.json."""
    snapshot = Path(snapshot)
    with open(snapshot / "config.json") as f:
        cfg = json.load(f)
    mel_bins = 128
    pp = snapshot / "preprocessor_config.json"
    if pp.exists():
        with open(pp) as f:
            mel_bins = json.load(f).get("feature_size", mel_bins)
    return cfg["audio_config"], cfg["text_config"]["hidden_size"], mel_bins


def load_audio_state_dict(snapshot: Path) -> dict[str, torch.Tensor]:
    """Audio tensors keyed `audio_tower.<...>` / `embed_audio.<...>` (bf16, as stored), `.linear.` folded."""
    from safetensors import safe_open

    out = {}
    for f in sorted(Path(snapshot).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if k.startswith(AUDIO_PREFIX) or k.startswith(EMBED_PREFIX):
                    out[k.replace(".linear.weight", ".weight")] = sf.get_tensor(k)
    if not out:
        raise FileNotFoundError(f"no `{AUDIO_PREFIX}*` tensors under {snapshot}")
    return out


def clip_bounds(state_dict: dict[str, torch.Tensor], name: str = "") -> ClipBounds:
    """ClipBounds of one clippable linear from its sub-state (keys `input_min`, ...)."""
    missing = [k for k in _CLIP_KEYS if k not in state_dict]
    if missing:
        raise KeyError(f"{name}: clipped-linear bounds {missing} missing from the checkpoint")
    vals = {k: float(state_dict[k].float().item()) for k in _CLIP_KEYS}
    if not (vals["input_min"] < vals["input_max"] and vals["output_min"] < vals["output_max"]):
        raise ValueError(f"{name}: invalid clip bounds {vals}")
    return ClipBounds(**vals)
