# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Setup-time loading of the vision tensors (`vision_tower.*`, `embed_vision.*`).

`Gemma4ClippableLinear` stores its weight as `<name>.linear.weight`; with use_clipped_linears off
the clip buffers are absent and the module is a plain bias-free Linear, so the loader renames
`<name>.linear.weight` -> `<name>.weight`. That lets the E1 `MLP` module load the vision MLP as is.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

VISION_PREFIX = "vision_tower."
EMBED_PREFIX = "embed_vision."


def load_vision_config_dicts(snapshot: Path) -> tuple[dict, int]:
    """(`vision_config` dict, text hidden size) from config.json."""
    with open(Path(snapshot) / "config.json") as f:
        cfg = json.load(f)
    return cfg["vision_config"], cfg["text_config"]["hidden_size"]


def load_vision_state_dict(snapshot: Path) -> dict[str, torch.Tensor]:
    """Vision tensors keyed `vision_tower.<...>` / `embed_vision.<...>` (bf16, as stored), `.linear.` folded."""
    from safetensors import safe_open

    out = {}
    for f in sorted(Path(snapshot).glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if k.startswith(VISION_PREFIX) or k.startswith(EMBED_PREFIX):
                    out[k.replace(".linear.weight", ".weight")] = sf.get_tensor(k)
    if not out:
        raise FileNotFoundError(f"no `{VISION_PREFIX}*` tensors under {snapshot}")
    clip = [k for k in out if k.endswith(("input_min", "input_max", "output_min", "output_max"))]
    if clip:
        raise ValueError(f"clipped-linear buffers present ({clip[:2]}...); clipped linears are not supported")
    return out
