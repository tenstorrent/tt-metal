# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side real-checkpoint weight loading for ACE-Step v1.5 (TTTv2 port).

Follows the Phi-4 weight_utils pattern (models/common/models/phi4/weight_utils.py): read the
genuine HF checkpoint tensors and hand them to the TT modules unchanged (the modules apply the
[out,in]->[in,out] transpose themselves via the test helpers).

We load specific tensors directly from `model.safetensors` rather than instantiating the whole
HF model, because AceStepConditionGenerationModel.__init__ builds a ResidualFSQ quantizer that
fails under meta-device init (`.item()` on meta tensors). Loading the raw state_dict sidesteps
that and lets us populate just the sub-module we want to validate against real weights.
"""

from __future__ import annotations

import glob
from pathlib import Path

import torch
from safetensors import safe_open

_HF_HUB = Path.home() / ".cache/huggingface/hub/models--ACE-Step--acestep-v15-base/snapshots"


def checkpoint_path() -> str:
    """Locate the downloaded model.safetensors in the HF snapshot."""
    hits = glob.glob(str(_HF_HUB / "*/model.safetensors"))
    assert hits, (
        "model.safetensors not found. Download it first:\n"
        '  python -c "from huggingface_hub import hf_hub_download; '
        "hf_hub_download('ACE-Step/acestep-v15-base','model.safetensors')\""
    )
    return hits[0]


def load_state_dict(prefix: str | None = None) -> dict[str, torch.Tensor]:
    """Load checkpoint tensors, optionally filtered to keys under `prefix` (prefix stripped)."""
    path = checkpoint_path()
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, "pt") as f:
        for key in f.keys():
            if prefix is None:
                out[key] = f.get_tensor(key)
            elif key.startswith(prefix):
                out[key[len(prefix) :]] = f.get_tensor(key)
    return out


def load_module_weights(reference_module: torch.nn.Module, prefix: str) -> None:
    """Copy real checkpoint weights (under `prefix`) into an instantiated reference nn.Module.

    Populates the module in-place so downstream code (and our TT weight extraction helpers) can
    read genuine trained tensors from `module.<...>.weight` exactly as with random init.
    """
    sub = load_state_dict(prefix)
    missing, unexpected = reference_module.load_state_dict(sub, strict=False)
    # Some buffers (rotary_emb.inv_freq) are non-persistent and legitimately absent — tolerate.
    real_missing = [k for k in missing if "inv_freq" not in k and "rotary" not in k]
    assert not real_missing, f"missing real weights for prefix '{prefix}': {real_missing[:8]}"
    assert not unexpected, f"unexpected keys for prefix '{prefix}': {unexpected[:8]}"
