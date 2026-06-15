# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Structural compatibility guard for running SulphurAI/Sulphur-2-base through the
LTX-2.3 distilled pipeline.

Sulphur is a fine-tune of Lightricks/LTX-2.3, so its distilled checkpoint must be a
drop-in for ``ltx-2.3-22b-distilled-1.1.safetensors``: the pipeline loader keys off the
prefixes below and silently drops anything else, so any renamed / missing / reshaped
tensor would load as zeros on device and surface only as garbage frames after a 20-minute
run. This reads safetensors headers only (no tensor data, no device) so the invariant is
checked in seconds before committing to an e2e.

Both 46 GB checkpoints must be on disk; the test skips otherwise.
"""

from __future__ import annotations

import os

import pytest
from safetensors import safe_open

# Prefixes the distilled pipeline's loaders consume, with the loader that owns each.
# Sulphur must supply every Lightricks tensor under these, identically shaped.
#   model.diffusion_model. → _build_transformer_state_dict (pipeline_ltx.py)
#   vae.                    → _vae_state_provider
#   audio_vae. / vocoder.   → audio decode path
#   text_embedding_projection. → connector
LOADER_PREFIXES = (
    "model.diffusion_model.",
    "vae.",
    "audio_vae.",
    "vocoder.",
    "text_embedding_projection.",
)

SULPHUR_DISTIL = os.environ.get(
    "SULPHUR_CHECKPOINT",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--SulphurAI--Sulphur-2-base/snapshots/"
        "8755af7871f0423eb784ff8e09c8c4d4d8438cb7/sulphur_distil_bf16.safetensors"
    ),
)
LIGHTRICKS_DISTIL = os.environ.get(
    "LIGHTRICKS_CHECKPOINT",
    "/home/kevinmi/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/"
    "76730e634e70a28f4e8d51f5e29c08e40e2d8e74/ltx-2.3-22b-distilled-1.1.safetensors",
)


def _shapes(path: str) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            out[k] = tuple(f.get_slice(k).get_shape())
    return out


@pytest.mark.skipif(
    not (os.path.exists(SULPHUR_DISTIL) and os.path.exists(LIGHTRICKS_DISTIL)),
    reason="needs both Sulphur and Lightricks distilled checkpoints on disk",
)
def test_sulphur_distil_is_drop_in_for_lightricks():
    sulphur = _shapes(SULPHUR_DISTIL)
    lightricks = _shapes(LIGHTRICKS_DISTIL)

    missing: list[str] = []
    mismatched: list[str] = []
    for prefix in LOADER_PREFIXES:
        ref = {k: v for k, v in lightricks.items() if k.startswith(prefix)}
        assert ref, f"no Lightricks keys under {prefix!r} — checkpoint layout changed"
        for k, shape in ref.items():
            if k not in sulphur:
                missing.append(k)
            elif sulphur[k] != shape:
                mismatched.append(f"{k}: lightricks {shape} vs sulphur {sulphur[k]}")

    assert not missing, f"{len(missing)} loader keys absent in Sulphur, e.g. {missing[:5]}"
    assert not mismatched, f"{len(mismatched)} shape mismatches, e.g. {mismatched[:5]}"
