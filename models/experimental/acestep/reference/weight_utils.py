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

import os as _os

_HF_HUB = Path.home() / ".cache/huggingface/hub/models--ACE-Step--acestep-v15-base/snapshots"

# Full generation pipeline (LM planner + text enc + turbo DiT + VAE). Weights live on-disk only
# (never committed). The pipeline is the HF repo `ACE-Step/Ace-Step1.5`, which bundles a `vae/`.
PIPELINE_REPO_ID = "ACE-Step/Ace-Step1.5"


def _resolve_pipeline_dir() -> Path | None:
    """Resolve the pipeline checkpoint root generically, in priority order:

    1. ``ACESTEP_PIPELINE_DIR`` env var (explicit local path — for custom/offline layouts).
    2. The HuggingFace cache snapshot for ``ACE-Step/Ace-Step1.5`` (the portable default:
       works for anyone who ran ``huggingface_hub.snapshot_download('ACE-Step/Ace-Step1.5')``).

    Returns None if neither is available (tests skip rather than hard-fail).
    """
    env = _os.environ.get("ACESTEP_PIPELINE_DIR")
    if env and Path(env).is_dir():
        return Path(env)
    try:
        from huggingface_hub import snapshot_download

        # local_files_only: resolve from cache without hitting the network during tests.
        return Path(snapshot_download(PIPELINE_REPO_ID, local_files_only=True))
    except Exception:
        return None


def pipeline_dir() -> Path:
    """Root of the full-pipeline checkpoints. Assert present with an actionable message."""
    p = _resolve_pipeline_dir()
    assert p is not None and p.is_dir(), (
        "ACE-Step pipeline checkpoints not found. Either set ACESTEP_PIPELINE_DIR to a local dir, "
        f'or download the bundle:\n  python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('{PIPELINE_REPO_ID}')\""
    )
    return p


def vae_dir() -> str:
    """Path to the diffusers AutoencoderOobleck VAE (latents -> 48kHz waveform).

    The VAE ships inside the pipeline bundle as ``vae/``. Fall back to the standalone repo
    ``ACE-Step/ace-step-v1.5-1d-vae-stable-audio-format`` cache if the bundle lacks it.
    """
    p = pipeline_dir() / "vae"
    if (p / "config.json").is_file():
        return str(p)
    raise AssertionError(f"VAE config not found under {p}; ensure the pipeline bundle includes vae/")


def have_pipeline() -> bool:
    """True iff the full pipeline (incl. VAE) is resolvable on disk — for test skip guards."""
    p = _resolve_pipeline_dir()
    return p is not None and p.is_dir() and (p / "vae" / "config.json").is_file()


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


def load_module_weights(reference_module: torch.nn.Module, prefix: str, *, allow_extra: bool = False) -> None:
    """Copy real checkpoint weights (under `prefix`) into an instantiated reference nn.Module.

    Populates the module in-place so downstream code (and our TT weight extraction helpers) can
    read genuine trained tensors from `module.<...>.weight` exactly as with random init.

    ``allow_extra`` tolerates checkpoint keys the (possibly layer-reduced) module doesn't have —
    e.g. loading a 2-layer DiT from a 24-layer checkpoint for a faster test. Every parameter the
    module DOES have is still required to be present (no silent partial loads).
    """
    sub = load_state_dict(prefix)
    missing, unexpected = reference_module.load_state_dict(sub, strict=False)
    # Some buffers (rotary_emb.inv_freq) are non-persistent and legitimately absent — tolerate.
    real_missing = [k for k in missing if "inv_freq" not in k and "rotary" not in k]
    assert not real_missing, f"missing real weights for prefix '{prefix}': {real_missing[:8]}"
    if not allow_extra:
        assert not unexpected, f"unexpected keys for prefix '{prefix}': {unexpected[:8]}"
