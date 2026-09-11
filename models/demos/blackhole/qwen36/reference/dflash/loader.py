# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint resolution and host model construction for Qwen3.6-27B-DFlash.

Two checkpoints are involved and they are *not* interchangeable:

* the **target**, ``$HF_MODEL`` (default ``Qwen/Qwen3.6-27B``) — the same env var the ``tt/`` port
  reads, so the reference and the device path always agree on which checkpoint is under test;
* the **drafter**, ``$DFLASH_HF_MODEL`` (default ``z-lab/Qwen3.6-27B-DFlash``) — 1.73B params,
  3.5 GB, a plain ``DFlashDraftModel``.

Both accept a hub id or a local directory; hub ids are ``snapshot_download``ed, matching
``Qwen36ModelArgs``' handling of ``HF_MODEL``.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoConfig

from models.demos.blackhole.qwen36.reference.dflash.dflash import DFlashDraftModel

# The config and checkpoint resolution live with the device port (tt/dflash) so there is exactly one
# definition; they are backend-neutral, and re-exported here because this is where the host path and
# its tests have always imported them from.
from models.demos.blackhole.qwen36.tt.dflash.config import (  # noqa: F401
    DEFAULT_DRAFTER,
    DEFAULT_TARGET,
    DRAFTER_ENV,
    TARGET_ENV,
    DFlashDrafterConfig,
    resolve_drafter_path,
    resolve_target_path,
)


def load_drafter(
    path: str | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    load_weights: bool = True,
) -> DFlashDraftModel:
    """Build the host drafter, in eval mode on CPU.

    fp32 by default: this is the golden reference a device port is PCC'd against, so it should not
    carry the target's bf16 rounding. Pass ``load_weights=False`` for a randomly initialised drafter
    (plumbing tests that must not download 3.5 GB).
    """
    path = path or resolve_drafter_path()
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    model = DFlashDraftModel(config)
    if load_weights:
        from safetensors.torch import load_file

        weights = load_file(os.path.join(path, "model.safetensors"))
        missing, unexpected = model.load_state_dict(weights, strict=False)
        # `fc`, `hidden_norm`, `norm` plus 11 tensors per layer — anything missing means we built a
        # different architecture than the checkpoint holds, which would silently draft from noise.
        if missing or unexpected:
            raise RuntimeError(
                f"drafter checkpoint {path} does not match DFlashDraftModel: "
                f"missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}"
            )
    return model.to(dtype).eval()


def load_target(path: str | None = None, *, dtype: torch.dtype = torch.bfloat16):
    """Build the host Qwen3.6-27B text model (``Qwen3_5ForCausalLM``) on CPU.

    Text-only: DFlash taps the language stack, and the vision tower plays no part in drafting. bf16
    by default because fp32 would be 108 GB of weights for no added fidelity in the verify path.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    path = path or resolve_target_path()
    text_config = Qwen3_5TextConfig.from_pretrained(path)
    text_config._attn_implementation = "sdpa"
    model = Qwen3_5ForCausalLM.from_pretrained(path, config=text_config, dtype=dtype)
    return model.eval()


def check_drafter_matches_target(cfg: DFlashDrafterConfig, target) -> None:
    """Fail loudly on a drafter/target mismatch, before it turns into a silent 0% acceptance rate."""
    tc = target.config.get_text_config()

    assert cfg.num_target_layers, (
        "drafter config declares no `num_target_layers`; cannot verify it attaches to this target "
        f"({type(target).__name__} with {tc.num_hidden_layers} layers)"
    )
    assert cfg.num_target_layers == tc.num_hidden_layers, (
        f"drafter was trained against a {cfg.num_target_layers}-layer target but this one has "
        f"{tc.num_hidden_layers} layers"
    )
    assert cfg.hidden_size == tc.hidden_size, (
        f"drafter hidden_size {cfg.hidden_size} != target hidden_size {tc.hidden_size}; the tap "
        "concat and `fc` input width would not line up"
    )
    assert cfg.vocab_size == tc.vocab_size, (
        f"drafter vocab {cfg.vocab_size} != target vocab {tc.vocab_size}; the drafter reuses the "
        "target's embedding and lm_head, so the two must agree"
    )
    assert (
        max(cfg.target_layer_ids) < tc.num_hidden_layers
    ), f"tap layer {max(cfg.target_layer_ids)} is out of range for a {tc.num_hidden_layers}-layer target"
    assert 0 <= cfg.mask_token_id < cfg.vocab_size, f"mask_token_id {cfg.mask_token_id} outside the vocab"
