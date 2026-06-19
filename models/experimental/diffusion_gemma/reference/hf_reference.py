# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF DiffusionGemma reference adapter (#47468).

The torch PCC oracle is the HF ``DiffusionGemmaForBlockDiffusion``
(``model_type=diffusion_gemma``). It is **not importable** in every environment
yet (it needs a ``transformers`` build that ships ``diffusion_gemma`` plus the
gated checkpoint), so the HF model load is guarded behind
:func:`is_hf_reference_available` / :func:`load_hf_reference`, which raise a
clear error when unavailable.

The *adapter* — wrapping any "canvas logits" model into the denoise-loop
``logits_fn`` and driving a reference trajectory — is environment-independent
and unit-tested against a mock model, so the real HF model (or the device model)
is a drop-in once available.
"""

from __future__ import annotations

import importlib.util
from typing import Callable, Optional

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory, NoiseFn, denoise_block

TRANSFORMERS_MODEL_TYPE = "diffusion_gemma"

# A canvas-logits model maps canvas token ids [B, L] (+ optional kwargs) to
# logits [B, L, vocab]. The HF reference and the device model both satisfy this.
CanvasLogitsModel = Callable[..., torch.Tensor]


def is_hf_reference_available() -> bool:
    """True iff ``transformers`` exposes the ``diffusion_gemma`` model."""
    return importlib.util.find_spec("transformers.models.diffusion_gemma") is not None


def load_hf_reference(model_id_or_path: str, *, dtype=None, device: str = "cpu"):
    """Load the HF DiffusionGemma torch reference (eval mode).

    Raises a clear ImportError when ``transformers`` lacks ``diffusion_gemma`` —
    do not silently fall back, since the PCC oracle must be the real reference.
    """
    if not is_hf_reference_available():
        raise ImportError(
            "transformers.models.diffusion_gemma is unavailable in this environment. "
            "Install a transformers build that ships DiffusionGemma "
            f"(model_type={TRANSFORMERS_MODEL_TYPE!r}) to load the HF PCC reference."
        )
    from transformers import AutoModelForCausalLM  # imported lazily; only when available

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, torch_dtype=dtype, trust_remote_code=True)
    return model.to(device).eval()


def make_logits_fn(model: CanvasLogitsModel, **model_kwargs):
    """Wrap a canvas-logits model into the ``denoise_block`` ``logits_fn(canvas, step)``."""

    def logits_fn(canvas: torch.Tensor, step: int) -> torch.Tensor:
        return model(canvas, **model_kwargs)

    return logits_fn


def run_reference_trajectory(
    model: CanvasLogitsModel,
    init_canvas: torch.Tensor,
    diffusion_config: DiffusionConfig,
    vocab_size: int,
    *,
    gumbel_noise_fn: Optional[NoiseFn] = None,
    noise_tokens_fn: Optional[NoiseFn] = None,
    **model_kwargs,
) -> DenoiseTrajectory:
    """Drive a full denoise trajectory from a canvas-logits model.

    Works for the pure-torch oracle, the HF reference, or the device model — the
    same trajectory can then be compared with ``tests/trajectory_pcc``.
    """
    return denoise_block(
        make_logits_fn(model, **model_kwargs),
        init_canvas,
        diffusion_config,
        vocab_size,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
    )
