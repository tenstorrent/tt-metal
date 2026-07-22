# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF DiffusionGemma reference adapter (#47468).

The torch PCC oracle is the HF ``DiffusionGemmaForBlockDiffusion``
(``model_type=diffusion_gemma``). It is **not importable** in every environment
yet (it needs a ``transformers`` build that ships ``diffusion_gemma`` plus the
gated checkpoint), so the HF model load is guarded behind
:func:`is_hf_reference_available` / :func:`load_hf_reference`.

Two distinct seams — **do not conflate them**:

1. :func:`hf_reference_generate` drives the **real** HF model. The real
   ``DiffusionGemmaForBlockDiffusion`` is NOT a canvas-logits callable: its
   ``forward`` takes the prompt as ``input_ids`` and the canvas as
   ``decoder_input_ids`` (plus ``past_key_values`` / ``self_conditioning_logits`` /
   decoder masks+positions), and its ``generate()`` owns the encode→denoise→commit
   loop internally. So the HF oracle is obtained by calling ``model.generate(...)``
   and reading ``sequences`` — NOT by feeding the canvas as the first positional.

2. :func:`make_logits_fn` / :func:`run_reference_trajectory` wrap a **canvas-logits
   callable** (``canvas[B,L] -> logits[B,L,vocab]``) — the mock, or the
   reconstructed-from-gemma4 oracle — into the denoise loop. These **reject a raw
   HF model** (it would silently mis-feed the canvas as ``input_ids``); use seam 1
   for the real model.
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


def load_hf_reference(model_id_or_path: str, *, dtype=None, device: str = "cpu", low_cpu_mem_usage: bool = True):
    """Load the HF DiffusionGemma torch reference (eval mode).

    ``DiffusionGemmaForBlockDiffusion`` is **not** an AutoModelForCausalLM —
    ``DiffusionGemmaConfig`` is not registered there because block-diffusion is a
    distinct generation paradigm. Load the class directly. Verified on transformers
    **5.14.1** (the pinned working env — `diffusion_gemma` ships since 5.12; 5.10.2
    lacked it) and 5.13.0.dev0 (main).

    Raises a clear ImportError when ``transformers`` lacks ``diffusion_gemma`` —
    do not silently fall back, since the PCC oracle must be the real reference.
    """
    if not is_hf_reference_available():
        raise ImportError(
            "transformers.models.diffusion_gemma is unavailable in this environment. "
            "Install a transformers build that ships DiffusionGemma "
            f"(model_type={TRANSFORMERS_MODEL_TYPE!r}) to load the HF PCC reference."
        )
    # imported lazily; only when available
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

    kwargs = {"low_cpu_mem_usage": low_cpu_mem_usage}
    if dtype is not None:
        kwargs["dtype"] = dtype  # transformers >=5.12: `dtype` is primary (`torch_dtype` kept for BC)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(model_id_or_path, **kwargs)
    return model.to(device).eval()


def _is_raw_hf_model(model) -> bool:
    """True for a raw ``DiffusionGemmaForBlockDiffusion`` (has ``generate`` +
    block-diffusion forward) — which must NOT be used as a canvas-logits callable."""
    return type(model).__name__ == "DiffusionGemmaForBlockDiffusion" or (
        hasattr(model, "generate")
        and hasattr(model, "config")
        and hasattr(getattr(model, "config", None), "canvas_length")
    )


def make_logits_fn(model: CanvasLogitsModel, **model_kwargs):
    """Wrap a **canvas-logits callable** (``canvas[B,L] -> logits[B,L,vocab]``) into
    the ``denoise_block`` ``logits_fn(canvas, step)``.

    Raises ``TypeError`` for a raw HF ``DiffusionGemmaForBlockDiffusion`` — that
    model's first positional is the prompt ``input_ids`` (the canvas is
    ``decoder_input_ids``), so feeding the canvas here would silently exercise the
    wrong path. Use :func:`hf_reference_generate` for the real HF model.
    """
    if _is_raw_hf_model(model):
        raise TypeError(
            "make_logits_fn/run_reference_trajectory expect a canvas-logits callable, not a raw "
            "DiffusionGemmaForBlockDiffusion (its canvas arg is `decoder_input_ids`, not the first "
            "positional). Use hf_reference_generate(model, input_ids, ...) for the real HF oracle."
        )

    def logits_fn(canvas: torch.Tensor, step: int) -> torch.Tensor:
        return model(canvas, **model_kwargs)

    return logits_fn


def run_reference_trajectory(
    model: CanvasLogitsModel,
    init_canvas: torch.Tensor,
    diffusion_config: DiffusionConfig,
    vocab_size: int,
    *,
    sampler: str = "multinomial",
    gumbel_noise_fn: Optional[NoiseFn] = None,
    noise_tokens_fn: Optional[NoiseFn] = None,
    generator: Optional["torch.Generator"] = None,
    **model_kwargs,
) -> DenoiseTrajectory:
    """Drive a full denoise trajectory from a **canvas-logits callable** (mock /
    reconstructed-from-gemma4 oracle / device wrapper).

    ``sampler`` is HF-faithful ``"multinomial"`` by default or ``"gumbel"`` (device
    path); inject ``gumbel_noise_fn`` for token-for-token determinism (R5). Rejects
    a raw HF model (see :func:`make_logits_fn`); use :func:`hf_reference_generate`.
    """
    return denoise_block(
        make_logits_fn(model, **model_kwargs),
        init_canvas,
        diffusion_config,
        vocab_size,
        sampler=sampler,
        gumbel_noise_fn=gumbel_noise_fn,
        noise_tokens_fn=noise_tokens_fn,
        generator=generator,
    )


def hf_reference_generate(model, input_ids, *, max_new_tokens=None, generation_config=None, **generate_kwargs):
    """Run the **real** HF DiffusionGemma generation oracle and return its output.

    Calls ``model.generate(input_ids, ...)`` — the canonical deterministic oracle
    (fixed seed + fixed schedule per #47468). The model owns the encode→denoise→
    commit loop internally, so this is the correct seam for the real HF model
    (unlike the canvas-logits adapter above). Returns the ``DiffusionGemmaGeneration``
    output; ``output.sequences`` are the committed token ids to PCC the device /
    reconstructed e2e generation against.

    NOTE: per-step trajectory records (entropy / accept / sampled) are not exposed
    by ``generate()`` directly — capture them with a draft-capable streamer
    (``streamer.put_draft``) or by hooking ``_denoising_step`` (follow-on); the
    committed-sequence comparison is the primary HF-oracle check here.
    """
    kwargs = dict(generate_kwargs)
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    return model.generate(input_ids, **kwargs)
