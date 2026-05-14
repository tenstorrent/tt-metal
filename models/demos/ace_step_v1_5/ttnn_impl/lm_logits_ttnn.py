# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN helpers for 5 Hz LM narrow CFG logits (valid audio-token slice only).

The ACE-Step 5 Hz causal LM stays on PyTorch (``AutoModelForCausalLM``). During CFG + codes
generation the handler restricts logits to the **valid audio token** subset (``[batch, K]``). That
slice is evaluated on TTNN:

  ``cfg = uncond + cfg_scale * (cond - uncond)``

Uses the same ``ttnn.from_torch`` / ``ttnn.to_torch`` staging style as ``full_pipeline`` and
``dit_decoder_core``. While this function runs, ``ttnn.CONFIG.throw_exception_on_fallback`` is set
to ``True`` when supported so host fallbacks are disabled (strict TTNN for this kernel path).

``run_prompt_to_wav`` attaches the open TTNN device via ``LocalFiveHzLMHandler.set_ttnn_logits_device``
(no environment variable). If no device is set, the handler uses PyTorch for this combine.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch


@contextlib.contextmanager
def _strict_ttnn_no_fallback() -> Iterator[None]:
    import ttnn

    cfg = getattr(ttnn, "CONFIG", None)
    if cfg is None or not hasattr(cfg, "throw_exception_on_fallback"):
        yield
        return
    prev = bool(cfg.throw_exception_on_fallback)
    cfg.throw_exception_on_fallback = True
    try:
        yield
    finally:
        cfg.throw_exception_on_fallback = prev


def cfg_linear_combination_bf16(
    cond: torch.Tensor,
    uncond: torch.Tensor,
    cfg_scale: float,
    *,
    device,
    memory_config=None,
) -> torch.Tensor:
    """Compute ``uncond + cfg_scale * (cond - uncond)`` on TTNN under strict config; return host float32.

    Args:
        cond: ``[B, K]`` conditional logits (typically float32 from upstream).
        uncond: ``[B, K]`` unconditional logits, same shape as ``cond``.
        cfg_scale: Classifier-free guidance scale.
        device: Open TTNN device (same object passed to ``TtQwen3EmbeddingEncoder``).
        memory_config: Optional TTNN memory config; defaults to ``DRAM_MEMORY_CONFIG``.

    Returns:
        Host ``torch.float32`` tensor with shape ``[B, K]``.
    """
    import ttnn

    if cond.shape != uncond.shape:
        raise ValueError(f"cond/uncond shape mismatch: {tuple(cond.shape)} vs {tuple(uncond.shape)}")
    if cond.dim() != 2:
        raise ValueError(f"expected rank-2 [B, K] logits, got {tuple(cond.shape)}")

    mem = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG

    c = cond.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()
    u = uncond.detach().to(dtype=torch.bfloat16, device="cpu").contiguous()

    with _strict_ttnn_no_fallback():
        tt_c = ttnn.from_torch(c, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        tt_u = ttnn.from_torch(u, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)
        diff = ttnn.subtract(tt_c, tt_u, memory_config=mem)
        ttnn.deallocate(tt_c)
        scaled = ttnn.multiply(diff, cfg_scale, memory_config=mem)
        ttnn.deallocate(diff)
        out_tt = ttnn.add(tt_u, scaled, memory_config=mem)
        ttnn.deallocate(tt_u)
        ttnn.deallocate(scaled)
        out = ttnn.to_torch(out_tt, dtype=torch.float32).contiguous()
        ttnn.deallocate(out_tt)
    return out


__all__ = ["cfg_linear_combination_bf16"]
