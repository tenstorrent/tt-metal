# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility layer when the real ``flash_attn`` CUDA package is not installed (e.g. no GPU).

Hub checkpoints such as ``rednote-hilab/dots.mocr`` may execute::

    from flash_attn import flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

before ``from_pretrained(..., _attn_implementation="eager")`` affects vision code paths.  We
register a namespace with valid :attr:`__spec__` and provide **CPU / PyTorch** reference
implementations so imports and forward runs succeed without CUDA FlashAttention.

This provides optional ``flash_attn`` import shims and eager fallbacks for Hub remote code
while keeping remote Hub files unmodified.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec

import torch
import torch.nn.functional as F

_SHIM_ATTR = "__dots_ocr_flash_attn_shim__"


def _make_pkg(name: str) -> types.ModuleType:
    """Namespace-like package with a valid ``__spec__`` (``find_spec`` / ``check_imports``-safe)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = []
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = []
    sys.modules[name] = mod
    parent_name, _, leaf = name.rpartition(".")
    if parent_name:
        parent = _make_pkg(parent_name)
        setattr(parent, leaf, mod)
    return mod


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, **kwargs) -> torch.Tensor:
    """RoPE apply (single tensor), CPU-safe — matches common ``flash_attn.layers.rotary`` usage."""
    # cos/sin broadcast to x (e.g. [S, 1, D/2] vs x [S, H, D])
    return (x * cos) + (_rotate_half(x) * sin)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Variable-length attention without FlashAttention: one SDPA call per packed segment.

    ``q``, ``k``, ``v`` layout matches FlashAttention: ``(total_tokens, n_heads, head_dim)``.
    """
    if softmax_scale is None:
        softmax_scale = float(q.shape[-1] ** -0.5)

    out = torch.empty_like(q)
    n_seq = int(cu_seqlens_q.numel() - 1)
    for i in range(n_seq):
        qs, qe = int(cu_seqlens_q[i].item()), int(cu_seqlens_q[i + 1].item())
        ks, ke = int(cu_seqlens_k[i].item()), int(cu_seqlens_k[i + 1].item())
        qi, ki, vi = q[qs:qe], k[ks:ke], v[ks:ke]
        # SDPA: (batch, n_heads, seq, dim)
        qq = qi.transpose(0, 1).unsqueeze(0)
        kk = ki.transpose(0, 1).unsqueeze(0)
        vv = vi.transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(
            qq,
            kk,
            vv,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)
    return out


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Batched FlashAttention-style call using :func:`torch.nn.functional.scaled_dot_product_attention`."""
    # Typical FA layout: (batch, seqlen, nheads, headdim) → SDPA (B, H, S, D)
    if q.dim() != 4:
        raise ValueError(f"flash_attn_func shim expects q dim 4, got {q.shape}")
    if softmax_scale is None:
        softmax_scale = float(q.shape[-1] ** -0.5)
    qq = q.transpose(1, 2)
    kk = k.transpose(1, 2)
    vv = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        qq,
        kk,
        vv,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


def _unimplemented(*_args, **_kwargs):
    raise NotImplementedError(
        "dots_ocr flash_attn shim: this FlashAttention entry point is not implemented on CPU. "
        "Install flash_attn for full support or use a model config that avoids this op."
    )


def install() -> None:
    """Register stub ``flash_attn`` packages if the real library is not importable."""
    try:
        import flash_attn as fa  # noqa: F401

        if not getattr(fa, _SHIM_ATTR, False):
            return
    except ImportError:
        pass

    if "flash_attn" in sys.modules:
        m = sys.modules["flash_attn"]
        if getattr(m, _SHIM_ATTR, False):
            return

    root = _make_pkg("flash_attn")
    setattr(root, _SHIM_ATTR, True)
    root.__version__ = "0.0.0+dots-ocr-compat"

    # Top-level re-exports (Hub: ``from flash_attn import flash_attn_varlen_func``)
    root.flash_attn_varlen_func = flash_attn_varlen_func
    root.flash_attn_func = flash_attn_func

    for sub in (
        "flash_attn.flash_attn_interface",
        "flash_attn.layers",
        "flash_attn.layers.rotary",
        "flash_attn.bert_padding",
    ):
        _make_pkg(sub)

    iface = sys.modules["flash_attn.flash_attn_interface"]
    iface.flash_attn_func = flash_attn_func
    iface.flash_attn_varlen_func = flash_attn_varlen_func
    for fn in (
        "flash_attn_with_kvcache",
        "flash_attn_qkvpacked_func",
        "flash_attn_kvpacked_func",
    ):
        setattr(iface, fn, _unimplemented)

    rotary = sys.modules["flash_attn.layers.rotary"]
    rotary.apply_rotary_emb = apply_rotary_emb

    # ``transformers`` flash-attention probes use ``PACKAGE_DISTRIBUTION_MAPPING["flash_attn"]``.
    # Some Python / metadata setups omit ``flash_attn`` from ``packages_distributions()``, which
    # raises KeyError when importing Qwen2 modeling (even with eager attention).
    try:
        from transformers.utils import import_utils

        m = import_utils.PACKAGE_DISTRIBUTION_MAPPING
        if isinstance(m, dict) and "flash_attn" not in m:
            import_utils.PACKAGE_DISTRIBUTION_MAPPING = {**m, "flash_attn": ("flash-attn",)}
    except Exception:
        pass
