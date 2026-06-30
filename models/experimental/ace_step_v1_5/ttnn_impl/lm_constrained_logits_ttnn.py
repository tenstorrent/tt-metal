# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN helpers for 5 Hz LM constrained decoding logits (ACE demo).

:class:`~models.experimental.ace_step_v1_5.ttnn_impl.five_hz_lm.five_hz_constrained_logits_processor.MetadataConstrainedLogitsProcessor`
keeps FSM / tokenizer logic on the host. When ``LocalFiveHzLMHandler.set_ttnn_logits_device`` has run,
the processor sets ``_ttnn_logits_device`` and these helpers run on TTNN (strict no-fallback when supported):

- **Whitelist** (only a set of token ids keep their logits; others → large negative) — :func:`logits_keep_allowed_bf16`
- **Logits + dense mask** (e.g. ``non_audio_code_mask``, ``audio_code_mask``) — :func:`logits_add_delta_bf16`
- **Temperature** (divide logits by temperature) — :func:`logits_divide_by_scalar_bf16`
- **Sequence concat** (``[B, S]`` + ``[B, T]`` along dim 1 for token ids / masks as ``int32``) — :func:`tensor_concat_dim1_int32_ttnn`
- **EOS / pad row mask** (``[B]`` bool per row), **any-reduction** on small bool masks, and **int32 identity round-trip** for host callbacks (streamer / FSM ``update_state``) — :func:`tokens_row_eos_or_pad_mask_int32_ttnn`, :func:`mask_any_true_int32_ttnn`, :func:`tensor_identity_roundtrip_int32_ttnn`

**Repetition penalty + top-k + top-p + temperature + sampling** were moved out of this
file and now run as one **fused on-device pipeline** in
:mod:`models.experimental.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers`
(subclasses of :class:`models.common.modules.sampling.penalties_1d.Penalties1D` and
:class:`models.common.modules.sampling.sampling_1d.Sampling1D`). Use
:meth:`LocalFiveHzLMHandler._postprocess_and_sample_ttnn_or_torch` as the entry point.

Host :mod:`tqdm` remains the progress UI (TTNN has no tqdm analogue); the handler tags the description when TTNN assist is active.

Single-index ``scores[0, i] = -inf`` tweaks inside the FSM still use PyTorch (cheap); extend this module if
those need to move too.

Reuses :func:`models.experimental.ace_step_v1_5.ttnn_impl.lm_logits_ttnn._strict_ttnn_no_fallback`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from models.experimental.ace_step_v1_5.ttnn_impl.lm_logits_ttnn import _strict_ttnn_no_fallback


def _dram_mem(ttnn):
    return getattr(ttnn, "DRAM_MEMORY_CONFIG", None)


def _row_to_tile_bf16(tt_tensor, ttnn):
    """Convert a device tensor to TILE (required for ``where`` / ``topk`` / softmax on some arch)."""
    tiled = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
    ttnn.deallocate(tt_tensor)
    return tiled


def _concat_dim1(ttnn, a, b):
    if hasattr(ttnn, "concat"):
        return ttnn.concat([a, b], dim=1)
    return ttnn.concatenate([a, b], dim=1)


def _torch_rows_to_int32_cpu(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().cpu()
    if x.dtype == torch.bool:
        return x.to(torch.int32)
    return x.to(torch.int32)


def tensor_concat_dim1_int32_ttnn(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    device,
    out_dtype: torch.dtype,
    memory_config=None,
) -> torch.Tensor:
    """Concat ``left`` and ``right`` on dim 1 (``[B, S]`` + ``[B, T]`` → ``[B, S+T]``) on TTNN; both rank-2, same ``B``."""
    import ttnn

    if left.dim() != 2 or right.dim() != 2:
        raise ValueError(f"expected rank-2 tensors, got {left.dim()} and {right.dim()}")
    if int(left.shape[0]) != int(right.shape[0]):
        raise ValueError(f"batch mismatch {tuple(left.shape)} vs {tuple(right.shape)}")

    a = _torch_rows_to_int32_cpu(left).contiguous()
    b = _torch_rows_to_int32_cpu(right).contiguous()

    mem = memory_config if memory_config is not None else _dram_mem(ttnn)
    mem_kw = dict(memory_config=mem) if mem is not None else {}

    with _strict_ttnn_no_fallback():
        tt_a = ttnn.from_torch(a, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        tt_b = ttnn.from_torch(b, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        out_tt = _concat_dim1(ttnn, tt_a, tt_b)
        ttnn.deallocate(tt_a)
        ttnn.deallocate(tt_b)
        out_i32 = ttnn.to_torch(out_tt).to(dtype=torch.int32).contiguous()
        ttnn.deallocate(out_tt)

    if out_dtype == torch.bool:
        return (out_i32 != 0).to(torch.bool)
    return out_i32.to(out_dtype)


def tensor_identity_roundtrip_int32_ttnn(
    tensor: torch.Tensor,
    *,
    device,
    memory_config=None,
) -> torch.Tensor:
    """Stage integer tensor to TTNN and read back (values preserved); returns **CPU** tensor."""
    import ttnn

    orig_shape = tuple(tensor.shape)
    orig_dtype = tensor.dtype
    flat = tensor.detach().cpu().reshape(-1).long().to(torch.int32).contiguous()
    n = int(flat.numel())
    if n == 0:
        return tensor.detach().cpu().reshape(orig_shape).to(orig_dtype)

    mem = memory_config if memory_config is not None else _dram_mem(ttnn)
    mem_kw = dict(memory_config=mem) if mem is not None else {}
    x2 = flat.reshape(1, n)

    with _strict_ttnn_no_fallback():
        tt = ttnn.from_torch(x2, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        out_i32 = ttnn.to_torch(tt).to(dtype=torch.int32).contiguous()
        ttnn.deallocate(tt)
    return out_i32.reshape(orig_shape).to(orig_dtype)


def mask_any_true_int32_ttnn(mask: torch.Tensor, *, device, memory_config=None) -> bool:
    """``True`` iff any element of ``mask`` (bool or 0/1) is true; reduced on TTNN."""
    import ttnn

    v = mask.detach().cpu().reshape(-1).to(torch.int32).contiguous()
    n = int(v.numel())
    if n == 0:
        return False

    mem = memory_config if memory_config is not None else _dram_mem(ttnn)
    mem_kw = dict(memory_config=mem) if mem is not None else {}

    with _strict_ttnn_no_fallback():
        tt = ttnn.from_torch(v.unsqueeze(0), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        tt_f = ttnn.typecast(tt, dtype=ttnn.bfloat16)
        ttnn.deallocate(tt)
        s = ttnn.sum(tt_f)
        ttnn.deallocate(tt_f)
        val = float(ttnn.to_torch(s).reshape(-1)[0].item())
        ttnn.deallocate(s)
    return val > 0.5


def tokens_row_eos_or_pad_mask_int32_ttnn(
    tokens: torch.Tensor,
    eos_token_id: int,
    pad_token_id: Optional[int],
    *,
    device,
    memory_config=None,
) -> torch.Tensor:
    """Host CPU ``torch.bool`` ``[B]``: ``(tokens == eos) | (tokens == pad)`` (pad branch optional)."""
    import ttnn

    t = tokens.detach().cpu().long().reshape(-1)
    b = int(t.numel())
    if b == 0:
        return torch.zeros(0, dtype=torch.bool)

    mem = memory_config if memory_config is not None else _dram_mem(ttnn)
    mem_kw = dict(memory_config=mem) if mem is not None else {}

    tok_cpu = t.to(torch.int32).contiguous().reshape(b, 1)
    eos_cpu = torch.full((b, 1), int(eos_token_id), dtype=torch.int32)

    with _strict_ttnn_no_fallback():
        tt_t = ttnn.from_torch(tok_cpu, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        tt_e = ttnn.from_torch(eos_cpu, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        eq_eos = ttnn.eq(tt_t, tt_e)
        ttnn.deallocate(tt_e)

        if pad_token_id is not None and int(pad_token_id) != int(eos_token_id):
            pad_cpu = torch.full((b, 1), int(pad_token_id), dtype=torch.int32)
            tt_p = ttnn.from_torch(pad_cpu, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
            eq_pad = ttnn.eq(tt_t, tt_p)
            ttnn.deallocate(tt_p)
            comb = ttnn.logical_or(eq_eos, eq_pad)
            ttnn.deallocate(eq_eos)
            ttnn.deallocate(eq_pad)
        else:
            comb = eq_eos

        ttnn.deallocate(tt_t)
        row = ttnn.to_torch(comb).reshape(b).to(torch.bool).contiguous()
        ttnn.deallocate(comb)
    return row


def tokens_any_eos_or_pad_int32_ttnn(
    tokens: torch.Tensor,
    eos_token_id: int,
    pad_token_id: Optional[int],
    *,
    device,
    memory_config=None,
) -> bool:
    """Whether **any** row in ``tokens`` (1-D ``[B]``) equals EOS or pad (same semantics as handler EOS check)."""
    row = tokens_row_eos_or_pad_mask_int32_ttnn(
        tokens, eos_token_id, pad_token_id, device=device, memory_config=memory_config
    )
    if row.numel() == 0:
        return False
    return mask_any_true_int32_ttnn(row, device=device, memory_config=memory_config)


# NOTE: ``logits_top_k_filter_bf16``, ``logits_top_p_nucleus_bf16``,
# ``logits_sample_indices_bf16``, and ``repetition_penalty_apply_bf16`` were removed
# from this file. The fused ``repetition_penalty → top-k → top-p → temperature → sample``
# pipeline now runs through
# :class:`~models.experimental.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers.AceStepPenalties1D`
# (subclass of :class:`models.common.modules.sampling.penalties_1d.Penalties1D`) and
# :class:`~models.experimental.ace_step_v1_5.ttnn_impl.lm_postprocess_tt_transformers.AceStepSampling1D`
# (subclass of :class:`models.common.modules.sampling.sampling_1d.Sampling1D`),
# orchestrated by ``apply_penalty_filter_sample`` and called from
# :meth:`LocalFiveHzLMHandler._postprocess_and_sample_ttnn_or_torch`.


def logits_add_delta_bf16(
    scores: torch.Tensor,
    delta: torch.Tensor,
    *,
    device,
    memory_config=None,
) -> torch.Tensor:
    """``scores + delta`` on TTNN; ``scores`` and ``delta`` same shape ``[B, V]`` float32."""
    import ttnn

    if scores.shape != delta.shape:
        raise ValueError(f"shape mismatch {tuple(scores.shape)} vs {tuple(delta.shape)}")
    if scores.dim() != 2:
        raise ValueError(f"expected [B, V], got {tuple(scores.shape)}")

    mem = memory_config if memory_config is not None else _dram_mem(ttnn)
    mem_kw = dict(memory_config=mem) if mem is not None else {}

    a = scores.detach().float().cpu().contiguous()
    b = delta.detach().float().cpu().contiguous()
    a_bf = a.to(dtype=torch.bfloat16)
    b_bf = b.to(dtype=torch.bfloat16)

    with _strict_ttnn_no_fallback():
        tt_a = ttnn.from_torch(a_bf, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        tt_b = ttnn.from_torch(b_bf, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        out_tt = ttnn.add(tt_a, tt_b, **mem_kw)
        ttnn.deallocate(tt_a)
        ttnn.deallocate(tt_b)
        out = ttnn.to_torch(out_tt, dtype=torch.float32).contiguous()
        ttnn.deallocate(out_tt)
    return out


def logits_keep_allowed_bf16(
    scores: torch.Tensor,
    allowed_tokens: Sequence[int],
    *,
    device,
    memory_config=None,
) -> torch.Tensor:
    """Whitelist on TTNN: same semantics as ``_apply_whitelist_inplace`` (batch must be 1)."""
    import ttnn

    if scores.dim() != 2 or int(scores.shape[0]) != 1:
        raise ValueError(f"logits_keep_allowed_bf16 expects [1, V], got {tuple(scores.shape)}")
    v = int(scores.shape[1])
    mem = memory_config if memory_config is not None else _dram_mem(ttnn)
    mem_kw = dict(memory_config=mem) if mem is not None else {}

    if not allowed_tokens:
        return torch.full((1, v), float("-inf"), dtype=torch.float32, device="cpu")

    keep = torch.zeros((1, v), dtype=torch.bool)
    for t in allowed_tokens:
        ti = int(t)
        if 0 <= ti < v:
            keep[0, ti] = True

    s = scores.detach().float().cpu().contiguous()
    s_bf = s.to(dtype=torch.bfloat16)
    neg = torch.full((1, v), -1e9, dtype=torch.bfloat16)
    w_bf = keep.to(dtype=torch.bfloat16)

    with _strict_ttnn_no_fallback():
        tt_s = ttnn.from_torch(s_bf, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        tt_neg = ttnn.from_torch(neg, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        tt_w = ttnn.from_torch(w_bf, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        tt_s = _row_to_tile_bf16(tt_s, ttnn)
        tt_neg = _row_to_tile_bf16(tt_neg, ttnn)
        tt_w = _row_to_tile_bf16(tt_w, ttnn)
        out_tt = ttnn.where(tt_w, tt_s, tt_neg, **mem_kw)
        ttnn.deallocate(tt_s)
        ttnn.deallocate(tt_neg)
        ttnn.deallocate(tt_w)
        out = ttnn.to_torch(out_tt, dtype=torch.float32).contiguous()
        ttnn.deallocate(out_tt)
    if int(out.shape[1]) != v:
        out = out[:, :v].contiguous()
    return out


def logits_divide_by_scalar_bf16(
    scores: torch.Tensor,
    temperature: float,
    *,
    device,
    memory_config=None,
) -> torch.Tensor:
    """``scores / temperature`` on TTNN; ``scores`` is ``[B, V]`` float32."""
    import ttnn

    if scores.dim() != 2:
        raise ValueError(f"expected [B, V], got {tuple(scores.shape)}")
    t = float(temperature)
    if t <= 0:
        t = 1e-6

    mem = memory_config if memory_config is not None else _dram_mem(ttnn)
    mem_kw = dict(memory_config=mem) if mem is not None else {}

    a = scores.detach().float().cpu().contiguous()
    a_bf = a.to(dtype=torch.bfloat16)

    with _strict_ttnn_no_fallback():
        tt_a = ttnn.from_torch(a_bf, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, **mem_kw)
        out_tt = ttnn.div(tt_a, t, **mem_kw)
        ttnn.deallocate(tt_a)
        out = ttnn.to_torch(out_tt, dtype=torch.float32).contiguous()
        ttnn.deallocate(out_tt)
    return out


__all__ = [
    "logits_add_delta_bf16",
    "logits_divide_by_scalar_bf16",
    "logits_keep_allowed_bf16",
    "mask_any_true_int32_ttnn",
    "tensor_concat_dim1_int32_ttnn",
    "tensor_identity_roundtrip_int32_ttnn",
    "tokens_any_eos_or_pad_int32_ttnn",
    "tokens_row_eos_or_pad_mask_int32_ttnn",
]
