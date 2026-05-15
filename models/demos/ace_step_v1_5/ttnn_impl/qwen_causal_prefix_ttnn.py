# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-first prefix for :class:`QwenModel` (ACE-Step experimental 5 Hz causal LM).

Replaces host ``embed_tokens[tokens]``, host ``torch.triu`` causal mask construction, and host
``torch.arange`` position ids with TTNN ops where the downstream stack still consumes torch tensors
for RoPE (HF ``Qwen3RotaryEmbedding``) and for the attention mask bridge inside ``Attention``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from models.common.auto_compose import to_torch_auto_compose


def embed_token_ids_ttnn(
    *,
    tokens: torch.Tensor,
    embed_tokens_tt: Any,
    device: Any,
    hidden_size: int,
) -> Any:
    """``input_ids`` (torch CPU) → hidden states on device via ``ttnn.embedding``.

    Returns a TTNN tensor shaped ``[1, batch, seq, hidden_size]`` (TILE layout) to match
    ``TransformerBlock`` expectations.
    """
    import ttnn

    batch_size, seq_len = int(tokens.shape[0]), int(tokens.shape[1])
    ids_tt = ttnn.from_torch(
        tokens.to(dtype=torch.int64),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    h_tt = ttnn.embedding(ids_tt, weight=embed_tokens_tt, dtype=ttnn.bfloat16)
    if hasattr(ttnn, "deallocate"):
        ttnn.deallocate(ids_tt)
    h_tt = ttnn.reshape(h_tt, (batch_size, seq_len, hidden_size))
    h_tt = ttnn.reshape(h_tt, (1, batch_size, seq_len, hidden_size))
    h_tt = ttnn.to_layout(h_tt, ttnn.TILE_LAYOUT)
    return h_tt


def prefill_causal_mask_torch(
    *,
    device: Any,
    seq_len: int,
) -> torch.Tensor:
    """Upper-triangular causal mask built with ``ttnn.full`` + ``ttnn.triu``; returned as torch ``[1,1,S,S]`` float32."""
    import ttnn

    mask_tt = ttnn.full(
        [seq_len, seq_len],
        float("-inf"),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    mask_tt = ttnn.triu(mask_tt, diagonal=1)
    mask_torch = to_torch_auto_compose(mask_tt).to(dtype=torch.float32)
    if hasattr(ttnn, "deallocate"):
        ttnn.deallocate(mask_tt)
    rows, cols = int(mask_torch.shape[-2]), int(mask_torch.shape[-1])
    if rows >= seq_len and cols >= seq_len:
        mask_torch = mask_torch[:seq_len, :seq_len]
    return mask_torch.unsqueeze(0).unsqueeze(0)


def prefill_causal_mask_ttnn_only(*, device: Any, seq_len: int) -> Any:
    """Upper-triangular causal additive mask on device, shape ``[1, 1, S, S]`` (TILE), for full-TTNN attention."""
    import ttnn

    mask_tt = ttnn.full(
        [seq_len, seq_len],
        float("-inf"),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    mask_tt = ttnn.triu(mask_tt, diagonal=1)
    mask_tt = ttnn.reshape(mask_tt, (1, 1, seq_len, seq_len))
    return ttnn.to_layout(mask_tt, ttnn.TILE_LAYOUT)


def build_prefix_full_device(
    *,
    tokens: torch.Tensor,
    embed_tokens_tt: Any,
    device: Any,
    hidden_size: int,
    start_pos: int,
) -> tuple[Any, Any]:
    """Embedding + optional prefill mask entirely on device (no host RoPE tensors)."""
    h_tt = embed_token_ids_ttnn(
        tokens=tokens,
        embed_tokens_tt=embed_tokens_tt,
        device=device,
        hidden_size=hidden_size,
    )
    bsz, seq_log = int(tokens.shape[0]), int(tokens.shape[1])
    s_hw = int(h_tt.shape[2])
    # TILE layout can pad sequence; run the decoder on logical tokens only so QKV volumes, KV cache,
    # and causal mask stay consistent (avoids TT_FATAL reshape in decomposed SDPA).
    if s_hw > seq_log:
        import ttnn

        hd = int(h_tt.shape[3])
        h_tt = ttnn.slice(h_tt, (0, 0, 0, 0), (1, bsz, seq_log, hd))
    mask_tt = None
    if seq_log > 1 and int(start_pos) == 0:
        mask_tt = prefill_causal_mask_ttnn_only(device=device, seq_len=seq_log)
    return h_tt, mask_tt


def position_ids_from_ttnn_arange(
    *,
    device: Any,
    start_pos: int,
    seq_len: int,
    batch_size: int,
) -> torch.Tensor:
    """Integer ``position_ids`` ``[batch, seq]`` (CPU long) with values produced via ``ttnn.arange`` on device."""
    import ttnn

    pos_tt = ttnn.arange(
        int(start_pos),
        int(start_pos + seq_len),
        1,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    pos_1d = to_torch_auto_compose(pos_tt).view(-1).long()
    if hasattr(ttnn, "deallocate"):
        ttnn.deallocate(pos_tt)
    if int(pos_1d.numel()) > seq_len:
        pos_1d = pos_1d[:seq_len]
    return pos_1d.unsqueeze(0).expand(int(batch_size), -1)


def rope_cos_sin_host(
    *,
    rotary_emb: Any,
    position_ids: torch.Tensor,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF rotary: cos/sin still computed in torch (dtype/device from a zero hidden-state stub)."""
    dummy_h = torch.zeros((int(batch_size), int(seq_len), int(hidden_size)), dtype=torch.bfloat16, device="cpu")
    cos, sin = rotary_emb(dummy_h, position_ids)
    return cos, sin


def build_prefix_ttnn(
    *,
    tokens: torch.Tensor,
    embed_tokens_tt: Any,
    device: Any,
    hidden_size: int,
    start_pos: int,
    rotary_emb: Any,
) -> tuple[Any, Optional[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Run embedding + optional prefill mask + RoPE inputs; returns ``(h_tt, mask_or_none, (cos, sin))``."""
    batch_size, seq_len = int(tokens.shape[0]), int(tokens.shape[1])
    h_tt = embed_token_ids_ttnn(
        tokens=tokens,
        embed_tokens_tt=embed_tokens_tt,
        device=device,
        hidden_size=hidden_size,
    )
    mask: Optional[torch.Tensor] = None
    if seq_len > 1 and start_pos == 0:
        mask = prefill_causal_mask_torch(device=device, seq_len=seq_len)
    position_ids = position_ids_from_ttnn_arange(
        device=device,
        start_pos=start_pos,
        seq_len=seq_len,
        batch_size=batch_size,
    )
    cos, sin = rope_cos_sin_host(
        rotary_emb=rotary_emb,
        position_ids=position_ids,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
    )
    return h_tt, mask, (cos, sin)


__all__ = [
    "build_prefix_full_device",
    "build_prefix_ttnn",
    "embed_token_ids_ttnn",
    "prefill_causal_mask_torch",
    "prefill_causal_mask_ttnn_only",
    "position_ids_from_ttnn_arange",
    "rope_cos_sin_host",
]
