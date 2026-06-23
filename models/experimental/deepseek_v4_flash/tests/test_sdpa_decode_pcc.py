# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the fused decode-attention op.

:meth:`DeepSeekV4Attention._sdpa_decode` (the fused
``scaled_dot_product_attention_decode`` op) is fed a single-token
``(q, kv, mask)`` + per-head sink and PCC-compared against a host fp32
softmax-with-sink reference, for the model's real attention shape
(``head_dim=256``, MQA with one shared K==V head, additive decode mask + sink).

This pins the compute-kernel config that the op requires: with
``fp32_dest_acc_en=True`` the kernel corrupts this shape (PCC ~0.36); the model's
``_HIFI4_SDPA`` (bf16 dest acc) matches the reference at PCC ~0.9999.

Run (ttnn venv)::

    pytest -s models/experimental/deepseek_v4_flash/tests/test_sdpa_decode_pcc.py
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.deepseek_v4_flash import DeepSeekV4Attention

_MASK_NEG = -1.0e9
PCC_THRESHOLD = 0.99


def _make_attention(device, num_heads: int, head_dim: int, sinks_torch: torch.Tensor) -> DeepSeekV4Attention:
    """Build a bare attention instance carrying only the state ``_sdpa_decode``
    touches (no projection weights — the test feeds q/kv/mask directly)."""
    attn = DeepSeekV4Attention.__new__(DeepSeekV4Attention)
    attn.device = device
    attn.num_heads = num_heads
    attn.head_dim = head_dim
    attn.scaling = head_dim**-0.5

    attn.sinks_torch = sinks_torch.reshape(1, num_heads, 1, 1).float()
    # Pre-divided, tile-padded sink for the fused op (see production __init__).
    sdpa_sink = attn.sinks_torch.reshape(num_heads, 1) / attn.scaling
    sdpa_sink = torch.nn.functional.pad(sdpa_sink, (0, ttnn.TILE_SIZE - 1), "constant", value=0.0)
    attn.sdpa_sinks_tt = ttnn.from_torch(sdpa_sink, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    attn._sdpa_pcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=32,
        exp_approx_mode=False,
        max_cores_per_head_batch=4,
    )
    return attn


def _torch_reference(
    q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor, sinks: torch.Tensor, scaling: float
) -> torch.Tensor:
    """fp32 softmax-with-sink attention. q [1,H,1,Dh], kv [1,1,Skv,Dh],
    mask [1,1,1,Skv], sinks [1,H,1,1]. Returns [1,H,1,Dh]."""
    h = q.shape[1]
    k = kv.expand(1, h, kv.shape[2], kv.shape[3])  # broadcast shared KV head
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaling + mask  # [1,H,1,Skv]
    m = torch.maximum(scores.amax(dim=-1, keepdim=True), sinks)
    num = torch.exp(scores - m)
    denom = num.sum(dim=-1, keepdim=True) + torch.exp(sinks - m)
    probs = num / denom
    return torch.matmul(probs, k)  # [1,H,1,Dh]


@pytest.mark.parametrize("num_heads,head_dim", [(64, 256)], ids=["64x256"])
@pytest.mark.parametrize("skv", [128, 512], ids=["skv128", "skv512"])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "masked"])
def test_sdpa_decode_pcc(device, reset_seeds, num_heads: int, head_dim: int, skv: int, masked: bool) -> None:
    torch.manual_seed(1234)

    q = torch.randn(1, num_heads, 1, head_dim) * 0.1
    kv = torch.randn(1, 1, skv, head_dim) * 0.1
    sinks = torch.randn(num_heads) * 0.5

    mask = torch.zeros(1, 1, 1, skv)
    if masked:
        # Mask out a trailing chunk (mimics not-yet-emittable / out-of-window cols).
        mask[..., skv // 2 :] = _MASK_NEG

    attn = _make_attention(device, num_heads, head_dim, sinks)

    def to_tt(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    fused = ttnn.to_torch(attn._sdpa_decode(to_tt(q), to_tt(kv), to_tt(mask))).float()
    reference = _torch_reference(q, kv, mask, attn.sinks_torch, attn.scaling)

    fus_ok, fus_pcc = comp_pcc(reference, fused, pcc=PCC_THRESHOLD)
    logger.info(f"fused _sdpa_decode vs fp32 ref : {fus_pcc}  {comp_allclose(reference, fused)}")

    assert fus_ok, f"fused _sdpa_decode vs fp32 ref PCC < {PCC_THRESHOLD}: {fus_pcc}"
