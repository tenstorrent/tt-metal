# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — non-tile-aligned sequence / head dim.

Exercises the alignment axis values added to SUPPORTED in Refinement 2:
  * w_non_aligned  — D % 32 != 0 (head_dim straddles the last tile).
  * h_non_aligned  — D aligned, S_q % 32 != 0 (seq straddles the last tile).

The kernel handles this natively (no ttnn.tilize / to_layout wrapper): TILE
tensors zero-pad the physical last tile, so
  - padded head_dim (D) columns of Q/K/V are zero -> QKᵀ / PV are exact;
  - padded S_kv key columns score 0 (Q·0) and would leak exp(0-rowmax) into
    the softmax row-sum. A partial SUM reduce scaler on the last kv-tile of
    the last KV block zeroes them out. (Row-MAX stays on the full scaler —
    an inflated max cancels in the normalized softmax; PV is unaffected
    because padded V rows are zero.)

The `device` fixture comes from the shared module-scoped conftest.

Thresholds mirror the golden suite (helpers.TOLERANCES): both PCC and a
relative-RMS (RMS / ref.std()) bound are checked, because the softmax-sum
defect this refinement fixes shows up as a per-row magnitude error that PCC
alone tolerates.
"""

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


# (dtype, pcc_min, relrms_max) — same as eval/golden_tests/.../helpers.TOLERANCES
# at fp32_dest_acc_en=True (the default resolved config).
TOL = {
    ttnn.bfloat16: (0.995, 0.05),
    ttnn.float32: (0.999, 0.02),
    ttnn.bfloat8_b: (0.99, 0.12),
}

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.bfloat16,
}


def _torch_reference(Q, K, V, *, attn_mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf = Kf.repeat_interleave(r, 1)
        Vf = Vf.repeat_interleave(r, 1)
    am = attn_mask.float() if attn_mask is not None else None
    return torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf, attn_mask=am, scale=scale)


def _pcc(ref, got):
    a, b = ref.flatten().float(), got.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _relrms(ref, got):
    a, b = ref.flatten().float(), got.flatten().float()
    return (((b - a) ** 2).mean().sqrt() / (a.std() + 1e-12)).item()


def _custom_mask(B, S_q, S_kv, torch_dtype):
    m = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    m.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return m


# (id, Q_shape, K_shape) — K and V share shape. Covers D-only non-aligned,
# S-only non-aligned, both, and non-aligned crossed with GQA / MQA /
# cross-attention (different S_kv), multi-head and multi-batch.
SHAPES = [
    ("w_D50_Saligned", (1, 1, 32, 50), (1, 1, 32, 50)),
    ("w_D47_multihead", (1, 8, 64, 47), (1, 8, 64, 47)),
    ("h_S47", (1, 1, 47, 64), (1, 1, 47, 64)),
    ("h_S100_multibatch", (2, 4, 100, 64), (2, 4, 100, 64)),
    ("both_S50_D50", (1, 1, 50, 50), (1, 1, 50, 50)),
    ("both_S33_D50_multihead", (1, 12, 33, 50), (1, 12, 33, 50)),
    ("h_S47_gqa", (1, 8, 47, 64), (1, 2, 47, 64)),
    ("h_S47_mqa", (1, 8, 47, 64), (1, 1, 47, 64)),
    ("both_cross_Sq100_Skv47_D50", (1, 4, 100, 50), (1, 4, 47, 50)),
    ("cross_Skv_nonaligned_only", (1, 4, 128, 64), (1, 4, 47, 64)),
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize("shape_id, q_shape, k_shape", SHAPES, ids=[s[0] for s in SHAPES])
@pytest.mark.parametrize("mask_mode", ["none", "custom"])
@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_scaled_dot_product_attention_alignment(device, dtype, shape_id, q_shape, k_shape, mask_mode, scale_mode):
    torch.manual_seed(0)
    torch_dtype = _TORCH_DTYPE[dtype]

    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(k_shape, dtype=torch_dtype)

    if mask_mode == "custom":
        B, _H, S_q, _D = q_shape
        S_kv = k_shape[-2]
        torch_mask = _custom_mask(B, S_q, S_kv, torch_dtype)
    else:
        torch_mask = None

    scale = 0.125 if scale_mode == "explicit" else None

    expected = _torch_reference(Q, K, V, attn_mask=torch_mask, scale=scale)

    def to_dev(t):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    ttnn_mask = to_dev(torch_mask) if torch_mask is not None else None
    ttnn_out = scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V), attn_mask=ttnn_mask, scale=scale)
    out = ttnn.to_torch(ttnn_out).float()

    assert list(out.shape) == list(q_shape), f"shape {tuple(out.shape)} != {q_shape}"
    pcc_min, relrms_max = TOL[dtype]
    pcc = _pcc(expected, out)
    relrms = _relrms(expected, out)
    assert pcc >= pcc_min and relrms <= relrms_max, (
        f"{shape_id} dtype={dtype} mask={mask_mode} scale={scale_mode}: "
        f"PCC={pcc:.5f} (>= {pcc_min}), relRMS={relrms:.4f} (<= {relrms_max})"
    )
