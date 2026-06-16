# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""L1-budget bounding for scaled_dot_product_attention (Refinement 4).

fp32 input @ D=1024 (d_t=32) OOMs L1: the input/output CBs (cb_q_in/k_in/v_in/
cb_out, sized io_factor*d_t pages) plus the fp32 accumulators (cb_o/cb_pv/
cb_o_resc/cb_q, d_t pages each) all SCALE WITH d_t. With fp32 tiles (4 KB) the
double-buffered (io_factor=2) footprint is ~1.58 MB > the 1.5 MB L1 budget.

R4 fix: the program descriptor projects the per-core CB footprint on the host
and single-buffers the input/output CBs (io_factor 2 -> 1, removing a d_t-
scaling term) only when the double-buffered layout would OOM — bringing
D=1024 fp32 to ~1.04 MB while keeping double-buffer pipelining for shapes that
fit. The failing golden cells are Q1x1x128x1024_KV1x1x128x1024 × {fp32, acc=True}
× {none,custom} × {auto,explicit}.

These tests assert the program now BUILDS + RUNS at D=1024 fp32 (no
RuntimeError: circular buffers grow beyond max L1 size) and is numerically
correct. bf16 at the same shape (which already fit) must keep working.
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _reference_sdpa(Q, K, V, *, attn_mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf)


# The R4 golden cells: D=1024 (d_t=32). Add D=512 (d_t=16) as a smaller
# fp32 case that already fit (double-buffered) to guard against regression.
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 128, 1024), id="D1024_d_t32"),  # R4 OOM golden cell
        pytest.param((1, 1, 128, 512), id="D512_d_t16"),  # already fit (control)
    ],
)
@pytest.mark.parametrize(
    "scale_mode",
    [pytest.param("auto", id="auto"), pytest.param("explicit", id="explicit")],
)
def test_fp32_large_d_no_oom(device, shape, scale_mode):
    """fp32 @ large D must BUILD (no L1 OOM) and be numerically correct."""
    B, H, S, D = shape
    torch.manual_seed(0)
    Q = torch.randn((B, H, S, D), dtype=torch.float32)
    K = torch.randn((B, H, S, D), dtype=torch.float32)
    V = torch.randn((B, H, S, D), dtype=torch.float32)

    scale = None if scale_mode == "auto" else 1.0 / math.sqrt(D)
    ref = _reference_sdpa(Q, K, V, scale=scale).float()

    tq = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    # Default config -> HiFi4 + fp32_dest_acc_en=True (fp32 + acc=False is EXCLUDED).
    out = scaled_dot_product_attention(tq, tk, tv, scale=scale)
    got = ttnn.to_torch(out).float()

    pcc_pass, pcc_str = comp_pcc(ref, got, pcc=0.99)
    _, allclose_str = comp_allclose(ref, got)
    print(f"\n[r4-l1] shape={shape} scale={scale_mode} | {pcc_str} | {allclose_str}")
    assert pcc_pass, f"PCC below floor: {pcc_str}"


@pytest.mark.parametrize(
    "scale_mode",
    [pytest.param("auto", id="auto"), pytest.param("explicit", id="explicit")],
)
def test_fp32_large_d_custom_mask(device, scale_mode):
    """fp32 @ D=1024 with a custom additive mask (the {custom} golden slice)."""
    B, H, S, D = 1, 1, 128, 1024
    torch.manual_seed(1)
    Q = torch.randn((B, H, S, D), dtype=torch.float32)
    K = torch.randn((B, H, S, D), dtype=torch.float32)
    V = torch.randn((B, H, S, D), dtype=torch.float32)
    mask = torch.randn((B, 1, S, S), dtype=torch.float32)

    scale = None if scale_mode == "auto" else 1.0 / math.sqrt(D)
    ref = _reference_sdpa(Q, K, V, attn_mask=mask, scale=scale).float()

    tq = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tm = ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tm, scale=scale)
    got = ttnn.to_torch(out).float()

    pcc_pass, pcc_str = comp_pcc(ref, got, pcc=0.99)
    print(f"\n[r4-l1-mask] scale={scale_mode} | {pcc_str}")
    assert pcc_pass, f"PCC below floor: {pcc_str}"


def test_bf16_large_d_still_works(device):
    """bf16 @ D=1024 fit before R4 (double-buffered) — must keep working."""
    B, H, S, D = 1, 1, 128, 1024
    torch.manual_seed(0)
    Q = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    K = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    V = torch.randn((B, H, S, D), dtype=torch.bfloat16)
    ref = _reference_sdpa(Q, K, V).float()

    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv)
    got = ttnn.to_torch(out).float()

    pcc_pass, pcc_str = comp_pcc(ref, got, pcc=0.99)
    print(f"\n[r4-l1-bf16] | {pcc_str}")
    assert pcc_pass, f"PCC below floor: {pcc_str}"
