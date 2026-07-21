# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Phase 6 correctness: ttnn KDA ops vs torch reference (PCC >= 0.98) on Blackhole.

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_delta_attention.torch_functional import kda_ops as ref
from models.experimental.kimi_delta_attention.torch_functional import KimiDeltaAttentionRef
from models.experimental.kimi_delta_attention.tt import ttnn_kda_ops as tt
from models.experimental.kimi_delta_attention.tt.ttnn_kda import TtKimiDeltaAttention
from models.common.utility_functions import comp_pcc

torch.manual_seed(0)


def _to_dev(x, device):
    return ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _mk_inputs(B, T, H, HV, K, V):
    """Post-gate contract: q,k L2-normed, g log-space (<=0), beta in (0,1)."""
    import torch.nn.functional as F

    q = ref.l2norm(torch.randn(B, T, H, K))
    k = ref.l2norm(torch.randn(B, T, H, K))
    v = torch.randn(B, T, HV, V)
    g = -F.softplus(torch.randn(B, T, HV, K))
    beta = torch.sigmoid(torch.randn(B, T, HV))
    return q, k, v, g, beta


@pytest.mark.parametrize("T", [1, 4, 16])
@pytest.mark.parametrize("HV,K,V", [(4, 64, 64), (8, 128, 128)])
def test_recurrent_kda_op(device, T, HV, K, V):
    B, H = 1, HV  # HV==H (48B KDA config uses no GVA)
    q, k, v, g, beta = _mk_inputs(B, T, H, HV, K, V)

    o_ref, _ = ref.naive_recurrent_kda(q, k, v, g, beta)

    o_tt, _ = tt.recurrent_kda_ttnn(
        _to_dev(q, device), _to_dev(k, device), _to_dev(v, device),
        _to_dev(g, device), _to_dev(beta, device), device=device,
    )
    o_tt = ttnn.to_torch(o_tt)

    ok, pcc = comp_pcc(o_ref, o_tt, pcc=0.98)
    logger.info(f"[recurrent_kda_op] T={T} HV={HV} K={K} V={V} PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"


@pytest.mark.parametrize("T,C", [(64, 64), (128, 64), (256, 64)])
@pytest.mark.parametrize("HV,K,V", [(4, 64, 64), (8, 128, 128)])
def test_chunk_kda_op(device, T, C, HV, K, V):
    """Chunked KDA prefill vs torch naive_chunk_kda (the perf path)."""
    from models.experimental.kimi_delta_attention.tt.ttnn_kda_chunk import chunk_kda_ttnn

    B, H = 1, HV
    q, k, v, g, beta = _mk_inputs(B, T, H, HV, K, V)
    o_ref, _ = ref.naive_chunk_kda(q, k, v, g, beta, chunk_size=C)
    o_tt, _ = chunk_kda_ttnn(
        _to_dev(q, device), _to_dev(k, device), _to_dev(v, device),
        _to_dev(g, device), _to_dev(beta, device), device=device, chunk_size=C,
    )
    ok, pcc = comp_pcc(o_ref, ttnn.to_torch(o_tt), pcc=0.98)
    logger.info(f"[chunk_kda_op] T={T} C={C} HV={HV} K={K} V={V} PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"


def test_kda_gate_op(device):
    B, T, HV, K = 1, 8, 8, 64
    g_pre = torch.randn(B, T, HV, K)
    A_log = torch.log(torch.empty(HV).uniform_(1, 16))
    dt_bias = torch.randn(HV * K)

    g_ref = ref.kda_gate(g_pre, A_log, dt_bias)

    g_tt = tt.kda_gate_ttnn(
        _to_dev(g_pre, device),
        _to_dev(A_log.view(HV, 1), device),
        _to_dev(dt_bias.view(HV, K), device),
    )
    g_tt = ttnn.to_torch(g_tt)

    ok, pcc = comp_pcc(g_ref, g_tt, pcc=0.99)
    logger.info(f"[kda_gate_op] PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"


def test_l2norm_op(device):
    x = torch.randn(1, 8, 8, 128)
    ok, pcc = comp_pcc(ref.l2norm(x), ttnn.to_torch(tt.l2norm_ttnn(_to_dev(x, device))), pcc=0.99)
    logger.info(f"[l2norm_op] PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"


@pytest.mark.parametrize("T", [1, 2, 8, 16])
def test_conv1d_op(device, T):
    """Isolate the on-device depthwise causal conv1d+SiLU (FIR) vs torch reference."""
    from models.experimental.kimi_delta_attention.torch_functional.kda_layer import causal_short_conv

    D, kernel = 256, 4
    x = torch.randn(1, T, D)
    w = torch.randn(D, kernel) * (kernel ** -0.5)
    y_ref = causal_short_conv(x, w)

    x_t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    taps = tt.conv_weight_taps(w, device)
    y_tt = ttnn.to_torch(tt.causal_conv1d_silu_ttnn(x_t, taps, kernel, device))

    ok, pcc = comp_pcc(y_ref, y_tt, pcc=0.98)
    logger.info(f"[conv1d_op] T={T} PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"


@pytest.mark.parametrize("T,use_conv", [(1, True), (8, True), (16, True), (8, False), (128, True), (128, False)])
def test_kda_layer(device, T, use_conv):
    """End-to-end ttnn KDA layer vs torch reference layer (shared weights)."""
    hidden, head_dim, nh = 256, 64, 4
    torch.manual_seed(1)
    m = KimiDeltaAttentionRef(
        hidden_size=hidden, head_dim=head_dim, num_heads=nh, num_v_heads=nh,
        conv_size=4, use_short_conv=use_conv, mode="recurrent",
    ).eval()
    x = torch.randn(1, T, hidden)
    with torch.no_grad():
        y_ref = m(x)

    tt_layer = TtKimiDeltaAttention(m, device)
    y_tt = tt_layer.forward(x)

    ok, pcc = comp_pcc(y_ref, y_tt, pcc=0.98)
    logger.info(f"[kda_layer] T={T} use_conv={use_conv} PCC={pcc}")
    assert ok, f"PCC too low: {pcc}"
