# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA (per-key-channel decay) vs the GDN ops of this build, random inputs, HF ``glm5_next`` references.

Deliverable 2 of ``doc/reuse_map.md``: can ``ttnn.transformer.chunk_gated_delta_rule`` (prefill) and
``fused_recurrent_gated_delta_rule`` (decode) compute Kimi's per-channel-decay delta rule, and what do they cost
against the d_p ttKDA recurrence (``ttnn.experimental.kda.*``) at our per-chip geometry (16 heads x K = V = 128)?

    TT_VISIBLE_DEVICES=3 pytest models/demos/glm53_flash/tests/test_kda_vs_gdn_ops.py -s

Facts the tests pin down (the op docstrings: ``g [B, T, HV]`` = ONE log decay per head and token):
* prefill: the GDN op with the per-head mean of the per-channel gate is NOT the KDA function (PCC logged); the
  same op with a channel-uniform gate matches the KDA reference (harness proof), and the d_p ttKDA recurrence
  matches the per-channel reference. Both are timed at T = 2048 / 8192.
* decode (T = 1): the per-channel decay is a row scale of the state, so ONE elementwise multiply on the state
  followed by the fused op with ``g = 0`` is exact (the ``kda_decode`` custom op fuses the same thing with the
  conv / norms / gates). The fused op needs a core per (user, head): B * H <= compute cores per call.
* the "gauge fold" (asymmetric read / write keys ``k * exp(+-G)``) that would express the per-channel rule on the
  scalar-gate chunk op is numerically dead past a few tokens (host check: ``exp(cumsum g)`` under/overflows).
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.kda import device_scalar_u32
from models.demos.glm53_flash.tt.ttkda.config import KDARecurrenceProgramConfig
from models.demos.glm53_flash.tt.ttkda.recurrence import KDARecurrence

H, K, V = 16, 128, 128  # TP4 per-chip head geometry of GLM-5.3-Flash KDA (64 heads over 4 columns)
GATE_LOWER_BOUND = -5.0  # g = -5 sigmoid(.) per key channel (GLM gate_lower_bound)
REGIMES = {"mild": -3.0, "strong": 0.0}  # sigmoid logit centre: mean log decay -0.3 / -2.5 per token


def _hf():
    from transformers.models.glm5_next.modeling_glm5_next import (
        chunk_kimi_delta_attention,
        l2norm,
        recurrent_kimi_delta_attention,
    )

    return chunk_kimi_delta_attention, recurrent_kimi_delta_attention, l2norm


def _inputs(B: int, T: int, regime: str, seed: int = 0):
    """Raw (unnormalized) bf16 q/k/v, the bf16 sigmoid gate values and beta logits, the fp32 initial state,
    plus the fp32 per-channel log decay ``g_pc = -5 * sigmoid`` and ``beta`` the references consume."""
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K).to(torch.bfloat16)
    k = torch.randn(B, T, H, K).to(torch.bfloat16)
    v = torch.randn(B, T, H, V).to(torch.bfloat16)
    sig = torch.sigmoid(torch.randn(B, T, H, K) + REGIMES[regime]).to(torch.bfloat16)
    g_pc = GATE_LOWER_BOUND * sig.float()  # exact in fp32
    beta_logits = torch.randn(B, T, H).to(torch.bfloat16)
    beta = torch.sigmoid(beta_logits.float())
    s0 = 0.1 * torch.randn(B, H, K, V)
    return q, k, v, sig, g_pc, beta_logits, beta, s0


def _const_tiles(device, chunk_size: int = 32):
    """chunk_gated_delta_rule's constant tiles (tests/ttnn/.../test_chunk_gated_delta_rule.py::_const_tiles)."""
    c = chunk_size
    eye = torch.eye(c, dtype=torch.float32)
    tril = torch.tril(torch.ones(c, c, dtype=torch.float32))
    ones = torch.ones(c, c, dtype=torch.float32)
    ii, jj = torch.arange(32).unsqueeze(1), torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

    def up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return dict(eye=up(eye), tril=up(tril), ones=up(ones), masks=up(masks))


def _dev(device, t, dtype):
    return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _timed(device, fn, iters: int):
    fn()  # compile / warm
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters * 1e3, out


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("regime", ["mild", "strong"])
@pytest.mark.parametrize("T", [2048, 8192])
def test_prefill_chunk_gdn_vs_ttkda(mesh_device, T, regime):
    """B = 1, 16 heads, K = V = 128, T tokens: ``chunk_gated_delta_rule`` (flat token-major path, chunk 32, the
    Qwen3.8 production call) fed the per-head MEAN of the per-channel gate, and the d_p ``KDARecurrence`` fed the
    per-channel gate, both vs HF ``recurrent_kimi_delta_attention`` (fp32, the exact KDA function)."""
    device = mesh_device
    chunk_ref, rec_ref, l2norm = _hf()
    B = 1
    q, k, v, sig, g_pc, beta_logits, beta, s0 = _inputs(B, T, regime)
    grid = device.compute_with_storage_grid_size()
    logger.info(f"grid {grid.x}x{grid.y} = {grid.x * grid.y} cores; BH = {B * H}")

    with torch.no_grad():
        o_ref, s_ref = rec_ref(q.float(), k.float(), v.float(), g_pc, beta, s0, True, use_qk_l2norm_in_kernel=True)
        g_head = g_pc.mean(-1)  # [B, T, H]: the only gate form the GDN op takes
        g_uniform = g_head[..., None].expand(B, T, H, K).contiguous()
        o_uni, s_uni = rec_ref(q.float(), k.float(), v.float(), g_uniform, beta, s0, True, use_qk_l2norm_in_kernel=True)
    o_ref_hm = o_ref.permute(0, 2, 1, 3).reshape(B * H, T, V)
    o_uni_hm = o_uni.permute(0, 2, 1, 3).reshape(B * H, T, V)
    logger.info(
        f"semantic gap of the per-head mean gate itself (reference vs reference): out pcc {pcc(o_ref, o_uni):.5f}"
    )

    q_f, k_f, v_f = (_dev(device, t.reshape(B, T, -1), ttnn.bfloat16) for t in (q, k, v))
    s0_dev = _dev(device, s0, ttnn.float32)
    consts = _const_tiles(device)
    g_head_dev, beta_dev = _dev(device, g_head, ttnn.float32), _dev(device, beta, ttnn.float32)

    # --- GDN op (scalar gate per head)
    def run_gdn():
        o, fs = ttnn.transformer.chunk_gated_delta_rule(
            q_f,
            k_f,
            v_f,
            g_head_dev,
            beta_dev,
            initial_state=s0_dev,
            output_final_state=True,
            chunk_size=32,
            output_head_major=True,
            **consts,
        )
        return o, fs

    ms_gdn, (o_g, fs_g) = _timed(device, run_gdn, 3)
    o_g_t, fs_g_t = ttnn.to_torch(o_g).float(), ttnn.to_torch(fs_g).float()
    p_gdn_pc, p_gdn_pc_s = pcc(o_ref_hm, o_g_t), pcc(s_ref, fs_g_t)
    p_gdn_uni, p_gdn_uni_s = pcc(o_uni_hm, o_g_t), pcc(s_uni, fs_g_t)
    logger.info(
        f"chunk_gated_delta_rule T={T} {regime}: {ms_gdn:.2f} ms/call; vs per-channel KDA ref out {p_gdn_pc:.5f} "
        f"state {p_gdn_pc_s:.5f}; vs uniform-gate ref out {p_gdn_uni:.5f} state {p_gdn_uni_s:.5f}"
    )

    # --- d_p ttKDA recurrence (per-channel gate; q/k normalized in-kernel; beta logits in a 32-wide column window)
    rec = KDARecurrence(
        device,
        KDARecurrenceProgramConfig(),
        sequence_parallel_axis=0,
        local_rows=T,
        heads=H,
        key_dim=K,
        value_dim=V,
        batch=B,
        gate_scale=GATE_LOWER_BOUND,
    )
    sig_f = _dev(device, sig.reshape(B, T, H * K), ttnn.bfloat16)
    beta_cols = torch.zeros(B, T, 32, dtype=torch.bfloat16)
    beta_cols[..., :H] = beta_logits
    beta_f = _dev(device, beta_cols, ttnn.bfloat16)
    start = device_scalar_u32(device, 0)

    def run_kda():
        r = rec(
            actual_start=start,
            q=q_f,
            k=k_f,
            v=v_f,
            gate=sig_f,
            beta=beta_f,
            initial_state=s0_dev,
            beta_logits_column_offset=0,
        )
        return r.output, r.final_state

    ms_kda, (o_k, fs_k) = _timed(device, run_kda, 3)
    o_k_t, fs_k_t = ttnn.to_torch(o_k).float(), ttnn.to_torch(fs_k).float().reshape(B, H, K, V)
    p_kda, p_kda_s = pcc(o_ref_hm, o_k_t), pcc(s_ref, fs_k_t)
    logger.info(
        f"ttKDA recurrence T={T} {regime}: {ms_kda:.2f} ms/call; vs per-channel KDA ref out {p_kda:.5f} state {p_kda_s:.5f}"
    )
    logger.info(
        f"SUMMARY T={T} {regime}: gdn {ms_gdn:.2f} ms (pcc {p_gdn_pc:.4f} per-channel / {p_gdn_uni:.4f} uniform), "
        f"ttkda {ms_kda:.2f} ms (pcc {p_kda:.4f}), ratio gdn/ttkda {ms_gdn / ms_kda:.2f}"
    )
    assert p_gdn_uni > 0.99 and p_gdn_uni_s > 0.99, "GDN op must match the uniform-gate reference (harness)"
    assert p_kda > 0.99 and p_kda_s > 0.99, "ttKDA must match the per-channel reference"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("B", [8, 32])
def test_decode_fused_recurrent_adapter(mesh_device, B):
    """T = 1, B users x 16 heads: ``S' = S * exp(g)[.., :, None]`` (one multiply) + ``fused_recurrent_gated_delta_rule``
    with ``g = 0`` on ``S'`` == KDA's per-channel decode step. The fused op takes one core per (user, head), so
    the users are processed in slices of ``cores // H`` per call (4 calls at B = 32 on a 130-core grid)."""
    device = mesh_device
    _, rec_ref, l2norm = _hf()
    T = 1
    q, k, v, sig, g_pc, beta_logits, beta, s0 = _inputs(B, T, "mild", seed=1)
    qn, kn = l2norm(q.float()), l2norm(k.float())  # the op wants host-normalized q/k
    with torch.no_grad():
        o_ref, s_ref = rec_ref(qn, kn, v.float(), g_pc, beta, s0, True, use_qk_l2norm_in_kernel=False)
    grid = device.compute_with_storage_grid_size()
    per_call = max(1, (grid.x * grid.y) // H)
    slices = [(s, min(s + per_call, B)) for s in range(0, B, per_call)]
    logger.info(f"grid {grid.x * grid.y} cores -> {per_call} users per fused call, {len(slices)} calls at B={B}")

    S = _dev(device, s0, ttnn.float32)  # [B, H, K, V]
    decay_col = _dev(device, g_pc.exp().reshape(B, H, K, 1), ttnn.float32)
    q_d, k_d, v_d = (_dev(device, t, ttnn.float32) for t in (qn, kn, v.float()))
    g0 = _dev(device, torch.zeros(B, T, H), ttnn.float32)
    beta_d = _dev(device, beta, ttnn.float32)

    def run():
        S_dec = ttnn.multiply(S, decay_col)  # per-channel decay = row scale of the state
        outs, states = [], []
        for a, b in slices:
            sl = lambda t, a=a, b=b: ttnn.slice(t, [a] + [0] * (len(t.shape) - 1), [b, *list(t.shape)[1:]])
            o, fs = ttnn.transformer.fused_recurrent_gated_delta_rule(
                sl(q_d), sl(k_d), sl(v_d), sl(g0), sl(beta_d), initial_state=sl(S_dec), output_final_state=True
            )
            outs.append(o)
            states.append(fs)
        o = outs[0] if len(outs) == 1 else ttnn.concat(outs, dim=0)
        fs = states[0] if len(states) == 1 else ttnn.concat(states, dim=0)
        return o, fs

    ms, (o, fs) = _timed(device, run, 5)
    o_t, fs_t = ttnn.to_torch(o).float().reshape(B, T, H, V), ttnn.to_torch(fs).float().reshape(B, H, K, V)
    p_o, p_s = pcc(o_ref, o_t), pcc(s_ref, fs_t)
    logger.info(
        f"decode adapter B={B} H={H}: {ms:.3f} ms/step eager ({len(slices)} fused calls); out pcc {p_o:.6f} state pcc {p_s:.6f}"
    )
    assert p_o > 0.999 and p_s > 0.999


@pytest.mark.parametrize("regime", ["mild", "strong"])
@pytest.mark.parametrize("T", [32, 256, 2048])
def test_gauge_fold_is_numerically_dead(T, regime):
    """Host only. The per-channel rule IS a plain (g = 0) delta rule in the basis ``S' = exp(-G) S``
    (``G`` = cumulative log decay) with READ key ``k * exp(+G)`` and WRITE key ``k * exp(-G)``: the gated_delta_attn_seq
    op takes separate q_decay / k_bd_sc / k_decay_t terms, so the asymmetry is expressible, but ``exp(+-G)`` leaves
    fp32 within a few hundred tokens (G ~ -0.3 .. -2.5 per token). Logged: PCC vs the exact rule per T."""
    _, rec_ref, l2norm = _hf()
    B = 1
    q, k, v, sig, g_pc, beta_logits, beta, s0 = _inputs(B, T, regime, seed=2)
    qn, kn = l2norm(q.float()), l2norm(k.float())
    with torch.no_grad():
        o_ref, s_ref = rec_ref(qn, kn, v.float(), g_pc, beta, s0, True, use_qk_l2norm_in_kernel=False)
    G = g_pc.cumsum(dim=1)  # [B, T, H, K]
    Sp = s0.clone()  # S' = exp(-G_0) S_0 with G before the first token = 0
    out = torch.zeros(B, T, H, V)
    scale = K**-0.5
    for t in range(T):
        eG = G[:, t].exp()  # [B, H, K]
        k_read, k_write, q_read = kn[:, t] * eG, kn[:, t] / eG, qn[:, t] * eG * scale
        kv = (Sp * k_read[..., None]).sum(-2)
        delta = (v[:, t].float() - kv) * beta[:, t][..., None]
        Sp = Sp + k_write[..., None] * delta[..., None, :]
        out[:, t] = (Sp * q_read[..., None]).sum(-2)
    S_final = G[:, -1].exp()[..., None] * Sp
    finite = bool(torch.isfinite(out).all() and torch.isfinite(S_final).all())
    p_o = pcc(o_ref, out) if finite else float("nan")
    p_s = pcc(s_ref, S_final) if finite else float("nan")
    logger.info(
        f"gauge fold T={T} {regime}: min G {G.min().item():.1f} (exp -> {G.min().exp().item():.2e}), finite={finite}, "
        f"out pcc {p_o:.5f} state pcc {p_s:.5f}"
    )
    if T == 32 and regime == "mild":  # exact only while exp(G) stays in fp32 range (one chunk, mild gate)
        assert finite and p_o > 0.99
