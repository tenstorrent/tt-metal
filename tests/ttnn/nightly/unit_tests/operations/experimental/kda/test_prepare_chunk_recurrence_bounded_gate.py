# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Repro: prepare_chunk_recurrence's decay factors lose precision under a bounded gate (gate_scale -5).

Kimi-K3 and GLM-5.3-Flash run KDA with a bounded gate: log decay = gate_lower_bound (-5) x sigmoid(...), passed to
this op as g in (0, 1) and gate_scale = -5. Within a 32-token chunk G = cumsum(gate_scale * g) then reaches about
-150. The kernel's prepare_gate_factors keeps G and G_last / 2 in FP32 but reads them back through the FPU source
registers (TF32, 10-bit mantissa) for G - G_last / 2: at |G| >= 128 the step is 0.125, so exp(G - G_last / 2) and
everything built from it (k_dec_t, final_decay, kd, q_decay, intra, t_inv) is off by up to ~6%.

The existing test (test_prepare_chunk_recurrence.py) draws g in [-0.051, -0.001] with gate_scale 1, so |G| < 2 and
the rounding never shows. Two regimes here, same shapes (head dim 128, the models'):
  - main_test_gate: the existing test's gate range - passes.
  - bounded_gate:   gate_scale -5, g = sigmoid output, slow and fast-decay heads - fails.

Run: pytest tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_prepare_chunk_recurrence_bounded_gate.py -s
"""

from __future__ import annotations

import pytest
import torch

import ttnn

CHUNK = 32
NAMES = ("v_beta", "kd", "q_decay", "intra", "k_dec_t", "final_decay", "t_inv")
HEADS, CHUNKS, DK, DV = 4, 4, 128, 128


def _inputs(regime: str, seed: int = 7):
    """q, k, v, g, beta_logits as the op takes them ([1, T, H * d] token-major, BF16 values), and gate_scale."""
    gen = torch.Generator().manual_seed(seed)
    T = CHUNKS * CHUNK
    q = (0.3 * torch.randn(1, T, HEADS * DK, generator=gen)).to(torch.bfloat16).float()
    k = (0.3 * torch.randn(1, T, HEADS * DK, generator=gen)).to(torch.bfloat16).float()
    v = (0.2 * torch.randn(1, T, HEADS * DV, generator=gen)).to(torch.bfloat16).float()
    beta = torch.randn(1, T, HEADS, generator=gen).to(torch.bfloat16).float()
    if regime == "main_test_gate":
        g = -0.001 - 0.05 * torch.rand(1, T, HEADS * DK, generator=gen)
        scale = 1.0
    else:
        # sigmoid outputs: heads 0-1 spread over (0, 1) (slow and fast channels), heads 2-3 near the bound (0.8-1.0)
        g = torch.rand(1, T, HEADS, DK, generator=gen)
        g[:, :, 2:] = 0.8 + 0.2 * g[:, :, 2:]
        g = g.reshape(1, T, HEADS * DK)
        scale = -5.0
    return (q, k, v, g.to(torch.bfloat16).float(), beta), scale


def _heads(t, d):  # [1, T, H * d] -> [H, chunks, 32, d]
    return t.double().reshape(CHUNKS * CHUNK, HEADS, d).permute(1, 0, 2).reshape(HEADS, CHUNKS, CHUNK, d)


def _oracle(inputs, scale):
    """The op's seven outputs in FP64, plus G (the scaled within-chunk cumsum) for binning the errors."""
    q, k, v, g, beta_logits = inputs
    beta = _heads(torch.sigmoid(beta_logits.double()), 1)
    q, k, v, g = _heads(q, DK), _heads(k, DK), _heads(v, DV), _heads(g, DK)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * DK**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    G = torch.cumsum(scale * g, dim=2)
    G_last = G[:, :, -1:]
    anchor = 0.5 * G_last
    v_beta = beta * v
    kd = beta * k * torch.exp(G)
    q_decay = q * torch.exp(G)
    k_dec_t = (k * torch.exp(G_last - G)).transpose(-1, -2)
    final_decay = torch.exp(G_last).transpose(-1, -2)
    left_k = beta * k * torch.exp(G - anchor)
    right = (k * torch.exp(anchor - G)).transpose(-1, -2)
    akk = left_k @ right
    intra = ((q * torch.exp(G - anchor)) @ right).tril()
    eye = torch.eye(CHUNK, dtype=torch.float64)
    t_inv = torch.linalg.inv(eye + torch.tril(akk, diagonal=-1))
    return (v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv), G


def _device(inputs, scale, device):
    tt = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in inputs]
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    outs = ttnn.experimental.kda.prepare_chunk_recurrence(
        *tt, HEADS, output_bf16_mask=0, compute_kernel_config=cfg, gate_scale=scale
    )
    return [ttnn.to_torch(o).double() for o in outs]


def _rel(exp, act, tril=False):
    """Max and median relative error over elements well inside FP32 range (|ref| > 1e-30, > 1e-6 of the max)."""
    if tril:
        m = torch.ones_like(exp, dtype=torch.bool).tril()
    else:
        m = torch.ones_like(exp, dtype=torch.bool)
    m &= (exp.abs() > 1e-30) & (exp.abs() > 1e-6 * exp.abs().max())
    r = ((act - exp).abs() / exp.abs())[m]
    nrmse = ((act - exp)[m].square().mean().sqrt() / exp[m].square().mean().sqrt()).item()
    return r.max().item(), r.median().item(), nrmse


@pytest.mark.parametrize("regime", ["main_test_gate", "bounded_gate"])
def test_prepare_chunk_recurrence_bounded_gate(device, regime):
    inputs, scale = _inputs(regime)
    ref, G = _oracle(inputs, scale)
    act = _device(inputs, scale, device)
    print(f"\n[{regime}] gate_scale {scale}, |G| max {G.abs().max().item():.1f}")
    print(f"  {'output':12s} {'max rel err':>12s} {'median rel':>12s} {'nrmse':>10s}")
    worst = {}
    for name, e, a in zip(NAMES, ref, act):
        a = a.reshape(e.shape)
        mx, md, nr = _rel(e, a, tril=name == "intra")
        worst[name] = mx
        print(f"  {name:12s} {mx:12.2e} {md:12.2e} {nr:10.2e}")

    # k_dec_t's error against the size of G, where the kernel rounds it: the step of a 10-bit mantissa at |G| is
    # 2^(floor(log2 |G|) - 10), half of it the worst rounding, exp() turns it into a relative error
    k_dec_t = act[NAMES.index("k_dec_t")].reshape(ref[NAMES.index("k_dec_t")].shape).transpose(-1, -2)
    k_ref = ref[NAMES.index("k_dec_t")].transpose(-1, -2)
    mag = torch.maximum(G.abs(), 0.5 * G[:, :, -1:].abs().expand_as(G))
    ok = (k_ref.abs() > 1e-30) & (k_ref.abs() > 1e-6 * k_ref.abs().amax(dim=(-1, -2), keepdim=True))
    r = (k_dec_t - k_ref).abs() / k_ref.abs()
    print("  k_dec_t rel error by max(|G|, |G_last| / 2):  bin  ->  measured max | TF32 rounding bound")
    for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256)]:
        sel = ok & (mag >= lo) & (mag < hi)
        if sel.any():
            bound = torch.expm1(
                torch.tensor((hi / 2) * 2.0**-10)
            ).item()  # (G and the anchor each off by half a step)
            print(f"    [{lo:3d}, {hi:3d})  ->  {r[sel].max().item():8.2e} | ~{bound:8.2e}")

    # main_test_gate shows the op's floor here (k_dec_t <= ~6e-3 at |G| < 2); 2.5x that as the bound
    assert worst["k_dec_t"] < 1.5e-2 and worst["final_decay"] < 1.5e-2, (
        f"{regime}: k_dec_t max rel err {worst['k_dec_t']:.3e}, final_decay {worst['final_decay']:.3e} "
        "(TF32 rounding of G in prepare_gate_factors)"
    )
