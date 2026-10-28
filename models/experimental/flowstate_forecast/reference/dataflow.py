# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU torch re-implementation of FlowState r1.1 in the device's decomposition (CONTRACT v1.1 §6 A).

Host front end (trim, NaN mask, appended [0, 1] steps, causal RevIN); embedding as x * w0 + flag * w1 + b; per layer
the B projection as one real matmul u @ [Bbar_r | Bbar_i]^T, the convolution as an FFT at a power-of-two N >= 2L - 1
with the reference-expression kernel, Re(C x) as one K = 1024 matmul [x_r | x_i] @ [C_r | -C_i]^T, the gate, + D u,
the MLP, LayerNorm, the residual; the last layer cut to the CPM-read rows; the decoder with lin and the Legendre basis
folded into one matrix; host CPM combine, de-normalization, torch.quantile and the weighted mean.

``scan_conv`` emulates route S (chunked complex scan, §6 B0) in any dtype, for the B0 error report.
"""

import math

import torch
import torch.nn.functional as F

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PATCH = 6  # decoder_patch_len
CONTEXT = 4096
DEC_DIM = 256
RANGE = (-1.0, 0.95)  # "legs" decoder


def bookkeeping(context, horizon, s):
    """FlowStateForPrediction.forward's index arithmetic."""
    fl = int(PATCH / s + 1e-6)
    mask_n = max(0, horizon - fl)
    max_context = min(16 * 1024, int(CONTEXT / s)) - mask_n
    ctx = min(context, max_context)
    return {
        "fl": fl,
        "mask_n": mask_n,
        "max_context": max_context,
        "L": ctx + mask_n,
        "min_context": ctx if mask_n > 0 else 0,
        "P": int(PATCH / s),
    }


def first_read_row(L, min_context):
    """First row the last layer keeps: o[min(mc, L) - 1:], which is the last row when mc is 0 or L."""
    return (min(min_context, L) - 1) % L


# ---------------------------------------------------------------- host front end and back end


def front_end(x, s, horizon):
    """x (B, T, 1) batch-first -> RevIN-normalized tokens (B, L, 2), stats (B, L, 1) each, bookkeeping."""
    bk = bookkeeping(x.shape[1], horizon, s)
    x = x[:, -bk["max_context"] :].float()
    flag = x.isnan().float()
    v = torch.nan_to_num(x, 0.0)
    if bk["mask_n"]:
        B = x.shape[0]
        v = torch.cat((v, torch.zeros(B, bk["mask_n"], 1)), 1)
        flag = torch.cat((flag, torch.ones(B, bk["mask_n"], 1)), 1)
    xin = torch.cat((v, flag), -1)
    # causal RevIN with missing values (FlowStateCausalRevIN._get_statistics), on both channels, then cut to one
    n = torch.cumsum(1 - flag[..., 0], 1).unsqueeze(-1)
    n = torch.where(n == 0, 1.0, n)
    mean2 = torch.cumsum(xin, 1) / n
    stdev2 = torch.sqrt(torch.clamp(torch.cumsum(((xin - mean2) * (1 - flag)) ** 2, 1) / n, min=1e-5))
    mean, stdev = mean2[..., :1], stdev2[..., :1]
    tok = (v - mean) / stdev * (1 - flag)
    return torch.cat((tok, flag), -1), mean, stdev, bk


def combine_cpm(pred, pred_len):
    """FlowStateForPrediction._combine_cpm_predictions, pred (n_preds, B, 9, fl, 1)."""
    n_preds, fl = pred.shape[0], pred.shape[3]
    lastp = pred[-1]
    rest = n_preds % fl - 1
    out = torch.cat([pred[i] for i in range(0, n_preds, fl)], 2)
    if rest > 0:
        out = torch.cat((out, lastp[:, :, -rest:]), 2)
    return out[:, :, :pred_len]


def back_end(dec, mean, stdev, horizon):
    """dec (rows, B, 9, P, 1) pre-denorm -> quantile_outputs (B, 9, H', 1), prediction_outputs (B, H', 1)."""
    rows = dec.shape[0]
    sd = stdev[:, -rows:].transpose(0, 1).unsqueeze(2).unsqueeze(-2)
    mu = mean[:, -rows:].transpose(0, 1).unsqueeze(2).unsqueeze(-2)
    pred = combine_cpm(dec * sd + mu, horizon)
    q = torch.tensor(QUANTILES)
    quantiles = torch.quantile(pred, q, 1).transpose(0, 1)
    w = 0.5 - (0.5 - q).abs()
    w = (w / w.sum()).view(1, -1, 1, 1)
    return quantiles, (quantiles * w).sum(1)


# ---------------------------------------------------------------- constants (the reference's own expressions)


def ssm_constants(ssm, s, L):
    """log Lbar (P,), Bbar (P, H) and the flipped kernel K_k = exp(k log Lbar), k = 0..L-1, all complex64."""
    lam = -torch.exp(ssm.log_Lambda_real) + 1j * ssm.Lambda_imag
    sf = torch.ones(1) * s
    log_lbar = (sf.repeat((lam.shape[0], 1)).T.unsqueeze(1) * lam[None, None, :] * torch.exp(ssm.log_Delta))[0, 0]
    K = torch.einsum("d,L->Ld", log_lbar, torch.arange(L)).exp()
    Bbar = (1 / lam * (K[1] - 1.0))[..., None] * (ssm.B_tilde_r + 1.0j * ssm.B_tilde_i)
    return log_lbar, Bbar, K


def legendre_w(s, P):
    """Folded-decoder basis (P, 256): FlowStateFunctionalBasisDecoder.get_kernel."""
    dt = s * (RANGE[1] - RANGE[0]) / PATCH
    t = torch.clamp(RANGE[0] + torch.arange(1, P + 1, dtype=torch.float) * dt, min=RANGE[0])
    f = torch.ones(P, DEC_DIM)
    f[:, 1] = t
    for i in range(1, DEC_DIM - 1):
        f[:, i + 1] = ((2 * i + 1) * t * f[:, i] - i * f[:, i - 1]) / (i + 1)
    return f / 4.0


# ---------------------------------------------------------------- convolution routes


def fft_conv(Bu, K, N=None):
    """x_t = sum_k K_k Bu_{t-k}; Bu (L, B, P), K (L, P) complex. N defaults to the power of two >= 2L - 1."""
    L = Bu.shape[0]
    N = N or 1 << (2 * L - 1 - 1).bit_length()
    return torch.fft.ifft(torch.fft.fft(K, n=N, dim=0).unsqueeze(1) * torch.fft.fft(Bu, n=N, dim=0), n=N, dim=0)[:L]


def recurrence(Bu, lbar):
    """The float64 truth: x_t = Lbar x_{t-1} + Bu_t in complex128."""
    Bu, lbar = Bu.to(torch.complex128), lbar.to(torch.complex128)
    x = torch.empty_like(Bu)
    h = torch.zeros_like(Bu[0])
    for t in range(Bu.shape[0]):
        h = lbar * h + Bu[t]
        x[t] = h
    return x


def scan_conv(Bu, lbar, c=64, dtype=torch.complex64, limit=1e30):
    """Route S emulation. Intra-chunk x = Lbar^t (Tri @ (Lbar^-t Bu)) where |Lbar|^-(c-1) max|Bu| < limit, a per-mode
    Toeplitz block Lbar^(t-k) (|entries| <= 1) elsewhere; powers by repeated multiplication; carry over chunks.
    Returns x (L, B, P) and the boolean mask of modes that took the bounded variant."""
    L, B, P = Bu.shape
    Bu, lbar = Bu.to(dtype), lbar.to(dtype)
    pw = torch.cumprod(torch.cat((torch.ones(1, P, dtype=dtype), lbar.expand(c, P))), 0)  # Lbar^0..Lbar^c
    inv = torch.cumprod(torch.cat((torch.ones(1, P, dtype=dtype), (1 / lbar).expand(c - 1, P))), 0)  # Lbar^-t
    # in log space: a complex inf times anything is NaN, and NaN >= limit is False
    log_amp = -(c - 1) * lbar.abs().double().log() + Bu.abs().amax((0, 1)).double().clamp(min=1e-300).log()
    bounded = log_amp >= math.log(limit)
    tri = torch.tril(torch.ones(c, c, dtype=dtype))
    idx = torch.arange(c)
    toe = torch.where(
        (idx[:, None] >= idx[None, :])[..., None], pw[(idx[:, None] - idx[None, :]).clamp(min=0)], 0
    )  # (c, c, P)
    nchunk = math.ceil(L / c)
    pad = torch.zeros(nchunk * c - L, B, P, dtype=dtype)
    u = torch.cat((Bu, pad)).view(nchunk, c, B, P)
    scaled = pw[:c, None] * torch.einsum("ts,jsbp->jtbp", tri, inv[:, None] * u)
    toep = torch.einsum("tsp,jsbp->jtbp", toe, u)
    x = torch.where(bounded, toep, scaled)
    h = torch.zeros(B, P, dtype=dtype)
    for j in range(nchunk):
        x[j] = x[j] + pw[1:, None] * h
        h = x[j, -1]
    return x.reshape(nchunk * c, B, P)[:L], bounded


# ---------------------------------------------------------------- the forward


def layer_forward(lay, u, s, min_context, last, conv="fft", N=None):
    """One FlowStateS5Layer; u (L, B, 512) fp32. Returns (out, parts) with parts = {ssm, mlp, x}."""
    L = u.shape[0]
    ssm = lay.ssm
    _, Bbar, K = ssm_constants(ssm, s, L)
    Bcat = torch.cat((Bbar.real, Bbar.imag), 0)  # (1024, 512)
    bu = u @ Bcat.T
    Bu = torch.complex(bu[..., :512], bu[..., 512:])
    x = fft_conv(Bu, K, N) if conv == "fft" else conv(Bu, K)
    r = first_read_row(L, min_context) if last else 0
    x, u = x[r:], u[r:]
    Ccat = torch.cat((ssm.C_tilde_r, -ssm.C_tilde_i), 1)  # (512, 1024)
    y = torch.cat((x.real, x.imag), -1) @ Ccat.T
    y = torch.sigmoid(F.linear(y, ssm.output_gate.weight, ssm.output_gate.bias)) * y
    y = y + ssm.D * u
    m = lay.mlp
    h = F.rms_norm(y, (512,), m.rms_weight, 1e-6)
    z = (
        F.linear(
            F.silu(F.linear(h, m.w1.weight, m.w1.bias)) * F.linear(h, m.w2.weight, m.w2.bias), m.w3.weight, m.w3.bias
        )
        + y
    )
    out = u + F.layer_norm(z, (512,), lay.norm.weight, lay.norm.bias, 1e-5)
    return out, {"ssm": y, "mlp": z, "x": x}


@torch.no_grad()
def forward(model, x, s, horizon, conv="fft", N=None):
    """model: the HF FlowStateForPrediction (weights only). x (B, T, 1). Returns a dict of intermediates + outputs."""
    m = model.model
    tok, mean, stdev, bk = front_end(x, s, horizon)
    w = m.embed.embed.weight  # (512, 2)
    h = (tok[..., :1] * w[:, 0] + tok[..., 1:] * w[:, 1] + m.embed.embed.bias).transpose(0, 1).contiguous()
    d = {"revin_mean": mean, "revin_stdev": stdev, "normalized": tok, "embed": h, **bk}
    layers = m.encoder.layers
    for i, lay in enumerate(layers):
        h, parts = layer_forward(lay, h, s, bk["min_context"], i == len(layers) - 1, conv, N)
        d[f"layer{i}"], d[f"ssm{i}"], d[f"mlp{i}"] = h, parts["ssm"], parts["mlp"]
    # folded decoder: (rows, B, 512) @ [lin per quantile composed with W^T] -> (rows, B, 9, P)
    W = legendre_w(s, bk["P"])  # (P, 256)
    lw = m.decoder.lin.weight.view(9, DEC_DIM, 512)
    fold_w = torch.einsum("ph,qhn->qpn", W, lw).reshape(9 * bk["P"], 512)
    fold_b = (m.decoder.lin.bias.view(9, DEC_DIM) @ W.T).reshape(-1)
    dec = (h @ fold_w.T + fold_b).view(*h.shape[:2], 9, bk["P"], 1)
    d["decoder_pre_denorm"] = dec
    d["combined_normalized"] = combine_cpm(dec, horizon)
    d["quantile_outputs"], d["prediction_outputs"] = back_end(dec, mean, stdev, horizon)
    return d
