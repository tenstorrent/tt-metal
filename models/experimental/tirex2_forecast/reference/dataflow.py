# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TiRex-2 forward in plain torch, written from the math in the decomposition the TTNN port uses.

This is not the upstream code (which needs flashrnn and mlstm_kernels); it is a re-derivation that the
device modules mirror op for op, checked against the upstream fp32 CPU goldens in
``tests/golden/test_dataflow.py``. The decomposition choices that matter on device:

* Every linear is ``x @ W`` with ``W`` stored ``[in, out]``; biases are added after the matmul.
* mLSTM projections are one packed matmul ``[q | k | v | o | i | f]`` (512 -> 1544).
* The mLSTM cell is the stabilized chunkwise form (chunk ``L``, default 64; the sequence is padded to a
  multiple of ``L``). The running max ``m_t`` is chunk-size invariant in exact arithmetic, so ``L`` is a
  free parameter; the ``+ eps`` after the denominator max is kept (it does not commute with dropping the
  stabilizer).
* The sLSTM headwise input projections are one block-diagonal ``[512, 2048]`` matmul whose output columns
  are ``(gate, head, d)`` with the gate slots in kernel order ``(i, f, z, o)``: slot 0 is the module named
  ``fgate`` and slot 1 the module named ``igate``. The gate bias is folded into ``Wx``. The recurrent
  matmul is one block-diagonal ``[512, 2048]`` matrix with the same column order.
* The sLSTM step uses ``m_t = pre_i`` at ``t = 0`` (the reference's global ``all(n == 0)`` rule, which
  only holds before the first step since ``n >= 1`` afterwards).
* MultiHeadLayerNorm is a per-head (128) mean/variance norm, eps 1e-6, times the [512] weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

D_MODEL = 512
HEADS = 4
PATCH = 32
CONTEXT_LEN = 8192
FUTURE_LEN = 928
NUM_TOKENS = (CONTEXT_LEN + FUTURE_LEN) // PATCH  # 285
NUM_BLOCKS = 12
QUANTILES = 9
QK_DIM = 64  # per head
V_DIM = 128  # per head
FFN_DIM = 1408
GATE_CAP = 15.0
MLSTM_EPS = 1e-6
NORM_EPS = 1e-6
OUT_NORM_EPS = 1e-7
READOUT_FIRST = (CONTEXT_LEN // PATCH) - 1  # token 255 predicts the first future patch
RAW_BAND_SCALE = 0.90
RAW_BAND_EXPONENT = 0.52


def is_mlstm(block: int) -> bool:
    return block % 2 == 0


# ---------------------------------------------------------------------------------------------- weights


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return {k: v.float() for k, v in load_file(str(path)).items()}


def default_checkpoint() -> Path:
    import os

    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    return (
        root
        / "models--NX-AI--TiRex-2-gifteval-pretrain"
        / "snapshots"
        / "54f88ac06ccfba9fe3e727aa8856b39775481f0f"
        / "model.safetensors"
    )


def _t(w: torch.Tensor) -> torch.Tensor:
    """torch Linear weight [out, in] -> [in, out]."""
    return w.t().contiguous()


@dataclass
class ResidualBlockW:
    w_hidden: torch.Tensor  # [in, h]
    b_hidden: torch.Tensor
    w_out: torch.Tensor  # [h, out]
    b_out: torch.Tensor
    w_res: torch.Tensor  # [in, out]
    b_res: torch.Tensor


@dataclass
class FFNW:
    w_gate: torch.Tensor  # [512, 1408] (proj_up_gate, through SiLU)
    w_up: torch.Tensor  # [512, 1408]
    w_down: torch.Tensor  # [1408, 512]


@dataclass
class MLSTMW:
    w_qkvoif: torch.Tensor  # [512, 256 + 256 + 512 + 512 + 4 + 4]
    b_if: torch.Tensor  # [8]: i bias (4), f bias (4)
    w_mhnorm: torch.Tensor  # [512]
    w_out: torch.Tensor  # [512, 512]


@dataclass
class SLSTMW:
    w_in: torch.Tensor  # [512, 2048] block-diagonal, columns (gate, head, d), gates (i, f, z, o)
    bias: torch.Tensor  # [2048], same column order
    r: torch.Tensor  # [512, 2048] block-diagonal recurrent kernel, same column order
    w_gnorm: torch.Tensor  # [512]


@dataclass
class VariateW:
    w_norm_attn: torch.Tensor
    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    w_norm_ffn: torch.Tensor
    w_wi: torch.Tensor  # [512, 2048]
    w_wo: torch.Tensor  # [2048, 512]


@dataclass
class BlockW:
    index: int
    w_norm_lstm: torch.Tensor
    w_norm_ffn: torch.Tensor
    ffn: FFNW
    cell: MLSTMW | SLSTMW
    variate: VariateW


@dataclass
class TiRex2W:
    embed: ResidualBlockW
    blocks: list[BlockW]
    w_out_norm: torch.Tensor
    head: ResidualBlockW
    extra: dict = field(default_factory=dict)


def _residual(sd, p) -> ResidualBlockW:
    return ResidualBlockW(
        _t(sd[f"{p}.hidden_layer.weight"]),
        sd[f"{p}.hidden_layer.bias"],
        _t(sd[f"{p}.output_layer.weight"]),
        sd[f"{p}.output_layer.bias"],
        _t(sd[f"{p}.residual_layer.weight"]),
        sd[f"{p}.residual_layer.bias"],
    )


def headwise_block_diag(weights: list[torch.Tensor]) -> torch.Tensor:
    """Gate weights ``[H, out_d, in_d]`` (LinearHeadwiseExpand) -> ``[H * in_d, G * H * out_d]``."""
    G = len(weights)
    H, Do, Di = weights[0].shape
    W = torch.zeros(H * Di, G * H * Do)
    for g, w in enumerate(weights):
        for h in range(H):
            W[h * Di : (h + 1) * Di, (g * H + h) * Do : (g * H + h + 1) * Do] = w[h].t()
    return W


def recurrent_block_diag(R: torch.Tensor) -> torch.Tensor:
    """flashrnn ``R [H, P, G, D]`` ("HPGD") -> ``[H * P, G * H * D]`` with ``y_prev @ out``."""
    H, P, G, D = R.shape
    W = torch.zeros(H * P, G * H * D)
    for h in range(H):
        for g in range(G):
            W[h * P : (h + 1) * P, (g * H + h) * D : (g * H + h + 1) * D] = R[h, :, g, :]
    return W


def build_weights(sd: dict[str, torch.Tensor]) -> TiRex2W:
    blocks = []
    for b in range(NUM_BLOCKS):
        p = f"stack.{b}.time_mixer"
        if is_mlstm(b):
            c = f"{p}.fwd_cell"
            w = torch.cat(
                [sd[f"{c}.{n}.weight"] for n in ("q", "k", "v", "ogate_preact", "igate_preact", "fgate_preact")], 0
            )
            cell = MLSTMW(
                _t(w),
                torch.cat([sd[f"{c}.igate_preact.bias"], sd[f"{c}.fgate_preact.bias"]]),
                sd[f"{c}.multihead_norm.weight"],
                _t(sd[f"{c}.out_proj.weight"]),
            )
        else:
            c = f"{p}.fwd_cell"
            # kernel slot order (i, f, z, o) = modules (fgate, igate, zgate, ogate)
            gates = [sd[f"{c}.{n}.weight"] for n in ("fgate", "igate", "zgate", "ogate")]
            bias = sd[f"{c}._bias_"]  # [H, G, D]
            cell = SLSTMW(
                headwise_block_diag(gates),
                bias.permute(1, 0, 2).reshape(-1).contiguous(),
                recurrent_block_diag(sd[f"{c}._recurrent_kernel_"]),
                sd[f"{c}.group_norm.weight"],
            )
        v = f"stack.{b}.variate_mixer"
        blocks.append(
            BlockW(
                b,
                sd[f"{p}.norm_lstm.weight"],
                sd[f"{p}.norm_ffn.weight"],
                FFNW(
                    _t(sd[f"{p}.ffn.proj_up_gate.weight"]),
                    _t(sd[f"{p}.ffn.proj_up.weight"]),
                    _t(sd[f"{p}.ffn.proj_down.weight"]),
                ),
                cell,
                VariateW(
                    sd[f"{v}.norm_attn.weight"],
                    _t(sd[f"{v}.attn.WQ.weight"]),
                    _t(sd[f"{v}.attn.WK.weight"]),
                    _t(sd[f"{v}.attn.WV.weight"]),
                    _t(sd[f"{v}.attn.WO.weight"]),
                    sd[f"{v}.norm_ffn.weight"],
                    _t(sd[f"{v}.ffn.wi.weight"]),
                    _t(sd[f"{v}.ffn.wo.weight"]),
                ),
            )
        )
    return TiRex2W(
        _residual(sd, "input_patch_embedding"),
        blocks,
        sd["stack_out_norm.weight"],
        _residual(sd, "output_patch_embedding"),
    )


# ------------------------------------------------------------------------------------------ host side


@dataclass
class Prepared:
    tokens: torch.Tensor  # [rows, 285, 64] (scaled values, observed flags)
    loc: torch.Tensor  # [rows, 1]
    scale: torch.Tensor  # [rows, 1]
    x_padded: torch.Tensor  # [rows, 9120]
    horizon: int
    group_ids: torch.Tensor  # [rows] (int), consecutive rows of a group
    target_mask: torch.Tensor  # [rows] bool


def pad_context(context: torch.Tensor, horizon: int, pad_context: bool = True) -> torch.Tensor:
    """``[rows, T]`` -> ``[rows, 9120]``: 928 NaN future steps, then left NaN pad (or truncation) to 9120."""
    x = F.pad(context.float(), (0, FUTURE_LEN), value=float("nan"))
    total = CONTEXT_LEN + FUTURE_LEN
    if x.shape[-1] < total and pad_context:
        x = F.pad(x, (total - x.shape[-1], 0), value=float("nan"))
    elif x.shape[-1] > total:
        x = x[..., -total:]
    return x


def scaler(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """nanmean and population nanstd over the whole padded row (eps 1e-8; all-NaN rows -> loc 0, scale 1)."""
    loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num(torch.nanmean((x - loc) ** 2, dim=-1, keepdim=True).sqrt(), nan=1.0)
    return loc, torch.clamp_min(scale, 1e-8)


def tokenize(x_padded: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, length = x_padded.shape
    assert length % PATCH == 0
    patches = ((x_padded - loc) / scale).reshape(rows, length // PATCH, PATCH)
    observed = (~torch.isnan(patches)).float()
    return torch.cat([torch.nan_to_num(patches, nan=0.0), observed], dim=-1)


def prepare(
    context: torch.Tensor,
    horizon: int,
    group_ids: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
    pad: bool = True,
) -> Prepared:
    """``context [rows, T]`` (targets and past covariates, consecutive rows per group)."""
    rows = context.shape[0]
    x = pad_context(context, horizon, pad)
    loc, scale = scaler(x)
    if group_ids is None:
        group_ids = torch.arange(rows)
    if target_mask is None:
        target_mask = torch.ones(rows, dtype=torch.bool)
    return Prepared(tokenize(x, loc, scale), loc, scale, x, horizon, group_ids, target_mask)


def readout_tokens(horizon: int) -> list[int]:
    return list(range(READOUT_FIRST, READOUT_FIRST + math.ceil(min(horizon, FUTURE_LEN) / PATCH)))


def head_to_quantiles(head_readout: torch.Tensor, loc, scale, horizon: int) -> torch.Tensor:
    """Head output at the read-out tokens ``[rows, n, 288]`` -> level-space ``[rows, 9, horizon]``."""
    rows, n, _ = head_readout.shape
    q = head_readout.reshape(rows, n, QUANTILES, PATCH).permute(0, 2, 1, 3).reshape(rows, QUANTILES, n * PATCH)
    q = q[..., :horizon]
    return q * scale[:, None] + loc[:, None]


def band_calibration(q: torch.Tensor) -> torch.Tensor:
    """Raw-row band calibration ``med + sqrt(0.9) * t^0.02 * (q - med)`` over ``t = 1..H``."""
    med = q[:, QUANTILES // 2 : QUANTILES // 2 + 1, :]
    t = torch.arange(1, q.shape[-1] + 1, dtype=q.dtype)
    return med + (RAW_BAND_SCALE**0.5) * t ** (RAW_BAND_EXPONENT - 0.5) * (q - med)


def postprocess(head_readout: torch.Tensor, prep: Prepared) -> torch.Tensor:
    """-> ``[targets, 9, H]`` single-pass forecast (inverse scaling, band calibration, target rows)."""
    q = band_calibration(head_to_quantiles(head_readout, prep.loc, prep.scale, prep.horizon))
    return q[prep.target_mask]


def sign_flip_average(q_pos: torch.Tensor, q_neg: torch.Tensor) -> torch.Tensor:
    """Average ``f(x)`` with ``-f(-x)`` mapped back (quantile axis reversed: q -> 1 - q)."""
    return (q_pos + (-q_neg.flip(1))) / 2


# -------------------------------------------------------------------------------------------- modules


def linear(x, w, b=None):
    y = x @ w
    return y if b is None else y + b


def residual_block(x, w: ResidualBlockW):
    h = F.silu(linear(x, w.w_hidden, w.b_hidden))
    return linear(h, w.w_out, w.b_out) + linear(x, w.w_res, w.b_res)


def rms_norm(x, w, eps=NORM_EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def mh_layer_norm(x, w, heads=HEADS, eps=NORM_EPS):
    """Per-head LayerNorm (no bias) of ``[..., heads * d]``."""
    shape = x.shape
    xh = x.reshape(*shape[:-1], heads, -1)
    xh = xh - xh.mean(-1, keepdim=True)
    xh = xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + eps)
    return xh.reshape(shape) * w


def ffn(x, w: FFNW):
    return linear(F.silu(linear(x, w.w_gate)) * linear(x, w.w_up), w.w_down)


def soft_cap(x, cap=GATE_CAP):
    return cap * torch.tanh(x / cap)


def mlstm_chunkwise(q, k, v, i, f, chunk: int = 64, eps: float = MLSTM_EPS, return_states: bool = False):
    """Stabilized chunkwise mLSTM. ``q, k [B, H, S, 64]``, ``v [B, H, S, 128]``, ``i, f [B, H, S]`` (capped).

    Per chunk of ``L`` steps, with ``b = cumsum(logsigmoid(f))`` inside the chunk and the carried state
    ``(C, n, m)`` (``m`` starts at 0):

        D[t, s]  = b_t - b_s + i_s  (s <= t, else -inf)
        m_t      = max(b_t + m_prev, max_s D[t, s])
        num_t    = exp(b_t + m_prev - m_t) (q_t / 8) C  + sum_s (q_t . k_s / 8) exp(D[t, s] - m_t) v_s
        den_t    = exp(b_t + m_prev - m_t) (q_t / 8) . n + sum_s (q_t . k_s / 8) exp(D[t, s] - m_t)
        h_t      = num_t / (max(|den_t|, exp(-m_t)) + eps)

    and the state update over the chunk (``g = b_{L-1}``, ``a_s = g - b_s + i_s``):

        m'  = max(g + m_prev, max_s a_s)
        C'  = exp(g + m_prev - m') C + sum_s exp(a_s - m') k_s v_s^T,   n' likewise with v_s -> 1
    """
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    pad = (-S) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        i = F.pad(i, (0, pad), value=-25.0)
        f = F.pad(f, (0, pad), value=25.0)
    nc = (S + pad) // chunk
    L = chunk
    qc = q.reshape(B, H, nc, L, DK) * (DK**-0.5)
    kc = k.reshape(B, H, nc, L, DK)
    vc = v.reshape(B, H, nc, L, DV)
    ic = i.reshape(B, H, nc, L)
    bc = F.logsigmoid(f).reshape(B, H, nc, L).cumsum(-1)
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    C = torch.zeros(B, H, DK, DV)
    n = torch.zeros(B, H, DK)
    m = torch.zeros(B, H)
    out = []
    states = []
    for c in range(nc):
        states.append((C, n, m))
        b_, i_, q_, k_, v_ = bc[:, :, c], ic[:, :, c], qc[:, :, c], kc[:, :, c], vc[:, :, c]
        D = (b_[..., :, None] - b_[..., None, :] + i_[..., None, :]).masked_fill(~causal, float("-inf"))
        m_t = torch.maximum(b_ + m[..., None], D.max(-1).values)  # [B, H, L]
        W = torch.exp(D - m_t[..., None]) * (q_ @ k_.transpose(-1, -2))  # [B, H, L, L]
        inter = torch.exp(b_ + m[..., None] - m_t)[..., None]  # [B, H, L, 1]
        num = inter * (q_ @ C) + W @ v_
        den = inter[..., 0] * (q_ @ n[..., None])[..., 0] + W.sum(-1)
        out.append(num / (torch.maximum(den.abs(), torch.exp(-m_t)) + eps)[..., None])
        g = b_[..., -1]
        a = g[..., None] - b_ + i_
        m_next = torch.maximum(g + m, a.max(-1).values)
        ka = k_ * torch.exp(a - m_next[..., None])[..., None]
        decay = torch.exp(g + m - m_next)
        C = decay[..., None, None] * C + ka.transpose(-1, -2) @ v_
        n = decay[..., None] * n + ka.sum(-2)
        m = m_next
    states.append((C, n, m))
    h = torch.cat(out, dim=2)[:, :, :S]
    return (h, states) if return_states else h


def mlstm_cell(xn, w: MLSTMW, chunk: int = 64, capture: dict | None = None):
    """``xn [B, S, 512]`` (normed) -> ``y [B, S, 512]``."""
    B, S, _ = xn.shape
    p = linear(xn, w.w_qkvoif)
    q, k, v, o, g = torch.split(p, [256, 256, 512, 512, 8], dim=-1)
    g = soft_cap(g + w.b_if)
    to_heads = lambda t: t.reshape(B, S, HEADS, -1).transpose(1, 2)
    i = g[..., :HEADS].transpose(1, 2)
    f = g[..., HEADS:].transpose(1, 2)
    h = mlstm_chunkwise(to_heads(q), to_heads(k), to_heads(v), i, f, chunk=chunk)
    if capture is not None:
        capture.update(q=to_heads(q), k=to_heads(k), v=to_heads(v), i=i, f=f, h=h)
    h = h.transpose(1, 2).reshape(B, S, -1)
    return linear(torch.sigmoid(o) * mh_layer_norm(h, w.w_mhnorm), w.w_out)


def slstm_input(xn, w: SLSTMW):
    """``Wx`` with the folded gate bias: ``[B, S, 2048]``, columns ``(gate i f z o, head, d)``."""
    return linear(xn, w.w_in, w.bias)


def slstm_recurrence(Wx, r, state=None, steps: int | None = None):
    """sLSTM time loop over ``Wx [B, S, 2048]``; returns ``y [B, S, 512]`` and per-step ``(y, c, n, m)``.

    ``state`` is ``(y, c, n, m)`` each ``[B, 512]`` or ``None`` (zeros; then step 0 uses ``m = pre_i``).
    """
    B, S, _ = Wx.shape
    D = D_MODEL
    steps = S if steps is None else steps
    first = state is None
    if first:
        y = c = n = m = torch.zeros(B, D)
    else:
        y, c, n, m = state
    ys, all_states = [], []
    for t in range(steps):
        pre = Wx[:, t] + y @ r
        pi, pf, pz, po = pre.split(D, dim=-1)
        lf = F.logsigmoid(pf) + m
        m_new = pi if (first and t == 0) else torch.maximum(pi, lf)
        ig = torch.exp(pi - m_new)
        fg = torch.exp(lf - m_new)
        c = fg * c + ig * torch.tanh(pz)
        n = torch.clamp_min(fg * n + ig, 1.0)
        y = torch.sigmoid(po) * c / n
        m = m_new
        ys.append(y)
        all_states.append((y, c, n, m))
    return torch.stack(ys, dim=1), all_states


def slstm_cell(xn, w: SLSTMW, capture: dict | None = None):
    Wx = slstm_input(xn, w)
    y, states = slstm_recurrence(Wx, w.r)
    if capture is not None:
        capture.update(Wx=Wx, states=states, y=y)
    return mh_layer_norm(y, w.w_gnorm)


def group_layout(group_ids: torch.Tensor, target_mask: torch.Tensor):
    """``(keep [rows] bool, allowed [rows, rows] bool)``: rows in groups of > 1 row, and the attention mask
    (same group, and covariate queries never read target keys)."""
    same = group_ids[:, None] == group_ids[None, :]
    counts = same.sum(-1)
    keep = counts > 1
    allowed = same & ~((~target_mask)[:, None] & target_mask[None, :])
    return keep, allowed


def variate_mixer(x, w: VariateW, group_ids, target_mask):
    """``x [rows, S, 512]``; attention across rows at every token, within groups. Singletons pass through."""
    keep, allowed = group_layout(group_ids, target_mask)
    if not bool(keep.any()):
        return x
    xt = x.transpose(0, 1)  # [S, rows, 512]
    xn = rms_norm(xt, w.w_norm_attn)
    S, R, _ = xt.shape
    heads = lambda t: t.reshape(S, R, HEADS, -1).transpose(1, 2)  # [S, H, rows, 128]
    q, k, v = heads(linear(xn, w.w_q)), heads(linear(xn, w.w_k)), heads(linear(xn, w.w_v))
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(D_MODEL // HEADS)
    scores = scores.masked_fill(~allowed, float("-inf"))
    attn = torch.softmax(scores, dim=-1) @ v
    attn = attn.transpose(1, 2).reshape(S, R, D_MODEL)
    y = xt + linear(attn, w.w_o)
    y = y + linear(F.silu(linear(rms_norm(y, w.w_norm_ffn), w.w_wi)), w.w_wo)
    y = torch.where(keep[None, :, None], y, xt)
    return y.transpose(0, 1)


def block(x, w: BlockW, group_ids=None, target_mask=None, chunk: int = 64, capture: dict | None = None):
    xn = rms_norm(x, w.w_norm_lstm)
    if isinstance(w.cell, MLSTMW):
        y = mlstm_cell(xn, w.cell, chunk=chunk, capture=capture)
    else:
        y = slstm_cell(xn, w.cell, capture=capture)
    if capture is not None:
        capture.update(norm_lstm=xn, cell_out=y)
    x = x + y
    x = x + ffn(rms_norm(x, w.w_norm_ffn), w.ffn)
    if group_ids is not None:
        x = variate_mixer(x, w.variate, group_ids, target_mask)
    return x


def t5_norm(x, w, eps=OUT_NORM_EPS):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def forward(weights: TiRex2W, prep: Prepared, chunk: int = 64, capture: dict | None = None) -> torch.Tensor:
    """Head output ``[rows, 285, 288]`` (scaled space, quantile-major)."""
    x = residual_block(prep.tokens, weights.embed)
    if capture is not None:
        capture["embed"] = x
    for w in weights.blocks:
        cap = {} if capture is not None else None
        x = block(x, w, prep.group_ids, prep.target_mask, chunk=chunk, capture=cap)
        if capture is not None:
            capture[f"block{w.index}"] = {**cap, "out": x}
    x = t5_norm(x, weights.w_out_norm)
    if capture is not None:
        capture["stack_out_norm"] = x
    return residual_block(x, weights.head)


def predict(weights: TiRex2W, context, horizon, group_ids=None, target_mask=None, sign_flip=False, chunk=64):
    """Single-pass (or sign-flip averaged) forecast ``[targets, 9, H]``."""

    def one(ctx):
        prep = prepare(ctx, horizon, group_ids, target_mask)
        head = forward(weights, prep, chunk=chunk)
        return postprocess(head[:, readout_tokens(horizon)], prep)

    q = one(context)
    return sign_flip_average(q, one(-context)) if sign_flip else q
