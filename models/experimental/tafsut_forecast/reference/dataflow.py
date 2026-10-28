# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Tafsut forward in plain torch, in the decomposition the TTNN port uses.

Differences from the reference (`tafsut.model.TafsutModel`), each exact in real arithmetic:
- the host buckets the context: NaN left padding up to the smallest of ``BUCKETS`` (keys masked, RoPE relative,
  time feature ``-k/1024`` from the end);
- token order ``[context | future | REG]``, RoPE positions following each token (reference order is
  ``[context | REG | future]``);
- S padded to a multiple of 128 (or the chunk table), pad rows masked as keys only, with a finite fill;
- fused QKV; future and REG embeddings are input-independent host constants;
- read-out cut: the last layer runs queries, O and the FFN only on the read-out rows, and the head on
  ``ceil(H / 32)`` tokens;
- the constant-series fallback is vectorized.

Runs in the reference venv and in tt-metal's python_env (torch + safetensors only).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

REPO = "Tafsut-FM/tafsut-univariate-base"
REVISION = "a17c92cbab2cccecbf033406644af88586f6731c"

D, NH, DH, DFF, NL = 768, 12, 64, 3072, 14
PATCH, NQ, NFUT = 32, 9, 32
MAX_CTX = 32768
TIME_SCALE = 1024.0
EPS = 1e-6
NORM_EPS = 1e-6
RTOL, ATOL, TAIL_FRAC = 1e-4, 1e-6, 0.5
ROPE_THETA = 10000.0
MEDIAN_IDX = 4
MASK_FILL = -1e9

# CONTRACT §7 item 1, option (a): bucket context lengths and their (padded S, masked SDPA chunk).
# ttnn SDPA returns wrong values (masked or not) when the q-chunk count s_pad / chunk is odd and > 1 at B = 16 or 32
# (fine at B = 2 and 64): 160/32, 384/128, 640/128, 1088/64, 1152/128 all fail (job 933). Every row here has an even
# count or a single chunk; tests/pcc/test_sdpa_mask.py checks each at the batches the model runs.
BUCKETS = (2048, 3072, 4096, 8192, 16384, 32768)
PAD_TABLE = {2048: (128, 128), 3072: (256, 128), 4096: (256, 128), 8192: (384, 64), 16384: (640, 64), 32768: (1088, 32)}


def chunk_count_ok(s_pad: int, chunk: int) -> bool:
    n = s_pad // chunk
    return s_pad % chunk == 0 and (n == 1 or n % 2 == 0)


def checkpoint_path() -> str:
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return os.path.join(hub, "models--" + REPO.replace("/", "--"), "snapshots", REVISION, "model.safetensors")


def load_state_dict() -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return {k: v.float() for k, v in load_file(checkpoint_path()).items()}


def bucket_len(t: int) -> int:
    t = min(t, MAX_CTX)
    for b in BUCKETS:
        if t <= b:
            return b
    return MAX_CTX


# ----------------------------------------------------------------------------------------------- host front end
def lower_median(v: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Row-wise ``torch.median`` (the lower middle) over the entries where ``keep``; NaN for empty rows."""
    s = torch.where(keep, v, torch.full_like(v, float("inf"))).sort(dim=-1).values
    n = keep.sum(-1)
    idx = ((n - 1).clamp(min=0) // 2).unsqueeze(-1)
    med = s.gather(-1, idx).squeeze(-1)
    return torch.where(n > 0, med, torch.full_like(med, float("nan")))


def constant_fallback(x: torch.Tensor, valid: torch.Tensor):
    """Vectorized `_constant_fallback_candidates`: ``([B] bool, [B] fp32 value)``. ``x`` fp32 ``[B, T]``."""
    x = x.to(torch.float32)
    xv = torch.where(valid, x, torch.zeros_like(x))
    n = valid.sum(-1)
    med = lower_median(x, valid)
    full_const = (torch.isclose(xv, med[:, None], rtol=RTOL, atol=ATOL) | ~valid).all(-1)
    last_pos = torch.where(valid, torch.arange(x.shape[1]).expand_as(x), torch.full_like(valid, -1, dtype=torch.long))
    last_pos = last_pos.max(-1).values.clamp(min=0)
    last = x.gather(-1, last_pos[:, None])
    near = torch.isclose(xv, last, rtol=RTOL, atol=ATOL) | ~valid
    run = near.flip(-1).to(torch.int64).cumprod(-1).flip(-1).bool() & valid
    tail_major = run.sum(-1) > TAIL_FRAC * n
    tail_med = lower_median(x, run)
    has = n > 0
    mask = has & (full_const | tail_major)
    value = torch.where(full_const, med, tail_med)
    value = torch.where(mask, value, torch.full_like(value, float("nan")))
    return mask, value


@dataclass
class Prepared:
    tokens: torch.Tensor  # [B, P, 96] fp32: [time | asinh values | mask]
    key_valid: torch.Tensor  # [B, P] bool
    loc: torch.Tensor  # [B, 1]
    scale: torch.Tensor  # [B, 1]
    fb_mask: torch.Tensor  # [B]
    fb_value: torch.Tensor  # [B]
    ctx_len: int  # bucket context length


def prepare_inputs(x: torch.Tensor, bucket: bool = True) -> Prepared:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x[None]
    x = x[:, -MAX_CTX:]
    valid = torch.isfinite(x)
    fb_mask, fb_value = constant_fallback(x, valid)
    xn = x.masked_fill(~valid, float("nan"))
    loc = torch.nan_to_num(torch.nanmean(xn, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num((xn - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0)
    scale = torch.where(scale == 0, EPS, scale)
    z = torch.arcsinh((xn - loc) / scale)
    B, T = x.shape
    T_pad = bucket_len(T) if bucket else math.ceil(T / PATCH) * PATCH
    pad = T_pad - T
    if pad:
        z = torch.cat([torch.full((B, pad), float("nan")), z], -1)
        valid = torch.cat([torch.zeros(B, pad, dtype=torch.bool), valid], -1)
    P = T_pad // PATCH
    m = valid.view(B, P, PATCH).float()
    v = torch.where(valid, z, torch.zeros_like(z)).view(B, P, PATCH)
    t = (torch.arange(-T_pad, 0, dtype=torch.float32) / TIME_SCALE).view(1, P, PATCH).expand(B, -1, -1)
    return Prepared(torch.cat([t, v, m], -1), m.sum(-1) > 0, loc, scale, fb_mask, fb_value, T_pad)


# ----------------------------------------------------------------------------------------------------- weights
def residual_block(sd, pre, x):
    hid = torch.relu(x @ sd[f"{pre}.hidden_layer.weight"].T + sd[f"{pre}.hidden_layer.bias"])
    out = hid @ sd[f"{pre}.output_layer.weight"].T + sd[f"{pre}.output_layer.bias"]
    return out + x @ sd[f"{pre}.residual_layer.weight"].T + sd[f"{pre}.residual_layer.bias"]


def future_tokens() -> torch.Tensor:
    t = (torch.arange(0, NFUT * PATCH, dtype=torch.float32) / TIME_SCALE).view(NFUT, PATCH)
    return torch.cat([t, torch.zeros(NFUT, 2 * PATCH)], -1)


class Weights:
    """Device-layout weights: fused QKV ``[D, 3D]``, transposed linears, host constants for future and REG."""

    def __init__(self, sd: dict[str, torch.Tensor] | None = None):
        sd = load_state_dict() if sd is None else sd
        self.sd = sd
        self.layers = []
        for i in range(NL):
            a, f = f"encoder.block.{i}.layer.0", f"encoder.block.{i}.layer.1"
            att = f"{a}.self_attention"
            self.layers.append(
                dict(
                    ln1=sd[f"{a}.layer_norm.weight"],
                    wqkv=torch.cat([sd[f"{att}.{n}.weight"] for n in "qkv"], 0).T.contiguous(),
                    wo=sd[f"{att}.o.weight"].T.contiguous(),
                    ln2=sd[f"{f}.layer_norm.weight"],
                    wi=sd[f"{f}.mlp.wi.weight"].T.contiguous(),
                    wo2=sd[f"{f}.mlp.wo.weight"].T.contiguous(),
                )
            )
        self.final_ln = sd["encoder.final_layer_norm.weight"]
        self.future = residual_block(sd, "input_patch_embedding", future_tokens())  # [32, D]
        self.reg = sd["shared.weight"][1]  # [D]


# ------------------------------------------------------------------------------------------------------ forward
def rms(x, g):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS) * g


def rope_tables(pos: torch.Tensor):
    inv = 1.0 / ROPE_THETA ** (torch.arange(0, DH, 2, dtype=torch.int64).float() / DH)
    f = pos.float()[:, None] * inv[None]
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def rot_half(x):
    return torch.cat([-x[..., DH // 2 :], x[..., : DH // 2]], -1)


def layout(P: int, s_pad: int):
    """Positions of the ``[context | future | REG | pad]`` rows (reference order ``[context | REG | future]``)."""
    pos = torch.cat([torch.arange(P), torch.arange(P + 1, P + 1 + NFUT), torch.tensor([P])])
    pos = torch.cat([pos, torch.full((s_pad - pos.numel(),), P + 1 + NFUT)])
    return pos


def forward(
    w: Weights,
    prep: Prepared,
    n_read: int = NFUT,
    read_cut: bool = True,
    s_pad: int | None = None,
    taps: list | None = None,
):
    """Device dataflow: ``[B, n_read, 288]`` head output in asinh space (``pcc_normalized``'s tensor).

    ``taps`` collects the residual stream after the embedding and after every block, in the device's tap layout
    (``TtTafsut.forward_device``): ``[context | future | REG | pad]``, the last block only its read-out rows."""
    tok = prep.tokens
    B, P, _ = tok.shape
    S = P + NFUT + 1
    if s_pad is None:
        s_pad = PAD_TABLE.get(prep.ctx_len, (math.ceil(S / 128) * 128, 128))[0]
    h = torch.zeros(B, s_pad, D)
    h[:, :P] = residual_block(w.sd, "input_patch_embedding", tok)
    h[:, P : P + NFUT] = w.future
    h[:, P + NFUT] = w.reg
    kv = torch.zeros(B, s_pad, dtype=torch.bool)
    kv[:, :P] = prep.key_valid
    kv[:, P:S] = True
    bias = torch.where(kv, 0.0, MASK_FILL)[:, None, None, :]  # [B, 1, 1, S_pad], keys only
    cos, sin = rope_tables(layout(P, s_pad))
    rd = slice(P, P + n_read)
    if taps is not None:
        taps.append(h.clone())
    for i, L in enumerate(w.layers):
        last = read_cut and i == NL - 1
        x = rms(h, L["ln1"])
        qkv = (x @ L["wqkv"]).view(B, s_pad, 3, NH, DH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = q * cos + rot_half(q) * sin, k * cos + rot_half(k) * sin
        if last:
            q, h = q[:, :, rd], h[:, rd]
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1.0)
        h = h + a.transpose(1, 2).reshape(B, -1, D) @ L["wo"]
        h = h + torch.relu(rms(h, L["ln2"]) @ L["wi"]) @ L["wo2"]
        if taps is not None:
            taps.append(h.clone())
    if not read_cut:
        h = h[:, rd]
    return residual_block(w.sd, "output_patch_embedding", rms(h, w.final_ln))


def postprocess(head: torch.Tensor, prep: Prepared, horizon: int) -> torch.Tensor:
    """``[B, n, 288]`` asinh-space head output -> ``[B, horizon, 9]`` data-space quantiles."""
    B, n, _ = head.shape
    q = head.view(B, n, NQ, PATCH).permute(0, 2, 1, 3).reshape(B, NQ, n * PATCH)
    fbz = torch.arcsinh((prep.fb_value[:, None] - prep.loc) / prep.scale)  # [B, 1]
    q = torch.where(prep.fb_mask[:, None, None], fbz[:, :, None].expand_as(q), q)
    out = torch.sinh(q) * prep.scale[:, :, None] + prep.loc[:, :, None]
    return out[:, :, :horizon].transpose(1, 2)


def forecast(w: Weights, x: torch.Tensor, horizon: int, bucket: bool = True, heads: list | None = None):
    """Median-feedback loop of `tafsut.inference.forecast`; ``heads`` collects each forward's head output."""
    x = torch.as_tensor(x, dtype=torch.float32)
    out, remaining = [], horizon
    while remaining > 0:
        take = min(remaining, NFUT * PATCH)
        prep = prepare_inputs(x, bucket=bucket)
        head = forward(w, prep, n_read=math.ceil(take / PATCH))
        if heads is not None:
            heads.append(head)
        cur = postprocess(head, prep, take)
        out.append(cur)
        remaining -= take
        if remaining > 0:
            x = torch.cat([x, cur[:, :, MEDIAN_IDX]], -1)[:, -MAX_CTX:]
    return torch.cat(out, 1)
