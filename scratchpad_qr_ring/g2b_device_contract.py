# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lock the MINIMAL device export contract for Q-gather.

Device change is minimal: the op keeps emitting today's NORMALIZED per-shard
output O_norm, and ADDS two [1,H,S,1] scalars per shard:
    m = max over local-selected of RAW q·k       (UNSCALED, exactly max_cur in-kernel)
    l = sum over local-selected exp(scale*(qk-m)) (the softmax denom = sum_cur in-kernel)

Merge (host or ttnn-eltwise), weights = softmax over shards of (scale*m):
    M      = max_i (scale * m_i)
    w_i    = exp(scale*m_i - M) * l_i
    out    = sum_i w_i * O_norm_i  /  sum_i w_i
Empty shard (no local hit for a row): m=-inf, l=0 -> w=0 (contributes nothing).

This is the device-faithful form of the online-softmax merge (G2 proved the
unnormalized variant; this proves the normalized-O + raw-(m,l) variant the op
will actually emit with the smallest possible kernel change).
"""
import sys

import torch

sys.path.insert(0, "tests/ttnn/unit_tests/operations/sdpa")
from sparse_sdpa_test_utils import MASKED_INDEX, golden, make_inputs  # noqa: E402

K_DIM, V_DIM = 576, 512


def shard_of(pos, sp, chunk_local):
    return (pos // chunk_local) % sp


def op_partial_normalized(q, kv, indices, scale, shard, sp, chunk_local):
    """Emulates the device op run over ONE shard's local KV with off-shard indices masked.
    Returns (O_norm [1,H,S,V], m_raw [1,H,S,1], l [1,H,S,1]) exactly as the kernel would."""
    B, H, S, Dk = q.shape
    k = indices.shape[-1]
    idx = indices.reshape(B, S, k)
    masked = idx == MASKED_INDEX
    idx_safe = torch.where(masked, torch.zeros_like(idx), idx).to(torch.int64)
    local = (shard_of(idx_safe, sp, chunk_local) == shard) & (~masked)  # this shard's live slots

    kv2 = kv[0, 0]
    sel = kv2[idx_safe.reshape(-1)].reshape(B, S, k, Dk)
    raw = torch.einsum("bhsd,bsjd->bhsj", q, sel)  # RAW q·k (no scale) — matches cb_qk_im
    neg = torch.finfo(torch.float32).min
    raw_local = raw.masked_fill(~local.view(B, 1, S, k), neg)

    m_raw = raw_local.max(dim=-1, keepdim=True).values  # [B,H,S,1] UNSCALED max (== max_cur)
    empty = ~local.any(dim=-1)  # [B,S] no local hit
    p = torch.exp((raw_local - m_raw) * scale)  # exp(scale*(qk - m)) == sub_exp
    p = p.masked_fill(~local.view(B, 1, S, k), 0.0)
    l = p.sum(dim=-1, keepdim=True)  # [B,H,S,1] == sum_cur
    O_unnorm = torch.einsum("bhsj,bsjd->bhsd", p, sel[..., :V_DIM])
    O_norm = O_unnorm / l.clamp_min(1e-30)  # == normalize_row_streaming output

    # identity for empty rows
    m_raw = m_raw.masked_fill(empty.view(B, 1, S, 1), float("-inf"))
    l = l.masked_fill(empty.view(B, 1, S, 1), 0.0)
    O_norm = O_norm.masked_fill(empty.view(B, 1, S, 1), 0.0)
    return O_norm, m_raw, l


def merge_lse(partials, scale, stats_dtype=torch.float32):
    """weights = softmax over shards of scale*m, scaled by l. out = sum w_i O_norm_i / sum w_i."""
    sm = torch.stack([(scale * p[1]).to(stats_dtype) for p in partials])  # [sp,...]
    M = sm.max(dim=0).values
    num = None
    den = 0.0
    for O_norm, m_raw, l in partials:
        w = torch.exp((scale * m_raw).to(stats_dtype) - M) * l.to(stats_dtype)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)  # empty shard -> 0
        term = w * O_norm.to(stats_dtype)
        num = term if num is None else num + term
        den = den + w
    return num / den.clamp_min(1e-30)


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    scale = K_DIM**-0.5
    print("Device contract: normalized-O + raw(m,l) LSE merge == full-T golden?\n")
    print(f"{'sp':>4} {'T':>7} {'topk':>5} {'nvalid':>7} {'stats':>6} {'PCC':>10}  gate>=0.99")
    print("-" * 56)
    for sp in [2, 4, 8]:
        T, chunk_local = sp * 256, 32
        for nv_fn, tag in [(lambda s: 10**9, "all"), (lambda s: 8, "8/row")]:
            q, kv, indices = make_inputs(32, 64, T, 64, K_DIM, nv_fn, seed=sp)
            ref = golden(q, kv, indices, scale, V_DIM)
            parts = [op_partial_normalized(q, kv, indices, scale, sh, sp, chunk_local) for sh in range(sp)]
            for sd, sdt in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
                out = merge_lse(parts, scale, stats_dtype=sd)
                p = pcc(out, ref)
                print(f"{sp:>4} {T:>7} {64:>5} {tag:>7} {sdt:>6} {p:>10.6f}  {'PASS' if p>=0.99 else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
