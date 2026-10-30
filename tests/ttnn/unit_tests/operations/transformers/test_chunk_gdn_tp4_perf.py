# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill device time and accuracy of ttnn.transformer.chunk_gated_delta_rule at the Qwen TP-4 shapes, called the
way the Qwen3.6-27B QB2 decoder calls it (models/demos/qwen38_27b_qb2/tt/decoder.py) for one 2048-token prefill chunk:

  q, k  [1, T, H*128]  bf16 TILE DRAM, flat token-major and NOT normalized (the op L2-normalizes them in the kernel)
  v     [1, T, HV*128] bf16 TILE DRAM, flat token-major
  g     [1, T, HV]     fp32 TILE DRAM (log-decay <= 0)
  beta  [1, T, HV]     bf16 TILE DRAM (sigmoid of the projection)
  initial_state [1, HV, 128, 128] fp32 TILE DRAM (non-zero: a prefill chunk after the first)
  chunk_size=32, output_head_major=True, output_final_state=True, program_config and wy_inverse left to the op.

Cases: bh12 = Qwen3.6-27B TP-4 (4 key heads, 12 value heads), bh16 = Qwen3.5-397B-A17B TP-4 (4 / 16).

Every case calls the op WARMUP + MEASURED times on the same inputs. A Tracy signpost `gdn_call` precedes each call and
`gdn_end` follows the last one, so the device time of a call is the sum over every device op it launches, whatever
path (fused / phased / mono) and pre- or post-processing ops the op uses. The device profiler is read after each call
so no call's markers are dropped. The outputs are read back only after `gdn_end`.

Accuracy: the first call's o and final state against a float64 torch golden of the whole computation (PCC >= 0.999),
and every later call bit-identical to the first (a hand-off race shows up as a mismatch). Logged for the campaign as
`DREAM_METRIC pcc_o=... pcc_fs=... max_abs_o=... repeat_exact=...`.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import is_blackhole

try:
    from tracy import signpost
except ImportError:  # no profiler tooling on the path: the test still checks accuracy

    def signpost(header, message=None):
        pass


pytestmark = pytest.mark.skipif(not is_blackhole(), reason="the fused chunk_gated_delta_rule path is Blackhole-only")

CHUNK = 32
D = 128  # key and value head size
T = 2048  # one prefill chunk of the 27B decoder (policy chunk_size 2048)
WARMUP = 3
MEASURED = 10


def _const_tiles(device):
    """The op's constant tiles, as the 27B decoder uploads them (self.delta_constants)."""
    c = CHUNK
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    masks = torch.cat([(lo_i & lo_j).float(), (~lo_i & ~lo_j).float(), (~lo_i & lo_j).float()], dim=1)

    def up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return {
        "eye": up(torch.eye(c)),
        "tril": up(torch.ones(c, c).tril()),
        "ones": up(torch.ones(c, c)),
        "masks": up(masks),
    }


def _golden(q, k, v, g, beta, s0, hk, hv):
    """Float64 torch golden of the whole op on the device's (bf16-rounded) inputs. q, k: [T, H*D] raw; v: [T, HV*D];
    g, beta: [T, HV]; s0: [HV, D, D]. Returns o [HV, T, D] (head-major) and the final state [HV, D, D]."""
    f = torch.float64
    G = hv // hk
    NC = T // CHUNK
    scale = D**-0.5
    qh = F.normalize(q.to(f).reshape(T, hk, D), dim=-1) * scale
    kh = F.normalize(k.to(f).reshape(T, hk, D), dim=-1)
    qh = qh.repeat_interleave(G, dim=1).permute(1, 0, 2).reshape(hv, NC, CHUNK, D)
    kh = kh.repeat_interleave(G, dim=1).permute(1, 0, 2).reshape(hv, NC, CHUNK, D)
    vh = v.to(f).reshape(T, hv, D).permute(1, 0, 2).reshape(hv, NC, CHUNK, D)
    g_c = g.to(f).permute(1, 0).reshape(hv, NC, CHUNK)
    b_c = beta.to(f).permute(1, 0).reshape(hv, NC, CHUNK)

    decay = g_c.cumsum(-1)
    decay_exp = decay.exp().unsqueeze(-1)
    v_beta = vh * b_c.unsqueeze(-1)
    k_beta = kh * b_c.unsqueeze(-1)
    l_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    mask_upper = torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ kh.transpose(-1, -2)) * l_mask).masked_fill(mask_upper, 0)
    for i in range(1, CHUNK):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    t_inv = attn + torch.eye(CHUNK, dtype=f)
    kd = k_beta * decay_exp
    q_decay = qh * decay_exp
    mask_causal = torch.triu(torch.ones(CHUNK, CHUNK, dtype=torch.bool), diagonal=1)
    intra = (qh @ kh.transpose(-1, -2) * l_mask).masked_fill(mask_causal, 0)
    k_dec_t = (kh * (decay[..., -1:] - decay).exp().unsqueeze(-1)).transpose(-1, -2)
    dl = decay[..., -1].exp()

    S = s0.to(f).clone()
    o = torch.zeros(hv, NC, CHUNK, D, dtype=f)
    for c in range(NC):
        v_new = t_inv[:, c] @ (v_beta[:, c] - kd[:, c] @ S)
        o[:, c] = q_decay[:, c] @ S + intra[:, c] @ v_new
        S = S * dl[:, c, None, None] + k_dec_t[:, c] @ v_new
    return o.reshape(hv, T, D), S


def _pcc(golden, actual):
    g = golden.to(torch.float64).flatten()
    a = actual.to(torch.float64).flatten()
    assert torch.isfinite(a).all(), "device output contains non-finite values"
    vg, va = g - g.mean(), a - a.mean()
    denom = vg.norm() * va.norm()
    return float((vg @ va) / denom) if denom > 0 else float(torch.equal(g, a))


@pytest.mark.parametrize("hk, hv", [(4, 12), (4, 16)], ids=["bh12", "bh16"])
def test_chunk_gdn_tp4_prefill(device, hk, hv):
    torch.manual_seed(20261019 + hv)
    # q/k/v come out of the decoder's causal conv + SiLU un-normalized; g = -exp(A) * softplus(a + dt_bias) <= 0
    q = F.silu(torch.randn(1, T, hk * D)).to(torch.bfloat16)
    k = F.silu(torch.randn(1, T, hk * D)).to(torch.bfloat16)
    v = F.silu(torch.randn(1, T, hv * D)).to(torch.bfloat16)
    g = -F.softplus(torch.randn(1, T, hv)) * 0.5
    beta = torch.sigmoid(torch.randn(1, T, hv)).to(torch.bfloat16)
    s0 = 0.05 * torch.randn(1, hv, D, D)

    def dev(t, dtype):
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    qt, kt, vt = dev(q, ttnn.bfloat16), dev(k, ttnn.bfloat16), dev(v, ttnn.bfloat16)
    gt, bt, st = dev(g, ttnn.float32), dev(beta, ttnn.bfloat16), dev(s0, ttnn.float32)
    consts = _const_tiles(device)
    ttnn.synchronize_device(device)

    outs = []
    for i in range(WARMUP + MEASURED):
        signpost("gdn_call", f"call {i}")
        o, fs = ttnn.transformer.chunk_gated_delta_rule(
            qt,
            kt,
            vt,
            gt,
            bt,
            initial_state=st,
            output_final_state=True,
            output_head_major=True,
            chunk_size=CHUNK,
            **consts,
        )
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        outs.append((o, fs))
    signpost("gdn_end")

    o0, fs0 = (ttnn.to_torch(t) for t in outs[0])
    repeat_exact = 1
    for i, (o, fs) in enumerate(outs[1:], start=1):
        if not (torch.equal(ttnn.to_torch(o), o0) and torch.equal(ttnn.to_torch(fs), fs0)):
            repeat_exact = 0
            print(f"call {i}: output differs from call 0")
    for o, fs in outs:
        ttnn.deallocate(o)
        ttnn.deallocate(fs)

    o_ref, fs_ref = _golden(q[0], k[0], v[0], g[0], beta[0].float(), s0[0], hk, hv)
    o_dev = o0.float().reshape(hv, T, D)
    fs_dev = fs0.float().reshape(hv, D, D)
    pcc_o, pcc_fs = _pcc(o_ref, o_dev), _pcc(fs_ref, fs_dev)
    max_abs_o = float((o_ref - o_dev.to(torch.float64)).abs().max())
    print(f"DREAM_METRIC pcc_o={pcc_o:.7f} pcc_fs={pcc_fs:.7f} max_abs_o={max_abs_o:.5f} repeat_exact={repeat_exact}")

    assert repeat_exact, "the op is not deterministic across calls on the same inputs (hand-off race?)"
    assert pcc_o >= 0.999, f"o: PCC {pcc_o} < 0.999"
    assert pcc_fs >= 0.999, f"final_state: PCC {pcc_fs} < 0.999"
