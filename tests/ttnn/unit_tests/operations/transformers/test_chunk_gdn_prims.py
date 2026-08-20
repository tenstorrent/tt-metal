# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Part-level tests for the phased GDN prims (F0 test pyramid, tier 2).

ttnn.transformer.chunk_gdn_prep produces the seven per-(head,chunk) fp32 intermediates
{v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv}; ttnn.transformer.chunk_gdn_scan consumes them
plus the initial state and carries the recurrence. Each prim is asserted against torch formulas
inlined from models/experimental/gated_attention_gated_deltanet/torch_functional/
delta_rule_ops.py:170-238 (tests must not import models/), and the composition prep->scan is
asserted BIT-IDENTICAL to the public op on the phased path — the seven intermediates are rounded
at pack time and the DRAM round trip is a byte copy, so any composition difference is a bug in
the prims' plumbing, not numerical noise.
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

CHUNK = 32  # Ct=1: the production chunk size; the prims' in-kernel WY inverse is exact here
KDIM = 128
VDIM = 128


def _pcc(golden, actual):
    g = golden.to(torch.float64).flatten()
    a = actual.to(torch.float64).flatten()
    assert torch.isfinite(a).all(), "device output contains non-finite values"
    if torch.equal(g, a):
        return 1.0
    vg = g - g.mean()
    va = a - a.mean()
    denom = vg.norm() * va.norm()
    if denom == 0:
        return 0.0
    return float((vg @ va) / denom)


def _const_tiles(device, chunk_size=CHUNK):
    """The op's constant tiles (mirrors chunk_gated_delta_rule.cpp build_const_tiles)."""
    c = chunk_size
    eye = torch.eye(c, dtype=torch.float32)
    tril = torch.tril(torch.ones(c, c, dtype=torch.float32))
    ones = torch.ones(c, c, dtype=torch.float32)
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    qtl = (lo_i & lo_j).float()
    qbr = (~lo_i & ~lo_j).float()
    qbl = (~lo_i & lo_j).float()
    masks = torch.cat([qtl, qbr, qbl], dim=1)  # [32, 96]

    def _up(t):
        return ttnn.from_torch(t.reshape(1, 1, *t.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    return (_up(eye), _up(tril), _up(ones), _up(masks))


def _make_inputs(bh, nc, seed, scale):
    """Head-major prim inputs in the op's numeric regime: q/k L2-normalized over the last dim
    (unnormalized keys NaN the recurrence on every path), q pre-scaled on host (the prim applies
    NO scale when qk_norm=False — mirroring the public op, which folds scale into q before prep),
    beta in (0,1), g <= 0."""
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(bh, nc, CHUNK, KDIM), dim=-1)
    k = F.normalize(torch.randn(bh, nc, CHUNK, KDIM), dim=-1)
    q = (q * scale).to(torch.bfloat16)
    k = k.to(torch.bfloat16)
    v = (0.5 * torch.randn(bh, nc, CHUNK, VDIM)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(bh, nc, CHUNK))
    g = -F.softplus(torch.randn(bh, nc, CHUNK)) * 0.5
    s0 = 0.05 * torch.randn(bh, KDIM, VDIM)
    return q, k, v, g, beta, s0


# ---------------------------------------------------------------------------
# Inlined torch reference (delta_rule_ops.py:170-205 prep formulas, :216-238 scan loop)
# ---------------------------------------------------------------------------


def _prep_reference(q, k, v, g, beta):
    """The seven prep outputs, fp32, from head-major per-chunk inputs.
    q,k,v: [BH,NC,C,D] fp32 (the exact bf16-rounded values fed to the device, q pre-scaled);
    g,beta: [BH,NC,C] fp32. Mirrors delta_rule_ops.py:170-205 with the phased op's
    UN-premultiplied WY hand-off: kd/v_beta are NOT multiplied by t_inv (the scan applies t_inv
    after the v_beta - kd@S subtraction)."""
    C = q.shape[-2]
    decay = g.cumsum(-1)  # [BH,NC,C]  (:189)
    decay_exp = decay.exp().unsqueeze(-1)  # [BH,NC,C,1]  (:190)
    v_beta = v * beta.unsqueeze(-1)  # :171
    k_beta = k * beta.unsqueeze(-1)  # :172
    # :193 — double-tril exp decay mask
    l_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()
    # :196-201 — WY inverse via forward substitution
    mask_upper = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ k.transpose(-1, -2)) * l_mask).masked_fill(mask_upper, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    t_inv = attn + torch.eye(C, dtype=torch.float32)  # :201
    kd = k_beta * decay_exp  # :205's operand, un-premultiplied
    q_decay = q * decay_exp  # :229 (scale already folded into q)
    mask_causal = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    intra = (q @ k.transpose(-1, -2) * l_mask).masked_fill(mask_causal, 0)  # :222
    # :237 — k * exp(decay_last - decay), transposed to [K,C]
    k_dec_t = (k * (decay[..., -1:] - decay).exp().unsqueeze(-1)).transpose(-1, -2)
    dl = decay[..., -1].exp().reshape(*decay.shape[:2], 1, 1)  # exp(g_sum): the scan's state decay
    return v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv


def _scan_reference(v_beta, kd, q_decay, intra, k_dec_t, dl, t_inv, s0):
    """The scan recurrence (delta_rule_ops.py:216-238 semantics in the phased prims'
    un-premultiplied form): v_new = t_inv @ (v_beta - kd@S) — mathematically identical to
    u - w@S with u=t_inv@v_beta, w=t_inv@kd."""
    bh, nc = v_beta.shape[:2]
    S = s0.clone()
    o = torch.zeros(bh, nc, CHUNK, v_beta.shape[-1], dtype=torch.float32)
    for c in range(nc):
        v_new = t_inv[:, c] @ (v_beta[:, c] - kd[:, c] @ S)
        o[:, c] = q_decay[:, c] @ S + intra[:, c] @ v_new  # :229-232
        S = S * dl[:, c] + k_dec_t[:, c] @ v_new  # :235-238
    return o, S


def _dev(device, t, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _clean_env(monkeypatch):
    # Neutralize ambient GDN debug/perf knobs that fork the kernel topology or corrupt outputs.
    monkeypatch.delenv("QWEN_GDN_SCAN_SERIAL", raising=False)
    monkeypatch.delenv("QWEN_GDN_PREP_SERIAL", raising=False)
    # Mcast on/off is documented bit-exact, but pin the shipped default topology anyway.
    monkeypatch.delenv("QWEN_GDN_SCAN_MCAST", raising=False)
    # QWEN_GDN_DUMP is read once via a function-local static; delenv helps only if the op has not
    # run yet in this process — kept for hygiene.
    monkeypatch.delenv("QWEN_GDN_DUMP", raising=False)


# ---------------------------------------------------------------------------
# (1) prep prim vs torch, each of the seven outputs
# ---------------------------------------------------------------------------


# (16, 8) = 128 (head, chunk) work-items: exercises the multi-core fan-out of distribute_prep
# (fills a full 8x8+ grid); (4, 4) is the small-shape smoke.
@pytest.mark.parametrize("bh, nc", [(4, 4), (16, 8)])
def test_prep_outputs_vs_torch(device, monkeypatch, bh, nc):
    _clean_env(monkeypatch)
    scale = KDIM**-0.5
    q, k, v, g, beta, _ = _make_inputs(bh, nc, seed=20260820, scale=scale)

    outs = ttnn.transformer.chunk_gdn_prep(
        _dev(device, q, ttnn.bfloat16),
        _dev(device, k, ttnn.bfloat16),
        _dev(device, v, ttnn.bfloat16),
        _dev(device, g.unsqueeze(-1), ttnn.float32),
        _dev(device, beta.unsqueeze(-1), ttnn.float32),
        *_const_tiles(device),
        chunk_size=CHUNK,
    )
    refs = _prep_reference(q.float(), k.float(), v.float(), g, beta)

    # Max-abs bounds. Golden is torch fp32 on the IDENTICAL bf16-rounded inputs, so the only
    # divergence sources are: the SFPU exp (non-approx, ~1e-6 relative vs libm), fp32 matmul
    # ki-accumulation-order rounding, reduced srcA/srcB precision on fp32 eltwise operands, and
    # fp32 pack rounding. Bounds are sized to each output's magnitude and op chain:
    #   v_beta  5e-3: one broadcast mul, |v_beta| <= |v| ~ 3 (0.5*randn tails); no exp involved
    #   kd      2e-3: |k_beta| <= 1 (normalized k, beta < 1) times decay_exp <= 1 (g <= 0); one exp
    #   q_decay 1e-3: |q*scale| <= ~0.1 (normalized rows), decay_exp <= 1; exp error dominates
    #   intra   1e-3: |q@k^T| <= scale (Cauchy-Schwarz on L2-normalized rows), L_mask <= 1;
    #                 128-term fp32 accumulation + one exp in the mask
    #   k_dec_t 5e-3: |k| <= 1 elementwise, decay factor exp(<=0) <= 1; one exp
    #   dl      1e-4: exp(sum g) <= 1; device forms it as exp(gsum-d0)*exp(d0) (two exps + a mul)
    #                 vs torch's single exp — pure relative error on a value <= 1
    #   t_inv   2.5e-3: both sides are mathematically exact inverses of the same matrix (device:
    #                 quadrant-split bounded Horner; golden: forward substitution); the device's
    #                 exp-derived L_mask feeds the matrix being inverted, so its ~1e-3 input error
    #                 is amplified through the inverse chain
    # k_dec_t and t_inv were calibrated on p150b hardware (fw 19.11): measured max-abs 2.2e-3 and
    # 1.1e-3 across the parametrized shapes; bounds carry ~2x headroom. PCC >= 0.999 remains the
    # primary gate for every output.
    names = ["v_beta", "kd", "q_decay", "intra", "k_dec_t", "dl", "t_inv"]
    bounds = [5e-3, 2e-3, 1e-3, 1e-3, 5e-3, 1e-4, 2.5e-3]
    assert len(outs) == 7
    for name, out, ref, bound in zip(names, outs, refs, bounds):
        got = ttnn.to_torch(out)
        assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
        pcc = _pcc(ref, got)
        assert pcc >= 0.999, f"{name}: PCC {pcc} < 0.999"
        max_abs = (got - ref).abs().max().item()
        assert max_abs <= bound, f"{name}: max-abs {max_abs} > {bound}"
        ttnn.deallocate(out)


# ---------------------------------------------------------------------------
# (2) scan prim vs torch, with torch-built (exact fp32) intermediates
# ---------------------------------------------------------------------------


def test_scan_vs_torch(device, monkeypatch):
    _clean_env(monkeypatch)
    bh, nc = 4, 4
    q, k, v, g, beta, s0 = _make_inputs(bh, nc, seed=20260821, scale=KDIM**-0.5)
    seven = _prep_reference(q.float(), k.float(), v.float(), g, beta)

    dev_seven = [_dev(device, t, ttnn.float32) for t in seven]
    o_d, fs_d = ttnn.transformer.chunk_gdn_scan(
        *dev_seven,
        initial_state=_dev(device, s0, ttnn.float32),
        chunk_size=CHUNK,
        output_final_state=True,
    )
    o = ttnn.to_torch(o_d)
    fs = ttnn.to_torch(fs_d)

    o_ref, fs_ref = _scan_reference(*seven, s0)
    # The seven inputs are bit-identical fp32 on both sides, and the scan is pure fp32 HiFi4
    # matmul/eltwise (no exp, no bf16 rounding) — only accumulation-order rounding differs.
    pcc_o = _pcc(o_ref, o)
    assert pcc_o >= 0.9999, f"o: PCC {pcc_o} < 0.9999"
    pcc_fs = _pcc(fs_ref, fs)
    assert pcc_fs >= 0.9999, f"final_state: PCC {pcc_fs} < 0.9999"


# ---------------------------------------------------------------------------
# (3) prep prim -> scan prim composition == public phased op, bit-exact
# ---------------------------------------------------------------------------


def test_composition_bit_exact(device, monkeypatch):
    """prep->scan on head-major inputs must be BYTE-IDENTICAL to the public op on the equivalent
    token-major inputs: the op's preprocessing is all bit-reproducible data movement (typecasts
    skipped for already-bf16/fp32 inputs; permute/reshape; pad==0 since T % 32 == 0), except the
    on-device q*scale multiply — neutralized here by passing scale=1.0, which is a numerical
    identity in bf16 (x*1.0 repacks to the same bits for the normal values randn+l2norm makes).
    G=1 (H==HV) so the GQA repeat_interleave is not in the path either."""
    torch.manual_seed(20260822)
    B, T, H = 1, 256, 8
    BH, NC = B * H, T // CHUNK
    grid = device.compute_with_storage_grid_size()
    if BH > grid.x * grid.y:
        pytest.skip(f"BH={BH} exceeds the {grid.x}x{grid.y} compute grid (scan needs a core per head)")

    monkeypatch.setenv("QWEN_GDN_PHASED", "1")
    _clean_env(monkeypatch)

    # Token-major op inputs (bf16 q/k/v so the op's typecast is a no-op; q/k host-normalized).
    q = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
    v = (0.5 * torch.randn(B, T, H, VDIM)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, H))
    g = -F.softplus(torch.randn(B, T, H)) * 0.5
    s0 = 0.05 * torch.randn(B, H, KDIM, VDIM)

    const_tiles = _const_tiles(device)
    # output_head_major=True: with pad==0 the op's o is a metadata-only reshape of the scan
    # prim's [BH,NC,C,V] output — no relayout between the compared tensors.
    o_op_d, fs_op_d = ttnn.transformer.chunk_gated_delta_rule(
        _dev(device, q, ttnn.bfloat16),
        _dev(device, k, ttnn.bfloat16),
        _dev(device, v, ttnn.bfloat16),
        _dev(device, g, ttnn.float32),
        _dev(device, beta, ttnn.float32),
        scale=1.0,
        initial_state=_dev(device, s0, ttnn.float32),
        output_final_state=True,
        chunk_size=CHUNK,
        output_head_major=True,
        eye=const_tiles[0],
        tril=const_tiles[1],
        ones=const_tiles[2],
        masks=const_tiles[3],
    )
    o_op = ttnn.to_torch(o_op_d)  # [BH, T, V] fp32
    fs_op = ttnn.to_torch(fs_op_d)  # [B, H, K, V] fp32

    # Replicate the op's preprocessing in torch (chunk_gated_delta_rule.cpp head_split_tile /
    # headvec_split_tile / to_chunks_tile): [B,T,H,D] -> [B,H,T,D] -> [BH,NC,C,D]; values are
    # moved, never re-rounded, so from_torch of these feeds prep bit-identical inputs.
    def head_major(x):
        return x.permute(0, 2, 1, 3).reshape(BH, NC, CHUNK, x.shape[-1]).contiguous()

    def headvec_major(x):
        return x.permute(0, 2, 1).reshape(BH, NC, CHUNK, 1).contiguous()

    prep = ttnn.transformer.chunk_gdn_prep(
        _dev(device, head_major(q), ttnn.bfloat16),
        _dev(device, head_major(k), ttnn.bfloat16),
        _dev(device, head_major(v), ttnn.bfloat16),
        _dev(device, headvec_major(g), ttnn.float32),
        _dev(device, headvec_major(beta), ttnn.float32),
        *const_tiles,
        chunk_size=CHUNK,
    )
    o_pr_d, fs_pr_d = ttnn.transformer.chunk_gdn_scan(
        *prep,
        initial_state=_dev(device, s0.reshape(BH, KDIM, VDIM), ttnn.float32),
        chunk_size=CHUNK,
        output_final_state=True,
    )
    o_pr = ttnn.to_torch(o_pr_d)  # [BH, NC, C, V] fp32
    fs_pr = ttnn.to_torch(fs_pr_d)  # [BH, K, V] fp32

    # The seven intermediates are rounded at pack time and the DRAM round trip is a byte copy:
    # composition must match the phased op bit-for-bit, not just numerically.
    assert torch.equal(o_pr.reshape(BH, T, VDIM), o_op), "prep->scan composition changed o (must be bit-identical)"
    assert torch.equal(
        fs_pr, fs_op.reshape(BH, KDIM, VDIM)
    ), "prep->scan composition changed final_state (must be bit-identical)"
