# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""F1 gates for the fused GDN prefill path (ttnn::prim::chunk_gdn_fused).

The fused prim runs prep and scan in ONE program: per head, a producer core runs the unchanged
prep reader+compute and NoC-writes the seven fp32 intermediates (v_beta, kd, q_decay, intra,
k_dec_t, dl, t_inv) straight into the paired scan core's CBs via the shipped ready/valid
handshake — zero DRAM intermediates. The compute kernels are byte-identical to the phased path
and the DRAM round trip they replace was byte-preserving, so the fused op must be BIT-IDENTICAL
to the phased op — any difference is a bug, not numerical noise.

Path selection is QWEN_GDN_PATH ("fused"|"phased"|"mono"), read fresh per call where the prim is
chosen (cache-safe: different prims hash to different program-cache entries). With no env set the
default is fused iff (BH >= 24 and 2*BH fits the compute grid), else phased. Because fused and
phased are bit-exact by design, torch.equal alone cannot prove WHICH path ran — every path proof
here rests on program-cache entry deltas (a new prim compiles a new program; a cache hit does not).
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn

CHUNK = 32  # Ct=1: the production chunk size
KDIM = 128
VDIM = 128
T_SMALL = 256  # NC=8 — enough chunks to exercise the recurrence, small enough to keep runtime down


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


def _clear_gdn_env(monkeypatch):
    """Neutralize every GDN path/debug knob; each test then sets QWEN_GDN_PATH explicitly (or
    leaves it unset to probe the default dispatch)."""
    monkeypatch.delenv("QWEN_GDN_PATH", raising=False)
    # Legacy selector — superseded by QWEN_GDN_PATH but still honored when PATH is unset.
    monkeypatch.delenv("QWEN_GDN_PHASED", raising=False)
    monkeypatch.delenv("QWEN_GDN_SCAN_SERIAL", raising=False)
    monkeypatch.delenv("QWEN_GDN_PREP_SERIAL", raising=False)
    monkeypatch.delenv("QWEN_GDN_SCAN_MCAST", raising=False)
    # QWEN_GDN_DUMP is read once via a function-local static; delenv helps only if the op has not
    # run yet in this process — kept for hygiene.
    monkeypatch.delenv("QWEN_GDN_DUMP", raising=False)


def _skip_unless_fused_fits(device, bh):
    grid = device.compute_with_storage_grid_size()
    if 2 * bh > grid.x * grid.y:
        pytest.skip(
            f"2*BH={2 * bh} exceeds the {grid.x}x{grid.y} compute grid (fused needs a producer+receiver pair per head)"
        )


def _make_inputs(device, batch, seq, num_k_heads, num_v_heads, with_initial_state, seed):
    """Token-major public-op inputs in the op's numeric regime: q/k L2-normalized on host (the op
    requires it — unnormalized keys NaN the recurrence on every path), beta in (0,1), g <= 0."""
    torch.manual_seed(seed)
    B, T, H, HV = batch, seq, num_k_heads, num_v_heads
    q = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(B, T, H, KDIM), dim=-1).to(torch.bfloat16)
    v = (0.5 * torch.randn(B, T, HV, VDIM)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, HV))
    g = -F.softplus(torch.randn(B, T, HV)) * 0.5
    s0 = 0.05 * torch.randn(B, HV, KDIM, VDIM) if with_initial_state else None

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    tensors = (
        dev(q, ttnn.bfloat16),
        dev(k, ttnn.bfloat16),
        dev(v, ttnn.bfloat16),
        dev(g, ttnn.float32),
        dev(beta, ttnn.float32),
    )
    s0_dev = dev(s0, ttnn.float32) if s0 is not None else None
    return (q, k, v, g, beta, s0), tensors, s0_dev


def _run_op(device, tensors, const_tiles, initial_state):
    q, k, v, g, beta = tensors
    eye, tril, ones, masks = const_tiles
    o, fs = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=CHUNK,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
    )
    o_t = ttnn.to_torch(o)
    fs_t = ttnn.to_torch(fs)
    ttnn.deallocate(o)
    ttnn.deallocate(fs)
    return o_t, fs_t


# ---------------------------------------------------------------------------
# Inlined torch golden of the WHOLE computation (delta_rule_ops.py:149-245; tests must not import
# models/, and --import-mode=importlib blocks importing the sibling test module, so the golden is
# copied verbatim from test_chunk_gated_delta_rule.py::_golden_chunk_gdn).
# ---------------------------------------------------------------------------


def _golden_chunk_gdn(q, k, v, g, beta, scale, s0, C):
    """q,k: [B,T,H,K] fp32 (exact bf16-rounded, L2-normalized); v: [B,T,HV,V] fp32 (bf16-rounded);
    g,beta: [B,T,HV] fp32; s0: [B,HV,K,V] fp32 or None. T must be a multiple of C (no padding).
    Returns o [B,T,HV,V], final_state [B,HV,K,V]."""
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    G = HV // H
    if G > 1:  # the op's GQA head expand (repeat_interleave over the head dim)
        q = q.repeat_interleave(G, dim=2)
        k = k.repeat_interleave(G, dim=2)
    # The op folds scale into q ON DEVICE in bf16 (ttnn::multiply on the bf16 tensor packs back
    # to bf16) — mirror that rounding so it doesn't count against the gates below.
    q = (q * scale).to(torch.bfloat16).to(torch.float32)

    BH, NC = B * HV, T // C
    q_c, k_c, v_c = (x.permute(0, 2, 1, 3).reshape(BH, NC, C, x.shape[-1]) for x in (q, k, v))
    g_c, beta_c = (x.permute(0, 2, 1).reshape(BH, NC, C) for x in (g, beta))

    decay = g_c.cumsum(-1)  # :189
    decay_exp = decay.exp().unsqueeze(-1)  # :190
    v_beta = v_c * beta_c.unsqueeze(-1)  # :171
    k_beta = k_c * beta_c.unsqueeze(-1)  # :172
    l_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().tril()  # :193
    # :196-201 — WY inverse via forward substitution
    mask_upper = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=0)
    attn = -((k_beta @ k_c.transpose(-1, -2)) * l_mask).masked_fill(mask_upper, 0)
    for i in range(1, C):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(
            -2
        )
    t_inv = attn + torch.eye(C, dtype=torch.float32)  # :201

    kd = k_beta * decay_exp
    q_decay = q_c * decay_exp  # :229
    mask_causal = torch.triu(torch.ones(C, C, dtype=torch.bool), diagonal=1)
    intra = (q_c @ k_c.transpose(-1, -2) * l_mask).masked_fill(mask_causal, 0)  # :222
    k_dec_t = (k_c * (decay[..., -1:] - decay).exp().unsqueeze(-1)).transpose(-1, -2)  # :237
    dl = decay[..., -1].exp()  # [BH,NC]

    # :216-238 scan loop in the phased op's un-premultiplied form — mathematically identical to
    # v_corrected/k_cumdecay premultiplication (v_new = t_inv @ (v_beta - kd@S) == u - w@S).
    S = s0.reshape(BH, K, V).clone() if s0 is not None else torch.zeros(BH, K, V, dtype=torch.float32)
    o = torch.zeros(BH, NC, C, V, dtype=torch.float32)
    for c in range(NC):
        v_new = t_inv[:, c] @ (v_beta[:, c] - kd[:, c] @ S)
        o[:, c] = q_decay[:, c] @ S + intra[:, c] @ v_new  # :229-232
        S = S * dl[:, c, None, None] + k_dec_t[:, c] @ v_new  # :235-238
    o = o.reshape(B, HV, T, V).permute(0, 2, 1, 3).contiguous()  # :243-244
    return o, S.reshape(B, HV, K, V)


# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_fused_bit_exact_vs_phased(device, monkeypatch, with_initial_state):
    """Primary F1 gate: fused output == phased output, bit for bit. The producer packs the seven
    intermediates at the same CB boundaries/formats the phased prep packs them at, and the DRAM
    round trip it eliminates was a byte copy — so torch.equal, not PCC. (F1 uses Option U for
    v_beta: the producer computes and ships it exactly as prep does, keeping the compute kernels
    byte-identical; the Option R recompute lands in F2.)"""
    B, num_k_heads, num_v_heads = 1, 16, 48  # single-device Qwen3.6 shape, GQA G=3
    BH = B * num_v_heads
    _skip_unless_fused_fits(device, BH)
    _clear_gdn_env(monkeypatch)

    _, tensors, s0 = _make_inputs(device, B, T_SMALL, num_k_heads, num_v_heads, with_initial_state, seed=20260820)
    const_tiles = _const_tiles(device)

    monkeypatch.setenv("QWEN_GDN_PATH", "phased")
    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0)
    n_phased = device.num_program_cache_entries()

    # A second phased run must be a full cache hit: it pins that the pre/postprocessing graph
    # around the prims is cache-stable, so the fused-run delta below is attributable purely to
    # the prim the dispatcher picked.
    o_ph2, fs_ph2 = _run_op(device, tensors, const_tiles, s0)
    n_phased2 = device.num_program_cache_entries()
    assert n_phased2 == n_phased, (
        f"repeated phased run compiled {n_phased2 - n_phased} new programs (expected 0): the op "
        "graph is not cache-stable, so program-cache deltas cannot prove which prim dispatched"
    )
    assert torch.equal(o_ph, o_ph2) and torch.equal(fs_ph, fs_ph2), "phased path is not deterministic"

    monkeypatch.setenv("QWEN_GDN_PATH", "fused")
    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0)
    n_fused = device.num_program_cache_entries()

    # The phased runs compiled prep+scan (2 prim programs) plus the shared pre/postprocessing;
    # the fused run reuses everything except the prim, which is ONE program (that is the point:
    # producer and receiver kernels live in a single fused program, zero DRAM intermediates).
    # A delta of 0 means QWEN_GDN_PATH was not read on this call and the "fused" run silently
    # reused the phased path — which would make the bit-exact comparison below vacuously compare
    # the phased op against itself. A delta of 2 means the fused branch dispatched prep+scan.
    assert n_fused - n_phased2 == 1, (
        f"QWEN_GDN_PATH phased->fused toggle compiled {n_fused - n_phased2} new programs "
        "(expected exactly 1, the single fused prim program): 0 => env not read per call "
        "(comparison vacuous), 2 => the fused branch ran the phased prims"
    )

    assert torch.equal(o_fu, o_ph), "fused path changed o (must be bit-identical to phased)"
    assert torch.equal(fs_fu, fs_ph), "fused path changed final_state (must be bit-identical to phased)"


@pytest.mark.parametrize(
    "num_k_heads, num_v_heads, expected_path",
    [
        (16, 48, "fused"),  # BH=48 >= 24 and 2*BH=96 cores fit -> default engages fused (F2 gate cleared)
        (4, 12, "phased"),  # BH=12 < 24 -> default falls back to phased
    ],
    ids=["bh48_fused", "bh12_phased"],
)
def test_fused_default_dispatch(device, monkeypatch, num_k_heads, num_v_heads, expected_path):
    """With NO path env set, the dispatcher must pick fused iff (BH >= 24 and 2*BH fits the grid),
    else phased. Since fused and phased are bit-exact, torch.equal cannot discriminate paths: the
    proof that the default took the expected path is a program-cache delta of ZERO after warming
    exactly that path explicitly (any other path would compile at least one new prim program)."""
    B = 1
    BH = B * num_v_heads
    grid = device.compute_with_storage_grid_size()
    if BH > grid.x * grid.y:
        pytest.skip(f"BH={BH} exceeds the {grid.x}x{grid.y} compute grid (scan needs a core per head)")
    if expected_path == "fused":
        _skip_unless_fused_fits(device, BH)  # on smaller grids the default itself falls back to phased
    _clear_gdn_env(monkeypatch)

    _, tensors, s0 = _make_inputs(device, B, T_SMALL, num_k_heads, num_v_heads, True, seed=20260821)
    const_tiles = _const_tiles(device)

    monkeypatch.setenv("QWEN_GDN_PATH", expected_path)
    o_exp, fs_exp = _run_op(device, tensors, const_tiles, s0)
    n_explicit = device.num_program_cache_entries()

    monkeypatch.delenv("QWEN_GDN_PATH", raising=False)
    o_def, fs_def = _run_op(device, tensors, const_tiles, s0)
    n_default = device.num_program_cache_entries()

    assert n_default - n_explicit == 0, (
        f"default dispatch (no QWEN_GDN_PATH) compiled {n_default - n_explicit} new programs after "
        f"an explicit '{expected_path}' run: the default did NOT take the {expected_path} path for "
        f"BH={BH} (the delta counts the unexpected branch's prim programs: fused or mono = 1, "
        "phased prep+scan = 2)"
    )
    assert torch.equal(o_def, o_exp), f"default dispatch o differs from explicit '{expected_path}' run"
    assert torch.equal(fs_def, fs_exp), f"default dispatch final_state differs from explicit '{expected_path}' run"


def test_fused_vs_torch_golden(device, monkeypatch):
    """Whole-computation correctness of the fused path against the inlined torch golden. The
    bit-exact gate above anchors fused==phased; this gate anchors the pair to the math (bf16
    q/k/v inputs dominate the error; kernel math is fp32/HiFi4 end-to-end). Initial state is
    exercised here (s0 flows through the receiver's DRAM read — the one input the producer does
    not ship); the s0=None case is covered bit-exactly vs phased above."""
    B, num_k_heads, num_v_heads = 1, 8, 24  # BH=24: the smallest default-fused shape; GQA G=3
    BH = B * num_v_heads
    _skip_unless_fused_fits(device, BH)
    _clear_gdn_env(monkeypatch)
    monkeypatch.setenv("QWEN_GDN_PATH", "fused")

    host, tensors, s0_dev = _make_inputs(device, B, T_SMALL, num_k_heads, num_v_heads, True, seed=20260822)
    q, k, v, g, beta, s0 = host
    const_tiles = _const_tiles(device)

    o_d, fs_d = _run_op(device, tensors, const_tiles, s0_dev)
    o = o_d.float()  # o is [B,T,HV,V]; final_state is [B,HV,K,V] — same shapes the golden returns
    fs = fs_d.float()

    scale = KDIM**-0.5
    o_ref, fs_ref = _golden_chunk_gdn(q.float(), k.float(), v.float(), g, beta, scale, s0, CHUNK)

    pcc_o = _pcc(o_ref, o)
    assert pcc_o >= 0.999, f"o: PCC {pcc_o} < 0.999"
    # fs gate is 0.999, looser than the phased whole-op test's 0.9999: fused==phased bit-exact is
    # pinned above, so this gate only anchors the math to torch — the looser bound avoids a flake
    # on this shape (BH=24, G=3), which the phased golden test does not cover at 0.9999.
    pcc_fs = _pcc(fs_ref, fs)
    assert pcc_fs >= 0.999, f"final_state: PCC {pcc_fs} < 0.999"
