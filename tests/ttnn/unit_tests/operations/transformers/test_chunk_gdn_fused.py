# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused GDN prefill path (ttnn::prim::chunk_gdn_fused).

The fused prim runs prep and scan in one program: per head, a producer core runs the unchanged
prep reader+compute and NoC-writes the seven fp32 intermediates (v_beta, kd, q_decay, intra,
k_dec_t, dl, t_inv) straight into the paired scan core's CBs via the shipped ready/valid
handshake — zero DRAM intermediates. The compute kernels are byte-identical to the phased path
and the DRAM round trip they replace was byte-preserving, so the fused op must be BIT-IDENTICAL
to the phased op — any difference is a bug, not numerical noise.

Path selection is QWEN_GDN_PATH ("fused"|"phased"|"mono"), read fresh per call where the prim is
chosen (cache-safe: different prims hash to different program-cache entries). With no env set the
default comes from the fused op's calibrated geometry cost model (chunk_gdn_fused_geometry): fused iff
a fused geometry fits the compute grid and its predicted time beats the phased reference. The model
itself, its feasibility predicate and the factory's core map are tested host-side, per grid, in
test_chunk_gdn_fused_geometry.py. Because fused and phased are bit-exact by design, torch.equal alone cannot prove WHICH path ran — every path proof
here rests on program-cache entry deltas (a new prim compiles a new program; a cache hit does not).
"""

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(
    not is_blackhole(), reason="the fused and phased chunk_gated_delta_rule paths are Blackhole-only"
)

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
    # F3a producer split and the NV receiver split (fused prim only; hashed via attrs.np / attrs.nv).
    monkeypatch.delenv("QWEN_GDN_NP", raising=False)
    monkeypatch.delenv("QWEN_GDN_NV", raising=False)
    # Hand-off CB depth and the unicast/multicast transport of the hand-off (fused prim only; hashed).
    monkeypatch.delenv("QWEN_GDN_HANDOFF_NBUF", raising=False)
    monkeypatch.delenv("QWEN_GDN_UNICAST", raising=False)
    monkeypatch.delenv("QWEN_GDN_POSTED", raising=False)
    monkeypatch.delenv("QWEN_GDN_PLACEMENT", raising=False)
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


def _cost_model_path(device, bh, nc):
    """The op's default path per its calibrated geometry cost model: fused iff a
    fused geometry fits the grid and its predicted time beats the phased reference."""
    from ttnn._ttnn.operations import transformer as _t

    grid = device.compute_with_storage_grid_size()
    nv, np_, pl, t_f, t_ph, pays = _t.chunk_gdn_fused_geometry(grid.x, grid.y, bh, nc, VDIM // 32)
    return "fused" if (nv >= 1 and pays) else "phased"


@pytest.mark.parametrize("nc", [8, 64], ids=["NC8", "NC64"])
@pytest.mark.parametrize(
    "num_k_heads, num_v_heads",
    [
        (16, 48),  # BH=48: the single-device shape
        (4, 12),  # BH=12: the 27B TP-4 shape
        (1, 4),  # BH=4: chain-bound
        (16, 64),  # BH=64: no fused geometry on a 110-core grid (needs >= 128 cores) -> phased
    ],
    ids=["bh48", "bh12", "bh4", "bh64"],
)
def test_fused_default_dispatch(device, monkeypatch, num_k_heads, num_v_heads, nc):
    """With NO path env set, the dispatcher must pick what the calibrated cost model says (N7): fused
    iff a fused geometry fits this grid and beats the phased reference. The choice depends on NC (the
    fill cost is amortized over the chunks), so both a short and the production chunk count run: on
    QB2 (11x10) the model picks fused for BH=48 at NC=8 and for BH in {4, 12, 48} at NC=64, and phased
    for the rest. Since fused and phased are bit-exact, torch.equal cannot discriminate paths: the
    proof that the default took the expected path is a program-cache delta of ZERO after warming
    exactly that path explicitly (any other path would compile at least one new prim program)."""
    B = 1
    BH = B * num_v_heads
    grid = device.compute_with_storage_grid_size()
    if BH > grid.x * grid.y:
        pytest.skip(f"BH={BH} exceeds the {grid.x}x{grid.y} compute grid (scan needs a core per head)")
    expected_path = _cost_model_path(device, BH, nc)
    _clear_gdn_env(monkeypatch)

    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20260821)
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


# ---------------------------------------------------------------------------
# F3a: NP > 1 producers per head (QWEN_GDN_NP). Producer p owns the round-robin chunks
# c = p, p+NP, ...; the receiver credits producers in rotation and the writer mcasts each chunk
# into an explicitly computed hand-off slot (global c % nbuf). Correctness is silent-failure
# territory (a wrong slot corrupts in-flight data without hanging), so the gate is torch.equal
# vs phased at NC values that stress the slot/rotation arithmetic: NC < NP (clamp), NC == NP,
# NC == NP+1 (first wraparound), NC == 2*NP+1 (odd/even slot alternation across producers), and
# a long-ish NC.
# ---------------------------------------------------------------------------

NP_BH_KV_HEADS = (4, 12)  # BH=12 (GQA G=3): leaves room for NP up to 8 on a 110-core grid


def _skip_unless_fused_np_fits(device, bh, np_req, nc):
    grid = device.compute_with_storage_grid_size()
    np_eff = min(np_req, nc)  # the op host clamps NP to NC
    if bh * (1 + np_eff) > grid.x * grid.y:
        pytest.skip(f"BH*(1+NP)={bh * (1 + np_eff)} exceeds the {grid.x}x{grid.y} compute grid")


@pytest.mark.parametrize(
    "np_producers, nc",
    [
        (2, 1),  # NC < NP: host clamps to NP=1 (degenerate, must still be exact)
        (2, 2),  # NC == NP: one chunk per producer
        (2, 3),  # NC == NP+1: first slot wraparound (chunk 2 reuses slot 0)
        (2, 5),  # NC == 2*NP+1
        (3, 4),  # NC == NP+1 at odd NP (producer/receiver slot parity diverges — the F2
        #          lockstep-breaking case the explicit destination addressing exists for)
        (3, 7),  # NC == 2*NP+1 at odd NP
        (5, 16),  # long run, NP does not divide NC
        (8, 16),  # max NP that fits BH=12 on a 110-core grid (12*9=108)
    ],
    ids=lambda v: str(v),
)
def test_fused_np_bit_exact_vs_phased(device, monkeypatch, np_producers, nc):
    """F3a primary gate: fused with NP>1 producers == phased, bit for bit, across the NC
    boundary cases. The compute kernels are untouched by the split (prep is chunk-independent),
    so as with F1 any difference is a protocol/addressing bug, not numerical noise."""
    B = 1
    num_k_heads, num_v_heads = NP_BH_KV_HEADS
    BH = B * num_v_heads
    _skip_unless_fused_np_fits(device, BH, np_producers, nc)
    _clear_gdn_env(monkeypatch)

    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20260823)
    const_tiles = _const_tiles(device)

    monkeypatch.setenv("QWEN_GDN_PATH", "phased")
    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0)
    o_ph2, fs_ph2 = _run_op(device, tensors, const_tiles, s0)
    n_phased = device.num_program_cache_entries()
    assert torch.equal(o_ph, o_ph2) and torch.equal(fs_ph, fs_ph2), "phased path is not deterministic"

    monkeypatch.setenv("QWEN_GDN_PATH", "fused")
    monkeypatch.setenv("QWEN_GDN_NP", str(np_producers))
    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0)
    n_fused = device.num_program_cache_entries()
    assert n_fused - n_phased == 1, (
        f"phased->fused(NP={np_producers}) toggle compiled {n_fused - n_phased} new programs "
        "(expected exactly 1, the fused prim): 0 => an env was not read per call and the "
        "comparison below is vacuous"
    )

    assert torch.equal(o_fu, o_ph), f"fused NP={np_producers} changed o (must be bit-identical to phased)"
    assert torch.equal(fs_fu, fs_ph), f"fused NP={np_producers} changed final_state (must be bit-identical to phased)"


def test_fused_np_cache_identity(device, monkeypatch):
    """Vacuity canary for the NP knob itself: QWEN_GDN_NP must be read fresh per call AND hashed
    (attrs.np), so each distinct NP compiles its own fused program, repeats are cache hits, and
    toggling back recompiles nothing. If the env were read in the factory instead of at attrs
    construction, the toggle deltas would be 0 and every NP 'A/B' would silently compare one
    cached program against itself. Deltas are asserted EXACTLY (never >=)."""
    B = 1
    num_k_heads, num_v_heads = NP_BH_KV_HEADS
    BH = B * num_v_heads
    nc = 16
    _skip_unless_fused_np_fits(device, BH, 3, nc)
    _clear_gdn_env(monkeypatch)
    monkeypatch.setenv("QWEN_GDN_PATH", "fused")

    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20260824)
    const_tiles = _const_tiles(device)

    o1, fs1 = _run_op(device, tensors, const_tiles, s0)  # NP unset -> the cost model's pick
    n1 = device.num_program_cache_entries()

    monkeypatch.setenv("QWEN_GDN_NP", "2")
    o2, fs2 = _run_op(device, tensors, const_tiles, s0)
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 1, f"np 1->2 compiled {n2 - n1} programs (expected 1: np must be hashed)"

    monkeypatch.setenv("QWEN_GDN_NP", "3")
    o3, fs3 = _run_op(device, tensors, const_tiles, s0)
    n3 = device.num_program_cache_entries()
    assert n3 - n2 == 1, f"np 2->3 compiled {n3 - n2} programs (expected 1)"

    monkeypatch.setenv("QWEN_GDN_NP", "2")
    _run_op(device, tensors, const_tiles, s0)
    n4 = device.num_program_cache_entries()
    assert n4 - n3 == 0, f"np 3->2 (already compiled) compiled {n4 - n3} programs (expected 0: cache hit)"

    monkeypatch.delenv("QWEN_GDN_NP")
    _run_op(device, tensors, const_tiles, s0)
    n5 = device.num_program_cache_entries()
    assert n5 - n4 == 0, f"np 2->unset (the model's pick, already compiled) compiled {n5 - n4} programs (expected 0)"

    # All NP variants of the same head must agree bit-for-bit with each other.
    assert torch.equal(o1, o2) and torch.equal(o1, o3), "o differs across NP values"
    assert torch.equal(fs1, fs2) and torch.equal(fs1, fs3), "final_state differs across NP values"


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


# ---------------------------------------------------------------------------
# NV > 1 receivers per head. Receiver (h, v) carries V columns
# [v*V/NV, (v+1)*V/NV); the six V-independent intermediates reach a head's 1xNV row rectangle as
# multicasts, v_beta as NV per-receiver slice writes into a receiver-side ring, and the producer
# sends only once all NV receivers have credited its per-head credit word. All of it is plumbing
# around byte-identical compute kernels, so the gate is torch.equal vs phased, with a per-V-block
# report on failure and program-cache deltas as the path proof.
# ---------------------------------------------------------------------------


def _skip_unless_geometry_fits(device, bh, nv, np_req, nc, placement=0):
    """The fused factory's own feasibility predicates (design §6.1): nv | Vt, the receiver rectangles
    fit the grid (row-major or row-local placement), and BH*(nv+np) cores exist (np is clamped to nc
    by the op host)."""
    grid = device.compute_with_storage_grid_size()
    vt = VDIM // 32
    if vt % nv != 0 or nv > grid.x:
        pytest.skip(f"NV={nv} does not divide Vt={vt} or exceeds the grid width")
    np_eff = min(np_req, nc)
    if placement == 0:
        hpr = grid.x // nv
        if bh > hpr * grid.y:
            pytest.skip(f"BH={bh} 1x{nv} receiver rectangles do not fit the {grid.x}x{grid.y} grid")
    else:
        L = nv + np_eff
        if L > grid.x:
            pytest.skip(f"row-local placement needs NV+NP={L} <= grid.x={grid.x}")
        k = grid.x // L
        if bh > k * grid.y:
            wl = grid.x - k * L
            rw = min(nv, wl) if wl else 0
            if wl < 1 or nv % rw != 0 or (bh - k * grid.y) * (nv // rw + -(-np_eff // wl)) > grid.y:
                pytest.skip(f"row-local placement: {bh - k * grid.y} leftover heads do not fit the {wl}-column block")
    if bh * (nv + np_eff) > grid.x * grid.y:
        pytest.skip(f"BH*(NV+NP)={bh * (nv + np_eff)} exceeds the {grid.x}x{grid.y} compute grid")


def _vblock_mismatches(o_ref, o_got, hv, nv):
    """Which (head, v-block) slices of o [B,T,HV,V] differ — localizes a wrong slice boundary or a
    wrong v_beta ring slot instead of a bare torch.equal failure."""
    vc = VDIM // nv
    bad = []
    for h in range(hv):
        for v in range(nv):
            if not torch.equal(o_ref[..., h, v * vc : (v + 1) * vc], o_got[..., h, v * vc : (v + 1) * vc]):
                bad.append((h, v))
    return bad


def _fused_vs_phased(device, monkeypatch, hk, hv, nc, nv, np_producers, seed):
    """Run phased (twice, for cache stability), then fused with the given geometry. Returns the
    outputs and the program-cache delta of the fused run (must be exactly 1: one fused program)."""
    B = 1
    _clear_gdn_env(monkeypatch)
    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, hk, hv, True, seed=seed)
    const_tiles = _const_tiles(device)

    monkeypatch.setenv("QWEN_GDN_PATH", "phased")
    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0)
    o_ph2, fs_ph2 = _run_op(device, tensors, const_tiles, s0)
    n_phased = device.num_program_cache_entries()
    assert torch.equal(o_ph, o_ph2) and torch.equal(fs_ph, fs_ph2), "phased path is not deterministic"

    monkeypatch.setenv("QWEN_GDN_PATH", "fused")
    monkeypatch.setenv("QWEN_GDN_NV", str(nv))
    monkeypatch.setenv("QWEN_GDN_NP", str(np_producers))
    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0)
    delta = device.num_program_cache_entries() - n_phased
    return (o_ph, fs_ph), (o_fu, fs_fu), delta, (tensors, const_tiles, s0)


@pytest.mark.parametrize(
    "nv, np_producers, nc",
    [
        (2, 1, 8),  # two receivers, one producer: the minimal NV>1 handshake
        (2, 3, 8),  # odd NP with NV=2: producer/receiver slot parity diverges at NV>1
        (4, 1, 8),  # four receivers (Vtl=1), one producer
        (4, 2, 3),  # NC == NP+1 at NV=4: first slot wraparound with four v_beta slices
        (4, 3, 1),  # NC=1: producers 2,3 clamp away (host clamps NP<=NC); single-chunk handshake
        (4, 5, 8),  # the QB2 production geometry (12*(4+5)=108 cores), short
        (4, 5, 64),  # the QB2 production geometry at the production chunk count (T=2048)
        (2, 7, 16),  # the NV=2 production candidate (12*(2+7)=108 cores)
    ],
    ids=lambda v: str(v),
)
def test_fused_nv_bit_exact_vs_phased(device, monkeypatch, nv, np_producers, nc):
    """Primary Phase-1 gate at the 27B TP-4 shape (BH=12): fused with NV receivers per head ==
    phased, bit for bit. On failure, name the (head, v-block) slices that differ."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260925
    )
    assert delta == 1, (
        f"phased->fused(NV={nv},NP={np_producers}) compiled {delta} new programs (expected exactly 1): "
        "0 => a knob was not read per call (comparison vacuous), 2 => the fused branch ran the phased prims"
    )
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"fused NV={nv} NP={np_producers}: o differs from phased in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph), "fused o differs from phased outside any single v-block (layout bug)"
    assert torch.equal(fs_fu, fs_ph), f"fused NV={nv} NP={np_producers}: final_state differs from phased"


@pytest.mark.parametrize("unicast", [1, 0])
@pytest.mark.parametrize(
    "nv, np_producers, nc, nbuf",
    [
        (2, 7, 64, 3),  # BH=12 NV=2 operating point (24 receivers + 84 producers), default ring
        (4, 5, 64, 2),  # BH=12 NV=4 gate geometry (48 + 60) at the shallowest pipelined ring (D = 1)
        (4, 5, 8, 4),  # short chain + deeper ring: every slot index is exercised on both sides
        (2, 3, 9, 3),  # NC not a multiple of NP or nbuf
        (2, 1, 7, 3),  # single producer per head: the same word is credited for chunks c and c+nbuf
        (4, 3, 5, 1),  # nbuf=1: one slot, D=1 — the pipelined reader degenerates to the Phase 1 loop
    ],
)
def test_fused_nv_transport_bit_exact(device, monkeypatch, nv, np_producers, nc, nbuf, unicast):
    """Phase 1/1b transport x ring-depth matrix (design D5, D6 v0.3): the hand-off shipped as NV plain
    unicast writes per item (QWEN_GDN_UNICAST=1, default) or as the linked multicast chain (0), with
    nbuf-1 hand-offs in flight per receiver, is bit-identical to phased. Transport and depth are
    hashed, so the fused program compiles fresh (delta == 1)."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    monkeypatch.setenv("QWEN_GDN_UNICAST", str(unicast))
    monkeypatch.setenv("QWEN_GDN_HANDOFF_NBUF", str(nbuf))
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260921
    )
    assert (
        delta == 1
    ), f"fused(NV={nv},NP={np_producers},nbuf={nbuf},unicast={unicast}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert (
        not bad
    ), f"fused NV={nv} NP={np_producers} nbuf={nbuf} unicast={unicast}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "fused differs from phased (transport matrix)"


@pytest.mark.parametrize(
    "nv, np_producers, nc",
    [
        (2, 7, 64),  # one head per row (9 cores) + 2 heads as 2x5 blocks in columns 9-10 (1x2 receivers)
        (4, 5, 16),  # same, the leftover heads' receivers as 2x2 rectangles
        (2, 3, 9),  # L=5: leftover heads as 1x2 receivers over 6 columns, 2 rows each
        (1, 9, 8),  # NV=1: L=10, leftover width 1
    ],
)
def test_fused_nv_row_local_placement_bit_exact(device, monkeypatch, nv, np_producers, nc):
    """Placement A/B (design D9 v0.3): row-local placement (QWEN_GDN_PLACEMENT=1) — one head per row,
    leftover heads as column blocks with 2-D receiver rectangles — is bit-identical to phased. The
    placement is hashed (delta == 1)."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=1)
    monkeypatch.setenv("QWEN_GDN_PLACEMENT", "1")
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260921
    )
    assert delta == 1, f"row-local fused(NV={nv},NP={np_producers}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"row-local fused NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "row-local fused differs from phased"


@pytest.mark.parametrize(
    "hk, hv, nv, np_producers, nc",
    [
        (4, 16, 1, 4, 16),  # 397B TP-4 shape (BH=16): the cost model's pick on 11x10 (L=5, 2 heads per row)
        (4, 16, 2, 3, 16),  # BH=16 at NV=2: 2 heads per row, 30 idle cores
        (8, 32, 1, 2, 8),  # BH=32 (397B TP-2): 3 heads per row + 2 heads in the leftover columns
        (2, 8, 4, 7, 16),  # BH=8 (Galaxy2 TP-8): one head per row, NV=4 chain-bound geometry
        (1, 4, 4, 7, 16),  # BH=4 (TP-16)
    ],
)
def test_fused_nv_row_local_shapes_bit_exact(device, monkeypatch, hk, hv, nv, np_producers, nc):
    """Row-local placement across the BH range the cost model dispatches (design D8/D9 v0.3, N12):
    k heads per row plus leftover column blocks, bit-identical to phased."""
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=1)
    monkeypatch.setenv("QWEN_GDN_PLACEMENT", "1")
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260921 + hv
    )
    assert delta == 1, f"row-local fused(BH={hv},NV={nv},NP={np_producers}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"row-local fused BH={hv} NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "row-local fused differs from phased"


@pytest.mark.parametrize(
    "nv, np_producers, nc, nbuf",
    [
        (2, 7, 64, 2),  # NV=2 operating point, the default ring depth
        (4, 5, 8, 3),  # NV=4, short chain, D=2 in flight
        (2, 1, 7, 3),  # single producer per head
    ],
)
def test_fused_nv_posted_bit_exact(device, monkeypatch, nv, np_producers, nc, nbuf):
    """Phase 1b transport A/B (design D5): posted unicast data writes with the VALID flag ordered
    behind them by same-VC in-order delivery (no per-item barrier) are bit-identical to phased."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    monkeypatch.setenv("QWEN_GDN_UNICAST", "1")
    monkeypatch.setenv("QWEN_GDN_POSTED", "1")
    monkeypatch.setenv("QWEN_GDN_HANDOFF_NBUF", str(nbuf))
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260921
    )
    assert delta == 1, f"posted fused(NV={nv},NP={np_producers},nbuf={nbuf}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"posted fused NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "posted fused differs from phased"


@pytest.mark.parametrize(
    "hk, hv, nv, np_producers",
    [
        (4, 16, 2, 4),  # 397B TP-4 shape on a 110-core grid: the NV=2 candidate (16*6=96 cores)
        (4, 16, 4, 2),  # same shape, NV=4 (16*6=96 cores; HPR=2 -> 8 receiver rows)
        (8, 24, 2, 2),  # BH=24 (batched 2x12 or 397B TP-2/2): 24*4=96 cores
        (2, 8, 4, 8),  # BH=8 (397B TP-8 / 35B TP-4): 8*12=96 cores, producer-rich
    ],
    ids=lambda v: str(v),
)
def test_fused_nv_shapes_bit_exact(device, monkeypatch, hk, hv, nv, np_producers):
    """Other target shapes: the geometry must place and stay bit-exact wherever it is feasible."""
    nc = 8
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260926
    )
    assert delta == 1, f"expected exactly one new (fused) program, got {delta}"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"BH={hv} NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "fused differs from phased"


def test_fused_nv_cache_identity(device, monkeypatch):
    """N1 for the NV knob: QWEN_GDN_NV must be read per call AND hashed (attrs.nv), so each distinct
    NV compiles its own fused program and a repeat is a cache hit. Deltas asserted EXACTLY."""
    hk, hv = NP_BH_KV_HEADS
    nc = 8
    _skip_unless_geometry_fits(device, hv, 4, 2, nc)
    _clear_gdn_env(monkeypatch)
    monkeypatch.setenv("QWEN_GDN_PATH", "fused")
    monkeypatch.setenv("QWEN_GDN_NP", "2")
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20260927)
    const_tiles = _const_tiles(device)

    o1, fs1 = _run_op(device, tensors, const_tiles, s0)  # NV unset -> the cost model's NV for NP=2
    n1 = device.num_program_cache_entries()
    monkeypatch.setenv("QWEN_GDN_NV", "2")
    o2, fs2 = _run_op(device, tensors, const_tiles, s0)
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 1, f"nv 1->2 compiled {n2 - n1} programs (expected 1: nv must be hashed)"
    monkeypatch.setenv("QWEN_GDN_NV", "4")
    o4, fs4 = _run_op(device, tensors, const_tiles, s0)
    n4 = device.num_program_cache_entries()
    assert n4 - n2 == 1, f"nv 2->4 compiled {n4 - n2} programs (expected 1)"
    monkeypatch.setenv("QWEN_GDN_NV", "2")
    _run_op(device, tensors, const_tiles, s0)
    n5 = device.num_program_cache_entries()
    assert n5 - n4 == 0, f"nv 4->2 (already compiled) compiled {n5 - n4} programs (expected 0: cache hit)"
    monkeypatch.delenv("QWEN_GDN_NV")
    _run_op(device, tensors, const_tiles, s0)
    n6 = device.num_program_cache_entries()
    assert n6 - n5 == 0, f"nv 2->unset (the model's pick, already compiled) compiled {n6 - n5} programs (expected 0)"
    assert torch.equal(o1, o2) and torch.equal(o1, o4), "o differs across NV values"
    assert torch.equal(fs1, fs2) and torch.equal(fs1, fs4), "final_state differs across NV values"


def test_fused_nv_repeats(device, monkeypatch):
    """N5: an NV-way handshake race is non-deterministic, so one comparison has little power. Re-run
    the production geometry (BH=12, NV=4, NP=5, T=2048) REPEATS times against the first result."""
    hk, hv = NP_BH_KV_HEADS
    nc, nv, np_producers = 64, 4, 5
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, (tensors, const_tiles, s0) = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260928
    )
    assert delta == 1 and torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "first fused run is not bit-exact"
    for rep in range(8):
        o_rep, fs_rep = _run_op(device, tensors, const_tiles, s0)
        assert torch.equal(o_rep, o_fu), f"fused NV=4 NP=5 o not reproducible on repeat {rep + 1}: race"
        assert torch.equal(fs_rep, fs_fu), f"fused NV=4 NP=5 final_state not reproducible on repeat {rep + 1}: race"


@pytest.mark.parametrize(
    "knob, a, b",
    [
        ("QWEN_GDN_HANDOFF_NBUF", "2", "3"),  # hand-off ring depth
        ("QWEN_GDN_UNICAST", "1", "0"),  # per-receiver unicast vs linked multicast hand-off
        ("QWEN_GDN_POSTED", "0", "1"),  # posted unicast data writes (requires UNICAST=1, the default)
        ("QWEN_GDN_PLACEMENT", "0", "1"),  # row-major vs row-local core map
    ],
    ids=lambda v: str(v),
)
def test_fused_knob_cache_identity(device, monkeypatch, knob, a, b):
    """N1 for the remaining geometry/transport knobs: each must be read at attrs construction AND
    hashed, so a -> b compiles exactly one new fused program and every revisit is a cache hit. A knob
    that is read but not hashed would serve a program built for the other setting — wrong answers
    that depend on test order. NV/NP are pinned to a geometry both placements accept."""
    hk, hv = NP_BH_KV_HEADS
    nc, nv, np_producers = 8, 2, 3
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=0)
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=1)
    _clear_gdn_env(monkeypatch)
    monkeypatch.setenv("QWEN_GDN_PATH", "fused")
    monkeypatch.setenv("QWEN_GDN_NV", str(nv))
    monkeypatch.setenv("QWEN_GDN_NP", str(np_producers))
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20260929)
    const_tiles = _const_tiles(device)

    monkeypatch.setenv(knob, a)
    o_a, fs_a = _run_op(device, tensors, const_tiles, s0)
    n_a = device.num_program_cache_entries()
    monkeypatch.setenv(knob, b)
    o_b, fs_b = _run_op(device, tensors, const_tiles, s0)
    n_b = device.num_program_cache_entries()
    assert n_b - n_a == 1, f"{knob} {a}->{b} compiled {n_b - n_a} programs (expected 1: the knob must be hashed)"
    monkeypatch.setenv(knob, a)
    _run_op(device, tensors, const_tiles, s0)
    monkeypatch.setenv(knob, b)
    _run_op(device, tensors, const_tiles, s0)
    n_rev = device.num_program_cache_entries()
    assert n_rev - n_b == 0, f"{knob}: revisiting {a} and {b} compiled {n_rev - n_b} programs (expected 0: cache hits)"
    assert torch.equal(o_a, o_b) and torch.equal(fs_a, fs_b), f"{knob}={a} and {knob}={b} disagree"


@pytest.mark.parametrize(
    "hk, hv, nv, np_producers, nc",
    [
        (4, 12, 2, 3, 8),  # BH=12: 1x2 rectangles, 5 heads per row
        (4, 12, 4, 5, 16),  # BH=12 at NV=4: 2 heads per row, 3 stranded columns per receiver row
        (4, 12, 2, 7, 64),  # BH=12 at the model's NV/NP, production chunk count
        (4, 16, 4, 2, 8),  # BH=16 (397B TP-4): 8 receiver rows
        (2, 8, 4, 8, 8),  # BH=8, producer-rich
    ],
    ids=lambda v: str(v),
)
def test_fused_nv_row_major_placement_bit_exact(device, monkeypatch, hk, hv, nv, np_producers, nc):
    """N3 for placement 0. The op defaults to row-local placement whenever it is feasible, so the
    row-major 1xNV rectangles — and their NOC_1 multicast orientation — only run when forced. A wrong
    orientation lands on another head's receivers: finite, wrong, no hang; the per-V-block report
    names the (head, vblock) slices that differ."""
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=0)
    monkeypatch.setenv("QWEN_GDN_PLACEMENT", "0")
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, monkeypatch, hk, hv, nc, nv, np_producers, 20260930 + hv
    )
    assert delta == 1, f"row-major fused(BH={hv},NV={nv},NP={np_producers}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"row-major fused BH={hv} NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "row-major fused differs from phased"


@pytest.mark.parametrize(
    "hk, hv",
    [(4, 12), (4, 16), (1, 4)],
    ids=["bh12", "bh16", "bh4"],
)
def test_fused_default_geometry_repeats(device, monkeypatch, hk, hv):
    """N5 at the geometry production actually runs: no NV/NP/PLACEMENT knob, so the op uses the cost
    model's pick (on QB2 at BH=12: NV=2, NP=7, row-local). Bit-exact vs phased, then 8 repeats against
    the first fused result — a handshake race is timing-dependent, so one comparison has little power."""
    nc = 64
    grid = device.compute_with_storage_grid_size()
    from ttnn._ttnn.operations import transformer as _t

    nv, np_producers, placement, _, _, _ = _t.chunk_gdn_fused_geometry(grid.x, grid.y, hv, nc, VDIM // 32)
    if nv == 0:
        pytest.skip(f"no fused geometry for BH={hv} on the {grid.x}x{grid.y} grid")
    _clear_gdn_env(monkeypatch)
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20260931 + hv)
    const_tiles = _const_tiles(device)

    monkeypatch.setenv("QWEN_GDN_PATH", "phased")
    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0)
    n_phased = device.num_program_cache_entries()
    monkeypatch.setenv("QWEN_GDN_PATH", "fused")
    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0)
    assert device.num_program_cache_entries() - n_phased == 1, "the fused run did not compile the fused prim"
    geom = f"BH={hv} NV={nv} NP={np_producers} placement={placement}"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"fused {geom}: o differs from phased in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), f"fused {geom} differs from phased"
    for rep in range(8):
        o_rep, fs_rep = _run_op(device, tensors, const_tiles, s0)
        assert torch.equal(o_rep, o_fu), f"fused {geom}: o not reproducible on repeat {rep + 1}: race"
        assert torch.equal(fs_rep, fs_fu), f"fused {geom}: final_state not reproducible on repeat {rep + 1}: race"


def test_phased_selectors_agree(device, monkeypatch):
    """QWEN_GDN_PATH=phased and the legacy QWEN_GDN_PHASED=1 must select the same prims: the second
    selector compiles nothing new and reproduces the first bit for bit. Callers that still set the
    legacy knob (and the prims tests) rely on it."""
    hk, hv = NP_BH_KV_HEADS
    _clear_gdn_env(monkeypatch)
    _, tensors, s0 = _make_inputs(device, 1, T_SMALL, hk, hv, True, seed=20260932)
    const_tiles = _const_tiles(device)
    monkeypatch.setenv("QWEN_GDN_PATH", "phased")
    o1, fs1 = _run_op(device, tensors, const_tiles, s0)
    n1 = device.num_program_cache_entries()
    monkeypatch.delenv("QWEN_GDN_PATH")
    monkeypatch.setenv("QWEN_GDN_PHASED", "1")
    o2, fs2 = _run_op(device, tensors, const_tiles, s0)
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 0, f"QWEN_GDN_PHASED=1 compiled {n2 - n1} programs after QWEN_GDN_PATH=phased (expected 0)"
    assert torch.equal(o1, o2) and torch.equal(fs1, fs2), "the two phased selectors disagree"
