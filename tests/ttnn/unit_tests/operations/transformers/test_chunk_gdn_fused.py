# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused GDN prefill path (ttnn::prim::chunk_gdn_fused).

The fused prim runs prep and scan in one program: per head, a producer core runs the unchanged
prep reader+compute and NoC-writes the seven fp32 intermediates (v_beta, kd, q_decay, intra,
k_dec_t, dl, t_inv) straight into the paired scan core's CBs.
The compute kernels are byte-identical to the phased path and the DRAM round trip they replace
was byte-preserving, so the fused op must be identical to the phased op.

The path is selected by the op's `program_config` (ttnn.ChunkGdnFusedProgramConfig /
ChunkGdnPhasedProgramConfig / ChunkGdnMonoProgramConfig): the alternative passed names the prim, its
fields the geometry. With program_config=None the op chooses by the fused op's calibrated geometry
cost model (chunk_gdn_fused_geometry): fused iff a fused geometry fits the compute grid and its
predicted time beats the phased reference. The model itself, its feasibility predicate and the
factory's core map are tested host-side, per grid, in test_chunk_gdn_fused_geometry.py. Because
fused and phased are bit-exact by design, every path proof here rests on program-cache entry
deltas (a new prim or a new config compiles a new program; a cache hit does not).
"""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(
    not is_blackhole(), reason="the fused and phased chunk_gated_delta_rule paths are Blackhole-only"
)

CHUNK = 32  # Ct=1: the production chunk size
KDIM = 128
VDIM = 128
T_SMALL = 256  # NC=8 — enough chunks to exercise the recurrence, small enough to keep runtime down

# The simulator runs much slower than silicon so the test repeats run on hardware only, as well as the large input cases.
_SIM = bool(os.environ.get("TT_METAL_SIMULATOR"))
_hw_only = pytest.mark.skipif(_SIM, reason="hardware test only")
# The chunk-64 phased prep program (Ct=2) exceeds the kernel-config buffer when the watcher is compiled in: its three
# compute binaries alone are 62 KB of the 69 KB there. The production chunk size is 32.
_no_watcher = pytest.mark.skipif(
    bool(os.environ.get("TT_METAL_WATCHER")),
    reason="chunk-64 prep binaries exceed the kernel-config buffer under the watcher",
)
# Trace capture is a fast-dispatch feature; the simulator SKUs run with TT_METAL_SLOW_DISPATCH_MODE set.
_fast_dispatch_only = pytest.mark.skipif(
    bool(os.environ.get("TT_METAL_SLOW_DISPATCH_MODE")), reason="trace capture needs fast dispatch"
)
REPEATS = 0 if _SIM else 8  # extra fused runs against the first result, to give a timing race a chance to show


def _phased(**kwargs):
    return ttnn.ChunkGdnPhasedProgramConfig(**kwargs)


def _fused(nv=None, np_producers=None, **kwargs):
    """A fused program config; None for nv / np_producers leaves that field to the cost model."""
    return ttnn.ChunkGdnFusedProgramConfig(num_receivers=nv, num_producers=np_producers, **kwargs)


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


def _run_op(device, tensors, const_tiles, initial_state, program_config=None, wy_inverse=None):
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
        program_config=program_config,
        wy_inverse=wy_inverse if wy_inverse is not None else ttnn.ChunkGdnWyInverse.AUTO,
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


def test_program_config_defaults():
    """The three program configs and their defaults, as the op and the model construct them."""
    f = ttnn.ChunkGdnFusedProgramConfig()
    assert (f.num_producers, f.num_receivers, f.row_local) == (None, None, None)
    assert (f.handoff_depth, f.unicast, f.posted, f.wave_heads) == (None, True, False, None)
    assert (f.producer_pool, f.pool_extra_share) == (None, None)
    f = ttnn.ChunkGdnFusedProgramConfig(num_receivers=4, num_producers=5, row_local=False, handoff_depth=3)
    assert (f.num_receivers, f.num_producers, f.row_local, f.handoff_depth) == (4, 5, False, 3)
    assert "num_receivers=4" in repr(f) and "row_local=False" in repr(f)
    f = ttnn.ChunkGdnFusedProgramConfig(num_receivers=2, num_producers=78, producer_pool=True, pool_extra_share=0.25)
    assert (f.num_producers, f.producer_pool) == (78, True) and abs(f.pool_extra_share - 0.25) < 1e-6
    assert "producer_pool=True" in repr(f) and "pool_extra_share=0.25" in repr(f)
    f = ttnn.ChunkGdnFusedProgramConfig(wave_heads=8)
    assert f.wave_heads == 8 and "wave_heads=8" in repr(f)
    p = ttnn.ChunkGdnPhasedProgramConfig()
    assert (p.use_mcast, p.scan_serial, p.prep_serial) == (True, False, False)
    assert repr(ttnn.ChunkGdnPhasedProgramConfig(use_mcast=False)) == (
        "ChunkGdnPhasedProgramConfig(use_mcast=False, scan_serial=False, prep_serial=False)"
    )
    assert repr(ttnn.ChunkGdnMonoProgramConfig()) == "ChunkGdnMonoProgramConfig()"


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_fused_bit_exact_vs_phased(device, with_initial_state):
    """Primary test: fused output == phased output, bit for bit. The producer packs the seven
    intermediates at the same CB boundaries/formats the phased prep packs them at, and the DRAM
    round trip it eliminates was a byte copy — so torch.equal."""
    B, num_k_heads, num_v_heads = 1, 16, 48  # single-device Qwen3.6 shape, GQA G=3
    BH = B * num_v_heads
    _skip_unless_fused_fits(device, BH)

    _, tensors, s0 = _make_inputs(device, B, T_SMALL, num_k_heads, num_v_heads, with_initial_state, seed=20260820)
    const_tiles = _const_tiles(device)

    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0, _phased())
    n_phased = device.num_program_cache_entries()

    # A second phased run must be a full cache hit: it pins that the pre/postprocessing graph
    # around the prims is cache-stable, so the fused-run delta below is attributable purely to
    # the prim the config selected.
    o_ph2, fs_ph2 = _run_op(device, tensors, const_tiles, s0, _phased())
    n_phased2 = device.num_program_cache_entries()
    assert n_phased2 == n_phased, (
        f"repeated phased run compiled {n_phased2 - n_phased} new programs (expected 0): the op "
        "graph is not cache-stable, so program-cache deltas cannot prove which prim dispatched"
    )
    assert torch.equal(o_ph, o_ph2) and torch.equal(fs_ph, fs_ph2), "phased path is not deterministic"

    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0, _fused())
    n_fused = device.num_program_cache_entries()

    # The phased runs compiled prep+scan (2 prim programs) plus the shared pre/postprocessing;
    # the fused run reuses everything except the prim, which is ONE program (that is the point:
    # producer and receiver kernels live in a single fused program, zero DRAM intermediates).
    # A delta of 0 means the program config was not threaded to the dispatch and the "fused" run
    # silently reused the phased path — which would make the bit-exact comparison below vacuously
    # compare the phased op against itself. A delta of 2 means the fused branch dispatched prep+scan.
    assert n_fused - n_phased2 == 1, (
        f"phased->fused program config compiled {n_fused - n_phased2} new programs "
        "(expected exactly 1, the single fused prim program): 0 => the config was not threaded "
        "(comparison vacuous), 2 => the fused branch ran the phased prims"
    )

    assert torch.equal(o_fu, o_ph), "fused path changed o (must be bit-identical to phased)"
    assert torch.equal(fs_fu, fs_ph), "fused path changed final_state (must be bit-identical to phased)"


def _cost_model_path(device, bh, nc):
    """The op's default path per its calibrated geometry cost model: fused iff its wave plan (one program,
    or several for the heads one program's core map cannot hold, each with the geometry — NP producers
    per head, or one producer pool — the model picks for its head count) is predicted faster than the
    phased reference, or, when BH exceeds the grid (phased needs one scan core per head, so it cannot run),
    whenever a plan exists at all."""
    from ttnn._ttnn.operations import transformer as _t

    grid = device.compute_with_storage_grid_size()
    waves, t_f, t_ph, pays = _t.chunk_gdn_fused_wave_plan(grid.x, grid.y, bh, nc, VDIM // 32)
    phased_fits = bh <= grid.x * grid.y
    return "fused" if waves and (pays or not phased_fits) else "phased"


def _wave_plan(device, bh, nc):
    from ttnn._ttnn.operations import transformer as _t

    grid = device.compute_with_storage_grid_size()
    return list(_t.chunk_gdn_fused_wave_plan(grid.x, grid.y, bh, nc, VDIM // 32)[0])


@pytest.mark.parametrize(
    "batch, num_k_heads, num_v_heads, nc",
    [
        pytest.param(1, 16, 48, 8, id="bh48-NC8"),  # BH=48: the single-device shape
        pytest.param(1, 16, 48, 64, id="bh48-NC64", marks=_hw_only),
        pytest.param(1, 4, 16, 8, id="bh16-NC8"),  # BH=16: the producer pool's shape
        pytest.param(1, 4, 16, 64, id="bh16-NC64", marks=_hw_only),
        pytest.param(1, 4, 12, 8, id="bh12-NC8"),  # BH=12: the 27B TP-4 shape
        pytest.param(1, 4, 12, 64, id="bh12-NC64"),  # the production chunk count; the one NC=64 case the simulator runs
        pytest.param(1, 1, 4, 8, id="bh4-NC8"),  # BH=4: chain-bound
        pytest.param(1, 1, 4, 64, id="bh4-NC64", marks=_hw_only),
        pytest.param(1, 16, 64, 8, id="bh64-NC8"),  # BH=64: no single program on a 110-core grid -> fused waves
        pytest.param(1, 16, 64, 64, id="bh64-NC64", marks=_hw_only),
        # Above a 110-core grid the phased scan cannot run (one core per head); at NC=1 the model still prices the
        # waves above the phased reference, so these pin that the dispatch takes the waves regardless.
        pytest.param(10, 4, 12, 1, id="bh120-NC1", marks=_hw_only),
        pytest.param(16, 4, 12, 1, id="bh192-NC1", marks=_hw_only),  # 27B at TP-4, batch 16, a 32-token prefill
    ],
)
def test_fused_default_dispatch(device, batch, num_k_heads, num_v_heads, nc):
    """With NO program config, the dispatcher must pick what the calibrated cost model says: fused
    iff its wave plan beats the phased reference, each program at the geometry (NP producers per head, or
    one producer pool) the model picks for its head count. The choice depends on NC (the fill cost is
    amortized over the chunks), so both a short and the production chunk count run; the plan oracle
    (chunk_gdn_fused_wave_plan) names the expected path per grid. Since fused and phased are bit-exact,
    torch.equal cannot discriminate paths: the
    proof that the default took the expected path is a program-cache delta of ZERO after warming
    exactly that path with an explicit config (any other path would compile at least one new prim
    program)."""
    B = batch
    BH = B * num_v_heads
    grid = device.compute_with_storage_grid_size()
    expected_path = _cost_model_path(device, BH, nc)
    if BH > grid.x * grid.y:
        assert expected_path == "fused", f"BH={BH} exceeds the {grid.x}x{grid.y} grid and no fused plan exists either"
    explicit = _fused() if expected_path == "fused" else _phased()  # free: the prim re-derives the plan's picks

    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20260821)
    const_tiles = _const_tiles(device)

    o_exp, fs_exp = _run_op(device, tensors, const_tiles, s0, explicit)
    n_explicit = device.num_program_cache_entries()

    o_def, fs_def = _run_op(device, tensors, const_tiles, s0, None)
    n_default = device.num_program_cache_entries()

    assert n_default - n_explicit == 0, (
        f"default dispatch (program_config=None) compiled {n_default - n_explicit} new programs after "
        f"an explicit '{expected_path}' run: the default did NOT take the {expected_path} path for "
        f"BH={BH} (the delta counts the unexpected branch's prim programs: fused or mono = 1, "
        "phased prep+scan = 2)"
    )
    assert torch.equal(o_def, o_exp), f"default dispatch o differs from explicit '{expected_path}' run"
    assert torch.equal(fs_def, fs_exp), f"default dispatch final_state differs from explicit '{expected_path}' run"
    if expected_path == "fused":
        # A fused config with every field unset is the fused side of that dispatch: the same program again.
        _run_op(device, tensors, const_tiles, s0, _fused())
        assert (
            device.num_program_cache_entries() - n_default == 0
        ), "ChunkGdnFusedProgramConfig() compiled a new program: it did not resolve to the default dispatch's geometry"


# ---------------------------------------------------------------------------
# NP > 1 producers per head (num_producers). Producer p owns the round-robin chunks
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
def test_fused_np_bit_exact_vs_phased(device, np_producers, nc):
    """Fused with NP>1 producers == phased, bit for bit, across the NC
    boundary cases. The compute kernels are untouched by the split (prep is chunk-independent),
    so any difference is a protocol/addressing bug, not numerical noise."""
    B = 1
    num_k_heads, num_v_heads = NP_BH_KV_HEADS
    BH = B * num_v_heads
    _skip_unless_fused_np_fits(device, BH, np_producers, nc)

    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20260823)
    const_tiles = _const_tiles(device)

    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0, _phased())
    if not _SIM:  # determinism of the phased reference: pinned by test_fused_bit_exact_vs_phased, re-checked on silicon
        o_ph2, fs_ph2 = _run_op(device, tensors, const_tiles, s0, _phased())
        assert torch.equal(o_ph, o_ph2) and torch.equal(fs_ph, fs_ph2), "phased path is not deterministic"
    n_phased = device.num_program_cache_entries()

    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0, _fused(np_producers=np_producers))
    n_fused = device.num_program_cache_entries()
    assert n_fused - n_phased == 1, (
        f"phased->fused(NP={np_producers}) compiled {n_fused - n_phased} new programs "
        "(expected exactly 1, the fused prim): 0 => the config was not threaded and the "
        "comparison below is vacuous"
    )

    assert torch.equal(o_fu, o_ph), f"fused NP={np_producers} changed o (must be bit-identical to phased)"
    assert torch.equal(fs_fu, fs_ph), f"fused NP={np_producers} changed final_state (must be bit-identical to phased)"


def test_fused_np_cache_identity(device):
    """Vacuity canary for num_producers: it must reach attrs.np, which is hashed, so each distinct NP
    compiles its own fused program, repeats are cache hits, and toggling back recompiles nothing. Were
    the field dropped on the way to the attributes (or read in the factory instead), the toggle deltas
    would be 0 and every NP 'A/B' would silently compare one cached program against itself. Deltas
    are asserted EXACTLY (never >=)."""
    B = 1
    num_k_heads, num_v_heads = NP_BH_KV_HEADS
    BH = B * num_v_heads
    nc = 16
    _skip_unless_fused_np_fits(device, BH, 3, nc)

    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20260824)
    const_tiles = _const_tiles(device)

    o1, fs1 = _run_op(device, tensors, const_tiles, s0, _fused())  # NP free -> the cost model's pick
    n1 = device.num_program_cache_entries()

    o2, fs2 = _run_op(device, tensors, const_tiles, s0, _fused(np_producers=2))
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 1, f"np free->2 compiled {n2 - n1} programs (expected 1: np must be hashed)"

    o3, fs3 = _run_op(device, tensors, const_tiles, s0, _fused(np_producers=3))
    n3 = device.num_program_cache_entries()
    assert n3 - n2 == 1, f"np 2->3 compiled {n3 - n2} programs (expected 1)"

    o2b, fs2b = _run_op(device, tensors, const_tiles, s0, _fused(np_producers=2))
    n4 = device.num_program_cache_entries()
    assert n4 - n3 == 0, f"np 3->2 (already compiled) compiled {n4 - n3} programs (expected 0: cache hit)"

    o1b, fs1b = _run_op(device, tensors, const_tiles, s0, _fused())
    n5 = device.num_program_cache_entries()
    assert n5 - n4 == 0, f"np 2->free (the model's pick, already compiled) compiled {n5 - n4} programs (expected 0)"

    # All NP variants of the same head must agree bit-for-bit with each other, and a cached program's
    # second launch with its first.
    assert torch.equal(o1, o2) and torch.equal(o1, o3), "o differs across NP values"
    assert torch.equal(fs1, fs2) and torch.equal(fs1, fs3), "final_state differs across NP values"
    assert torch.equal(o2, o2b) and torch.equal(fs2, fs2b), "the cached NP=2 program's second launch differs"
    assert torch.equal(o1, o1b) and torch.equal(fs1, fs1b), "the cached free-NP program's second launch differs"


def test_fused_vs_torch_golden(device):
    """Whole-computation correctness of the fused path against the inlined torch golden. The
    bit-exact gate above anchors fused==phased; this gate anchors the pair to the math (bf16
    q/k/v inputs dominate the error; kernel math is fp32/HiFi4 end-to-end). Initial state is
    exercised here (s0 flows through the receiver's DRAM read — the one input the producer does
    not ship); the s0=None case is covered bit-exactly vs phased above."""
    B, num_k_heads, num_v_heads = 1, 8, 24  # BH=24: the smallest default-fused shape; GQA G=3
    BH = B * num_v_heads
    _skip_unless_fused_fits(device, BH)

    host, tensors, s0_dev = _make_inputs(device, B, T_SMALL, num_k_heads, num_v_heads, True, seed=20260822)
    q, k, v, g, beta, s0 = host
    const_tiles = _const_tiles(device)

    o_d, fs_d = _run_op(device, tensors, const_tiles, s0_dev, _fused())
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
    """The fused factory's own feasibility predicates: nv | Vt, the receiver rectangles
    fit the grid (row-major or row-local placement), and BH*(nv+np) cores exist (np is clamped to nc
    by the op host). `placement` is the factory's int (0 row-major, 1 row-local) = row_local as 0/1."""
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


def _fused_vs_phased(device, hk, hv, nc, nv, np_producers, seed, wy_inverse=None, **fused_kwargs):
    """Run phased (twice, for cache stability), then fused with the given geometry and any further
    fused-config fields, both with the same WY-inverse method. Returns the outputs and the program-cache
    delta of the fused run (must be exactly 1: one fused program)."""
    B = 1
    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, hk, hv, True, seed=seed)
    const_tiles = _const_tiles(device)
    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0, _phased(), wy_inverse)

    if not _SIM:  # determinism of the phased reference: pinned by test_fused_bit_exact_vs_phased, re-checked on silicon
        o_ph2, fs_ph2 = _run_op(device, tensors, const_tiles, s0, _phased(), wy_inverse)
        assert torch.equal(o_ph, o_ph2) and torch.equal(fs_ph, fs_ph2), "phased path is not deterministic"
    n_phased = device.num_program_cache_entries()

    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0, _fused(nv, np_producers, **fused_kwargs), wy_inverse)
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
        pytest.param(4, 5, 64, marks=_hw_only),  # the QB2 production geometry at the production chunk count (T=2048)
        (2, 7, 16),  # the NV=2 production candidate (12*(2+7)=108 cores)
    ],
    ids=lambda v: str(v),
)
def test_fused_nv_bit_exact_vs_phased(device, nv, np_producers, nc):
    """Primary gate at the 27B TP-4 shape (BH=12): fused with NV receivers per head ==
    phased, bit for bit. On failure, name the (head, v-block) slices that differ."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(device, hk, hv, nc, nv, np_producers, 20260925)
    assert delta == 1, (
        f"phased->fused(NV={nv},NP={np_producers}) compiled {delta} new programs (expected exactly 1): "
        "0 => the config was not threaded (comparison vacuous), 2 => the fused branch ran the phased prims"
    )
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"fused NV={nv} NP={np_producers}: o differs from phased in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph), "fused o differs from phased outside any single v-block (layout bug)"
    assert torch.equal(fs_fu, fs_ph), f"fused NV={nv} NP={np_producers}: final_state differs from phased"


@pytest.mark.parametrize("unicast", [True, False], ids=["unicast", "mcast"])
@pytest.mark.parametrize(
    "nv, np_producers, nc, nbuf",
    [
        pytest.param(
            2, 7, 64, 3, marks=_hw_only
        ),  # BH=12 NV=2 operating point (24 receivers + 84 producers), default ring
        pytest.param(
            4, 5, 64, 2, marks=_hw_only
        ),  # BH=12 NV=4 gate geometry (48 + 60) at the shallowest pipelined ring (D = 1)
        (4, 5, 8, 4),  # short chain + deeper ring: every slot index is exercised on both sides
        (2, 3, 9, 3),  # NC not a multiple of NP or nbuf
        (2, 1, 7, 3),  # single producer per head: the same word is credited for chunks c and c+nbuf
        (4, 3, 5, 1),  # nbuf=1: one slot, D=1 — the pipelined reader degenerates to the unpipelined loop
    ],
)
def test_fused_nv_transport_bit_exact(device, nv, np_producers, nc, nbuf, unicast):
    """The hand-off shipped as NV plain unicast writes per item (unicast=True, the default) or
    as the linked multicast chain (False), with nbuf-1 hand-offs in flight per receiver,
    is bit-identical to phased. Transport and depth are hashed, so the fused program compiles
    fresh (delta == 1)."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20260921, unicast=unicast, handoff_depth=nbuf
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
        pytest.param(2, 7, 64, marks=_hw_only),  # one head per row (9 cores) + 2 heads as 2x5 blocks in columns 9-10
        (4, 5, 16),  # same, the leftover heads' receivers as 2x2 rectangles
        (2, 3, 9),  # L=5: leftover heads as 1x2 receivers over 6 columns, 2 rows each
        (1, 9, 8),  # NV=1: L=10, leftover width 1
    ],
)
def test_fused_nv_row_local_placement_bit_exact(device, nv, np_producers, nc):
    """Placement test: row-local placement (row_local=True) — one head per row,
    leftover heads as column blocks with 2-D receiver rectangles — is bit-identical to phased. The
    placement is hashed (delta == 1)."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=1)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20260921, row_local=True
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
def test_fused_nv_row_local_shapes_bit_exact(device, hk, hv, nv, np_producers, nc):
    """Row-local placement across the BH range the cost model dispatches:
    k heads per row plus leftover column blocks, bit-identical to phased."""
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=1)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20260921 + hv, row_local=True
    )
    assert delta == 1, f"row-local fused(BH={hv},NV={nv},NP={np_producers}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"row-local fused BH={hv} NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "row-local fused differs from phased"


@pytest.mark.parametrize(
    "nv, np_producers, nc, nbuf",
    [
        pytest.param(2, 7, 64, 2, marks=_hw_only),  # NV=2 operating point, the default ring depth
        (4, 5, 8, 3),  # NV=4, short chain, D=2 in flight
        (2, 1, 7, 3),  # single producer per head
    ],
)
def test_fused_nv_posted_bit_exact(device, nv, np_producers, nc, nbuf):
    """Posted unicast data writes with the VALID flag ordered
    behind them by same-VC in-order delivery (no per-item barrier) are bit-identical to phased."""
    hk, hv = NP_BH_KV_HEADS
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20260921, unicast=True, posted=True, handoff_depth=nbuf
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
def test_fused_nv_shapes_bit_exact(device, hk, hv, nv, np_producers):
    """Other target shapes: the geometry must place and stay bit-exact wherever it is feasible."""
    nc = 8
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(device, hk, hv, nc, nv, np_producers, 20260926)
    assert delta == 1, f"expected exactly one new (fused) program, got {delta}"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"BH={hv} NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "fused differs from phased"


def test_fused_nv_cache_identity(device):
    """Num_receivers: it must reach attrs.nv, which is hashed, so each distinct NV compiles its
    own fused program and a repeat is a cache hit. Deltas asserted EXACTLY."""
    hk, hv = NP_BH_KV_HEADS
    nc = 8
    _skip_unless_geometry_fits(device, hv, 4, 2, nc)
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20260927)
    const_tiles = _const_tiles(device)

    o1, fs1 = _run_op(device, tensors, const_tiles, s0, _fused(np_producers=2))  # NV free -> the model's NV for NP=2
    n1 = device.num_program_cache_entries()
    o2, fs2 = _run_op(device, tensors, const_tiles, s0, _fused(2, 2))
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 1, f"nv free->2 compiled {n2 - n1} programs (expected 1: nv must be hashed)"
    o4, fs4 = _run_op(device, tensors, const_tiles, s0, _fused(4, 2))
    n4 = device.num_program_cache_entries()
    assert n4 - n2 == 1, f"nv 2->4 compiled {n4 - n2} programs (expected 1)"
    _run_op(device, tensors, const_tiles, s0, _fused(2, 2))
    n5 = device.num_program_cache_entries()
    assert n5 - n4 == 0, f"nv 4->2 (already compiled) compiled {n5 - n4} programs (expected 0: cache hit)"
    _run_op(device, tensors, const_tiles, s0, _fused(np_producers=2))
    n6 = device.num_program_cache_entries()
    assert n6 - n5 == 0, f"nv 2->free (the model's pick, already compiled) compiled {n6 - n5} programs (expected 0)"
    assert torch.equal(o1, o2) and torch.equal(o1, o4), "o differs across NV values"
    assert torch.equal(fs1, fs2) and torch.equal(fs1, fs4), "final_state differs across NV values"


@_hw_only
def test_fused_nv_repeats(device):
    """NV-way handshake race is non-deterministic, so one comparison has little power. Re-run
    the production geometry (BH=12, NV=4, NP=5, T=2048) REPEATS times against the first result."""
    hk, hv = NP_BH_KV_HEADS
    nc, nv, np_producers = 64, 4, 5
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, (tensors, const_tiles, s0) = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20260928
    )
    assert delta == 1 and torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "first fused run is not bit-exact"
    cfg = _fused(nv, np_producers)
    for rep in range(REPEATS):
        o_rep, fs_rep = _run_op(device, tensors, const_tiles, s0, cfg)
        assert torch.equal(o_rep, o_fu), f"fused NV=4 NP=5 o not reproducible on repeat {rep + 1}: race"
        assert torch.equal(fs_rep, fs_fu), f"fused NV=4 NP=5 final_state not reproducible on repeat {rep + 1}: race"


@pytest.mark.parametrize(
    "field, a, b",
    [
        ("handoff_depth", 2, 3),  # hand-off ring depth
        ("unicast", True, False),  # per-receiver unicast vs linked multicast hand-off
        ("posted", False, True),  # posted unicast data writes (requires unicast, the default)
        ("row_local", False, True),  # row-major vs row-local core map
    ],
    ids=lambda v: str(v),
)
def test_fused_knob_cache_identity(device, field, a, b):
    """Remaining geometry/transport fields: each must reach the hashed attributes, so
    a -> b compiles exactly one new fused program and every revisit is a cache hit. A field that is
    accepted but not hashed would serve a program built for the other setting — wrong answers that
    depend on test order. NV/NP are pinned to a geometry both placements accept."""
    hk, hv = NP_BH_KV_HEADS
    nc, nv, np_producers = 8, 2, 3
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=0)
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=1)
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20260929)
    const_tiles = _const_tiles(device)
    cfg_a = _fused(nv, np_producers, **{field: a})
    cfg_b = _fused(nv, np_producers, **{field: b})

    o_a, fs_a = _run_op(device, tensors, const_tiles, s0, cfg_a)
    n_a = device.num_program_cache_entries()
    o_b, fs_b = _run_op(device, tensors, const_tiles, s0, cfg_b)
    n_b = device.num_program_cache_entries()
    assert n_b - n_a == 1, f"{field} {a}->{b} compiled {n_b - n_a} programs (expected 1: the field must be hashed)"
    _run_op(device, tensors, const_tiles, s0, cfg_a)
    _run_op(device, tensors, const_tiles, s0, cfg_b)
    n_rev = device.num_program_cache_entries()
    assert n_rev - n_b == 0, f"{field}: revisiting {a} and {b} compiled {n_rev - n_b} programs (expected 0: cache hits)"
    assert torch.equal(o_a, o_b) and torch.equal(fs_a, fs_b), f"{field}={a} and {field}={b} disagree"


@pytest.mark.parametrize(
    "hk, hv, nv, np_producers, nc",
    [
        (4, 12, 2, 3, 8),  # BH=12: 1x2 rectangles, 5 heads per row
        (4, 12, 4, 5, 16),  # BH=12 at NV=4: 2 heads per row, 3 stranded columns per receiver row
        pytest.param(4, 12, 2, 7, 64, marks=_hw_only),  # BH=12 at the model's NV/NP, production chunk count
        (4, 16, 4, 2, 8),  # BH=16 (397B TP-4): 8 receiver rows
        (2, 8, 4, 8, 8),  # BH=8, producer-rich
    ],
    ids=lambda v: str(v),
)
def test_fused_nv_row_major_placement_bit_exact(device, hk, hv, nv, np_producers, nc):
    """Test for row_local=False. The op defaults to row-local placement whenever it is feasible, so the
    row-major 1xNV rectangles — and their NOC_1 multicast orientation — only run when forced. A wrong
    orientation lands on another head's receivers: finite, wrong, no hang; the per-V-block report
    names the (head, vblock) slices that differ."""
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=0)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20260930 + hv, row_local=False
    )
    assert delta == 1, f"row-major fused(BH={hv},NV={nv},NP={np_producers}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"row-major fused BH={hv} NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "row-major fused differs from phased"


@_hw_only
@pytest.mark.parametrize(
    "hk, hv",
    [(4, 12), (4, 16), (1, 4)],
    ids=["bh12", "bh16", "bh4"],
)
def test_fused_default_geometry_repeats(device, hk, hv):
    """A fused config with every field free, so the op
    uses the cost model's pick (on QB2 at BH=12: NV=2, NP=7, row-local, hand-off depth 3). Bit-exact vs
    phased, then 8 repeats against the first fused result — a handshake race is timing-dependent, so one
    comparison has little power."""
    nc = 64
    grid = device.compute_with_storage_grid_size()
    from ttnn._ttnn.operations import transformer as _t

    nv, np_producers, placement, nbuf = _t.chunk_gdn_fused_geometry(grid.x, grid.y, hv, nc, VDIM // 32)[:4]
    if nv == 0:
        pytest.skip(f"no fused geometry for BH={hv} on the {grid.x}x{grid.y} grid")
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20260931 + hv)
    const_tiles = _const_tiles(device)

    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0, _phased())
    n_phased = device.num_program_cache_entries()
    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0, _fused(producer_pool=placement == 2))
    assert device.num_program_cache_entries() - n_phased == 1, "the fused run did not compile the fused prim"
    geom = f"BH={hv} NV={nv} NP={np_producers} placement={placement} depth={nbuf}"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"fused {geom}: o differs from phased in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), f"fused {geom} differs from phased"
    for rep in range(REPEATS):
        o_rep, fs_rep = _run_op(device, tensors, const_tiles, s0, _fused())
        assert torch.equal(o_rep, o_fu), f"fused {geom}: o not reproducible on repeat {rep + 1}: race"
        assert torch.equal(fs_rep, fs_fu), f"fused {geom}: final_state not reproducible on repeat {rep + 1}: race"


def test_fused_config_pinned_geometry_matches_free(device):
    """Pinning the cost model's own pick explicitly (geometry, placement and hand-off depth) must be the
    SAME program as leaving the fields free (the free config is what the op's default dispatch builds),
    and a free config after a pinned one is a cache hit. Pins the equivalence the default-dispatch test
    relies on."""
    hk, hv = NP_BH_KV_HEADS
    nc = 16
    grid = device.compute_with_storage_grid_size()
    from ttnn._ttnn.operations import transformer as _t

    nv, np_producers, placement, nbuf = _t.chunk_gdn_fused_geometry(grid.x, grid.y, hv, nc, VDIM // 32)[:4]
    if nv == 0:
        pytest.skip(f"no fused geometry for BH={hv} on the {grid.x}x{grid.y} grid")
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20260933)
    const_tiles = _const_tiles(device)
    pooled = placement == 2
    pinned = _fused(
        nv,
        np_producers,
        row_local=None if pooled else bool(placement),
        handoff_depth=nbuf,
        producer_pool=pooled,
    )
    o_pin, fs_pin = _run_op(device, tensors, const_tiles, s0, pinned)
    n_pin = device.num_program_cache_entries()
    o_free, fs_free = _run_op(device, tensors, const_tiles, s0, _fused(producer_pool=pooled))
    assert (
        device.num_program_cache_entries() == n_pin
    ), "a free fused config compiled a new program after its own pinned geometry: the model's pick is not the default"
    assert torch.equal(o_pin, o_free) and torch.equal(fs_pin, fs_free), "pinned and free geometries disagree"


# ---------------------------------------------------------------------------
# WY-inverse methods (wy_inverse = ttnn.ChunkGdnWyInverse.AUTO | HORNER | FORWARD_SUBSTITUTION; AUTO = the solve on
# Blackhole at chunk_size 32, Horner elsewhere). The SFPU forward-substitution solve changes the arithmetic
# of T_inv, but the phased prep and the fused producer compile the same body for a given method, so
# fused == phased stays bit-exact for every method. Every other test in this file runs AUTO, so the solve is
# what they exercise; the tests below pin each method explicitly. The solver's own accuracy is tested on the
# prep prim (test_chunk_gdn_prims.py).
# ---------------------------------------------------------------------------

AUTO, HORNER, FORWARD_SUBSTITUTION = (
    ttnn.ChunkGdnWyInverse.AUTO,
    ttnn.ChunkGdnWyInverse.HORNER,
    ttnn.ChunkGdnWyInverse.FORWARD_SUBSTITUTION,
)


@pytest.mark.parametrize("method", [HORNER, FORWARD_SUBSTITUTION], ids=["horner", "forward_substitution"])
@pytest.mark.parametrize(
    "hk, hv, nv, np_producers, nc, placement",
    [
        (4, 12, 2, 7, 64, 1),  # BH=12 (27B TP-4) at the model's geometry, T=2048
        (4, 12, 4, 5, 16, 1),  # NV=4 receivers (Vtl=1)
        (4, 12, 2, 3, 8, 0),  # row-major placement
        (16, 48, 1, 1, 8, 1),  # BH=48 (single-device shape), one producer per head
        (1, 4, 4, 7, 16, 1),  # BH=4
    ],
    ids=lambda v: str(v),
)
def test_fused_tinv_bit_exact_vs_phased(device, method, hk, hv, nv, np_producers, nc, placement):
    """fused == phased, bit for bit, with the same WY-inverse method pinned on both paths."""
    _skip_unless_geometry_fits(device, hv, nv, np_producers, nc, placement=placement)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, np_producers, 20261001 + hv, wy_inverse=method, row_local=bool(placement)
    )
    assert delta == 1, f"{method}: fused compiled {delta} new programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"{method} BH={hv} NV={nv} NP={np_producers}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), f"{method}: fused differs from phased"


def test_fused_tinv_vs_horner(device):
    """End to end at the 27B TP-4 shape (BH=12, T=2048, the model's default fused geometry): the default
    WY-inverse (AUTO, the SFPU solve on this device) against pinned Horner, and against the torch golden. The
    T_inv difference is ~1e-3 (prims test); across the 64-chunk recurrence it must stay PCC-class."""
    hk, hv, nc = 4, 12, 64
    host, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20261002)
    const_tiles = _const_tiles(device)
    o_h, fs_h = _run_op(device, tensors, const_tiles, s0, _fused(), HORNER)
    o_s, fs_s = _run_op(device, tensors, const_tiles, s0, _fused(), AUTO)
    assert not torch.equal(o_s, o_h), "AUTO output identical to Horner — the SFPU solve did not run"
    q, k, v, g, beta, s0_host = host
    o_ref, fs_ref = _golden_chunk_gdn(q.float(), k.float(), v.float(), g, beta, KDIM**-0.5, s0_host, CHUNK)
    for name, got, horner, ref in (("o", o_s, o_h, o_ref), ("final_state", fs_s, fs_h, fs_ref)):
        pcc_h = _pcc(horner.float(), got.float())
        assert pcc_h >= 0.99999, f"{name}: PCC vs Horner {pcc_h} < 0.99999"
        pcc_ref, pcc_ref_h = _pcc(ref, got.float()), _pcc(ref, horner.float())
        assert pcc_ref >= 0.999, f"{name}: PCC vs torch golden {pcc_ref} < 0.999"
        # no worse than the Horner reference against the golden, beyond PCC noise
        assert pcc_ref >= pcc_ref_h - 1e-5, f"{name}: PCC vs golden {pcc_ref} < Horner's {pcc_ref_h}"


def test_fused_tinv_cache_identity(device):
    """Cache identity for wy_inverse on both prims: AUTO resolves to FORWARD_SUBSTITUTION on this device (the explicit form is the same
    program and the same bits), HORNER compiles its own fused program and its own phased prep program (the
    scan is unchanged, so phased compiles exactly one), and revisits are cache hits."""
    hk, hv = NP_BH_KV_HEADS
    _, tensors, s0 = _make_inputs(device, 1, T_SMALL, hk, hv, True, seed=20261003)
    const_tiles = _const_tiles(device)
    for cfg in (_fused(), _phased()):
        o_def, fs_def = _run_op(device, tensors, const_tiles, s0, cfg, AUTO)
        n = device.num_program_cache_entries()
        o_exp, fs_exp = _run_op(device, tensors, const_tiles, s0, cfg, FORWARD_SUBSTITUTION)
        assert (
            device.num_program_cache_entries() == n
        ), f"{cfg}: explicit FORWARD_SUBSTITUTION compiled a new program — AUTO must resolve to it on this device"
        assert torch.equal(o_def, o_exp) and torch.equal(
            fs_def, fs_exp
        ), f"{cfg}: AUTO != explicit FORWARD_SUBSTITUTION"
        _run_op(device, tensors, const_tiles, s0, cfg, HORNER)
        n2 = device.num_program_cache_entries()
        assert (
            n2 - n == 1
        ), f"{cfg}: FORWARD_SUBSTITUTION->HORNER compiled {n2 - n} programs (expected 1: the method must be hashed)"
        for method in (AUTO, FORWARD_SUBSTITUTION, HORNER):
            _run_op(device, tensors, const_tiles, s0, cfg, method)
        assert device.num_program_cache_entries() == n2, f"{cfg}: revisiting the methods compiled new programs"


@_no_watcher
def test_fused_tinv_chunk64(device, expect_error):
    """The SFPU solve is a single-tile (chunk_size == 32) routine. At chunk_size 64 AUTO falls back to Horner
    (the default is the solve wherever it is supported, and that program is the pinned-Horner one), but an
    explicit FORWARD_SUBSTITUTION must be refused, not silently downgraded (which would make any A/B vacuous)."""
    _, tensors, s0 = _make_inputs(device, 1, 256, 4, 12, True, seed=20261004)
    q, k, v, g, beta = tensors
    eye, tril, ones, masks = _const_tiles(device, chunk_size=64)

    def run(wy_inverse):
        o, fs = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=s0,
            output_final_state=True,
            chunk_size=64,
            program_config=_phased(),
            wy_inverse=wy_inverse,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        return ttnn.to_torch(o), ttnn.to_torch(fs)

    o_def, fs_def = run(AUTO)
    n = device.num_program_cache_entries()
    o_h, fs_h = run(HORNER)
    assert device.num_program_cache_entries() == n, "chunk 64: AUTO compiled a different program than HORNER"
    assert torch.equal(o_def, o_h) and torch.equal(fs_def, fs_h), "chunk 64: AUTO is not the Horner inverse"
    with expect_error(RuntimeError, "needs chunk_size == 32"):
        run(FORWARD_SUBSTITUTION)


def test_mono_tinv_horner_only(device, expect_error):
    """The mono program has no forward-substitution solve: AUTO resolves to Horner there (bit-identical to an
    explicit HORNER) and an explicit FORWARD_SUBSTITUTION is refused rather than silently downgraded."""
    hk, hv = NP_BH_KV_HEADS
    _, tensors, s0 = _make_inputs(device, 1, T_SMALL, hk, hv, True, seed=20261005)
    const_tiles = _const_tiles(device)
    mono = ttnn.ChunkGdnMonoProgramConfig()
    o_auto, fs_auto = _run_op(device, tensors, const_tiles, s0, mono, AUTO)
    o_h, fs_h = _run_op(device, tensors, const_tiles, s0, mono, HORNER)
    assert torch.equal(o_auto, o_h) and torch.equal(fs_auto, fs_h), "mono: AUTO is not the Horner inverse"
    with expect_error(RuntimeError, "Horner only"):
        _run_op(device, tensors, const_tiles, s0, mono, FORWARD_SUBSTITUTION)


# ---------------------------------------------------------------------------
# Producer pool (producer_pool=True, placement 2). One pool of P producers for every head: the receivers
# and BH*NPH home producers in the row-local map of the largest NPH the pool allows, every other core of
# the pool an extra producer serving all heads with the share pool_extra_share of each head's chunks
# (default NX/P). Which producer computes which (head, chunk) is one formula shared by the factory and
# the three dataflow kernels (chunk_gdn_fused_map.hpp; its partition and owner properties are tested
# host-side in test_chunk_gdn_fused_geometry.py). The gate here is the same as everywhere: torch.equal
# vs phased, and a protocol error hangs rather than corrupts.
# ---------------------------------------------------------------------------


def _skip_unless_pool_fits(device, bh, nv, pool, nc):
    from ttnn._ttnn.operations import transformer as _t

    grid = device.compute_with_storage_grid_size()
    if VDIM // 32 % nv != 0 or nv > grid.x:
        pytest.skip(f"NV={nv} does not divide Vt or exceeds the grid width")
    if pool > bh * nc:
        pytest.skip(f"pool of {pool} exceeds the BH*NC={bh * nc} items")
    if not _t.chunk_gdn_fused_pool_feasible(grid.x, grid.y, bh, nv, pool):
        pytest.skip(f"a producer pool of {pool} for BH={bh} NV={nv} does not fit the {grid.x}x{grid.y} grid")


@pytest.mark.parametrize(
    "hk, hv, nv, pool, nc",
    [
        (4, 16, 2, 78, 8),  # BH=16 NV=2: 3 home producers per head + 30 extras (the target geometry)
        (4, 16, 2, 78, 9),  # NC not a multiple of anything
        (4, 16, 2, 78, 64),  # the production chunk count
        (4, 16, 1, 94, 8),  # BH=16 NV=1: 4 per head + 30 extras
        (4, 16, 2, 48, 8),  # no extras: the row-local NV=2 NP=3 map through the pool kernels
        (4, 12, 2, 86, 8),  # BH=12 NV=2: 7 per head + 2 extras
        (4, 12, 2, 40, 16),  # BH=12 NV=2, a small pool: 3 per head + 4 extras
        (1, 4, 2, 102, 32),  # BH=4 NV=2: 9 per head + 66 extras, most items on the extras (NC=32: one item per core)
        (1, 4, 4, 94, 32),  # BH=4 NV=4: 7 per head + 66 extras
        (16, 48, 1, 62, 8),  # BH=48 NV=1: 1 per head + 14 extras
    ],
)
def test_fused_pool_bit_exact_vs_phased(device, hk, hv, nv, pool, nc):
    """Fused with a producer pool == phased, bit for bit, at the pool geometries of the target shapes."""
    _skip_unless_pool_fits(device, hv, nv, pool, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, pool, 20261001 + hv + pool, producer_pool=True
    )
    assert delta == 1, f"pooled fused(NV={nv},P={pool}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"pooled fused BH={hv} NV={nv} P={pool}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), "pooled fused differs from phased"


@pytest.mark.parametrize("share", [0.0, 1.0 / 3.0, 1.0], ids=["none", "third", "all"])
def test_fused_pool_extra_share_bit_exact(device, share):
    """pool_extra_share at its extremes (the extras idle; every chunk on the extras, the home producers
    idle) and at the 1/3 of the design note is bit-exact: idle producers of either kind are legal."""
    hk, hv, nv, pool, nc = 4, 16, 2, 78, 8
    _skip_unless_pool_fits(device, hv, nv, pool, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, pool, 20261002, producer_pool=True, pool_extra_share=share
    )
    assert delta == 1, f"pooled fused(share={share}) compiled {delta} programs (expected 1)"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), f"pooled fused share={share} differs from phased"


@pytest.mark.parametrize(
    "hk, hv, nv, pool, nc, kwargs",
    [
        (4, 16, 2, 78, 8, dict(unicast=False)),  # the linked multicast chain to each item's head
        (4, 16, 2, 78, 8, dict(posted=True)),  # posted unicast writes, VALID ordered by delivery
        (4, 16, 2, 78, 8, dict(handoff_depth=3)),  # two hand-offs in flight per receiver
        (4, 12, 2, 86, 9, dict(unicast=False, handoff_depth=3)),  # NC not a multiple of the ring
        (1, 4, 4, 94, 32, dict(posted=True, handoff_depth=3)),  # NV=4, most items on the extras
    ],
    ids=lambda v: str(sorted(v.items())) if isinstance(v, dict) else str(v),
)
def test_fused_pool_transport_bit_exact(device, hk, hv, nv, pool, nc, kwargs):
    """The pool's item map under the other transports and a deeper ring — the linked multicast chain
    (unicast=False), posted unicast writes (posted=True), hand-off depth 3 — is bit-identical to phased."""
    _skip_unless_pool_fits(device, hv, nv, pool, nc)
    (o_ph, fs_ph), (o_fu, fs_fu), delta, _ = _fused_vs_phased(
        device, hk, hv, nc, nv, pool, 20261005 + hv + pool, producer_pool=True, **kwargs
    )
    assert delta == 1, f"pooled fused(NV={nv},P={pool},{kwargs}) compiled {delta} programs (expected 1)"
    bad = _vblock_mismatches(o_ph, o_fu, hv, nv)
    assert not bad, f"pooled fused BH={hv} NV={nv} P={pool} {kwargs}: o differs in (head, vblock) slices {bad}"
    assert torch.equal(o_fu, o_ph) and torch.equal(fs_fu, fs_ph), f"pooled fused {kwargs} differs from phased"


def test_fused_pool_cache_identity(device):
    """producer_pool and pool_extra_share must reach the hashed attributes: per-head -> pool compiles one
    program, a different share another, and every revisit is a cache hit. Pinning the model's own pool
    size (every core the receivers leave) is the same program as leaving num_producers free."""
    from ttnn._ttnn.operations import transformer as _t

    hk, hv = NP_BH_KV_HEADS
    nc, nv = 8, 2
    grid = device.compute_with_storage_grid_size()
    pool = min(grid.x * grid.y - hv * nv, hv * nc)
    _skip_unless_pool_fits(device, hv, nv, pool, nc)
    _skip_unless_geometry_fits(device, hv, nv, 3, nc, placement=1)
    _, tensors, s0 = _make_inputs(device, 1, nc * CHUNK, hk, hv, True, seed=20261003)
    const_tiles = _const_tiles(device)

    o_h, fs_h = _run_op(device, tensors, const_tiles, s0, _fused(nv, 3))
    n0 = device.num_program_cache_entries()
    o_p, fs_p = _run_op(device, tensors, const_tiles, s0, _fused(nv, pool, producer_pool=True))
    n1 = device.num_program_cache_entries()
    assert n1 - n0 == 1, f"per-head -> pool compiled {n1 - n0} programs (expected 1)"
    o_s, fs_s = _run_op(device, tensors, const_tiles, s0, _fused(nv, pool, producer_pool=True, pool_extra_share=0.5))
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 1, f"pool_extra_share compiled {n2 - n1} programs (expected 1: the share must be hashed)"
    o_f, fs_f = _run_op(device, tensors, const_tiles, s0, _fused(nv, producer_pool=True))  # the model's P == pool
    o_h2, fs_h2 = _run_op(device, tensors, const_tiles, s0, _fused(nv, 3))
    o_p2, fs_p2 = _run_op(device, tensors, const_tiles, s0, _fused(nv, pool, producer_pool=True))
    n3 = device.num_program_cache_entries()
    assert n3 - n2 == 0, f"revisits compiled {n3 - n2} programs (expected 0: cache hits, free P == pinned {pool})"
    assert torch.equal(o_h, o_p) and torch.equal(o_h, o_s), "o differs between the per-head and pool forms"
    assert torch.equal(fs_h, fs_p) and torch.equal(fs_h, fs_s), "final_state differs between the forms"
    for tag, (o_b, fs_b) in (("pool, free P", (o_f, fs_f)), ("per-head", (o_h2, fs_h2)), ("pool", (o_p2, fs_p2))):
        assert torch.equal(o_p, o_b) and torch.equal(fs_p, fs_b), f"the cached {tag} program's second launch differs"


def test_fused_pool_infeasible_raises(device, expect_error):
    """A pool smaller than a home producer per head, or larger than the grid, is refused. Both are pinned to
    one wave: left to the plan, the batch would run as waves of 8 heads (the small pool as is, the large one
    clamped to the wave's BH*NC items)."""
    hk, hv = 4, 16
    _, tensors, s0 = _make_inputs(device, 1, T_SMALL, hk, hv, True, seed=20261004)
    const_tiles = _const_tiles(device)
    grid = device.compute_with_storage_grid_size()
    with expect_error(RuntimeError, "no fused geometry fits"):
        _run_op(device, tensors, const_tiles, s0, _fused(2, hv - 1, producer_pool=True, wave_heads=hv))
    with expect_error(RuntimeError, "no fused geometry fits"):
        _run_op(device, tensors, const_tiles, s0, _fused(2, grid.x * grid.y, producer_pool=True, wave_heads=hv))
    with expect_error(RuntimeError, "pool_extra_share must be in"):
        _run_op(device, tensors, const_tiles, s0, _fused(2, None, producer_pool=True, pool_extra_share=1.5))
    with expect_error(RuntimeError, "pool_extra_share must be in"):  # a pool with no extras (P = BH*NPH)
        _run_op(device, tensors, const_tiles, s0, _fused(2, 3 * hv, producer_pool=True, pool_extra_share=-0.1))


# ---------------------------------------------------------------------------
# Head waves: heads beyond what one program's core map holds run as further fused programs (waves) into
# shared outputs, each wave addressing its head range of the same tensors. The gate is torch.equal against
# ONE phased run of the whole batch (the phased scan needs a core per head, so these shapes stay <= 110).
# ---------------------------------------------------------------------------


def _run_op_kw(device, tensors, const_tiles, initial_state, program_config, **kwargs):
    """_run_op with the public op's output options; returns (o, final_state or None) as torch tensors."""
    q, k, v, g, beta = tensors
    eye, tril, ones, masks = const_tiles
    out = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        chunk_size=CHUNK,
        program_config=program_config,
        wy_inverse=ttnn.ChunkGdnWyInverse.AUTO,
        eye=eye,
        tril=tril,
        ones=ones,
        masks=masks,
        **kwargs,
    )
    o, fs = out
    o_t = ttnn.to_torch(o)
    fs_t = ttnn.to_torch(fs) if fs is not None else None
    ttnn.deallocate(o)
    if fs is not None:
        ttnn.deallocate(fs)
    return o_t, fs_t


def _make_flat_inputs(device, batch, seq, num_k_heads, num_v_heads, with_initial_state, seed):
    """Rank-3 token-major inputs of the model's flat path: q/k unnormalized (the kernel normalizes), v, g,
    beta as [B, T, heads * dim] / [B, T, HV]."""
    torch.manual_seed(seed)
    B, T, H, HV = batch, seq, num_k_heads, num_v_heads
    q = torch.randn(B, T, H * KDIM).to(torch.bfloat16)
    k = torch.randn(B, T, H * KDIM).to(torch.bfloat16)
    v = (0.5 * torch.randn(B, T, HV * VDIM)).to(torch.bfloat16)
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
    return tensors, (dev(s0, ttnn.float32) if s0 is not None else None)


def _head_mismatches(golden, actual, num_v_heads):
    """(batch, head) pairs whose rows differ, for token-major [B, T, HV, V] or state [B, HV, K, V] tensors."""
    bad = []
    for b in range(golden.shape[0]):
        for h in range(num_v_heads):
            g = golden[b, :, h] if golden.dim() == 4 and golden.shape[2] == num_v_heads else golden[b, h]
            a = actual[b, :, h] if actual.dim() == 4 and actual.shape[2] == num_v_heads else actual[b, h]
            if not torch.equal(g, a):
                bad.append((b, h))
    return bad


def _skip_unless_phased_fits(device, bh):
    grid = device.compute_with_storage_grid_size()
    if bh > grid.x * grid.y:
        pytest.skip(f"BH={bh} exceeds the {grid.x}x{grid.y} grid: no phased reference (one scan core per head)")


@pytest.mark.parametrize(
    "batch, num_k_heads, num_v_heads, nc, wave_heads",
    [
        pytest.param(2, 4, 12, 8, 12, id="bh24-2x12"),  # pinned: two waves of one batch row each
        pytest.param(4, 4, 12, 8, None, id="bh48-plan"),  # the plan: BH 48 -> two waves on QB2
        pytest.param(2, 16, 48, 8, None, id="bh96-plan"),  # single-device shape at batch 2 -> four waves on QB2
        pytest.param(4, 4, 12, 64, None, id="bh48-plan-NC64", marks=_hw_only),
        pytest.param(2, 16, 48, 64, None, id="bh96-plan-NC64", marks=_hw_only),
    ],
)
def test_fused_waves_bit_exact_vs_phased(device, batch, num_k_heads, num_v_heads, nc, wave_heads):
    """Waved fused output == one phased run of the whole batch, bit for bit: every wave reads its heads of the
    same inputs and writes its rows of the same outputs, and the per-wave programs are the fused programs
    of their head count."""
    BH = batch * num_v_heads
    _skip_unless_phased_fits(device, BH)
    waves = _wave_plan(device, BH, nc) if wave_heads is None else None
    if wave_heads is None and len(waves) < 2:
        pytest.skip(f"the plan runs BH={BH} as one program on this grid ({waves}); nothing to wave")

    _, tensors, s0 = _make_inputs(device, batch, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20261012)
    const_tiles = _const_tiles(device)
    o_ph, fs_ph = _run_op(device, tensors, const_tiles, s0, _phased())
    o_fu, fs_fu = _run_op(device, tensors, const_tiles, s0, _fused(wave_heads=wave_heads))
    assert torch.equal(
        o_fu, o_ph
    ), f"waved o differs from phased at (batch, head) {_head_mismatches(o_ph, o_fu, num_v_heads)}"
    assert torch.equal(
        fs_fu, fs_ph
    ), f"waved final_state differs from phased at (batch, head) {_head_mismatches(fs_ph, fs_fu, num_v_heads)}"


@pytest.mark.parametrize("flat", [False, True], ids=["rank4", "flat"])
@pytest.mark.parametrize("with_initial_state", [False, True], ids=["s0=0", "s0=rand"])
def test_fused_waves_straddle_batch_rows(device, flat, with_initial_state):
    """Waves of 8 heads over B=2 x HV=12: the middle wave holds heads of both batch rows, on the head-major
    (rank-4) inputs and on the flat token-major inputs the prep reader addresses by batch and head itself.
    The rank-4 case pads T to the chunk size (T=300 -> 320), which the waves must not disturb."""
    B, num_k_heads, num_v_heads = 2, 4, 12
    _skip_unless_phased_fits(device, B * num_v_heads)
    const_tiles = _const_tiles(device)
    if flat:
        tensors, s0 = _make_flat_inputs(device, B, T_SMALL, num_k_heads, num_v_heads, with_initial_state, seed=20261013)
    else:
        _, tensors, s0 = _make_inputs(device, B, 300, num_k_heads, num_v_heads, with_initial_state, seed=20261013)
    o_ph, fs_ph = _run_op_kw(device, tensors, const_tiles, s0, _phased(), output_final_state=True)
    o_fu, fs_fu = _run_op_kw(device, tensors, const_tiles, s0, _fused(wave_heads=8), output_final_state=True)
    assert torch.equal(o_fu, o_ph), f"o differs at (batch, head) {_head_mismatches(o_ph, o_fu, num_v_heads)}"
    assert torch.equal(
        fs_fu, fs_ph
    ), f"final_state differs at (batch, head) {_head_mismatches(fs_ph, fs_fu, num_v_heads)}"


@pytest.mark.parametrize("output_head_major", [False, True], ids=["token_major", "head_major"])
@pytest.mark.parametrize("output_final_state", [False, True], ids=["o_only", "o+state"])
def test_fused_waves_output_options(device, output_head_major, output_final_state):
    """The output options of the public op on a waved run: head-major o (a metadata reshape of the shared
    output) and the final state left out, each against the phased run with the same options."""
    B, num_k_heads, num_v_heads = 2, 4, 12
    _skip_unless_phased_fits(device, B * num_v_heads)
    const_tiles = _const_tiles(device)
    _, tensors, s0 = _make_inputs(device, B, T_SMALL, num_k_heads, num_v_heads, True, seed=20261014)
    kw = dict(output_head_major=output_head_major, output_final_state=output_final_state)
    o_ph, fs_ph = _run_op_kw(device, tensors, const_tiles, s0, _phased(), **kw)
    o_fu, fs_fu = _run_op_kw(device, tensors, const_tiles, s0, _fused(wave_heads=8), **kw)
    assert torch.equal(o_fu, o_ph)
    assert (fs_fu is None) == (fs_ph is None) == (not output_final_state)
    if output_final_state:
        assert torch.equal(fs_fu, fs_ph)


def test_fused_waves_cache_identity(device):
    """The waves of one plan share a program: the head offset is not part of the program-cache key but is
    re-applied on every launch, so a two-wave batch compiles ONE fused program and a repeat compiles
    nothing. Deltas are asserted EXACTLY (the inputs' shapes stay in the key, as for every op)."""
    num_k_heads, num_v_heads = 4, 12
    nc = 8
    _skip_unless_phased_fits(device, 4 * num_v_heads)
    if _wave_plan(device, 4 * num_v_heads, nc) != [24, 24]:
        pytest.skip(
            f"the plan for BH=48 on this grid is {_wave_plan(device, 4 * num_v_heads, nc)}, not two waves of 24"
        )
    const_tiles = _const_tiles(device)
    _, tensors4, s0_4 = _make_inputs(device, 4, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20261016)

    _run_op(device, tensors4, const_tiles, s0_4, _phased())  # the shared pre/postprocessing graph, warmed
    n0 = device.num_program_cache_entries()
    o4, fs4 = _run_op(device, tensors4, const_tiles, s0_4, _fused())  # BH 48 -> waves [24, 24]
    n1 = device.num_program_cache_entries()
    assert n1 - n0 == 1, f"two waves of 24 compiled {n1 - n0} fused programs (expected 1: one per wave SIZE)"
    o4b, fs4b = _run_op(device, tensors4, const_tiles, s0_4, _fused())
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 0, f"the repeated waved run compiled {n2 - n1} programs (expected 0)"
    assert torch.equal(o4, o4b) and torch.equal(fs4, fs4b), "waved run is not deterministic"


@_fast_dispatch_only
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_fused_waves_trace(device):
    """A waved call captured in a trace and replayed: each wave's program is re-enqueued with its own
    head offset, which the replay must reproduce (the offset rides on the per-launch runtime args)."""
    B, num_k_heads, num_v_heads = 2, 4, 12
    nc = 8
    _skip_unless_phased_fits(device, B * num_v_heads)
    const_tiles = _const_tiles(device)
    _, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20261017)
    q, k, v, g, beta = tensors
    eye, tril, ones, masks = const_tiles
    cfg = _fused(wave_heads=8)  # three waves, the middle one across the batch rows

    def run():
        return ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            initial_state=s0,
            output_final_state=True,
            output_head_major=True,  # a metadata reshape: the trace holds the fused programs only
            chunk_size=CHUNK,
            program_config=cfg,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )

    o_ref, fs_ref = run()  # eager, also warms the program cache for the capture
    o_ref_t, fs_ref_t = ttnn.to_torch(o_ref), ttnn.to_torch(fs_ref)
    ttnn.deallocate(o_ref)
    ttnn.deallocate(fs_ref)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    o_tr, fs_tr = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert torch.equal(ttnn.to_torch(o_tr), o_ref_t), "trace replay o differs from the eager run"
        assert torch.equal(ttnn.to_torch(fs_tr), fs_ref_t), "trace replay final_state differs from the eager run"
    ttnn.release_trace(device, trace_id)


# ---------------------------------------------------------------------------
# Batches above the grid, and the prim's head range on its own outputs.
# ---------------------------------------------------------------------------


@_hw_only
def test_fused_waves_above_grid_bit_exact_per_row(device):
    """A batch with more heads than the grid has cores (the 27B TP-4 shape at batch 16: 192 heads on 110 cores)
    has no phased reference of the whole batch, so the waved default dispatch is checked one batch row at a
    time: every head's scan is independent and fused == phased bit for bit at any geometry, so the 16-row
    result must equal the 16 single-row phased runs of the same inputs, stacked."""
    B, num_k_heads, num_v_heads = 16, 4, 12
    nc = 8
    BH = B * num_v_heads
    grid = device.compute_with_storage_grid_size()
    if BH <= grid.x * grid.y:
        pytest.skip(f"BH={BH} fits the {grid.x}x{grid.y} grid; this check is for batches above it")
    assert _cost_model_path(device, BH, nc) == "fused"
    host, tensors, s0 = _make_inputs(device, B, nc * CHUNK, num_k_heads, num_v_heads, True, seed=20261026)
    const_tiles = _const_tiles(device)
    o_all, fs_all = _run_op(device, tensors, const_tiles, s0, None)  # the default dispatch: the plan's waves
    q, k, v, g, beta, s0_host = host
    dtypes = (ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, ttnn.float32)
    for b in range(B):
        row = tuple(
            ttnn.from_torch(t[b : b + 1], dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
            for t, dt in zip((q, k, v, g, beta), dtypes)
        )
        s0_row = ttnn.from_torch(s0_host[b : b + 1], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        o_b, fs_b = _run_op(device, row, const_tiles, s0_row, _phased())
        assert torch.equal(o_all[b : b + 1], o_b), f"batch row {b}: waved o differs from its single-row phased run"
        assert torch.equal(fs_all[b : b + 1], fs_b), f"batch row {b}: waved final_state differs from its phased run"


def _make_prim_inputs(device, rows, nc, seed):
    """Head-major inputs of the fused prim, as the public op hands them over after its preprocessing: q/k
    [rows, NC, C, K] bf16 L2-normalized (scale 1), v [rows, NC, C, V] bf16, the token-major gates as one batch
    row of `rows` heads [1, T, rows], and initial_state [rows, K, V] fp32."""
    torch.manual_seed(seed)
    T = nc * CHUNK
    q = F.normalize(torch.randn(rows, nc, CHUNK, KDIM), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(rows, nc, CHUNK, KDIM), dim=-1).to(torch.bfloat16)
    v = (0.5 * torch.randn(rows, nc, CHUNK, VDIM)).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, T, rows))
    g = -F.softplus(torch.randn(1, T, rows)) * 0.5
    s0 = 0.05 * torch.randn(rows, KDIM, VDIM)

    def dev(t, dtype):
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    return (
        dev(q, ttnn.bfloat16),
        dev(k, ttnn.bfloat16),
        dev(v, ttnn.bfloat16),
        dev(g, ttnn.float32),
        dev(beta, ttnn.float32),
        dev(s0, ttnn.float32),
    )


def test_fused_prim_head_range_on_own_outputs(device):
    """A fused prim call for heads [head_offset, head_offset + num_heads) WITHOUT preallocated outputs returns
    compact [num_heads, ...] outputs holding exactly that head range at rows 0..num_heads-1: the inputs are read
    at the global rows, the outputs written at the wave-local ones. Checked on a program-cache miss (the 8-head
    program's first launch, at offset 0) and on a hit (offset 16 of the same program: head_offset is not part
    of the cache key, so the writer's row is re-applied per launch) against one 24-head run of the same inputs."""
    from ttnn._ttnn.operations import transformer as _t

    rows, nc = 24, 8
    _skip_unless_fused_fits(device, rows)
    q, k, v, g, beta, s0 = _make_prim_inputs(device, rows, nc, seed=20261027)
    eye, tril, ones, masks = _const_tiles(device)

    def prim(num_heads, head_offset):
        o, fs = _t.chunk_gdn_fused(
            q, k, v, g, beta, eye, tril, ones, masks, s0, chunk_size=CHUNK, num_heads=num_heads, head_offset=head_offset
        )
        o_t, fs_t = ttnn.to_torch(o), ttnn.to_torch(fs)
        ttnn.deallocate(o)
        ttnn.deallocate(fs)
        return o_t, fs_t

    o_full, fs_full = prim(0, 0)  # every row, one program
    assert tuple(o_full.shape) == (rows, nc, CHUNK, VDIM) and tuple(fs_full.shape) == (rows, KDIM, VDIM)

    n0 = device.num_program_cache_entries()
    o_lo, fs_lo = prim(8, 0)  # heads 0..7: the 8-head program's cache miss
    n1 = device.num_program_cache_entries()
    assert n1 - n0 == 1, f"the 8-head program compiled {n1 - n0} programs (expected 1)"
    o_hi, fs_hi = prim(8, 16)  # heads 16..23: a cache hit of that program with a new head offset
    n2 = device.num_program_cache_entries()
    assert n2 - n1 == 0, f"the offset-16 launch compiled {n2 - n1} programs (expected a cache hit)"

    assert tuple(o_lo.shape) == (8, nc, CHUNK, VDIM) and tuple(fs_lo.shape) == (8, KDIM, VDIM)
    assert torch.equal(o_lo, o_full[0:8]) and torch.equal(fs_lo, fs_full[0:8]), "heads 0..7 differ from the full run"
    assert torch.equal(o_hi, o_full[16:24]), "heads 16..23 (cache hit): o is not the full run's rows 16..23"
    assert torch.equal(fs_hi, fs_full[16:24]), "heads 16..23 (cache hit): final_state is not rows 16..23"
