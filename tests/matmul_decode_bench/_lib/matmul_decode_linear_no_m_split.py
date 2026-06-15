# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MatmulDecodeLinearNoMSplit + COPIED M-aware plan_matmul_decode for the fork's
now-FIXED (commit e476d1ce) width-sharded ``ttnn.matmul_decode`` device op.

This is the M=T-direct matmul primitive for ``pi05_chunked_mmdecode``: each
projection / MLP matmul is fed its M dimension WHOLE (the per-stage T-row query
chunk, or the full sequence S for QKV, or 1 for adaRMS modulation) -- there is NO
internal ``ceil(M/32)`` 32-row re-split, for BOTH FULL and PARTIAL modes. The fix
(deep-work/matmul_decode_Mfix_execution.md) made the M-tile compute math correct
at all M, so feeding M whole is exact.

The planner (plan_matmul_decode / _full_ws_l1_bytes / the PARTIAL helpers) is
COPIED here (deep-plan_4 s4.2: the new dir OWNS its planner; the parent
``pi05_blocked_mmdecode.matmul_decode_linear`` is NOT edited) and threaded with an
``m_tiles`` parameter so the FULL-WS K-split budget G-search auto-grows G with M
against the conservative budget knob 1_310_720. The flat per-core L1 sum is a
POST-SEARCH backstop only, NOT the fit gate (the round-2 HW finding: the flat sum
under-predicts the real CB-region clash, so the conservative budget gate is what
forces VLM-QKV@M=288 to G=2).

``MatmulDecodeLinearNoMSplit`` subclasses the parent ``MatmulDecodeLinear`` (import
only; parent untouched) and overrides ONLY ``__call__`` (remove the ceil(M/32)
loop, feed M whole) and ``__init__`` (thread m_tiles into the COPIED planner). All
the staging / shard / matmul helpers (stage / _a_shard_mc / _b_l1_mc_* /
_call_full / _call_partial / _call_one_mtile / describe-fields) are REUSED verbatim
from the parent -- the only M-coupling was the removed loop (and, pre-fix, the
kernel).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import ttnn

from .matmul_decode_linear import (
    DEFAULT_CORE_CAP,
    DEFAULT_L1_FULL_WS_BUDGET_B,
    M_TILE,
    P150_L1_BANK_BYTES,
    MatmulDecodeLinear,
    _A_TILE_BYTES,
    _MATMUL_DECODE_HAS_STREAM_K,
    _device_usable_cores,
    _full_ws_l1_bytes,
    _matmul_decode_kwargs,
    _plan_chunk,
    _plan_n_chunks,
    _tile_bytes,
    _to_torch_KN,
    _wide_n_out_cores,
)

# deep-plan_12 S0: refreshed from the stale b56f57e5 to the live clean-fork HEAD.
TT_METAL_COMMIT = "e4500c1fae97c103b16fc24fc7010b852992a9e6"

# HW-observed FULL static-CB-region ceiling (deep-plan_4 s5.2; program.cpp:1476).
# The flat L1 sum under-predicts the real CB-region-vs-L1-buffer clash. The
# K-split G-search additionally requires (full_in0 + in1) per group <= this cap.
# Calibrated from on-device clashes: vlm_down G=4 @M=96 (full_in0+in1=925,696)
# PASSES; siglip_fc2 K=4320 G=1 @M=96 (976,320) CLASHES (program.cpp:1476).
DEFAULT_CB_REGION_CAP_B = 940_000


# ----------------------------------------------------------------------------
# COPIED + M-aware planner (deep-plan_4 s4.2). _full_ws_l1_bytes / the FULL
# G-search are threaded with m_tiles so K-split auto-grows G with M.
# ----------------------------------------------------------------------------
def _full_ws_l1_bytes_m(K: int, in1_dtype, m_tiles: int = 1, *, a_cores: int = 2,
                        npc: int = 1, stream_k: bool = False,
                        k_slice_tiles: int = 16) -> int:
    """Per-core FULL-WS L1 floor for ONE call at M_tiles=m_tiles (plan_5 s5.2 fix).
    Delegates to the parent's corrected helper: in0 = m*(k_tiles/a_cores)*A_tile (not the old
    'conservative 1'), in1/out carry the npc multiplier, and full_in0 is the double-buffered
    streamed slice when stream_k. Mirrors full_width_sharded_program_factory.cpp."""
    return _full_ws_l1_bytes(K, in1_dtype, m_tiles, a_cores=a_cores, npc=npc,
                             stream_k=stream_k, k_slice_tiles=k_slice_tiles)


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def _clamp_k_slice(k_tiles: int, a_cores: int, want: int) -> int:
    """Reproduce the factory's K_slice clamp (full_width_sharded_program_factory.cpp:101-114):
    K_slice <= inA_K_tiles_per_core (= k_tiles // a_cores), then decremented until it divides
    BOTH k_tiles AND inA_K_tiles_per_core."""
    in_a_k = max(1, k_tiles // a_cores)
    ks = min(want, in_a_k)
    while ks > 1 and (k_tiles % ks != 0 or in_a_k % ks != 0):
        ks -= 1
    return max(1, ks)


def _a_cores_candidates(K: int, m_tiles: int, k_slice: int, weight_dtype, budget: int,
                        usable: int, npc: int = 1, out_cores: Optional[int] = None):
    """deep-plan_6 §4.1: divisors of K//32 (<=usable) whose STREAMED full_in0 floor fits the
    budget, descending (largest first -> smallest resident in0). Each entry is (a_cores, clamped
    K_slice). Used by full_seq_mode to pick the fit default while exposing the candidate set for
    the §6c perf sweep.

    DEVICE-VALIDATED CONSTRAINT (deep-plan_6 execution): a_cores MUST be <= out_cores. When the
    A-sender grid extends BEYOND the output (compute-consumer) grid, the extra sender cores are
    ORPHANS inside the multicast bbox; the streamed gather then corrupts (fc2 a_cores=48>36
    out_cores -> PCC 0.88; a_cores=72 -> 0.75; a_cores<=36 -> 0.99998. down a_cores=64==out_cores
    -> 0.99997). Keeping a_cores<=out_cores keeps the senders a subset of the consumer grid."""
    k_tiles = K // 32
    cap = usable if out_cores is None else min(usable, out_cores)
    out = []
    for ac in sorted([d for d in _divisors(k_tiles) if d <= cap], reverse=True):
        ks = _clamp_k_slice(k_tiles, ac, k_slice)
        floor = _full_ws_l1_bytes_m(K, weight_dtype, m_tiles, a_cores=ac, npc=npc,
                                    stream_k=True, k_slice_tiles=ks)
        if floor <= budget:
            out.append((ac, ks))
    return out


def plan_matmul_decode_m(K: int, N: int, usable_cores: int, *,
                         force_partial: bool = False,
                         weight_dtype=ttnn.bfloat8_b,
                         l1_full_ws_budget_B: int = DEFAULT_L1_FULL_WS_BUDGET_B,
                         pin_k_split: Optional[int] = None,
                         pad_n_to: Optional[int] = None,
                         pad_k_to: Optional[int] = None,
                         m_tiles: int = 1,
                         a_cores: int = 2,
                         core_cap: int = DEFAULT_CORE_CAP,
                         stream_a_cores: int = 8,
                         stream_k_slice_tiles: int = 16,
                         full_seq_mode: bool = False) -> dict:
    """M-aware port of pi05_blocked_mmdecode.plan_matmul_decode with the plan_5 collapse:
    for M_tiles<=3 (M<=96) FULL projections the per-projection N-chunk + K-split loops move
    INTO the fork device op -- ONE matmul_decode call per projection:
      * wide-N (gate/up, N=16384): single FULL-WS call, out_cores capped via _wide_n_out_cores
        (== device-op divisor-cap rule), n_chunks=1, k_split_G=1.
      * large-K (down, K=16384): single FULL-WS call, stream_k=True (temporal K-streaming +
        forced fp32 in the factory), a_cores>=8, k_split_G=1.
    M=S QKV (m_tiles>=8) is EXCLUDED -- it keeps the legacy K-split routing + the
    eff_K==2048 && m_tiles>=8 -> G>=2 clamp. The eff_K==16384 -> G>=4 clamp is MOOT for M<=96
    (down is now K-B) and only applies on the legacy path."""
    eff_N = pad_n_to if pad_n_to else N
    eff_K = pad_k_to if pad_k_to else K
    assert eff_K % 32 == 0 and eff_N % 32 == 0, (eff_K, eff_N)

    n_tiles_total = (eff_N + 31) // 32
    out_cores = n_tiles_total
    is_small_m = m_tiles <= 3  # M <= 96 collapse regime
    # deep-plan_6 §4.1: full_seq_mode turns the M<=96 hard cap into a MODE SELECTOR. When the
    # wrapper runs T=S (n_chunks=1) the collapse branches engage at ANY M (up to M=S) and route
    # through stream_k (the streamed M-block DST-tiling kernel handles M_tiles up to 9). The
    # M<=96 path stays byte-identical (full_seq_mode defaults False).
    collapse = (is_small_m or full_seq_mode) and not force_partial

    # ---- COLLAPSED wide-N (gate/up, N=16384): single FULL-WS call, capped out_cores ----
    if collapse and out_cores > usable_cores:
        oc = _wide_n_out_cores(eff_N, core_cap)
        npc = n_tiles_total // oc
        assert n_tiles_total % oc == 0, (n_tiles_total, oc)
        # M<=96: one-shot full_in0 fits -> non-streamed wide-N (byte-identical to plan_5).
        # full_seq (M up to S): one-shot full_in0 = m*K_tiles*A_tile OOMs at M_tiles>=8 even for
        # K=2048 -> MUST stream_k. Divisor-search a_cores for the streamed floor (§3.2: gate/up
        # a_cores=8, K_slice=8, Npc=8 -> out_block_h=1, num_blocks_h=9, total 1,146,880 < bank).
        one_shot = _full_ws_l1_bytes_m(eff_K, weight_dtype, m_tiles, a_cores=a_cores, npc=npc)
        # deep-plan_12 G0a: on the CLEAN fork the op has NO stream_k. The full-WS
        # factory gathers the WHOLE A (M_tiles*K_tiles) per core and the compute
        # kernel loops M_tiles -- so a non-stream wide-N call is correct + fits for
        # the small-M denoise shapes (M=64 -> M_tiles=2, full_in0 ~128 KB/core). When
        # the op cannot stream we MUST take the non-stream branch (it would otherwise
        # build a stream_k=True plan the op rejects). Guard with the one_shot fit.
        _no_stream = not _MATMUL_DECODE_HAS_STREAM_K
        if one_shot <= l1_full_ws_budget_B and (not full_seq_mode or _no_stream):
            return {"mode": "full", "n_chunks": 1, "chunk_N": eff_N,
                    "k_blocks": 1, "n_blocks": oc, "kc": eff_K, "nc": npc * 32,
                    "b_cores": oc, "k_split_G": 1, "Kc_call": eff_K,
                    "in1_dtype": weight_dtype, "pad_N_to": pad_n_to, "pad_K_to": pad_k_to,
                    "partial": False, "wide_n": True, "npc": npc, "stream_k": False,
                    "a_cores": a_cores, "m_tiles": m_tiles}
        cands = _a_cores_candidates(eff_K, m_tiles, stream_k_slice_tiles, weight_dtype,
                                    P150_L1_BANK_BYTES, usable_cores, npc=npc, out_cores=oc)
        assert cands, (f"wide-N stream K={eff_K} N={eff_N} npc={npc} m_tiles={m_tiles}: "
                       f"no a_cores fits bank")
        sac, ks = cands[0]
        return {"mode": "full", "n_chunks": 1, "chunk_N": eff_N,
                "k_blocks": 1, "n_blocks": oc, "kc": eff_K, "nc": npc * 32,
                "b_cores": oc, "k_split_G": 1, "Kc_call": eff_K,
                "in1_dtype": weight_dtype, "pad_N_to": pad_n_to, "pad_K_to": pad_k_to,
                "partial": False, "wide_n": True, "npc": npc, "stream_k": True,
                "a_cores": sac, "k_slice_tiles": ks, "m_tiles": m_tiles,
                "a_cores_candidates": cands}

    # ---- COLLAPSED large-K / full-M (down, O): single FULL-WS K-B streamed call ----
    # Trigger: a NON-streamed one-shot full_in0 (= m*K_tiles*A_tile) would exceed the budget.
    # Fires for down (K=16384) at any M, and for O/QKV-class shapes at full M=S (one-shot 1.42 MB
    # > budget). The legacy `pin_k_split` is IGNORED -- the K-B stream supersedes it.
    # deep-plan_12 G0a: SKIP this streamed branch when the op cannot stream -- fall
    # through to the non-stream FULL-WS K-split path (L223+), which fits L1 by splitting
    # K (G>1) and reduces across calls. Correct for the small-M denoise down (K=4096).
    if _MATMUL_DECODE_HAS_STREAM_K and collapse and out_cores <= usable_cores and (
            _full_ws_l1_bytes_m(eff_K, weight_dtype, m_tiles, a_cores=a_cores) > l1_full_ws_budget_B):
        cands = _a_cores_candidates(eff_K, m_tiles, stream_k_slice_tiles, weight_dtype,
                                    P150_L1_BANK_BYTES, usable_cores, npc=1, out_cores=out_cores)
        if cands:
            sac, ks = cands[0]
        else:
            # fall back to the legacy floor-search (divisor of K/32, descending from stream_a_cores)
            sac = stream_a_cores
            while eff_K % (sac * 32) != 0 and sac > 1:
                sac -= 1
            ks = _clamp_k_slice(eff_K // 32, sac, stream_k_slice_tiles)
        floor = _full_ws_l1_bytes_m(eff_K, weight_dtype, m_tiles, a_cores=sac,
                                    stream_k=True, k_slice_tiles=ks)
        assert floor <= P150_L1_BANK_BYTES, (
            f"K-B stream down K={eff_K} N={eff_N} a_cores={sac} m_tiles={m_tiles} "
            f"floor {floor} > bank")
        return {"mode": "full", "n_chunks": 1, "chunk_N": eff_N,
                "k_blocks": 1, "n_blocks": out_cores, "kc": eff_K, "nc": 32,
                "b_cores": out_cores, "k_split_G": 1, "Kc_call": eff_K,
                "in1_dtype": weight_dtype, "pad_N_to": pad_n_to, "pad_K_to": pad_k_to,
                "partial": False, "wide_n": False, "npc": 1, "stream_k": True,
                "a_cores": sac, "k_slice_tiles": ks, "m_tiles": m_tiles,
                "a_cores_candidates": cands}

    fits_full = out_cores <= usable_cores
    want_partial = force_partial or not fits_full

    # ---- FULL-WS path (single N call), maybe M-aware K-split (legacy / QKV@M=S) ----
    if not want_partial:
        if pin_k_split is not None:
            k_split_G = pin_k_split
        else:
            def _cb_region(Kc):
                kt = Kc // 32
                return m_tiles * kt * _A_TILE_BYTES + kt * _tile_bytes(weight_dtype)

            def _fits(Kc):
                return (_full_ws_l1_bytes_m(Kc, weight_dtype, m_tiles) <= l1_full_ws_budget_B
                        and _cb_region(Kc) <= DEFAULT_CB_REGION_CAP_B)

            g = 1
            while not _fits(eff_K // g):
                g *= 2
                if (eff_K // g) % 32 != 0 or g > eff_K // 32:
                    raise RuntimeError(
                        f"FULL-WS K={eff_K} N={eff_N} m_tiles={m_tiles} "
                        f"cannot power-of-2 K-split to fit L1/CB even at G={g} "
                        f"(Kc not tile-aligned); pad K to a split-friendly width")
            k_split_G = g
            # HW-verified-floor clamps for the kernel/runtime/I2S/CB-region reserve
            # the analytic flat floor omits (deep-plan_4 s5.2, R2/R3; round-2 HW):
            #  * K=16384 bf8_b: G=2 OOM'd on HW at m=1 -> G>=4 (deep-plan_1 s6.3).
            #  * K=2048 FULL at the M=S QKV regime (m_tiles>=8, M>=256): G=1 FATALs
            #    program.cpp:1476 (CB region clashes the L1 buffer) although the flat
            #    floor is sub-budget -> G>=2 (HW: G=2 PCC 0.99986). The flat sum
            #    under-predicts the real CB ceiling, so the clamp encodes it.
            if eff_K == 16384 and weight_dtype == ttnn.bfloat8_b:
                k_split_G = max(k_split_G, 4)
            if eff_K == 2048 and m_tiles >= 8:
                k_split_G = max(k_split_G, 2)
        Kc_call = eff_K // k_split_G
        assert Kc_call % 32 == 0, f"FULL-WS Kc_call {Kc_call} not tile-aligned"
        assert out_cores <= usable_cores, (out_cores, usable_cores)
        # POST-SEARCH backstop only (loud): the chosen-G per-group floor must be
        # under the conservative budget knob. NOT the gate (the gate is the loop).
        floor = _full_ws_l1_bytes_m(Kc_call, weight_dtype, m_tiles)
        assert floor <= l1_full_ws_budget_B, (
            f"FULL-WS post-search backstop: K={eff_K} N={eff_N} G={k_split_G} "
            f"m_tiles={m_tiles} per-group floor {floor} > budget {l1_full_ws_budget_B}")
        return {"mode": "full", "n_chunks": 1, "chunk_N": eff_N,
                "k_blocks": 1, "n_blocks": out_cores, "kc": Kc_call, "nc": 32,
                "b_cores": out_cores, "k_split_G": k_split_G, "Kc_call": Kc_call,
                "in1_dtype": weight_dtype, "pad_N_to": pad_n_to, "pad_K_to": pad_k_to,
                "partial": False, "wide_n": False, "npc": 1, "stream_k": False,
                "a_cores": a_cores, "m_tiles": m_tiles}

    # ---- PARTIAL-WS path with M-aware OUTPUT N-chunking ----
    # The N-chunk count grows ONLY if a chunk's M-scaled partial CB floor would
    # overflow. In PARTIAL mode K is split across kb k-blocks, so the per-core
    # gathered-A (full_in0) holds only kc = K/kb rows -> floor = m*(kc/32)*2048 +
    # in1(kc/32 * nc_tiles). (Using full K here would massively over-count and
    # spuriously grow the chunk count.)
    n_chunks = _plan_n_chunks(eff_K, eff_N, usable_cores, True)
    while True:
        chunk_N = eff_N // n_chunks
        ck = _plan_chunk(eff_K, chunk_N, usable_cores, True)
        kc_tiles = ck["kc"] // 32
        nc_tiles = ck["nc"] // 32  # PARTIAL nc is 32 -> 1 tile
        kb = ck["k_blocks"]
        # plan_5 s5.2 DEFENSIVE fix (moot under the converged design -- no M<=96 partial path --
        # but kept correct for any future partial caller): the partial-WS factory allocates SIX
        # CBs on the base core (c_0..c_5), not just c_0+c_1. Count them all:
        #   c_0 in0 (m*kc) + full_in0 (m*kc, slimmed) + c_1 in1 (kc*nc) + c_2 out (m*nc)
        #   + c_4 partial (m*nc) + c_5 reduce (kb*m*nc).  block = m*nc tiles.
        block = m_tiles * nc_tiles
        partial_floor = (m_tiles * kc_tiles * _A_TILE_BYTES            # c_0 in0
                         + m_tiles * kc_tiles * _A_TILE_BYTES           # c_3 full_in0 (slimmed)
                         + kc_tiles * nc_tiles * _tile_bytes(weight_dtype)  # c_1 in1
                         + block * 2048                                 # c_2 out
                         + block * 2048                                 # c_4 partial
                         + kb * block * 2048)                           # c_5 reduce
        if partial_floor <= l1_full_ws_budget_B or n_chunks >= eff_N // 32:
            break
        nxt = None
        for c in range(n_chunks + 1, eff_N // 32 + 1):
            if eff_N % c == 0 and (eff_N // c) % 32 == 0:
                nxt = c
                break
        if nxt is None:
            break
        n_chunks = nxt
    chunk_N = eff_N // n_chunks
    chunk = _plan_chunk(eff_K, chunk_N, usable_cores, True)
    kb, nb = chunk["k_blocks"], chunk["n_blocks"]
    kc, nc = chunk["kc"], chunk["nc"]
    assert chunk["mode"] == "partial", (eff_K, eff_N, chunk)
    assert kb % 2 == 0, f"partial k_blocks {kb} not even"
    assert kc % 32 == 0, f"partial Kc {kc} not tile-aligned"
    assert nc == 32, f"partial Nc {nc} != 32"
    assert kb * nb == chunk["b_cores"], (kb, nb, chunk["b_cores"])
    assert nb == chunk_N // 32, (nb, chunk_N)
    assert (chunk_N // 32) <= usable_cores, "partial chunk OUTPUT cores > cap"
    return {"mode": "partial", "n_chunks": n_chunks, "chunk_N": chunk_N,
            "k_blocks": kb, "n_blocks": nb, "kc": kc, "nc": nc,
            "b_cores": chunk["b_cores"], "k_split_G": 1, "Kc_call": eff_K,
            "in1_dtype": weight_dtype, "pad_N_to": pad_n_to, "pad_K_to": pad_k_to,
            "partial": True, "wide_n": False, "npc": 1, "stream_k": False,
            "a_cores": a_cores, "m_tiles": m_tiles}


# ----------------------------------------------------------------------------
# MatmulDecodeLinearNoMSplit -- M-whole __call__ (both modes), M-aware plan
# ----------------------------------------------------------------------------
class MatmulDecodeLinearNoMSplit(MatmulDecodeLinear):
    """A single [K, N] linear executed via ttnn.matmul_decode with M fed WHOLE.

    Constructed at move_weights_to_device_impl time with ``m_rows`` = the routed M
    for this instance (T for in-chunk roles, S for full-seq QKV, 64 for the suffix
    action chunk, 1 for adaRMS modulation). The COPIED M-aware planner derives
    K-split G / PARTIAL N-chunks for that M. ``__call__`` feeds M whole (no
    ceil(M/32) loop). All staging/shard/matmul helpers are inherited from the
    parent verbatim.
    """

    def __init__(self, device, weight, *, m_rows: int, bias=None,
                 out_dtype=ttnn.bfloat16, weight_dtype=ttnn.bfloat8_b,
                 role: str = "generic", a_cores: int = 2,
                 core_cap: int = DEFAULT_CORE_CAP, force_partial: bool = False,
                 pin_k_split: Optional[int] = None,
                 pad_n: Optional[int] = None, pad_k: Optional[int] = None,
                 fp32_dest_acc: bool = True, full_seq_mode: bool = False,
                 resident_weight: Optional[bool] = None,
                 t_rows: Optional[int] = None):
        # deep-plan_12 WS-A A.2 -- SURGICAL T-KNOB (NO kernel edit).
        # When ``t_rows`` (T) is set, __call__ slices the incoming M=m_rows activation
        # into ceil(m_rows / T) row-chunks of T rows each and calls matmul_decode per
        # chunk REUSING the SAME L1-resident weight (the per-call weight memcfg
        # _b_l1_mc_full/_partial is M-INDEPENDENT -> SAME _resident_b cache entry, no
        # re-stage between chunks). The op's full-WS factory adapts to M_tiles=ceil(T/32)
        # per call (program-cache keyed on attributes) -- NO .cpp edit. The planner is
        # built for the CHUNK's M (T rows), so M_tiles matches the per-chunk packing
        # (T=32 -> M_tiles=1, T=64 -> M_tiles=2). Default (t_rows None) = monolithic m_rows.
        self.t_rows = int(t_rows) if t_rows else None
        plan_m_rows = self.t_rows if self.t_rows else int(m_rows)
        # Replicate the parent __init__ body (do NOT call super().__init__, which
        # would call the parent m_tiles-less planner AND stage() twice). The ONLY
        # change is the M-aware planner call with m_tiles=ceil(plan_m_rows/32).
        self.device = device
        self.role = role
        self.out_dtype = out_dtype
        self.weight_dtype = weight_dtype
        self.bias_tt = bias
        self.m_rows = int(m_rows)
        # The planner / shards are sized to the CHUNK M (T) when the T-knob is active.
        self.plan_m_rows = plan_m_rows
        self.m_tiles = max(1, math.ceil(plan_m_rows / M_TILE))
        # pi05_chunked_mmdecode OPTS IN to fp32 DST accumulation by default: it needs the
        # higher-precision K-reduction for the Tier-4 T=32 e2e >=0.99. fp32 is now an
        # OPT-IN compute_kernel_config on the fork's ttnn.matmul_decode (default OFF at the
        # op level), so we build the enabling config here and thread it (inherited
        # _call_full / _call_partial read self.compute_kernel_config) to every call.
        self.fp32_dest_acc = bool(fp32_dest_acc)
        self.compute_kernel_config = (
            ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                math_approx_mode=False,
            )
            if self.fp32_dest_acc
            else None
        )

        w_KN = weight if isinstance(weight, torch.Tensor) else _to_torch_KN(weight)
        w_KN = w_KN.to(torch.bfloat16).contiguous()
        self.K_orig, self.N_orig = int(w_KN.shape[0]), int(w_KN.shape[1])

        pad_n_to = pad_n
        pad_k_to = pad_k
        if pad_n and pad_n > self.N_orig:
            w_KN = torch.nn.functional.pad(w_KN, (0, pad_n - self.N_orig))
        if pad_k and pad_k > self.K_orig:
            w_KN = torch.nn.functional.pad(w_KN, (0, 0, 0, pad_k - self.K_orig))
        self.K, self.N = int(w_KN.shape[0]), int(w_KN.shape[1])
        self._w_KN = w_KN
        assert self.K % 32 == 0 and self.N % 32 == 0, (self.K, self.N)

        self.full_seq_mode = bool(full_seq_mode)
        self.usable_cores = _device_usable_cores(device, core_cap)
        self.core_cap = core_cap
        self.plan = plan_matmul_decode_m(
            self.K, self.N, self.usable_cores, force_partial=force_partial,
            weight_dtype=weight_dtype, pin_k_split=pin_k_split,
            pad_n_to=pad_n_to, pad_k_to=pad_k_to, m_tiles=self.m_tiles,
            a_cores=a_cores, core_cap=core_cap, full_seq_mode=self.full_seq_mode)
        self.mode = self.plan["mode"]
        self.partial = self.plan["partial"]
        self.n_chunks = self.plan["n_chunks"]
        self.chunk_N = self.plan["chunk_N"]
        self.k_split_G = self.plan["k_split_G"]
        self.Kc_call = self.plan["Kc_call"]
        self.wide_n = self.plan.get("wide_n", False)
        self.npc = self.plan.get("npc", 1)
        self.stream_k = self.plan.get("stream_k", False)
        self.k_slice_tiles = self.plan.get("k_slice_tiles", 16)

        # K-B streamed down forces a_cores>=8 (planner sets self.plan["a_cores"]).
        self.a_cores = self.plan.get("a_cores", a_cores)
        self._grid = device.compute_with_storage_grid_size()

        self.b_chunks: List = []
        self.b_groups: List = []
        self._staged = False
        self.stage()  # ONCE (outside any trace)

        # deep-plan_7 §3 STEP-10: cross-call weight residency. The parent
        # _call_full/_call_partial re-DMA the weight DRAM->L1 (ttnn.to_memory_config)
        # AND ttnn.deallocate(b_l1) on EVERY call, so the weight is re-read from DRAM
        # every one of the 10 denoise steps -- the reuse win is unrealized. When
        # resident_weight is ON we route the weight through an L1-resident per-mc cache
        # (_resident_b) and DROP the per-call dealloc for the cached tensor (transient
        # A-slice / output tensors are still freed).
        #
        # DEFAULT OFF (deep-plan_7 §0.3 bound + STEP-14 measured OOM): in the e2e VLM/SigLIP
        # block, gate+up+down are THREE separate NoMSplit instances; if each pins its 557 KB
        # weight resident across the block, three resident weights + activations + the fp32
        # c_4 accumulator bust the ~1.5 MB L1 bank (observed: bank_manager.cpp:462 OOM, 50 KB
        # free). The reuse win is bounded to ONE projection's weight shard resident at a time
        # PER INSTANCE and is the per-call DRAM-bytes-saved ACROSS the denoise loop (calls
        # 1..10 of the SAME instance) -- it is MEASURED on an isolated single-projection loop
        # (STEP-14 harness), NOT by pinning every projection of the e2e block simultaneously.
        # So default OFF here (e2e PCC/fit byte-identical to iter-6); the STEP-14 reuse harness
        # passes resident_weight=True explicitly on a single isolated instance.
        self.resident_weight = (
            False if resident_weight is None else bool(resident_weight)
        )
        self._b_l1_cache = {}  # (id(weight_t), repr(mc)) -> L1-resident weight tensor

    def describe(self) -> dict:
        d = super().describe()
        d["m_rows"] = self.m_rows
        d["t_rows"] = self.t_rows if self.t_rows else self.m_rows
        d["t_knob_active"] = self.t_rows is not None
        d["plan_m_rows"] = self.plan_m_rows
        d["m_tiles"] = self.m_tiles
        d["G"] = self.k_split_G
        d["n_chunks_derived"] = self.n_chunks
        d["resident_weight"] = self.resident_weight
        return d

    # ------------------------------------------------------------------
    # deep-plan_7 §3 STEP-10: cross-call L1-resident weight cache
    # ------------------------------------------------------------------
    def _resident_b(self, weight_t, mc):
        """Return the weight L1-resident under memory-config ``mc``. On the first
        call (or after the cached tensor was freed at a boundary) a single
        DRAM->L1 ``to_memory_config`` stages it; calls 2..N reuse the SAME L1
        buffer (no DRAM read). The cache key is ``(id(weight_t), repr(mc))``:
        the staged weight tensors are held on the instance for its lifetime so
        ``id`` cannot be recycled while the cache exists, and ``repr(mc)`` +
        ``is_allocated()`` make the key sound across re-stage."""
        key = (id(weight_t), repr(mc))
        t = self._b_l1_cache.get(key)
        if t is None or not t.is_allocated():
            t = ttnn.to_memory_config(weight_t, mc)  # one-time DRAM->L1
            self._b_l1_cache[key] = t
        return t  # reused on subsequent calls (NO DRAM read)

    def _call_full(self, a_i, rows: int):
        """OVERRIDE of the parent (which was inherited verbatim and re-DMA'd +
        deallocated the weight every call). Identical EXCEPT the weight is routed
        through the L1-resident cache and NOT deallocated when resident_weight is
        ON. Transient A-slice / output tensors are still freed exactly as parent."""
        if not self.resident_weight:
            return super()._call_full(a_i, rows)
        G = self.k_split_G
        Kc = self.Kc_call
        N_chunk = self.N
        acc = None
        for g in range(G):
            if G > 1:
                a_g = ttnn.slice(a_i, [0, g * Kc], [rows, (g + 1) * Kc])  # [rows, Kc]
            else:
                a_g = a_i
            a_sh = ttnn.interleaved_to_sharded(a_g, self._a_shard_mc(rows, Kc))
            b_l1 = self._resident_b(self.b_groups[g], self._b_l1_mc_full(Kc, N_chunk))
            y = ttnn.matmul_decode(a_sh, b_l1, partial_width_sharded=False,
                                   dtype=self.out_dtype,
                                   compute_kernel_config=self.compute_kernel_config,
                                   **_matmul_decode_kwargs(self.stream_k,
                                                           self.k_slice_tiles))
            y_il = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            # NOTE: do NOT ttnn.deallocate(b_l1) -- the cache owns it (cross-call reuse).
            ttnn.deallocate(a_sh)
            if G > 1:
                ttnn.deallocate(a_g)
            if acc is None:
                acc = y_il
            else:
                nacc = ttnn.add(acc, y_il)
                ttnn.deallocate(acc)
                ttnn.deallocate(y_il)
                acc = nacc
        return acc  # [rows, N]

    def _call_partial(self, a_i, rows: int):
        """OVERRIDE of the parent: route the per-N-chunk weight through the
        L1-resident cache; skip the per-call weight dealloc when resident_weight
        is ON. Transient tensors freed as parent."""
        if not self.resident_weight:
            return super()._call_partial(a_i, rows)
        a_sh = ttnn.interleaved_to_sharded(a_i, self._a_shard_mc(rows, self.K))
        b_mc = self._b_l1_mc_partial()
        chunk_outs: List = []
        for c in range(self.n_chunks):
            b_l1 = self._resident_b(self.b_chunks[c], b_mc)
            y = ttnn.matmul_decode(a_sh, b_l1, partial_width_sharded=True,
                                   dtype=self.out_dtype,
                                   compute_kernel_config=self.compute_kernel_config)
            y_il = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            # NOTE: do NOT ttnn.deallocate(b_l1) -- the cache owns it.
            chunk_outs.append(y_il)
        ttnn.deallocate(a_sh)
        if len(chunk_outs) == 1:
            return chunk_outs[0]
        out = ttnn.concat(chunk_outs, dim=-1)
        for t in chunk_outs:
            ttnn.deallocate(t)
        return out

    def __call__(self, x_dev):
        """deep-plan_12 WS-A T-knob dispatch. When ``t_rows`` is set and the total M
        exceeds T, slice the activation into ceil(M/T) row-chunks of T rows, call the
        per-chunk matmul_decode (REUSING the SAME L1-resident weight -- the weight
        memcfg is M-independent so _resident_b returns the SAME cached L1 buffer), and
        concatenate outputs along the row (M) dimension. NO kernel edit: the op's
        factory adapts to M_tiles=ceil(T/32) per call. Pure-TTNN, no torch.* in the loop.

        Default (t_rows None) -> the monolithic M-whole path (byte-identical to before).
        """
        if self.t_rows is None:
            return self._call_monolithic(x_dev)

        orig = list(x_dev.shape)
        K_in = orig[-1]
        M = orig[0] if len(orig) == 2 else math.prod(orig[:-1])
        T = self.t_rows
        if M <= T:
            return self._call_monolithic(x_dev)
        # Flatten leading dims to [M, K_in] so we can slice rows.
        x2d = ttnn.reshape(x_dev, [M, K_in]) if len(orig) != 2 else x_dev
        n_chunks = math.ceil(M / T)
        chunk_outs = []
        for c in range(n_chunks):
            r0 = c * T
            r1 = min((c + 1) * T, M)
            x_c = ttnn.slice(x2d, [r0, 0], [r1, K_in])  # [<=T, K_in]
            y_c = self._call_monolithic(x_c)            # [<=T, N_orig]
            ttnn.deallocate(x_c)
            chunk_outs.append(y_c)
        if len(orig) != 2 and x2d is not x_dev:
            ttnn.deallocate(x2d)
        if len(chunk_outs) == 1:
            y = chunk_outs[0]
        else:
            y = ttnn.concat(chunk_outs, dim=0)  # concat along rows (M)
            for t in chunk_outs:
                ttnn.deallocate(t)
        # restore leading dims
        if len(orig) >= 3:
            y = ttnn.reshape(y, list(orig[:-1]) + [self.N_orig])
        return y

    def _call_monolithic(self, x_dev):
        """y = x_dev @ W (+ bias), M fed WHOLE. Pure-TTNN, preserves leading dims."""
        orig = list(x_dev.shape)
        N = self.N_orig
        K = self.K
        M = orig[0] if len(orig) == 2 else math.prod(orig[:-1])

        # --- K-pad branch ---
        x2d_owned = False
        if orig[-1] == K:
            x2d = ttnn.reshape(x_dev, [M, K])
        else:
            x_padded = ttnn.pad(x_dev, [(0, 0)] * (len(orig) - 1) + [(0, K - orig[-1])], 0.0)
            # ttnn.reshape that preserves the last dim + total element count returns a
            # NEW Python wrapper that ALIASES x_padded's device buffer (so `x2d is
            # x_padded` is False yet they share storage). Eagerly deallocating
            # x_padded on the `is` check therefore frees x2d's buffer too, which a
            # later device op (e.g. the G>=2 K-split ttnn.slice in _call_full) hits as
            # "Tensor is not allocated" -- the SigLIP fc2 K->4608 G=2 @ T=96 bug.
            # Clone the reshape into a standalone owned buffer, then free x_padded so
            # there is exactly one owning handle (x2d) with no aliasing double-free.
            x2d_view = ttnn.reshape(x_padded, [M, K])
            x2d = ttnn.clone(x2d_view)
            # x2d_view aliases x_padded's buffer; free that buffer exactly once via
            # the original owning handle (x_padded). x2d is now a standalone owned
            # buffer carrying the K-padded activation.
            ttnn.deallocate(x_padded)
            x2d_owned = True

        # --- tail-pad to a tile (for M=1 adaRMS-mod, or a non-tile remainder; for
        #     pi05 (S,T) chunks M%32==0 so this is the M=1 / defensive path) ---
        M_pad = ((M + M_TILE - 1) // M_TILE) * M_TILE
        a = x2d
        a_owned = x2d_owned
        if M_pad != M:
            a_pad = ttnn.pad(x2d, [(0, M_pad - M), (0, 0)], 0.0)
            # ttnn.pad on a tiled tensor whose logical rows already fit the
            # tile-padded physical buffer (e.g. M=1) is metadata-only and returns a
            # VIEW sharing x2d's buffer. When x2d is UNOWNED (it reshaped the
            # caller's input, no K-pad), that view aliases the caller's tensor
            # (reused across the 2 modulation calls / the chunk loop), so a later
            # deallocate would free the caller's input. Force an owned clone in that
            # case; only free x2d when WE own it and pad made a genuine new tensor.
            if not x2d_owned:
                a_pad = ttnn.clone(a_pad)
            elif a_pad is not x2d:
                ttnn.deallocate(x2d)
            a = a_pad
            a_owned = True

        # --- NO ceil(M/32) loop: feed M_pad WHOLE; routes via _call_full/_partial ---
        out_full = self._call_one_mtile(a, M_pad)
        if a_owned:
            ttnn.deallocate(a)
        if M_pad != M:
            y2d = ttnn.slice(out_full, [0, 0], [M, out_full.shape[-1]])
            ttnn.deallocate(out_full)
        else:
            y2d = out_full

        # --- N-pad-column drop + reshape + bias-add tail IDENTICAL to parent ---
        if y2d.shape[-1] != N:
            y_sl = ttnn.slice(y2d, [0, 0], [M, N])
            ttnn.deallocate(y2d)
            y2d = y_sl

        if len(orig) >= 3:
            y = ttnn.reshape(y2d, list(orig[:-1]) + [N])
        else:
            y = ttnn.reshape(y2d, [M, N])

        if self.bias_tt is not None:
            y2 = ttnn.add(y, self.bias_tt)
            ttnn.deallocate(y)
            return y2
        return y
