# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""In-source MatmulDecodeLinear + pure-Python plan_matmul_decode for the fork's
REAL width-sharded ``ttnn.matmul_decode`` device op.

This is the single matmul primitive for the pi05_blocked_mmdecode variant: every
projection / MLP matmul in the routed pipeline (SigLIP patch-embed / QKV / O /
fc1 / fc2; multimodal projector; Gemma VLM + expert QKV / O / gate / up / down;
adaRMS modulation; suffix action_in/out, time_mlp_in/out) goes through one of
these.  The only matmuls left native are the QK^T / scores*V inside flash SDPA.

Design (deep-plan_1 s2, s6, s7):
  * The device op is ``matmul_decode(a, b, *, partial_width_sharded=False,
    dtype=None)`` -- no bias / memory_config / core_grid args. Both A and B are
    TILE + WIDTH_SHARDED. A is M x K width(K)-sharded; output is ALWAYS
    width(N)-sharded across div_up(N, 32) cores. OUTPUT-core cap 104 on P150.
  * The partial-WS device guards are COMMENTED OUT on this commit, so this module
    enforces them HOST-SIDE in ``plan_matmul_decode`` (loud asserts).
  * FULL when div_up(N,32) <= usable_cores; else PARTIAL with the fewest equal
    tile-aligned OUTPUT N-chunks. Huge-K FULL shapes that OOM L1 are K-split
    (the VLM down-proj K=16384 PINS G=4, HW-verified).
  * ``__call__`` is PURE-TTNN (no torch round-trip), trace-capturable: M-split +
    N-chunk + K-split, on-device reassembly via ttnn.slice / concat / add. All B
    staging (DRAM-interleaved, partial host reshape, K-split slices) happens once
    in ``stage()`` (called from move_weights_to_device_impl), OUTSIDE any trace.

The device-free planner helpers (_chunk_fits / _plan_n_chunks / _partial_k_blocks
/ _plan_chunk / plan_matmul_decode / _device_usable_cores / _to_torch_KN) are
lifted from the recovered ``8c75526:tests/models/pi05/blaze/auto_matmul_decode.py``
with three modifications (s7.2): FULL-preferred routing, host-side constraint
enforcement, and a NEW L1-fit + K-split branch keyed on the B-operand dtype tile.
The recovered ``__call__`` round-tripped through host torch and is NOT lifted --
it is re-authored from scratch here as pure-TTNN.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import ttnn

# deep-plan_12 S0: live HEAD of the CLEAN fork checkout this module ran against.
# (Stale b56f57e5 refreshed.) The clean fork's matmul_decode op has NO stream_k /
# k_slice_tiles params (those lived only on the dirty e7023ed9 fork) -- detected below.
TT_METAL_COMMIT = "e4500c1fae97c103b16fc24fc7010b852992a9e6"

# deep-plan_12 G0a finding: the clean-fork ttnn.matmul_decode signature is
#   (a, b, *, partial_width_sharded=False, dtype=None, compute_kernel_config=None)
# -- it does NOT accept stream_k / k_slice_tiles. Detect support once so the
# call sites stay byte-identical on a fork that DOES carry them.
try:
    _MATMUL_DECODE_HAS_STREAM_K = "stream_k" in (ttnn.matmul_decode.__doc__ or "")
except Exception:
    _MATMUL_DECODE_HAS_STREAM_K = False

# deep-plan_14 Lever 0/2 split: the docstring now ADVERTISES stream_k (Lever 0 plumbing),
# so _HAS_STREAM_K flips True and the op ACCEPTS the kwargs. But the TEMPORAL KERNEL BODY
# (Lever 2, checkpoint-gated) may NOT be built yet -- the factory currently ignores k_stream
# and runs the one-shot full-K gather, which OOMs on large-K (vlm_down K=16384). So the
# ROUTING decision to take the stream_k path is gated on a SEPARATE functional flag, default
# OFF, flipped to "1" ONLY after the Lever-2 temporal kernel is built + PCC-validated. Until
# then large-K stays on the documented host-G fallback (the deep-plan_13 PCC-green route).
import os as _os

_MATMUL_DECODE_STREAM_K_FUNCTIONAL = (
    _MATMUL_DECODE_HAS_STREAM_K and _os.environ.get("MMD_STREAM_K_FUNCTIONAL", "0") == "1"
)


# deep-plan_15 SCOPE-A gather_in0 fat-fill role map (per_core_N<=2, GATE-LEGAL fat).
# m_rows = per-stage full-seq S; fp32/oh/ow = the PCC-validated legal fat subblock
# (SigLIP o S=256: fp32 oh=4 fat PCC 0.999986; VLM o S=288: fp32 oh=3 fat PCC 0.999954;
#  VLM qkv S=288: bf16 oh=3 ow=2 fat PCC 0.999674 -- fp32 cap 4 would force oh=1 thin).
_WS2D_GATHER_ROLES = {
    "siglip_o": {"m_rows": 256, "num_cores": 64, "fp32": True, "oh": 4, "ow": 1},
    "vlm_o": {"m_rows": 288, "num_cores": 64, "fp32": True, "oh": 3, "ow": 1},
    "vlm_qkv": {"m_rows": 288, "num_cores": 64, "fp32": False, "oh": 3, "ow": 2},
}


def _matmul_decode_kwargs(stream_k, k_slice_tiles):
    """Build the stream_k/k_slice_tiles kwargs ONLY when the op supports them."""
    if _MATMUL_DECODE_HAS_STREAM_K:
        return {"stream_k": bool(stream_k), "k_slice_tiles": int(k_slice_tiles)}
    return {}


# matmul_decode output is one M-tile (32-row decode regime) per call.
M_TILE = 32
# Conservative usable OUTPUT-core cap. 128-core OUTPUT configs FATAL on P150.
DEFAULT_CORE_CAP = 104
# Allocator-reported usable per-core L1 bank on P150 (probe).
P150_L1_BANK_BYTES = 1_461_760
# Conservative FULL-WS L1 budget knob (s6.3); resident-L1 disabled (CB clash).
DEFAULT_L1_FULL_WS_BUDGET_B = 1_310_720
DEFAULT_L1_RESIDENT_FRACTION = 0.0

_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.bfloat8_b: 1088,
    ttnn.bfloat4_b: 576,
}
_A_TILE_BYTES = 2048  # A is always bf16 (residual / hidden stream).


def _tile_bytes(dtype) -> int:
    return _TILE_BYTES.get(dtype, 2048)


def _device_usable_cores(device, cap: int = DEFAULT_CORE_CAP) -> int:
    grid = device.compute_with_storage_grid_size()
    total = int(grid.x) * int(grid.y)
    return min(total, cap)


def _wide_n_out_cores(N: int, cap: int = DEFAULT_CORE_CAP) -> int:
    """Output-core count for a FULL-WS (wide-N) call, IDENTICAL to the fork device op's
    compute_output_specs cap rule (plan_5 s1.1): if N_tiles <= cap -> N_tiles cores (1
    N-tile/core, byte-identical to the pre-cap spec); else the LARGEST DIVISOR of N_tiles
    that is <= cap (even N-tiling, no pad). The wrapper MUST reproduce this so the B shard's
    core grid equals the output grid (the full-WS factory's inputB==output FATAL guard)."""
    n_tiles = (N + 31) // 32
    if n_tiles <= cap:
        return n_tiles
    chosen = 1
    for c in range(1, min(cap, n_tiles) + 1):
        if n_tiles % c == 0:
            chosen = c
    return chosen


# ----------------------------------------------------------------------------
# Pure-Python, device-free planner (lifted + 3 modifications, deep-plan_1 s7.2)
# ----------------------------------------------------------------------------
def _partial_k_blocks(K: int, cn: int, usable_cores: int):
    """Minimal EVEN k_blocks with (K // kb) % 32 == 0 and
    kb * (cn // 32) <= usable_cores. Returns None if none fits.

    HARDWARE-MEASURED (deep-plan_1 OQ3 fallback): contrary to s2.10's optimistic
    reading, the input-B-core count (kb * n_blocks) DOES hit the 110-core grid
    cap on this fork build -- a 144-input config FATALs at
    work_split.cpp:98 ('target_num_cores <= total_available_cores'), and 128
    (the 8x2048 VLM gate/up config) also exceeds 110. So we re-enforce the
    input-B-core bound against ``usable_cores`` (104). This yields the
    deep-plan_1 s5 PRIMARY factorizations: VLM gate/up 16x1024 (b_cores=64),
    SigLIP QKV 3x1536 (b_cores=96), expert gate/up 2x2048 (b_cores=128 ->
    rejected -> 4x1024 b_cores=64). The 8x2048/2x2048-with-128-cores split from
    s6.1 is the over-optimistic path; 16x1024/4x1024 is the validated fallback."""
    n_blocks = cn // 32
    for kb in range(2, K // 32 + 1, 2):
        if K % kb or (K // kb) % 32:
            continue
        if kb * n_blocks <= usable_cores:
            return kb
    return None


def _chunk_fits(K: int, cn: int, usable_cores: int, want_partial: bool) -> bool:
    # OUTPUT-core cap (cn//32 <= usable_cores) AND input-B-core cap
    # (kb * n_blocks <= usable_cores, enforced inside _partial_k_blocks).
    if cn // 32 > usable_cores:
        return False
    if want_partial:
        return _partial_k_blocks(K, cn, usable_cores) is not None
    return True


def _plan_n_chunks(K: int, N: int, usable_cores: int, want_partial: bool) -> int:
    for chunks in range(1, 1025):
        if N % chunks != 0:
            continue
        cn = N // chunks
        if cn % 32 != 0:
            continue
        if _chunk_fits(K, cn, usable_cores, want_partial):
            return chunks
    raise RuntimeError(f"cannot N-chunk K={K} N={N} under {usable_cores} cores (want_partial={want_partial})")


def _plan_chunk(K: int, cn: int, usable_cores: int, want_partial: bool) -> dict:
    cn_tiles = cn // 32
    if want_partial:
        kb = _partial_k_blocks(K, cn, usable_cores)
        if kb is not None:
            return {
                "mode": "partial",
                "k_blocks": kb,
                "n_blocks": cn_tiles,
                "kc": K // kb,
                "nc": 32,
                "b_cores": kb * cn_tiles,
            }
    return {"mode": "full", "k_blocks": 1, "n_blocks": cn_tiles, "kc": K, "nc": 32, "b_cores": cn_tiles}


def _full_ws_l1_bytes(
    K: int,
    in1_dtype,
    m_tiles: int = 1,
    *,
    a_cores: int = 2,
    npc: int = 1,
    stream_k: bool = False,
    k_slice_tiles: int = 16,
) -> int:
    """Per-core FULL-WS L1 floor for one call, mirroring full_width_sharded_program_factory.cpp
    (plan_5 s5.1 fixes). CBs:
      c_0 in0 (resident A slice, K-sharded across a_cores): m_tiles*(k_tiles/a_cores)*A_tile
      c_1 in1 (resident B, npc N-tiles/core):               k_tiles*npc*B_tile
      c_2 out:                                              m_tiles*npc*2048
      c_3 full_in0: one-shot gather m_tiles*k_tiles*A_tile, OR streamed 2*m_tiles*k_slice*A_tile
    The OLD helper understated in0 ('conservative 1') and omitted the npc multiplier on in1/out,
    which under-counts wide-N (npc>1) and large-K, green-lighting an OOM."""
    k_tiles = K // 32
    ac = max(1, a_cores)
    in0 = m_tiles * max(1, k_tiles // ac) * _A_TILE_BYTES
    in1 = k_tiles * npc * _tile_bytes(in1_dtype)
    out = m_tiles * npc * 2048
    if stream_k:
        full_in0 = 2 * m_tiles * k_slice_tiles * _A_TILE_BYTES
    else:
        full_in0 = m_tiles * k_tiles * _A_TILE_BYTES
    # deep-plan_7 STEP-1: the iter-7 streamed branch holds the full M x N partial state in a
    # NEW per-core fp32 L1 accumulator CB (c_4) across the K-OUTER-once loop (PACKER_L1_ACC).
    # fp32 tile = 4096 B; size = m_tiles * npc tiles. Non-streamed branch has no accumulator.
    acc = (m_tiles * npc * 4096) if stream_k else 0  # fp32 c_4 accumulator
    return full_in0 + in1 + in0 + out + acc


def plan_matmul_decode(
    K: int,
    N: int,
    usable_cores: int,
    *,
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
) -> dict:
    """Auto-select the full matmul_decode execution plan for a [K, N] weight.

    plan_5 collapse (M<=96): the per-projection N-chunk + K-split loops move INTO the fork
    device op. For M_tiles<=3 (M<=96) projections:
      * wide-N (gate/up, N=16384): ONE FULL-WS call. out_cores capped via _wide_n_out_cores
        (largest divisor of N_tiles <= cap; identical rule to the device op so the B shard grid
        == output grid). n_chunks=1, k_split_G=1.
      * large-K (down, K=16384): ONE FULL-WS call with stream_k=True (temporal K-streaming +
        forced fp32 inside the factory), a_cores>=8. No K-split, no N-chunk.
      * else (O, small-N): plain FULL-WS single call (K-split only if L1 truly overflows).
    QKV@M=S (M_tiles>=8) is EXCLUDED -- it keeps the legacy FULL/partial K-split routing.
    """
    eff_N = pad_n_to if pad_n_to else N
    eff_K = pad_k_to if pad_k_to else K
    assert eff_K % 32 == 0 and eff_N % 32 == 0, (eff_K, eff_N)

    n_tiles_total = (eff_N + 31) // 32
    out_cores_uncapped = n_tiles_total
    is_small_m = m_tiles <= 3  # M <= 96 collapse regime

    # ---- COLLAPSED M<=96 wide-N (gate/up): single FULL-WS call, capped out_cores ----
    if is_small_m and not force_partial and out_cores_uncapped > usable_cores:
        out_cores = _wide_n_out_cores(eff_N, core_cap)
        npc = n_tiles_total // out_cores
        assert n_tiles_total % out_cores == 0, (n_tiles_total, out_cores)
        floor = _full_ws_l1_bytes(eff_K, weight_dtype, m_tiles, a_cores=a_cores, npc=npc)
        assert (
            floor <= P150_L1_BANK_BYTES
        ), f"wide-N FULL K={eff_K} N={eff_N} npc={npc} m_tiles={m_tiles} floor {floor} > bank"
        return {
            "mode": "full",
            "n_chunks": 1,
            "chunk_N": eff_N,
            "k_blocks": 1,
            "n_blocks": out_cores,
            "kc": eff_K,
            "nc": npc * 32,
            "b_cores": out_cores,
            "k_split_G": 1,
            "Kc_call": eff_K,
            "in1_dtype": weight_dtype,
            "pad_N_to": pad_n_to,
            "pad_K_to": pad_k_to,
            "partial": False,
            "wide_n": True,
            "npc": npc,
            "stream_k": False,
            "a_cores": a_cores,
            "m_tiles": m_tiles,
        }

    # ---- COLLAPSED M<=96 large-K (down): single FULL-WS K-B streamed call ----
    # deep-plan_13 sec 4.5 fallback ordering: WIDTH-temporal k_stream is the PRIMARY route for
    # large-K (vlm_down K=16384). When the fork op CARRIES stream_k (_MATMUL_DECODE_HAS_STREAM_K)
    # we emit a single streamed call. When it does NOT (Phase 2 temporal not yet built on this
    # fork checkout), we fall back to the documented host G-split last resort: G separate WIDTH
    # one-shot calls over K/G slices + host ttnn.add over the G outputs (the surviving
    # _call_full k_split_G path). Both are functional; neither is a scheme switch.
    if (
        is_small_m
        and not force_partial
        and out_cores_uncapped <= usable_cores
        and _full_ws_l1_bytes(eff_K, weight_dtype, m_tiles, a_cores=a_cores) > l1_full_ws_budget_B
    ):
        if _MATMUL_DECODE_STREAM_K_FUNCTIONAL:
            sac = stream_a_cores
            while eff_K % (sac * 32) != 0 and sac > 1:
                sac -= 1
            floor = _full_ws_l1_bytes(
                eff_K, weight_dtype, m_tiles, a_cores=sac, stream_k=True, k_slice_tiles=stream_k_slice_tiles
            )
            assert floor <= P150_L1_BANK_BYTES, (
                f"K-B stream down K={eff_K} N={eff_N} a_cores={sac} m_tiles={m_tiles} " f"floor {floor} > bank"
            )
            return {
                "mode": "full",
                "n_chunks": 1,
                "chunk_N": eff_N,
                "k_blocks": 1,
                "n_blocks": out_cores_uncapped,
                "kc": eff_K,
                "nc": 32,
                "b_cores": out_cores_uncapped,
                "k_split_G": 1,
                "Kc_call": eff_K,
                "in1_dtype": weight_dtype,
                "pad_N_to": pad_n_to,
                "pad_K_to": pad_k_to,
                "partial": False,
                "wide_n": False,
                "npc": 1,
                "stream_k": True,
                "a_cores": sac,
                "k_slice_tiles": stream_k_slice_tiles,
                "m_tiles": m_tiles,
            }
        # Host G-split fallback: smallest G (power-of-2, K/G tile-aligned) that fits L1.
        g = pin_k_split if pin_k_split is not None else 1
        while _full_ws_l1_bytes(eff_K // g, weight_dtype, m_tiles, a_cores=a_cores) > l1_full_ws_budget_B:
            g *= 2
            if (eff_K // g) % 32 != 0 or g > eff_K // 32:
                raise RuntimeError(f"host-G fallback K={eff_K} cannot fit L1 even at G={g}")
        Kc_call = eff_K // g
        return {
            "mode": "full",
            "n_chunks": 1,
            "chunk_N": eff_N,
            "k_blocks": 1,
            "n_blocks": out_cores_uncapped,
            "kc": Kc_call,
            "nc": 32,
            "b_cores": out_cores_uncapped,
            "k_split_G": g,
            "Kc_call": Kc_call,
            "in1_dtype": weight_dtype,
            "pad_N_to": pad_n_to,
            "pad_K_to": pad_k_to,
            "partial": False,
            "wide_n": False,
            "npc": 1,
            "stream_k": False,
            "a_cores": a_cores,
            "m_tiles": m_tiles,
        }

    out_cores = out_cores_uncapped
    fits_full = out_cores <= usable_cores
    want_partial = force_partial or not fits_full

    # ---- FULL-WS path (single N call), maybe K-split for huge K (legacy / QKV@M=S) ----
    if not want_partial:
        k_split_G = 1
        if pin_k_split is not None:
            k_split_G = pin_k_split
        else:
            # Budget-derived smallest power-of-2 G (K/G tile-aligned) that fits.
            g = 1
            while _full_ws_l1_bytes(eff_K // g, weight_dtype, m_tiles, a_cores=a_cores) > l1_full_ws_budget_B:
                g *= 2
                if (eff_K // g) % 32 != 0 or g > eff_K // 32:
                    raise RuntimeError(f"FULL-WS K={eff_K} N={eff_N} cannot fit L1 even at G={g}")
            k_split_G = g
        Kc_call = eff_K // k_split_G
        # Host-side FULL-WS asserts: each group is a full call over Kc_call rows.
        assert Kc_call % 32 == 0, f"FULL-WS Kc_call {Kc_call} not tile-aligned"
        assert out_cores <= usable_cores, (out_cores, usable_cores)
        return {
            "mode": "full",
            "n_chunks": 1,
            "chunk_N": eff_N,
            "k_blocks": 1,
            "n_blocks": out_cores,
            "kc": Kc_call,
            "nc": 32,
            "b_cores": out_cores,
            "k_split_G": k_split_G,
            "Kc_call": Kc_call,
            "in1_dtype": weight_dtype,
            "pad_N_to": pad_n_to,
            "pad_K_to": pad_k_to,
            "partial": False,
            "wide_n": False,
            "npc": 1,
            "stream_k": False,
            "a_cores": a_cores,
            "m_tiles": m_tiles,
        }

    # ---- PARTIAL-WS path with OUTPUT N-chunking ----
    n_chunks = _plan_n_chunks(eff_K, eff_N, usable_cores, True)
    chunk_N = eff_N // n_chunks
    chunk = _plan_chunk(eff_K, chunk_N, usable_cores, True)
    # Host-side PARTIAL asserts (device guards commented out -- s2.7):
    kb, nb = chunk["k_blocks"], chunk["n_blocks"]
    kc, nc = chunk["kc"], chunk["nc"]
    assert chunk["mode"] == "partial", (eff_K, eff_N, chunk)
    assert kb % 2 == 0, f"partial k_blocks {kb} not even"
    assert kc % 32 == 0, f"partial Kc {kc} not tile-aligned"
    assert nc == 32, f"partial Nc {nc} != 32"
    assert kb * nb == chunk["b_cores"], (kb, nb, chunk["b_cores"])
    assert nb == chunk_N // 32, (nb, chunk_N)
    assert (chunk_N // 32) <= usable_cores, "partial chunk OUTPUT cores > cap"
    return {
        "mode": "partial",
        "n_chunks": n_chunks,
        "chunk_N": chunk_N,
        "k_blocks": kb,
        "n_blocks": nb,
        "kc": kc,
        "nc": nc,
        "b_cores": chunk["b_cores"],
        "k_split_G": 1,
        "Kc_call": eff_K,
        "in1_dtype": weight_dtype,
        "pad_N_to": pad_n_to,
        "pad_K_to": pad_k_to,
        "partial": True,
        "wide_n": False,
        "npc": 1,
        "stream_k": False,
        "a_cores": a_cores,
        "m_tiles": m_tiles,
    }


def _to_torch_KN(w_tt) -> torch.Tensor:
    """Pull a resident [K, N] ttnn weight to a torch [K, N] bf16 tensor."""
    try:
        dev = ttnn.get_device_tensors(w_tt)[0]
    except Exception:
        dev = w_tt
    return ttnn.to_torch(dev).to(torch.bfloat16).contiguous()


# ----------------------------------------------------------------------------
# MatmulDecodeLinear -- pure-TTNN device call + weight-time-only B staging
# ----------------------------------------------------------------------------
class MatmulDecodeLinear:
    """A single [K, N] linear (y = x @ W + bias) executed via ttnn.matmul_decode.

    Build at move_weights_to_device_impl time from a torch or resident-ttnn [K, N]
    weight. ``stage()`` materializes all B operands ONCE (DRAM-interleaved). The
    ``__call__`` is pure-TTNN (M-split + N-chunk + K-split, on-device reassembly),
    trace-capturable.
    """

    def __init__(
        self,
        device,
        weight,
        *,
        bias=None,
        out_dtype=ttnn.bfloat16,
        weight_dtype=ttnn.bfloat8_b,
        role: str = "generic",
        a_cores: int = 2,
        core_cap: int = DEFAULT_CORE_CAP,
        force_partial: bool = False,
        pin_k_split: Optional[int] = None,
        pad_n: Optional[int] = None,
        pad_k: Optional[int] = None,
        fp32_dest_acc: bool = False,
    ):
        # deep-plan_15 SCOPE-A: env-gated gather_in0 fat-fill delegation. Read the gate
        # INSIDE __init__ so the harness _build stays byte-untouched (a one-env-var delta).
        # WS2D_GATHER=1 routes the per_core_N<=2 prefill projections (siglip_o, vlm_o,
        # vlm_qkv) through GatherIn0FatLinear (Matmul1D -> MatmulDeviceOperation, resident
        # WIDTH_SHARDED-L1 in1, GATE-LEGAL fat subblock). Default OFF -> byte-identical.
        self._gather_delegate = None
        if _os.environ.get("WS2D_GATHER", "0") == "1":
            cfg = _WS2D_GATHER_ROLES.get(role)
            if cfg is not None:
                from models.experimental.pi0_5.tt.mmdecode.gather_in0_fat_linear import (
                    GatherIn0FatLinear,
                )

                w_KN = weight if isinstance(weight, torch.Tensor) else _to_torch_KN(weight)
                m_rows = cfg["m_rows"]
                self._gather_delegate = GatherIn0FatLinear(
                    device,
                    w_KN.to(torch.bfloat16).contiguous(),
                    m_rows=m_rows,
                    num_cores=cfg["num_cores"],
                    weight_dtype=weight_dtype,
                    out_dtype=out_dtype,
                    pad_n=pad_n,
                    pad_k=pad_k,
                    role=role,
                    resident_weight=True,
                    fp32_dest_acc=cfg["fp32"],
                    oh_override=cfg.get("oh"),
                    ow_override=cfg.get("ow"),
                )
                self.role = role
                self.out_dtype = out_dtype
                self.weight_dtype = weight_dtype
                self.K_orig = self._gather_delegate.K_orig
                self.N_orig = self._gather_delegate.N_orig
                self.K = self._gather_delegate.K
                self.N = self._gather_delegate.N
                self.npc = self._gather_delegate.per_core_N
                self.stream_k = False
                return
        self.device = device
        self.role = role
        self.out_dtype = out_dtype
        self.weight_dtype = weight_dtype
        self.bias_tt = bias
        # fp32 DST accumulation is now an OPT-IN compute_kernel_config on the fork's
        # ttnn.matmul_decode (mirroring ttnn.matmul), default OFF. When enabled we build a
        # DeviceComputeKernelConfig with fp32_dest_acc_en=True and HiFi4 (the op's fidelity
        # floor) and pass it to every matmul_decode call; when off we pass None so the op
        # resolves to its default (fp32_dest_acc_en=False, HiFi4). This recovers the
        # ~0.007 bf16 reduction-order drift on the deep path at a device-time cost.
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

        # Pad N (fc1 4304->4320) / K (fc2 4304->4320) with zeros, host-side.
        pad_n_to = pad_n
        pad_k_to = pad_k
        if pad_n and pad_n > self.N_orig:
            w_KN = torch.nn.functional.pad(w_KN, (0, pad_n - self.N_orig))
        if pad_k and pad_k > self.K_orig:
            w_KN = torch.nn.functional.pad(w_KN, (0, 0, 0, pad_k - self.K_orig))
        self.K, self.N = int(w_KN.shape[0]), int(w_KN.shape[1])
        self._w_KN = w_KN
        assert self.K % 32 == 0 and self.N % 32 == 0, (self.K, self.N)

        self.usable_cores = _device_usable_cores(device, core_cap)
        self.core_cap = core_cap
        # The blocked wrapper M-splits into 32-row tiles in __call__, so EVERY device call is
        # M_tiles=1 -> the planner is keyed on m_tiles=1 (the M<=96 collapse regime applies).
        self.plan = plan_matmul_decode(
            self.K,
            self.N,
            self.usable_cores,
            force_partial=force_partial,
            weight_dtype=weight_dtype,
            pin_k_split=pin_k_split,
            pad_n_to=pad_n_to,
            pad_k_to=pad_k_to,
            m_tiles=1,
            a_cores=a_cores,
            core_cap=core_cap,
        )
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

        # A width-shard: K (or per-group Kc_call) divisible by a_cores * 32. For the K-B
        # streamed down path a_cores is forced to >=8 by the planner (self.plan["a_cores"]).
        self.a_cores = self.plan.get("a_cores", a_cores)
        self._grid = device.compute_with_storage_grid_size()

        # deep-plan_14 Lever 1: whole-M (no-M-split) mode. When MMD_WHOLE_M=1 and the role is a
        # prefill projection (M=256/288) AND the full-M one-shot fits the L1 budget at the actual
        # m_tiles, __call__ feeds M WHOLE so the device op runs ONE fat call (out_h>1 M-fill auto-
        # derived) instead of ceil(M/32) thin 32-row calls. Collapses the M-split dispatch tax.
        # Default OFF (the M0 inventory + all existing callers stay byte-identical). Only WIDTH
        # full mode (not partial, not host-G stream) qualifies for the simple whole-M path.
        self._whole_m = _os.environ.get("MMD_WHOLE_M", "0") == "1"

        self.b_chunks: List = []  # PARTIAL: per-N-chunk staged B
        self.b_groups: List = []  # FULL K-split: per-group staged B (or 1 elem for plain FULL)
        self._staged = False
        self.stage()

    # -- weight-time-only B staging (OUTSIDE any trace) --
    def stage(self):
        if self._staged:
            return
        w = self._w_KN
        if self.partial:
            kb, nb, kc, nc = (self.plan["k_blocks"], self.plan["n_blocks"], self.plan["kc"], self.plan["nc"])
            for c in range(self.n_chunks):
                sub = w[:, c * self.chunk_N : (c + 1) * self.chunk_N].contiguous()
                sub = sub.reshape(kb, kc, self.chunk_N).permute(1, 0, 2).reshape(kc, self.chunk_N * kb).contiguous()
                self.b_chunks.append(
                    ttnn.from_torch(
                        sub,
                        dtype=self.weight_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
        else:
            G = self.k_split_G
            Kc = self.Kc_call
            for g in range(G):
                sub = w[g * Kc : (g + 1) * Kc, :].contiguous()  # [Kc, N]
                self.b_groups.append(
                    ttnn.from_torch(
                        sub,
                        dtype=self.weight_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
        self._w_KN = None  # release host weight
        self._staged = True

    def describe(self) -> dict:
        if self._gather_delegate is not None:
            gd = self._gather_delegate.describe()
            return {
                "role": self.role,
                "K": self.K,
                "N": self.N,
                "mode": "gather_in0",
                "n_chunks": 1,
                "k_split_G": 1,
                "a_cores": gd["num_cores"],
                "K_orig": self.K_orig,
                "N_orig": self.N_orig,
                "gather": gd,
            }
        return {
            "role": self.role,
            "K": self.K,
            "N": self.N,
            "mode": self.mode,
            "n_chunks": self.n_chunks,
            "chunk_N": self.chunk_N,
            "k_blocks": self.plan["k_blocks"],
            "n_blocks": self.plan["n_blocks"],
            "kc": self.plan["kc"],
            "nc": self.plan["nc"],
            "b_cores": self.plan["b_cores"],
            "a_cores": self.a_cores,
            "k_split_G": self.k_split_G,
            "Kc_call": self.Kc_call,
            "in1_dtype": str(self.weight_dtype),
            "weight_dtype_actual": str(self.weight_dtype),
            "fp32_dest_acc": self.fp32_dest_acc,
            "K_orig": self.K_orig,
            "N_orig": self.N_orig,
        }

    # -- per-call helpers (device ops only, trace-safe) --
    def _a_shard_mc(self, m_rows: int, k: int):
        ac = self.a_cores
        while k % (ac * 32) != 0 and ac > 1:
            ac -= 1
        a_crs = ttnn.num_cores_to_corerangeset(ac, self._grid, True)
        return ttnn.create_sharded_memory_config(
            (m_rows, k // ac),
            core_grid=a_crs,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _b_l1_mc_partial(self):
        kc, nc, b_cores = self.plan["kc"], self.plan["nc"], self.plan["b_cores"]
        b_crs = ttnn.num_cores_to_corerangeset(b_cores, self._grid, True)
        return ttnn.create_sharded_memory_config(
            (kc, nc),
            core_grid=b_crs,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _b_l1_mc_full(self, K_call: int, N_chunk: int):
        # plan_5 s1.4: out_cores via the SAME divisor-cap rule as the device op so the B shard
        # grid == output grid (the full-WS factory inputB==output FATAL guard). For N<=cap*32
        # this is N//32 cores @ 1 N-tile/core (byte-identical to the pre-cap spec); for wide-N
        # it caps to the largest divisor <= cap with N_tiles_per_core>1.
        out_cores = _wide_n_out_cores(N_chunk, self.core_cap)
        npc = (N_chunk // 32) // out_cores
        b_crs = ttnn.num_cores_to_corerangeset(out_cores, self._grid, True)
        return ttnn.create_sharded_memory_config(
            (K_call, npc * 32),
            core_grid=b_crs,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    def _matmul_one(self, a_dev, m_rows: int, k_call: int, b_l1, partial: bool):
        """One matmul_decode -> interleaved L1 [m_rows, chunk_N] (sharded->interleaved)."""
        y = ttnn.matmul_decode(
            a_dev,
            b_l1,
            partial_width_sharded=partial,
            dtype=self.out_dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        return y  # WIDTH(N)-sharded ROW_MAJOR

    def __call__(self, x_dev):
        """y = x_dev @ W (+ bias). Pure-TTNN, preserves leading dims."""
        if self._gather_delegate is not None:
            # deep-plan_15 SCOPE-A: route the whole forward through gather_in0 fat-fill.
            # Feed [M, K]; return [1, M, N_orig] L1 interleaved (drop pad cols).
            orig = list(x_dev.shape)
            M = 1
            for d in orig[:-1]:
                M *= d
            x2d = ttnn.reshape(x_dev, [M, x_dev.shape[-1]])
            y_sh = self._gather_delegate(x2d)
            y_il = ttnn.sharded_to_interleaved(y_sh, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y_sh)
            N = self.N_orig
            if y_il.shape[-1] != N:
                y_sl = ttnn.slice(y_il, [0, 0], [M, N])
                ttnn.deallocate(y_il)
                y_il = y_sl
            if len(orig) >= 3:
                return ttnn.reshape(y_il, list(orig[:-1]) + [N])
            return ttnn.reshape(y_il, [M, N])
        orig = list(x_dev.shape)
        N = self.N_orig  # output width returned to caller (drop any pad columns)
        K = self.K
        if len(orig) == 2:
            M = orig[0]
        else:
            M = 1
            for d in orig[:-1]:
                M *= d
        # x2d is a reshape (view) of the caller's tensor when no K-pad is needed;
        # in that case we must NOT deallocate it (the caller owns x_dev). Only the
        # K-pad branch allocates a fresh tensor we own + free.
        x2d_owned = False
        if orig[-1] == K:
            x2d = ttnn.reshape(x_dev, [M, K])
        else:
            x_padded = ttnn.pad(x_dev, [(0, 0)] * (len(orig) - 1) + [(0, K - orig[-1])], 0.0)
            x2d = ttnn.reshape(x_padded, [M, K])
            if x2d is not x_padded:
                ttnn.deallocate(x_padded)
            x2d_owned = True

        n_mtiles = math.ceil(M / M_TILE)

        # deep-plan_14 Lever 1: WHOLE-M one-shot fat-M-fill path. Feed all M_tiles in ONE device
        # call (out_h>1 auto-derived) instead of looping n_mtiles thin 32-row calls. Gated on
        # MMD_WHOLE_M=1 + WIDTH-full (not partial, single K-group) + the full-M one-shot fits L1.
        if self._whole_m and not self.partial and self.k_split_G == 1 and M > M_TILE:
            m_tiles_whole = math.ceil(M / M_TILE)
            M_pad = m_tiles_whole * M_TILE
            # Pick the smallest a_cores (divisor of K_tiles) whose whole-M one-shot fits the L1
            # budget. Larger a_cores shrinks each core's gathered-A slice (full_in0 dominates at
            # large m_tiles). Bounded by 32 (the gather sender cap); start from the plan a_cores.
            k_tiles = self.K // 32
            whole_ac = self.a_cores
            floor = _full_ws_l1_bytes(self.K, self.weight_dtype, m_tiles_whole, a_cores=whole_ac, npc=self.npc)
            while floor > DEFAULT_L1_FULL_WS_BUDGET_B and whole_ac < 32:
                nac = whole_ac * 2
                while k_tiles % nac != 0 and nac < 32:
                    nac += 1
                if k_tiles % nac != 0:
                    break
                whole_ac = nac
                floor = _full_ws_l1_bytes(self.K, self.weight_dtype, m_tiles_whole, a_cores=whole_ac, npc=self.npc)
            self._whole_m_a_cores = whole_ac
            if floor <= DEFAULT_L1_FULL_WS_BUDGET_B:
                if M_pad != M:
                    a_w = ttnn.pad(x2d, [(0, M_pad - M), (0, 0)], 0.0)
                    a_w_owned = a_w is not x2d
                else:
                    a_w = x2d
                    a_w_owned = False
                self.a_cores = whole_ac  # whole-M A-shard uses the fitting a_cores
                out_w = self._call_one_mtile(a_w, M_pad)  # ONE fat call, M_tiles>1
                if a_w_owned:
                    ttnn.deallocate(a_w)
                if M_pad != M:
                    y2d = ttnn.slice(out_w, [0, 0], [M, out_w.shape[-1]])
                    ttnn.deallocate(out_w)
                else:
                    y2d = out_w
                if x2d_owned:
                    ttnn.deallocate(x2d)
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

        m_outs: List = []
        for mt in range(n_mtiles):
            r0 = mt * M_TILE
            rows = min((mt + 1) * M_TILE, M) - r0
            # ttnn.slice may return a VIEW aliasing x2d (and thus the caller's
            # input) when the slice spans a full tile dim, so track ownership and
            # never deallocate a tensor that aliases the input.
            if rows == M and r0 == 0:
                # The single M-tile spans all of x2d (e.g. M<=32: the AdaRMS
                # modulation cond [b,width] with M=1, or any M<=32 call). Reuse
                # x2d directly instead of a full-extent ttnn.slice, which would
                # return a VIEW aliasing x2d (and thus the caller's x_dev) -- a
                # later deallocate of that view frees the caller's input, which is
                # reused across the 18 expert blocks ("Tensor is not allocated").
                a_i = x2d
                a_i_owned = False
            else:
                a_i = ttnn.slice(x2d, [r0, 0], [r0 + rows, K])  # [rows, K]
                a_i_owned = True
            # The device op requires a single M-tile (M=32). Pad short tail tiles
            # to 32 rows (the phantom rows produce phantom output rows we slice
            # off below) -- the A width-shard height must be tile-aligned.
            if rows < M_TILE:
                a_pad = ttnn.pad(a_i, [(0, M_TILE - rows), (0, 0)], 0.0)
                # ttnn.pad on a tiled tensor whose logical rows (e.g. M=1) already
                # fit inside the tile-padded physical buffer is metadata-only and
                # returns a VIEW sharing a_i's buffer. When a_i is unowned (it
                # aliases x2d/x_dev), that view also aliases the caller's input, so
                # deallocating it later would free the caller's tensor (reused
                # across the 18 expert blocks). Force a real owned copy in that
                # case so the per-tile deallocate cannot touch x_dev.
                if not a_i_owned:
                    a_pad = ttnn.clone(a_pad)
                if a_i_owned:
                    ttnn.deallocate(a_i)
                a_i = a_pad
                a_i_owned = True
            out_mt = self._call_one_mtile(a_i, M_TILE)  # full 32-row tile
            if a_i_owned:
                ttnn.deallocate(a_i)
            if rows < M_TILE:
                out_real = ttnn.slice(out_mt, [0, 0], [rows, out_mt.shape[-1]])
                ttnn.deallocate(out_mt)
                out_mt = out_real
            m_outs.append(out_mt)
        if mt == 0 and n_mtiles == 1:
            y2d = m_outs[0]
        else:
            y2d = ttnn.concat(m_outs, dim=-2)
            for t in m_outs:
                ttnn.deallocate(t)
        if x2d_owned:
            ttnn.deallocate(x2d)

        # Drop pad columns (fc1 N pad) if any, keep N_orig.
        if y2d.shape[-1] != N:
            y_sl = ttnn.slice(y2d, [0, 0], [M, N])
            ttnn.deallocate(y2d)
            y2d = y_sl

        if len(orig) >= 3:
            y = ttnn.reshape(y2d, list(orig[:-1]) + [N])
        else:
            y = ttnn.reshape(y2d, [M, N])
        # The final reshape returns a metadata view sharing y2d's buffer; do NOT
        # deallocate y2d (it would free y). The buffer is freed when the caller
        # deallocates the returned tensor.

        if self.bias_tt is not None:
            y2 = ttnn.add(y, self.bias_tt)
            ttnn.deallocate(y)
            return y2
        return y

    def _call_one_mtile(self, a_i, rows: int):
        """Run one M-tile (<=32 rows) over the plan's chunks/groups; return
        interleaved L1 TILE [rows, N] (padded N)."""
        if self.partial:
            return self._call_partial(a_i, rows)
        return self._call_full(a_i, rows)

    def _call_full(self, a_i, rows: int):
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
            b_l1 = ttnn.to_memory_config(self.b_groups[g], self._b_l1_mc_full(Kc, N_chunk))
            y = ttnn.matmul_decode(
                a_sh,
                b_l1,
                partial_width_sharded=False,
                dtype=self.out_dtype,
                compute_kernel_config=self.compute_kernel_config,
                **_matmul_decode_kwargs(self.stream_k, self.k_slice_tiles),
            )
            y_il = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            ttnn.deallocate(b_l1)
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
        a_sh = ttnn.interleaved_to_sharded(a_i, self._a_shard_mc(rows, self.K))
        b_mc = self._b_l1_mc_partial()
        chunk_outs: List = []
        for c in range(self.n_chunks):
            b_l1 = ttnn.to_memory_config(self.b_chunks[c], b_mc)
            y = ttnn.matmul_decode(
                a_sh,
                b_l1,
                partial_width_sharded=True,
                dtype=self.out_dtype,
                compute_kernel_config=self.compute_kernel_config,
            )
            y_il = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(y)
            ttnn.deallocate(b_l1)
            chunk_outs.append(y_il)
        ttnn.deallocate(a_sh)
        if len(chunk_outs) == 1:
            return chunk_outs[0]
        out = ttnn.concat(chunk_outs, dim=-1)
        for t in chunk_outs:
            ttnn.deallocate(t)
        return out
