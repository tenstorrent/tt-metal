# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Flag-gated drop-in for the GDN chunk-prefill delta-rule core using the fully-fused
C++ op ``ttnn.transformer.chunk_gated_delta_rule``.

Same operation and I/O contract as ``chunk_gated_delta_rule_seq_adapter`` (preprocessing
+ WY inverse + inter/intra-chunk scan -> o, final_state) but computed in ONE device op
instead of Python/ttnn preprocessing + the separate ``gated_delta_attn_seq`` scan kernel.

Enabled by env flag ``QWEN_GDN_FUSED_CHUNK=1`` (see tp.py gate). Falls back to the seq
adapter for masked buckets (``valid_len is not None``), which the fused op does not
handle yet.

Notes:
* The fused op does not L2-normalize q/k (its contract), so we normalize here — identical
  to what the seq adapter does internally.
* The fused op runs at chunk_size=32 (chunk=128 exceeds the L1 CB budget). chunk size is an
  internal tiling choice; the result is identical to chunk=128. At 32 each per-chunk WY matrix
  is a single 32x32 tile whose (I + strictly_lower)^-1 is computed by the 16x16-blocked inverse
  (mirroring FLA solve_tril's merge_16x16_to_32x32) — numerically exact-to-PCC across seeds.
  chunk_size=64 splits the WY matrix into a 2x2 tile-block whose bottom-right 32x32 sub-block can
  be ill-conditioned enough that the fp32 block inverse loses precision on some chunks; 32 avoids
  that with identical math (see tests/.../test_gdn_phased_perchunk.py).
* GVA (Nk<Nv) head expansion is done inside the fused op; we pass q/k with Nk heads.
"""
import os

import torch
from loguru import logger

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

# The chunk size the fused op runs at (same math as 128, different internal tiling).
_FUSED_CHUNK_SIZE = 32


# GDN prefill runs the fast fused path by DEFAULT — no env vars needed. The two flags below default
# ON and exist only as opt-outs for benchmarking/debug (set to 0). Everything the demo needs (phased
# prep+scan, fp32 o output, fp32 state, flat token-major q/k/v with in-kernel L2-norm) is the default.


def fused_chunk_enabled():
    """Route GDN prefill through the fused ttnn.transformer.chunk_gated_delta_rule op (fast path).
    Always on; the seq adapter is used only for decode (valid_len set), selected in tp.py."""
    return True


def phased_enabled():
    """Chunk-parallel phase-split GDN (prep fanned across the grid + V-block scan). Default ON;
    QWEN_GDN_PHASED=0 falls back to the monolithic fused op (benchmark/debug only)."""
    return not os.environ.get("QWEN_GDN_PHASED", "1").startswith("0")


def flat_qkv_enabled():
    """Flat token-major q/k/v reading + in-kernel L2-norm (OPT-A/B): the prep reader tile-addresses
    each head's chunk straight out of the flat tensors, eliminating the head-split relayouts and the
    host l2_norm (the ~70% preprocessing win). Default ON; QWEN_GDN_FLAT_QKV=0 disables it (falls back
    to head-split + host l2_norm). Requires the phased path + chunk_size==32."""
    return not os.environ.get("QWEN_GDN_FLAT_QKV", "1").startswith("0")


_logged_path = False


def build_fused_const_tiles(device, chunk_size=_FUSED_CHUNK_SIZE):
    """Build the device-resident constant tiles the fused op consumes, replicated across the mesh:
    eye/tril/ones [1,1,C,C] fp32 and the [1,1,32,96] quadrant masks. Built ONCE by the caller (the
    GDN layer, in __init__) and passed into ttnn.transformer.chunk_gated_delta_rule so the op stays
    stateless. Owning them on the layer ties their lifetime to the model — freed before the device
    closes — instead of a process-lifetime C++ static that would deallocate after the device is gone
    and segfault at exit. Values mirror the op's make_const_cc / make_quadrant_masks builders exactly.
    """
    C = chunk_size
    eye = torch.eye(C, dtype=torch.float32)
    tril = torch.tril(torch.ones(C, C, dtype=torch.float32))
    ones = torch.ones(C, C, dtype=torch.float32)
    # Three 32x32 quadrant masks packed into [32,96]: block0=top-left, block1=bottom-right,
    # block2=bottom-left (matches make_quadrant_masks: m[i, b*32+j]).
    ii = torch.arange(32).unsqueeze(1)
    jj = torch.arange(32).unsqueeze(0)
    lo_i, lo_j = ii < 16, jj < 16
    qtl = (lo_i & lo_j).float()
    qbr = (~lo_i & ~lo_j).float()
    qbl = (~lo_i & lo_j).float()
    masks = torch.cat([qtl, qbr, qbl], dim=1)  # [32, 96]

    def _up(t):
        return ttnn.from_torch(
            t.reshape(1, 1, *t.shape),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

    return (_up(eye), _up(tril), _up(ones), _up(masks))


def chunk_gated_delta_rule_fused_adapter(
    q,  # [B, T, Nk, Dk] or flat [B, T, Nk*Dk] (with qkv_head_dims)
    k,  # [B, T, Nk, Dk] or flat
    v,  # [B, T, Nv, Dv] or flat
    beta,  # [B, T, Nv]
    g,  # [B, T, Nv]
    chunk_size=128,  # accepted for drop-in parity; the fused op runs at _FUSED_CHUNK_SIZE
    scale=None,
    initial_state=None,  # [B, Nv, Dk, Dv] or None
    device=None,
    cached_masks=None,  # unused (the fused op builds its own constants)
    valid_len=None,  # must be None here; caller gates masked buckets to the seq adapter
    qkv_head_dims=None,  # (Nk, Dk, Nv, Dv) when q/k/v are flat
    return_o_bh=False,  # True: return o as [B*Nv, T, V]; else [B, T, Nv, V]
    const_tiles=None,  # (eye, tril, ones, masks) device tensors built once by the caller (layer);
    # passed to the op so it stays stateless. Required under trace (the op's internal build does a
    # host upload, illegal under trace); if None, the op builds them eagerly.
):
    global _logged_path
    if not _logged_path:
        logger.info(
            "[GDN] fused chunk_gated_delta_rule active: "
            f"path={'PHASED (chunk-parallel prep + V-block scan)' if phased_enabled() else 'monolithic'}, "
            f"chunk_size={_FUSED_CHUNK_SIZE}, flat_qkv={flat_qkv_enabled()}, "
            f"input q/k/v dtype={q.dtype}/{k.dtype}/{v.dtype}"
        )
        _logged_path = True

    B = q.shape[0]
    T = q.shape[1]

    if qkv_head_dims is not None:
        Nk, Dk, Nv, Dv = qkv_head_dims

        # Split flat [B,T,H*D] TILE -> [B,T,H,D] TILE via ROW_MAJOR. A *direct* TILE reshape
        # puts H in a tile dim (H padded to 32) and is ~2-5x slower (measured: v-reshape 4.05ms
        # vs 0.83ms at T=8192, Nv=12); doing the reshape untilized avoids the padded relayout.
        # Keep the whole split in DRAM: the op's static CBs (~1.36MB/core) clash with any
        # L1-resident intermediate that lingers on a core during the kernel (the untilize
        # intermediate does, at higher head counts). DRAM untilize is still far cheaper than a
        # direct TILE reshape with H in the tile dim.
        def _split_flat(t, H, D):
            t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            t = ttnn.reshape(t, [B, T, H, D])
            return ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # OPT-A: when flat, leave q/k/v FLAT token-major [B,T,H*D]; the phased prep reader tile-addresses
        # each head's chunk directly (skips the untilize/reshape/tilize + op-internal head-split permute).
        # q/k flat also requires the in-kernel L2-norm (handled below + in the op). Otherwise split here.
        if not flat_qkv_enabled():
            q = _split_flat(q, Nk, Dk)
            k = _split_flat(k, Nk, Dk)
            v = _split_flat(v, Nv, Dv)
    else:
        Nk, Dk = q.shape[2], q.shape[3]
        Nv, Dv = v.shape[2], v.shape[3]

    beta = ttnn.reshape(beta, [B, T, Nv])
    g = ttnn.reshape(g, [B, T, Nv])

    # L2-norm q/k at Nk heads (fused op does not; matches the seq adapter's internal norm).
    # OPT-B: when in-kernel q/k norm is enabled (QWEN_GDN_QK_NORM, or implied by QWEN_GDN_FLAT_QKV),
    # the prep compute normalizes q/k over K and folds in `scale`, so the host norm is skipped. This
    # is also REQUIRED for flat q/k: a flat [B,T,Nk*Dk] tensor can't be L2-normed over D on host.
    if not flat_qkv_enabled():
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)
        # l2_norm_ttnn returns L1-resident tensors for T<=512; the fused op's static CBs (~1.36MB/core)
        # clash with any L1-resident kernel input, so force q/k to DRAM. No-op for T>512 (the demo runs
        # T=2048 chunks, where l2_norm already lands in DRAM).
        if T <= 512:
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)

    s0 = None
    if initial_state is not None:
        s0 = ttnn.reshape(initial_state, [B, Nv, Dk, Dv])
        if s0.dtype != ttnn.float32:
            s0 = ttnn.typecast(s0, ttnn.float32)

    # output_head_major=return_o_bh: the op natively produces o head-major, so when the caller
    # wants [BH,T,V] we get it directly (TILE) and skip the token<->head permute round-trip that
    # otherwise dominates this adapter's cost (~10 ms/call at T=8192, Nv=12 — measured).
    _eye, _tril, _ones, _masks = const_tiles if const_tiles is not None else (None, None, None, None)
    o, final_state = ttnn.transformer.chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=s0,
        output_final_state=True,
        chunk_size=_FUSED_CHUNK_SIZE,
        output_head_major=return_o_bh,
        eye=_eye,
        tril=_tril,
        ones=_ones,
        masks=_masks,
    )

    if return_o_bh:
        # Op already returned head-major [B*Nv, T, Dv] in TILE — nothing to relayout.
        pass
    else:
        # Token-major [B, T, Nv, Dv]; op returns ROW_MAJOR, tilize to match the seq adapter.
        o = ttnn.to_layout(o, ttnn.TILE_LAYOUT)

    # Final state is returned fp32 (the validated default; matches the seq path and the op's fp32 s0).
    if final_state.dtype != ttnn.float32:
        final_state = ttnn.typecast(final_state, ttnn.float32)

    return o, final_state
