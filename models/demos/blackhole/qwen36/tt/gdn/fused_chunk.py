# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Flag-gated drop-in for the GDN chunk-prefill delta-rule core using the fully-fused
C++ op ``ttnn.transformer.chunk_gated_delta_rule``.

Same operation and I/O contract as ``chunk_gated_delta_rule_seq_adapter`` (preprocessing
+ WY inverse + inter/intra-chunk scan -> o, final_state) but computed in ONE device op
instead of Python/ttnn preprocessing + the separate ``gated_delta_attn_seq`` scan kernel.

Always the GDN prefill path (``fused_chunk_enabled()`` is hardcoded on; the seq adapter is
decode-only). Handles masked buckets (``valid_len is not None``) by zeroing beta/g past valid_len
before the op — padded positions become identity state updates, so o (causal) and final_state stay correct.

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

import torch
from loguru import logger

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

# The chunk size the fused op runs at (same math as 128, different internal tiling).
_FUSED_CHUNK_SIZE = 32


def fused_chunk_enabled():
    """Route GDN prefill through the fused ttnn.transformer.chunk_gated_delta_rule op (fast path).
    Always on; the seq adapter is used only for decode (valid_len set), selected in tp.py."""
    return True


def phased_enabled():
    """Chunk-parallel phase-split GDN (prep fanned across the grid + V-block scan)."""
    return True


def flat_qkv_enabled():
    """Flat token-major q/k/v + in-kernel L2-norm; needs phased path and chunk_size==32."""
    return True


_logged_path = False


def build_fused_const_tiles(device, chunk_size=_FUSED_CHUNK_SIZE):
    """Mesh-replicated fused-op constant tiles (eye/tril/ones + quadrant masks). Owned by the GDN layer."""
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
    valid_len=None,  # scalar: zero padded positions >= valid_len (masked-bucket prefill)
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

        # Split [B,T,H*D]->[B,T,H,D] via ROW_MAJOR in DRAM (direct TILE reshape pads H; L1 untilize clashes with op CBs).
        def _split_flat(t, H, D):
            t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            t = ttnn.reshape(t, [B, T, H, D])
            return ttnn.to_layout(t, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # OPT-A: leave q/k/v flat [B,T,H*D] for phased prep (+ in-kernel L2-norm); else split here.
        if not flat_qkv_enabled():
            q = _split_flat(q, Nk, Dk)
            k = _split_flat(k, Nk, Dk)
            v = _split_flat(v, Nv, Dv)
    else:
        Nk, Dk = q.shape[2], q.shape[3]
        Nv, Dv = v.shape[2], v.shape[3]

    beta = ttnn.reshape(beta, [B, T, Nv])
    g = ttnn.reshape(g, [B, T, Nv])

    # Host L2-norm q/k (skipped when in-kernel norm via QWEN_GDN_QK_NORM / flat QKV — required for flat).
    if not flat_qkv_enabled():
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)
        # Force DRAM: l2_norm can land in L1 (T<=512) and clash with fused-op static CBs.
        if T <= 512:
            q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)

    # valid_len: zero beta/g past pad so state updates are identity; final_state = state at valid_len.
    # Scalar (one length for all rows) or a per-row list/tuple of B lengths (grouped batched prefill:
    # each user its own real length within the shared bucket).
    _is_per_row = isinstance(valid_len, (list, tuple))
    if _is_per_row or (valid_len is not None and valid_len < T):
        _dram = ttnn.DRAM_MEMORY_CONFIG  # op CBs clash with L1 inputs at small buckets
        _mt = torch.zeros(B, T, 1, dtype=torch.float32)
        if _is_per_row:
            for _b in range(B):
                _mt[_b, : int(valid_len[_b]), :] = 1.0
        else:
            _mt[:, :valid_len, :] = 1.0
        _m = ttnn.from_torch(_mt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        beta = ttnn.multiply(beta, _m, memory_config=_dram)  # beta/g fp32 (op contract) — load-bearing
        g = ttnn.multiply(g, _m, memory_config=_dram)
        # Also mask q/k/v for bit-parity with seq (beta/g alone is enough for correctness).
        _mq_t = _mt if len(q.shape) == 3 else _mt.reshape(B, T, 1, 1)
        _mq = ttnn.from_torch(_mq_t, dtype=q.dtype, layout=ttnn.TILE_LAYOUT, device=device)
        q = ttnn.multiply(q, _mq, memory_config=_dram)
        k = ttnn.multiply(k, _mq, memory_config=_dram)
        v = ttnn.multiply(v, _mq, memory_config=_dram)
        ttnn.deallocate(_m)
        ttnn.deallocate(_mq)

    s0 = None
    if initial_state is not None:
        s0 = ttnn.reshape(initial_state, [B, Nv, Dk, Dv])
        if s0.dtype != ttnn.float32:
            s0 = ttnn.typecast(s0, ttnn.float32)

    # output_head_major=return_o_bh: skip token<->head permute when caller wants [BH,T,V].
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
