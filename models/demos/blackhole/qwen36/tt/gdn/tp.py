# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel Gated DeltaNet for Qwen3.5.

Recurrence is per value-head (no cross-device comms inside); all-reduce after row-parallel out.
Reuses `recurrent_gated_delta_rule_decode_ttnn`; weights interleaved. GDN norm uses raw weight
(no +1) + SiLU(z) gate — distinct from QK/layer norms.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import weight_prefetch
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce


def gdn_fused_decode_enabled():
    """QWEN36_GDN_FUSED_DECODE (default "1"): single-user (max batch 1) GDN decode through the fused kernels
    (qkv_causal_conv1d_silu + chunk_gated_delta_rule + sigmoid_gated_rms_norm). "0" keeps the original
    shift-register / recurrent_gated_delta_rule_decode_ttnn path byte-for-byte."""
    return os.environ.get("QWEN36_GDN_FUSED_DECODE", "1") != "0"


_fused_decode_logged = False


def gdn_decode_step_op_enabled():
    """QWEN36_GDN_DECODE_STEP_OP (default on): decode through ttnn.experimental.kda.gdn_decode_step (conv + gating +
    recurrence + gated norm in one op, packed conv history). 0 reverts to the fused-batched / original paths."""
    return os.environ.get("QWEN36_GDN_DECODE_STEP_OP", "1") == "1"


def gdn_decode_step_fp32_state_enabled():
    """QWEN36_GDN_STEP_FP32_STATE (default on): gdn_decode_step keeps the recurrent-state decay and add in exact fp32
    (SFPU) instead of FPU SrcA, which truncates the fp32 state every token and biases long free-running decode.
    0 reverts to the FPU path."""
    return os.environ.get("QWEN36_GDN_STEP_FP32_STATE", "1") == "1"


def gdn_verify_spec_step_enabled():
    """QWEN36_VERIFY_SPEC_STEP (default "1"): run the batched spec verify AND its T = 1 seed through
    ttnn.experimental.kda.gdn_spec_step -- ONE fused op per GDN layer (conv-window rebuild + depthwise conv + SiLU +
    l2norms + gates + T-step gated delta rule with per-token ring writes + gated RMSNorm + silu(z)) with
    gdn_decode_step's fast_mode / fp32_state arithmetic, in place of the ~70-op composite chain of
    _verify_fullbatch / _seed_fullbatch. "1" (default) uses the op; "0" keeps the composite
    path byte for byte. With the op, plain decode stays on gdn_decode_step (set_gdn_fused_decode(True) is ignored),
    so decode, seed and verify share one kernel family."""
    return os.environ.get("QWEN36_VERIFY_SPEC_STEP", "1") == "1"


def gdn_spec_step_value_blocks():
    """QWEN36_SPEC_STEP_VALUE_BLOCKS (default "1" = off): gdn_spec_step's value_blocks -- split each (user, value
    head)'s recurrent state columns over up to this many cores (tt-lab's 4-core GDN head split). The op takes the
    largest divisor of Dv/32 <= the value whose B*Nv*blocks cores fit the grid (TP = 8, Nv = 6, >= 96-core grid: 2
    at the B = 8 bucket, 4 at B <= 4) and stays bit-identical (out, ring, window) to one core per (user, head). Only
    the T > 1 verify splits; the T = 1 seed keeps one core per (user, head) (faster there). Values: 1, 2 or 4; 1
    passes nothing to the op, so a ttnn build without the attribute keeps working."""
    env = os.environ.get("QWEN36_SPEC_STEP_VALUE_BLOCKS", "1").strip() or "1"
    vb = int(env)
    if vb not in (1, 2, 4):
        raise ValueError(f"QWEN36_SPEC_STEP_VALUE_BLOCKS must be 1, 2 or 4, got {env!r}")
    return vb


# gdn_spec_step kernel_opts bits (GdnSpecStepKernelOpt in gdn_spec_step_device_operation_types.hpp)
SPEC_STEP_OPT_FUSED_DECAY = 1  # the next token's exact-fp32 state decay fused into this token's state add
SPEC_STEP_OPT_GATE_SCALARS = 2  # gate math on the writer, scalar tiles filled in DST (no reader broadcast-tile fills)
SPEC_STEP_OPT_EARLY_SRC = 4  # the reader hands the conv inputs to the compute before the taps / masks
SPEC_STEP_OPT_HOIST_INV = 8  # rsqrt(q stats + eps) * scale once per op instead of once per token
SPEC_STEP_OPT_QKSHARE = 16  # value_blocks > 1: the head's blocks split the q/k conv and swap the tiles over the NoC
SPEC_STEP_OPT_A3_HALF = 32  # the conv's SFPU tail on the tile half that holds the user's rows
SPEC_STEP_OPT_GSC_SPLIT = 64  # with 2 and value_blocks > 1: the head's blocks split the gate math and swap the results
SPEC_STEP_OPT_VS_TAIL = 128  # value_blocks > 1: each block normalizes / gates only its own output columns
SPEC_STEP_OPT_KNOWN = (
    SPEC_STEP_OPT_FUSED_DECAY
    | SPEC_STEP_OPT_GATE_SCALARS
    | SPEC_STEP_OPT_EARLY_SRC
    | SPEC_STEP_OPT_HOIST_INV
    | SPEC_STEP_OPT_QKSHARE
    | SPEC_STEP_OPT_A3_HALF
    | SPEC_STEP_OPT_GSC_SPLIT
    | SPEC_STEP_OPT_VS_TAIL
)


def gdn_spec_step_kernel_opts():
    """QWEN36_SPEC_STEP_OPT (default "0" = off): gdn_spec_step's kernel_opts bitmask of bit-identical kernel
    restructurings of the fast_mode + fp32_state program (out, ring and window stay torch.equal to 0):
    1 = fused next-token decay (no hn32 buffer / reload per token); 2 = gate scalars (the writer computes the
    beta / decay gates and the compute fills the scalar tiles in DST: the reader's 2*T broadcast-tile fills were the
    op's critical path); 4 = early src (the conv inputs reach the compute before the taps / masks are built);
    8 = hoisted q scale (rsqrt(q stats + eps) * scale once per op, not per token); 16 = q/k share (with
    QWEN36_SPEC_STEP_VALUE_BLOCKS > 1: the head's value blocks split the q/k conv and swap the exact conv tiles over
    the NoC); 32 = the conv's SFPU tail (tap sum + silu) on the tile half holding the user's rows; 64 = gate split
    (with 2 and value blocks: the head's blocks split the gate math); 128 = each value block normalizes / gates only
    its own output columns. Recommended: 250 (= 2|8|16|32|64|128) with QWEN36_SPEC_STEP_VALUE_BLOCKS=4 (fits the
    70656 B kernel config buffer at TP = 8 / 4 for B 1..8, T 1..12); all eight bits at once (255) overflows it at
    value_blocks 2 (B = 8), and 250 does at TP = 2 (Nv = 24) with T = 12 and value blocks.
    0 passes nothing to the op, so a ttnn build without the attribute keeps working. Decimal or 0x-prefixed; unknown
    bits are refused."""
    env = os.environ.get("QWEN36_SPEC_STEP_OPT", "0").strip() or "0"
    try:
        opts = int(env, 0)
    except ValueError:
        raise ValueError(f"QWEN36_SPEC_STEP_OPT must be an integer bitmask, got {env!r}") from None
    if opts < 0 or opts & ~SPEC_STEP_OPT_KNOWN:
        raise ValueError(f"QWEN36_SPEC_STEP_OPT has unknown bits: {env!r} (known mask {SPEC_STEP_OPT_KNOWN:#x})")
    return opts


def pack_head_tiles(rows, Nv, Nk, Dk, Dv, parity=0, both_parities=False):
    """rows: list of 4 torch [C] vectors ([q | k | v] channels of one device) -> [Nv, 4, 32, 32] bf16 packed head
    tiles for gdn_decode_step: tile (h, s) holds rows[s]'s channels [q_hk | k_hk | v_h] (hk = h // rf) as 32-wide
    chunks, chunk c in row 2c + parity (user b's history uses parity b & 1; taps are stored on both parities)."""
    rf = Nv // Nk
    kd = Nk * Dk
    out = torch.zeros(Nv, 4, 32, 32, dtype=torch.bfloat16)
    for h in range(Nv):
        hk = h // rf
        for j, r in enumerate(rows):
            r = r.reshape(-1).to(torch.bfloat16)
            chunks = torch.cat(
                [
                    r[hk * Dk : (hk + 1) * Dk],
                    r[kd + hk * Dk : kd + (hk + 1) * Dk],
                    r[2 * kd + h * Dv : 2 * kd + (h + 1) * Dv],
                ]
            ).reshape(-1, 32)
            n = chunks.shape[0]
            for par in (0, 1) if both_parities else (parity,):
                out[h, j, par : 2 * n + par : 2, :] = chunks
    return out


def _log_fused_decode_once(msg):
    """One info line per process (the layer is built 48x)."""
    global _fused_decode_logged
    if not _fused_decode_logged:
        _fused_decode_logged = True
        logger.info(msg)


_spec_step_logged = set()


def _log_spec_step_once(key, msg):
    """One info line per process and key for the QWEN36_VERIFY_SPEC_STEP path (the layer is built 48x)."""
    if key not in _spec_step_logged:
        _spec_step_logged.add(key)
        logger.info(msg)


def _shard_small_fp32(torch_tensor, mesh, cache_path, dim=-1):
    """tpc.shard_small without the bf16 round-trip: per-head fp32 tensor -> fp32 TILE sharded on `dim`."""
    t = torch_tensor.float()
    if t.dim() == 1:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 2:
        t = t.unsqueeze(0)
    return ttnn.as_tensor(
        t,
        dtype=ttnn.float32,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def _replace(old, new):
    """Free ``old`` unless ``new`` is a view of it; returns ``new``."""
    if not tpc.aliases(new, old):
        ttnn.deallocate(old)
    return new


def _softplus_add(a, bias):
    """g-gate: softplus(a + bias) fused into one op (softplus as a post-activation on the add)."""
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def _silu_mul(x, z, memory_config, dtype=None):
    """out-gate: x * silu(z). NOT fused into one op: fusing silu via input_tensor_b_activations
    overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small inputs).
    dtype: optional output dtype (bf16 for the column-parallel prefill out-proj; default = x's)."""
    s = ttnn.silu(z, memory_config=memory_config)
    if dtype is None:
        return ttnn.multiply(x, s, memory_config=memory_config)
    return ttnn.multiply(x, s, memory_config=memory_config, dtype=dtype)


def kda_channel_chunk_size(channels, cap=512):
    """channel_chunk_size for qkv_causal_conv1d_silu: the largest tile-aligned divisor of `channels` not above
    `cap`. 512 at the 27B TP-4 width (2560 -> 5 blocks x 64 tile rows = 320 work items over the grid), the
    configuration the op was measured at."""
    for c in range(min(cap, channels) - min(cap, channels) % 32, 0, -32):
        if channels % c == 0:
            return c
    raise ValueError(f"no tile-aligned channel chunk divides {channels}")


_EXACT_SELECT_CFG = None


def exact_select_cfg():
    """HiFi4 + fp32 accumulation compute config for the one-hot row-select matmuls (bf16 values x 0/1 -> the
    selected row is reproduced bit-exactly; lower fidelities may drop mantissa bits of the value operand)."""
    global _EXACT_SELECT_CFG
    if _EXACT_SELECT_CFG is None:
        _EXACT_SELECT_CFG = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return _EXACT_SELECT_CFG


def kda_conv_prefill(
    qkv,
    T,
    history,
    taps,
    widths,
    actual_start,
    xin_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    conv_sel=None,
    emit_state=True,
):
    """Depthwise causal conv (K=4) + SiLU + q/k/v split in ONE program (ttnn.experimental.kda.qkv_causal_conv1d_silu).

    qkv:          [1, T, C] bf16 TILE, the projection's conv columns (q | k | v).
    history:      [1, 3, C] bf16, the three rows preceding this chunk (zeros from scratch). TILE or ROW_MAJOR;
                  the op reads ROW_MAJOR, so a TILE carry is converted here (three rows).
    taps:         four [1, 1, C] bf16 TILE tensors in kernel-position order, tap j multiplying row t-3+j
                  (tw["conv_taps"], exactly the op's tap0..tap3 contract).
    widths:       (q_width, k_width, v_width), tile-aligned, summing to C.
    actual_start: uint32 [1] device tensor holding 0, allocated once before any trace capture.
    conv_sel:     optional [1, 3, T] bf16 TILE one-hot row selector (traced masked bucket: valid_len is a persistent
                  device input, so the "last three real rows" cannot be a static slice). new_state is then
                  matmul(conv_sel, qkv) = rows valid_len-3..valid_len-1 of the conv input (all-zero rows where
                  valid_len-3+j < 0, exact for a from-scratch history of zeros). None: the static last-3-rows slice.
    Returns q [1,T,Q], k [1,T,K], v [1,T,V] bf16 TILE DRAM, and new_state [1, 3, C] bf16 TILE DRAM: the chunk's
    last three INPUT rows, i.e. the next chunk's history (the op does not emit it).
    emit_state=False skips new_state (returned as None), for callers that keep their own carry (the spec verify).
    """
    C = qkv.shape[-1]
    n_hist = history.shape[1]  # K - 1 = 3 rows, the op's fixed history depth
    assert qkv.shape[1] == T, f"kda_conv_prefill: T={T} but qkv has {qkv.shape[1]} rows"
    _dram = ttnn.DRAM_MEMORY_CONFIG
    # The op's input contract is row-major: one untilize of the conv columns.
    xin = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=xin_memory_config)
    hist = history if history.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(history, ttnn.ROW_MAJOR_LAYOUT)
    q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        xin,
        hist,
        *taps,
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=kda_channel_chunk_size(C)),
        actual_start=actual_start,
        # No sequence parallelism on the TP mesh (sequence_parallel_axis=0 on a 1xN mesh): the op validates
        # predecessor_carry against history and never reads it, so alias history as its docstring says.
        predecessor_carry=hist,
        memory_config=_dram,
    )
    if hist is not history:
        ttnn.deallocate(hist)
    if not emit_state:
        ttnn.deallocate(xin)
        return q, k, v, None
    # Next chunk's history: the last three input rows, taken from the row-major copy (page reads, no
    # untilize), then re-tiled because the layer-wide carry (conv_carry, decode seeding, per-user assembly)
    # is TILE.
    if conv_sel is not None:
        assert conv_sel.shape[1] == n_hist and conv_sel.shape[-1] == T, (conv_sel.shape, n_hist, T)
        ttnn.deallocate(xin)
        # bf16 one-hot x bf16 values from the TILE projection output (same values the RM copy holds), HiFi4 + fp32
        # accumulate: exactly one nonzero term per output element -> bit-exact row selection.
        # DRAM operand: an L1-resident qkv could clash with the matmul's static CBs (the op sees a 3-row LHS).
        qkv_d = qkv if qkv.memory_config().buffer_type == ttnn.BufferType.DRAM else ttnn.to_memory_config(qkv, _dram)
        new_state = ttnn.matmul(
            conv_sel, qkv_d, memory_config=_dram, compute_kernel_config=exact_select_cfg(), dtype=ttnn.bfloat16
        )
        if qkv_d is not qkv:
            ttnn.deallocate(qkv_d)
        return q, k, v, new_state
    new_state = ttnn.slice(xin, (0, T - n_hist, 0), (1, T, C), memory_config=_dram)
    ttnn.deallocate(xin)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT, memory_config=_dram)
    return q, k, v, new_state


def kda_pack_gather(n_users, T, kc, rows_padded):
    """One-hot bf16 [1, n_users*T, rows_padded] that picks token rows out of a packed KDA conv output.

    The packed input holds n_users windows end to end, each L = kc-1+T rows ([kc-1 carry ; T tokens]).
    Output row u*T + j selects packed row u*L + (kc-1) + j: user u's token j."""
    L = kc - 1 + T
    assert n_users * L <= rows_padded, f"{n_users} windows of {L} rows exceed {rows_padded}"
    g = torch.zeros(1, n_users * T, rows_padded, dtype=torch.bfloat16)
    r = torch.arange(n_users * T)
    g[0, r, r // T * L + kc - 1 + r % T] = 1.0
    return g


# --------------------------------------------------------------------------- #
# Batched (multi-user) spec-decode host helpers (pure torch)
# --------------------------------------------------------------------------- #
# A batched verify packs B users x T = K+1 candidate rows user-major into one 32-row decode tile
# (row r = u*T + j). With mi[u] the index of user u's last accepted row, the host commits by
# selection: the next replay reads user u's GDN state from the ring block mi[u] wrote and rebuilds
# its conv shift register from the window rows mi[u] wrote. These helpers build the selector
# tensors that carry mi into the trace.


HOLD_SENTINEL = (
    0xFFFFFFFF  # ring block index that makes gdn_spec_step skip a head's state writes (window copied through)
)


def spec_state_blk_idx(mi, n_users, nv, hold=None):
    """Ring block index per (user, value-head) for the fused recurrent op's deferred-select mode.

    The ring holds T*n_users*nv state blocks laid out (token slot, user, head)-major, so the state
    user u has after accepting through row mi[u] is block (mi[u]*n_users + u)*nv + h, which is
    entry u*nv + h of the result.

    mi: per-user last accepted row (len n_users, each in [0, T)). Returns torch.int32 [n_users*nv].
    hold: optional set of users whose every head gets the HOLD sentinel (int32 -1 == uint32 0xFFFFFFFF):
    gdn_spec_step then skips their ring writes. The engine does not use it today (a row left out of a step is
    parked on host, see MTPSpecEngine.verify); it is here so a ctrl page can express the op's full contract.
    """
    assert len(mi) == n_users, f"need one mi per user: got {len(mi)} for {n_users} users"
    blk = torch.as_tensor(mi, dtype=torch.int64) * n_users + torch.arange(n_users)
    out = (blk[:, None] * nv + torch.arange(nv)).reshape(-1).to(torch.int32)
    for u in set(hold or ()):
        out[u * nv : (u + 1) * nv] = -1
    return out


def spec_ctrl_words(n_users, nv):
    """Words in one ``gdn_spec_step`` ctrl page: 1 (window parity) + B (mi) + B*Nv (ring block per (user, head)),
    rounded up to a multiple of 16 so the page is a whole number of 64 B lines (the op's N*4 % 64 == 0 contract)."""
    n = 1 + n_users + n_users * nv
    return (n + 15) // 16 * 16


def spec_ctrl_page(mi, n_users, nv, parity, hold=None):
    """The ONE control page a fused ``gdn_spec_step`` verify reads (QWEN36_VERIFY_SPEC_STEP=1) in place of the
    (spec_state_blk_idx, spec_conv_sel) pair.

    word 0        = conv-window parity: which half of the layer's window ping-pong pair is E_prev (the op reads
                    it and writes the rebuilt window into the other half);
    words 1..B    = mi[u], the accepted-prefix index (the carry rows are E_prev[u, mi+1 : mi+K]); 0 for a held user
                    (unread: its window is copied through unchanged);
    words 1+B..   = the initial ring block per (user, head), exactly ``spec_state_blk_idx`` -- the HOLD sentinel for
                    EVERY head of a held user (the op skips that user's ring writes; the q/k window chunks are owned
                    per key-head group, so a partial hold is not a valid page).
    Pure data, shared by every GDN layer (all flip parity in lockstep) and not a program-hash input: one trace
    serves every commit pattern. Returns torch.int32 [1, N] (N = spec_ctrl_words); staged as uint32, where the
    int32 -1 of the sentinel is 0xFFFFFFFF. mi is NOT range-checked here (the op clamps mi >= T to T-1 silently), so
    callers check it against their T.
    """
    assert len(mi) == n_users, f"need one mi per user: got {len(mi)} for {n_users} users"
    assert parity in (0, 1), f"window parity must be 0 or 1, got {parity}"
    hold = set(hold or ())
    words = torch.zeros(spec_ctrl_words(n_users, nv), dtype=torch.int32)
    words[0] = int(parity)
    if tpc.spec_hostset_enabled() and not hold:  # QWEN36_SPEC_HOSTSET: one slice assignment instead of B scalar ones
        words[1 : 1 + n_users] = torch.as_tensor(mi, dtype=torch.int32)
    else:
        for u in range(n_users):
            words[1 + u] = 0 if u in hold else int(mi[u])
    words[1 + n_users : 1 + n_users + n_users * nv] = spec_state_blk_idx(mi, n_users, nv, hold=hold)
    return words.reshape(1, -1)


def spec_conv_sel(mi, n_users, T, kc):
    """One-hot row selector that rebuilds every user's conv window for the next verify.

    The verify concatenates the previous window with the new qkv rows:
        concat = cat([E_prev [n_users, kc-1+T, C], qkv_new [n_users, T, C]], dim=1)
    and left-multiplies it by this selector, so for user u:
        out row r        (r < kc-1) <- concat row mi[u] + 1 + r      (E_prev, post-commit window)
        out row kc-1 + j            <- concat row (kc-1+T) + j       (new qkv row j)
    i.e. the kc-1 conv taps that survive the commit, followed by the T new inputs.

    Returns torch.bfloat16 [n_users, kc-1+T, kc-1+2T] (one-hot, so the matmul is exact).
    """
    assert len(mi) == n_users, f"need one mi per user: got {len(mi)} for {n_users} users"
    m = torch.as_tensor(mi, dtype=torch.int64)
    assert bool(((m >= 0) & (m < T)).all()), f"mi={m.tolist()} out of range [0,{T})"
    sel = torch.zeros(n_users, kc - 1 + T, kc - 1 + 2 * T, dtype=torch.bfloat16)
    r, j = torch.arange(kc - 1), torch.arange(T)
    sel[torch.arange(n_users)[:, None], r, m[:, None] + 1 + r] = 1.0
    sel[:, kc - 1 + j, kc - 1 + T + j] = 1.0
    return sel


def fresh_row_select(fresh, n_users, T, kc):
    """One-hot row selector that seeds the conv window E_prev of FRESH users (QWEN36_SPEC_FAST_JOIN=1,
    ``TPGatedDeltaNet.seed_spec_rows_select``) while every other user's window is kept:

        cat = cat([E_prev [n_users, kc-1+T, C], taps [n_users, kc-1, C]], dim=1)      (taps = the kc-1 most
        E_new = sel @ cat                                                               recent raw inputs, oldest first)

    For a running user u, E_new[u] = E_prev[u] (the identity on the first kc-1+T columns). For a fresh user u,
    E_new[u, 1 + r] = taps[u, r] for r < kc-1 and every other row is 0: a mi = 0 verify (spec_conv_sel / the
    gdn_spec_step ctrl page) reads rows 1 .. kc-1 as the surviving conv carry, exactly the rows seed_spec_state /
    seed_spec_rows fill from conv taps 1 .. kc-1; row 0 and rows kc .. are never read at mi = 0.

    Returns torch.bfloat16 [n_users, kc-1+T, (kc-1+T) + (kc-1)] (one-hot, so the matmul is exact).
    """
    L = kc - 1 + T
    f = {int(u) for u in fresh}
    assert all(0 <= u < n_users for u in f), f"fresh rows {sorted(f)} outside range({n_users})"
    sel = torch.zeros(n_users, L, L + kc - 1, dtype=torch.bfloat16)
    for u in range(n_users):
        if u in f:
            sel[u, 1 + torch.arange(kc - 1), L + torch.arange(kc - 1)] = 1.0
        else:
            sel[u, torch.arange(L), torch.arange(L)] = 1.0
    return sel


def fresh_row_mask(fresh, n_users, Nv, Dk):
    """Row mask of ``TPGatedDeltaNet.seed_spec_rows_masked`` (QWEN36_SPEC_JOIN_TRACE=1): torch.float32
    [1, n_users*Nv, Dk, 1], 1.0 on the Nv head blocks of every fresh user (rows u*Nv .. (u+1)*Nv-1), 0.0 elsewhere.
    The user index space is the ring / rec_state row space of fresh_row_select (engine rows)."""
    f = {int(u) for u in fresh}
    assert all(0 <= u < n_users for u in f), f"fresh rows {sorted(f)} outside range({n_users})"
    m = torch.zeros(1, n_users * Nv, Dk, 1, dtype=torch.float32)
    for u in sorted(f):
        m[:, u * Nv : (u + 1) * Nv] = 1.0
    return m


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's linear_attn.* weights across the mesh."""
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp
    # In-proj weight dtype (QWEN36_BFP4_GDN_IN, opt-in): BFP4. Out-proj (QWEN36_BFP4_GDN_OUT, opt-in): BFP4.
    _in_dtype = ttnn.bfloat4_b if tpc.bfp4_gdn_in_enabled() else ttnn.bfloat8_b
    _out_dtype = ttnn.bfloat4_b if tpc.bfp4_gdn_out_enabled() else ttnn.bfloat8_b

    if cache_dir is not None:
        import os

        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    # State-dict keys vary by loader: optional linear_attn. prefix; conv1d may be fused or q/k/v split.
    P = "linear_attn." if any(k.startswith("linear_attn.") for k in sd) else ""

    def first_key(*names):
        for n in names:
            if (P + n) in sd:
                return sd[P + n]
        raise KeyError(f"none of {[P + n for n in names]} found in GDN state dict")

    # Fused QKV+Z (column-parallel)
    qkv_w = first_key("in_proj_qkv.weight", "qkv_proj.weight")
    if (P + "conv1d.weight") in sd:
        conv1d_w = sd[P + "conv1d.weight"]
    else:  # bf16 remap: reassemble fused conv1d from q/k/v streams
        conv1d_w = torch.cat([sd[P + "q_conv.weight"], sd[P + "k_conv.weight"], sd[P + "v_conv.weight"]], dim=0)
    qkv_re = tpc.prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp)
    z_w = sd[P + "in_proj_z.weight"]
    a_w, b_w = sd[P + "in_proj_a.weight"], sd[P + "in_proj_b.weight"]
    tw = {}
    # Column-parallel qkvz (DRAM-sharded decode matmul when enabled); distinct .dramshard cache
    qkvz_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
    # Fold a/b into qkvz → one matmul outputs [qkv|z|a|b] (default when DRAM-sharded)
    fuse_ab = qkvz_sharded
    if fuse_ab:
        fused = torch.cat(
            [
                torch.cat(
                    [
                        qkv_re[d * qkv_per : (d + 1) * qkv_per],
                        z_w[d * z_per : (d + 1) * z_per],
                        a_w[d * nv_per : (d + 1) * nv_per],
                        b_w[d * nv_per : (d + 1) * nv_per],
                    ],
                    dim=0,
                )
                for d in range(tp)
            ],
            dim=0,
        )
        # proj_1d_decode: interleaved weight (fast small-grid 1D decode matmul; prefill AGMM verified
        # bit-identical on interleaved). Distinct cache suffix.
        _proj1d = getattr(args, "proj_1d_decode", False)
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.gdn_qkvzab_weight_memcfg,
            cache_path=c("qkvzab" + (".il" if _proj1d else ".dramshard")),
            dtype=_in_dtype,
        )
        if _proj1d:
            # QWEN36_WEIGHT_PREFETCH (gdn_in): receiver-contiguous decode copy for the Tensor prefetcher.
            _pc = getattr(args, "gdn_qkvz_decode_1d_progcfg", None)
            weight_prefetch.attach_twin(
                mesh,
                tw["qkvz"],
                weight_prefetch.load_prefetched(
                    "gdn_in", "gdn.qkvzab", mesh, fused, -1, _in_dtype, c("qkvzab.il"), _pc, tp
                ),
                _pc,
            )
    else:
        fused = torch.cat(
            [
                torch.cat([qkv_re[d * qkv_per : (d + 1) * qkv_per], z_w[d * z_per : (d + 1) * z_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        qkvz_mc = args.gdn_qkvz_weight_memcfg if qkvz_sharded else ttnn.DRAM_MEMORY_CONFIG
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=qkvz_mc,
            cache_path=c("qkvz" + (".dramshard" if qkvz_sharded else "")),
            dtype=_in_dtype,
        )
        # Separate A+B projection (column-parallel fallback)
        ab = torch.cat(
            [
                torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        tw["ab"] = tpc.shard_w(
            ab, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab"), dtype=_in_dtype
        )
    # Row-parallel out projection: DRAM-width-sharded (like the in-proj) — decode tput win.
    _out_sharded = getattr(args, "gdn_out_weight_memcfg", None) is not None
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.gdn_out_weight_memcfg if _out_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out.dramshard" if _out_sharded else "out"),
        dtype=_out_dtype,
    )
    if not _out_sharded and getattr(args, "proj_1d_decode", False):
        # QWEN36_WEIGHT_PREFETCH (gdn_out): receiver-contiguous decode copy for the Tensor prefetcher.
        _pc = getattr(args, "gdn_out_decode_1d_progcfg", None)
        weight_prefetch.attach_twin(
            mesh,
            tw["out"],
            weight_prefetch.load_prefetched(
                "gdn_out", "gdn.out_proj", mesh, sd[P + "out_proj.weight"], 0, ttnn.bfloat8_b, c("out"), _pc, tp
            ),
            _pc,
        )
    if getattr(args, "num_devices", 1) > 1:
        # COLUMN-parallel copy of the out-proj for prefill.
        # Decode keeps the row-sharded tw["out"] (matmul + all-reduce).
        tw["out_colpar"] = tpc.shard_w(
            sd[P + "out_proj.weight"],
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=c("out.colpar"),
            dtype=_out_dtype,
        )
    # Per-head params
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    # Conv taps (4), sharded per Q/K/V head grouping
    taps = tpc.prepare_conv_taps(conv1d_w, key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    if gdn_fused_decode_enabled() and args.gdn_conv_kernel_size == 4:
        # Fused single-user decode (QWEN36_GDN_FUSED_DECODE): the same constants the validated fused decoder
        # uses. dt_bias / -exp(A_log) stay fp32 (tw["dt_bias"] / tw["neg_exp_A"] are bf16), and the
        # sigmoid_gated_rms_norm op wants the norm weight as a rank-1 [Dv] bf16 TILE tensor (tw["norm_w"] is [1,1,Dv]).
        tw["dt_bias_fp32"] = _shard_small_fp32(sd[P + "dt_bias"], mesh, c("dt_bias_fp32"))
        tw["neg_exp_A_fp32"] = _shard_small_fp32(-torch.exp(sd[P + "A_log"].float()), mesh, c("neg_exp_A_fp32"))
        tw["norm_w_1d"] = ttnn.from_torch(
            sd[P + "norm.weight"].reshape(-1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if gdn_decode_step_op_enabled() and hasattr(ttnn.experimental, "kda"):
            # gdn_decode_step conv taps: per device the 4 taps packed into [Nv_tp, 4, 32, 32] head tiles (both
            # parities), sharded on dim 0 -> each device holds its own [Nv_tp, 4, 32, 32].
            nv_tp, nk_tp = nv // tp, nk // tp
            C_tp = 2 * nk_tp * dk + nv_tp * dv
            per_dev = torch.cat(
                [
                    pack_head_tiles(
                        [taps[j][d * C_tp : (d + 1) * C_tp] for j in range(4)], nv_tp, nk_tp, dk, dv, both_parities=True
                    )
                    for d in range(tp)
                ],
                dim=0,
            )
            tw["conv_taps_packed"] = ttnn.from_torch(
                per_dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # gdn_spec_step conv taps (QWEN36_VERIFY_SPEC_STEP): the SAME per-device tap rows as ONE [1, 4, C_tp] bf16
            # tile per device (tap j in row j), the fused spec verify's row-broadcast conv operand. Same bf16 rounding
            # of the same fp32 rows as the packed tiles, so the two ops see identical tap values.
            tap_rows = torch.stack(
                [torch.stack([taps[j][d * C_tp : (d + 1) * C_tp] for j in range(4)]) for d in range(tp)]
            )  # [tp, 4, C_tp]
            tw["conv_taps_rows"] = ttnn.from_torch(
                tap_rows.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
    return tw


class TPGatedDeltaNet:
    """Standalone TP GDN decode (per-device value-head recurrence + all-reduce)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        # DRAM-shard the row-parallel out projection (decode tput win; matches loader gate).
        self._out_sharded = getattr(self.args, "gdn_out_weight_memcfg", None) is not None
        self.B = args.max_batch_size
        self.Nk = args.gdn_nk_tp
        self.Nv = args.gdn_nv_tp
        self.Dk = args.gdn_dk
        self.Dv = args.gdn_dv
        self.qkv_dim_tp = args.gdn_qkv_dim_tp
        self.qkvz_dim_tp = args.gdn_qkvz_dim_tp
        self.key_dim_tp = args.gdn_key_dim_tp
        self.value_dim_tp = args.gdn_value_dim_tp
        # Flat q/k/v into adapter (skips prefill head-split reshapes)
        self._gdn_flat_qkv = True
        # Fuse adapter output relayout with rms_norm + head-flatten
        self._gdn_fuse_out = True
        self.gdn_program_config = getattr(args, "gdn_program_config", None)
        # WY-inverse arithmetic (ttnn.ChunkGdnWyInverse); None = the op's AUTO (the forward-substitution solve on Blackhole).
        self.gdn_wy_inverse = getattr(args, "gdn_wy_inverse", None)
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
        self.cfg = tpc.COMPUTE_HIFI2
        # In-proj decode matmul only (prefill keeps self.cfg = HiFi2)
        self.cfg_in = tpc.COMPUTE_LOFI if tpc.bfp4_gdn_in_enabled() else self.cfg
        # Must match load_gdn_weights_tp gates
        self._dram_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
        self._fuse_ab = self._dram_sharded
        # Fuse prefill norm-allgather + qkvzab in-proj into all_gather_minimal_matmul_async.
        # Requires the folded qkvzab weight; norm's post-AG is disabled in layer.py (GDN, prefill).
        self._fuse_agmm = self._fuse_ab
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1
        # PREFILL out-proj as column-parallel AG+matmul (takes precedence over the MMRS arm when the
        # col-sharded weight was loaded).
        self._out_colpar_prefill = "out_colpar" in tw
        # Pre-build chunk masks once (trace-safe; avoids from_torch inside captured trace)
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        # Prefill fused-op constant tiles, owned by this layer (avoids process-lifetime C++ cache vs device lifetime).
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

        self._fused_const_tiles = build_fused_const_tiles(mesh, _FUSED_CHUNK_SIZE)
        self.conv_states = None
        self.rec_state = None
        # ---- Batched (multi-user) spec-decode verify buffers, allocated by prepare_spec_verify. ----
        # (n_users, T) the spec buffers are sized for; None until prepare_spec_verify runs.
        self._spec_shape = None
        # Per-token recurrent-state ring, fp32 TILE [T*B*Nv, Dk, Dv]. The fused recurrent op reads each
        # (user, head)'s initial state from the block `state_blk_idx` names and writes every token's
        # state back in place, so a commit is just the host choosing the next iteration's index.
        self._spec_ring = None
        # E_prev: the conv window each verify consumes and rewrites, bf16 TILE [B, K-1+T, C].
        self._verify_win_buf = None
        # Zero rows [B, T-1, C] that seed_spec_state uses to fill _verify_win_buf's tail (never read:
        # the first replay runs with mi = 0).
        self._spec_win_pad = None
        # The durable shift register as one [B, K, qkv_dim_tp] tensor, mirroring conv_states[0..K-1]
        # with the user axis folded in (tap j of user u == _conv_win_buf[u, j, :]).
        self._conv_win_buf = None
        # One-hot matmul config: HiFi4 + fp32 accumulate keep a 0/1 selector matmul exact in bf16.
        self._cfg_onehot = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # Which of the two conv mirrors is authoritative. The traced batched verify advances only the
        # window buffers (refilling the K conv_states taps in the trace would cost ~10 ms/iteration
        # over 48 layers, and nothing in the spec loop reads them), so after a spec loop
        # the taps are behind: _conv_taps_stale. Anything that writes the taps from outside (prefill
        # capture_state, reset, slot edits, decode's own shift register) leaves the window behind:
        # _conv_win_stale. At most one is set; each setter clears the other. sync_conv_taps()
        # (window -> taps) runs at every tap consumer; sync_conv_win() (taps -> window) runs lazily
        # in _ensure_conv_win, which only ever runs eagerly.
        self._conv_taps_stale = False
        self._conv_win_stale = False
        # Spec decode only (set by SpeculativeDecoder): run the ONE fused recurrent device op in
        # forward_decode instead of the composite, so decode and hybrid verify share GDN math.
        # Full-batch (B == self.B) only — it has no bucketed B<Bmax state slice/writeback.
        self.use_fused_recurrent_decode = False
        self.conv_carry = None  # cross-chunk prefill conv carry [1, K-1, qkv_dim_tp]
        # Which causal conv runs the single-sequence prefill when valid_len is None (masked buckets always
        # keep the MAC FIR, whose one-hot new_state selection they need):
        #   QWEN_GDN_CONV=kda  (default) ttnn.experimental.kda.qkv_causal_conv1d_silu: conv + SiLU + q/k/v
        #                      split + tilize in ONE program (kda_conv_prefill);
        #   QWEN_GDN_CONV=fir  the shifted multiply-accumulate FIR everywhere.
        self._conv_impl = os.environ.get("QWEN_GDN_CONV", "kda")
        if self._conv_impl not in {"kda", "fir"}:
            raise ValueError(f"QWEN_GDN_CONV must be 'kda' or 'fir', got {self._conv_impl!r}")
        # The KDA op is fixed at four taps.
        self._gdn_kda_conv = self._conv_impl == "kda" and self.K == 4
        # KDA conv constants, allocated once by _ensure_kda_consts (host writes, so before any trace capture):
        # the op's actual_start scalar and the row-major zero history of a from-scratch chunk.
        self._kda_actual_start = None
        self._kda_zero_history = None
        self._kda_gather = {}  # (n_users, T) -> constant one-hot row gather for _kda_conv_packed
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        # Persistent init buffers the SHORT prefill trace copies into rec_state/conv_carry (zeros, or a carried/restored
        # state for a resumed prefill; see model._prepare_short_trace_gdn_init). Allocated before any trace capture.
        self._prefill_init_rec = None
        self._prefill_init_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill
        # Fused decode (QWEN36_GDN_FUSED_DECODE, default on). Decided here, once, so the state format is fixed for
        # the model's lifetime. max_batch_size == 1: the single-user path (_forward_decode_fused, active width 1);
        # max_batch_size > 1: the batched path (_fused_batched, every width 1..Bmax); otherwise the original path.
        #   _conv_hist_rm: the fused decode's conv history, [Bmax, K-1, qkv_dim_tp] ROW_MAJOR bf16 (Bmax == 1: the
        #                  single-user [1, K-1, C]; row u = user u in batched mode), the three raw
        #                  conv inputs preceding the next token, oldest first (the layout/order the kda conv op
        #                  reads). Allocated once (reset_state / first write) and only ever updated IN PLACE
        #                  (ttnn.copy), so the decode and prefill traces keep a fixed address. Decode maintains
        #                  it INSTEAD of conv_states; every code path that writes conv_states for decode
        #                  (prefill capture_state, assemble_batched_state, write_slot, forward_prefill_batched)
        #                  also refreshes it. It is deliberately NOT part of the model's per-binding state swaps
        #                  (model.py rebinds rec_state/conv_states/conv_carry on prefill scratch): with
        #                  max_batch_size == 1 there is one user, so the latest prefill IS the decode state.
        self._conv_hist_rm = None
        # Batched fused decode (max_batch_size 2..N): the same kernels over per-user 32-row blocks, see
        # _forward_decode_fused_batched. self.B is swapped by the model's scratch bindings (B=1 prefill scratch,
        # B=bg group scratch), so the decode capacity is remembered here: _conv_hist_rm is [_decode_B, K-1, C]
        # and only the decode binding (self.B == _decode_B) owns it. _scan_max_b: the widest decode width that
        # runs the fused chunk_gated_delta_rule (one scan core per value head per user); wider widths use the
        # kda conv + the original recurrent kernel on the same rec_state.
        self._decode_B = args.max_batch_size
        self._scan_max_b = self._fused_scan_max_width()
        self._nea_mask = {}  # per-width [1, 32*B, Nv] fp32 -exp(A_log) at each user's live row, 0 elsewhere
        self._fused_decode = self._fused_decode_supported(args, tw)
        self._fused_batched = self._fused_decode and self._decode_B > 1
        # gdn_decode_step decode op (QWEN36_GDN_DECODE_STEP_OP): conv + gating + recurrence + gated norm in ONE op over
        # a packed conv history [Bmax, Nv, 4, 32, 32] (_conv_hist_packed, updated in place by the op). The RM
        # history stays the canonical format of every non-decode writer; conversions are on-device embedding gathers.
        self._conv_hist_packed = None
        self._decode_step_fp32_state = gdn_decode_step_fp32_state_enabled()
        self._decode_op = self._decode_step_op_supported(args, tw)
        # QWEN36_VERIFY_SPEC_STEP=1: the batched spec verify AND its T = 1 seed run ttnn.experimental.kda.gdn_spec_step
        # -- ONE fused op per layer in place of the composite chains of _verify_fullbatch / _seed_fullbatch, with the
        # fast_mode / fp32_state arithmetic of the gdn_decode_step plain decode (same kernel family, so decode and
        # verify agree bit for bit per token). Default OFF: every OFF-path op below is untouched.
        self._spec_fused = self._spec_step_supported(args, tw)
        # Fused mode keeps E_prev as a ping-pong PAIR [W0, W1] of [B, K-1+T, C] instead of _verify_win_buf: a verify
        # reads pair[_spec_win_par] and writes the rebuilt window into the other half, and the host flips
        # _spec_win_par after every replay (model.verify_traced), so pair[_spec_win_par] is always the current E_prev
        # (verify_win_cur()). Every layer flips in lockstep: one ctrl page serves all of them.
        self._verify_win_pair = None
        self._spec_win_par = 0
        # Fused T = 1 seed constants (allocated eagerly by prepare_spec_verify): the scratch half of its [B, K, C]
        # window pair and its identity ctrl page (parity 0, mi 0, block u*Nv + h).
        self._seed_win_scratch = None
        self._seed_ctrl = None
        # Batched mode keeps TWO conv-state formats, one per decode width class (see prepare_decode_width):
        #   hist   _conv_hist_rm [Bmax, K-1, C] RM: read/written by the fused path (width <= SCAN_MAX)
        #   states conv_states[1..K-1] [1, Bmax, C] TILE: read/written by the original path (width > SCAN_MAX)
        # (conv_states[j] row u == hist[u, j-1]: after a shift-register step st[j]<-st[j+1], st[K-1]<-new, so
        # st[1..K-1] are the last K-1 raw inputs, oldest first; st[0] is overwritten by the shift before it is read.)
        # _conv_fmt names which formats are currently valid for the DECODE binding: "both" / "hist" / "states".
        # Whole-batch writers set "both"; partial writers (write_slot, remap_slots) write both formats for their rows
        # and leave it unchanged; prepare_decode_width(B) syncs if needed and sets it to the class it is about to run.
        # "packed" (decode_op only): _conv_hist_packed is the authoritative history (the op advances it); RM hist and
        # conv_states are stale. Every non-decode reader/writer first converts packed -> RM (_unpack_conv_hist).
        self._conv_fmt = "both"

    def _decode_step_op_supported(self, args, tw):
        # max_batch_size == 1 takes the op too (its single binding owns a [1, Nv, 4, 32, 32] packed history) instead of
        # the single-user fused kernels (_forward_decode_fused); QWEN36_GDN_DECODE_STEP_OP=0 reverts both.
        if not gdn_decode_step_op_enabled() or not self._fused_decode:
            return False
        reasons = []
        if not (1 <= self._decode_B <= 32):
            reasons.append(f"max_batch_size {self._decode_B} not in [1, 32]")
        if self.K != 4 or self.Dk != self.Dv or self.Nv % self.Nk or 2 * self.Nv > 32 or self.Nv > 16:
            reasons.append("unsupported GDN head geometry")
        if self.qkvz_dim_tp != self.qkv_dim_tp + self.Nv * self.Dv:
            reasons.append("qkvz_dim != qkv_dim + Nv*Dv")
        if not hasattr(ttnn.experimental.kda, "gdn_decode_step"):
            reasons.append("ttnn.experimental.kda.gdn_decode_step unavailable")
        if not (self._fuse_ab and getattr(args, "proj_1d_decode", False)):
            reasons.append("fused [qkv|z|a|b] 1D decode projection unavailable")
        if "conv_taps_packed" not in tw:
            reasons.append("conv_taps_packed missing from tw")
        if reasons:
            logger.info("[GDN] gdn_decode_step op NOT used: " + "; ".join(reasons))
            return False
        logger.info(
            "[GDN] decode path: gdn_decode_step op (QWEN36_GDN_DECODE_STEP_OP=0 reverts); "
            f"fp32_state={'on' if self._decode_step_fp32_state else 'off'} (QWEN36_GDN_STEP_FP32_STATE=0 reverts)"
        )
        return True

    def _spec_step_supported(self, args, tw):
        """QWEN36_VERIFY_SPEC_STEP=1 is an explicit opt-in: refuse loudly when the fused spec verify cannot run with
        the same arithmetic as the plain decode (anything else would silently fork decode from verify)."""
        if not gdn_verify_spec_step_enabled():
            return False
        reasons = []
        if not self._decode_op:
            reasons.append(
                "plain decode is not on gdn_decode_step (QWEN36_GDN_DECODE_STEP_OP / QWEN36_GDN_FUSED_DECODE off, or "
                "an unsupported geometry): verify and decode must share one kernel family"
            )
        if not (hasattr(ttnn.experimental, "kda") and hasattr(ttnn.experimental.kda, "gdn_spec_step")):
            reasons.append("ttnn.experimental.kda.gdn_spec_step unavailable in this ttnn build")
        if "conv_taps_rows" not in tw:
            reasons.append("conv_taps_rows missing from tw")
        if self.K != 4:
            reasons.append(f"conv kernel size {self.K} != 4")
        if reasons:
            raise RuntimeError(
                "QWEN36_VERIFY_SPEC_STEP=1 but the fused GDN spec verify cannot run: " + "; ".join(reasons)
            )
        _log_spec_step_once(
            "path",
            "[GDN] spec verify path: gdn_spec_step op (QWEN36_VERIFY_SPEC_STEP=1; fast_mode, "
            f"fp32_state={'on' if self._decode_step_fp32_state else 'off'}); plain decode stays on gdn_decode_step",
        )
        return True

    def _fused_scan_max_width(self):
        """Widest decode width (users) taking the fused chunk_gated_delta_rule: grid cores // Nv_tp (9 on a 11x10
        grid at Nv_tp=12), capped at the 32-row placement limit. QWEN36_GDN_FUSED_SCAN_MAXB overrides."""
        env = os.environ.get("QWEN36_GDN_FUSED_SCAN_MAXB")
        if env is not None:
            return max(0, min(32, int(env)))
        try:
            grid = self.mesh.compute_with_storage_grid_size()
            return max(1, min(32, (grid.x * grid.y) // max(1, self.Nv)))
        except Exception:  # noqa: BLE001 - mesh without a grid query (unit-test doubles)
            return 9

    def _fused_decode_supported(self, args, tw):
        """Whether this layer takes the fused single-user decode path (QWEN36_GDN_FUSED_DECODE). Logs one info line
        per process stating the active path."""
        reasons = []
        if not gdn_fused_decode_enabled():
            reasons.append("QWEN36_GDN_FUSED_DECODE=0")
        else:
            if self.K != 4:
                reasons.append(f"conv kernel size {self.K} != 4 (the kda conv op is fixed at four taps)")
            if self.Dk != self.Dv or self.Dv % tpc.TILE_SIZE or self.key_dim_tp % tpc.TILE_SIZE:
                reasons.append(f"head dims Dk={self.Dk}/Dv={self.Dv} unsupported by the flat chunk_gated_delta_rule")
            if os.environ.get("QWEN35_GDN_STATE_BF16") == "1" or os.environ.get("QWEN35_GDN_DECODE_BF16") == "1":
                reasons.append("QWEN35_GDN_STATE_BF16/QWEN35_GDN_DECODE_BF16 request the bf16 recurrence")
            if any(k not in tw for k in ("dt_bias_fp32", "neg_exp_A_fp32", "norm_w_1d")):
                reasons.append("fused-decode weights missing from tw (built only when the flag is on at load time)")
            if not hasattr(ttnn.experimental, "kda") or not hasattr(ttnn.experimental.kda, "sigmoid_gated_rms_norm"):
                reasons.append("ttnn.experimental.kda.sigmoid_gated_rms_norm unavailable in this ttnn build")
        if reasons:
            _log_fused_decode_once(
                "[GDN] decode path: ORIGINAL (shift-register conv + recurrent kernel): " + "; ".join(reasons)
            )
            return False
        if args.max_batch_size == 1:
            _log_fused_decode_once(
                "[GDN] decode path: FUSED for max_batch_size == 1 (qkv_causal_conv1d_silu + chunk_gated_delta_rule + "
                "sigmoid_gated_rms_norm; QWEN36_GDN_FUSED_DECODE=0 reverts). conv_states is not maintained during "
                "decode; out-proj input is bf16."
            )
        else:
            _log_fused_decode_once(
                f"[GDN] decode path: FUSED-BATCHED for max_batch_size={args.max_batch_size} (widths <= "
                f"SCAN_MAX={self._scan_max_b}: qkv_causal_conv1d_silu over per-user 32-row blocks + "
                "chunk_gated_delta_rule + sigmoid_gated_rms_norm on the RM conv history; wider widths run the ORIGINAL "
                "decode on conv_states; prepare_gdn_decode_width(B) syncs the two conv formats on a width-class "
                "change; QWEN36_GDN_FUSED_DECODE=0 reverts, QWEN36_GDN_FUSED_SCAN_MAXB overrides SCAN_MAX)."
            )
        return True

    def _release_spec_bufs(self):
        """Free the batched spec-verify buffers and forget their shape (prepare_spec_verify reallocs)."""
        for name in ("_spec_ring", "_verify_win_buf", "_spec_win_pad"):
            buf = getattr(self, name, None)
            if buf is not None:
                ttnn.deallocate(buf)
            setattr(self, name, None)
        for buf in getattr(self, "_verify_win_pair", None) or ():  # fused mode's window ping-pong pair
            ttnn.deallocate(buf)
        self._verify_win_pair = None
        self._spec_win_par = 0
        self._spec_shape = None

    def reset_state(self):
        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        # fp32 recurrent state by default (QWEN35_GDN_STATE_BF16=1 reverts)
        if os.environ.get("QWEN35_GDN_STATE_BF16") != "1":
            self.rec_state = ttnn.from_torch(
                torch.zeros(self.B, self.Nv, self.Dk, self.Dv, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        else:
            self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv))
        # Cross-chunk conv carry + persistent zero sources (created before any trace)
        self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        self._zero_conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_rec = z((self.B, self.Nv, self.Dk, self.Dv))
        if self._gdn_kda_conv or self._fused_decode:
            self._ensure_kda_consts()
        if self._fused_decode and self._owns_conv_hist():
            # Fused-decode conv history: allocated once, zeroed IN PLACE on every later reset (it may already be
            # baked into a decode/prefill trace, and reset_state also runs for the B=1 prefill scratch). In
            # batched mode only the decode binding (self.B == _decode_B) owns it: the B=1 / group scratch bindings
            # must not touch the decode users' history.
            _had_hist = self._conv_hist_rm is not None
            self._ensure_conv_hist()
            if _had_hist:
                ttnn.copy(self._hist_zero_source(), self._conv_hist_rm)
            self._conv_fmt = "both"  # zeros in both formats
            if self._decode_op:
                self._ensure_conv_hist_packed()  # allocated once here (host write), before any trace capture
                self._pack_tables()
            if self._fused_batched and not self._decode_op:
                # Persistent width constants (placement / selection matrices, zero blocks, masked -exp(A_log)),
                # allocated here (host writes) so no decode width allocates inside a trace capture.
                self._prepare_batched_widths()
        # Chunk-outer batched-prefill conv left-context (allocated lazily by forward_prefill_batched).
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
        self._batched_conv_carry = None
        # rec_state/conv_states have new addresses, so the spec buffers seeded from the old ones are
        # stale; the next prepare_spec_verify reallocates them.
        self._release_spec_bufs()
        if self._conv_win_buf is not None:  # mirrors the now-stale conv_states; re-seeded at capture
            ttnn.deallocate(self._conv_win_buf)
            self._conv_win_buf = None
        # Fresh (zero) taps and no window: neither mirror is behind.
        self._conv_taps_stale = False
        self._conv_win_stale = False

    def reset_state_inplace(self):
        """Zero conv + recurrent state in place (preserves trace buffer addresses).

        Copies from preallocated _zero_* buffers only — never allocates during an active trace.
        """
        # Drop any chunk-outer batched-prefill conv left-context so the next sequence starts clean.
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
            self._batched_conv_carry = None
        if self.conv_states is None:
            self.reset_state()
            return
        # Zero sources must exist (reset_state runs first; no lazy alloc during trace)
        assert (
            self._zero_conv0 is not None and self._zero_conv_carry is not None and self._zero_rec is not None
        ), "zero sources missing; reset_state must run before reset_state_inplace"
        for cs in self.conv_states:
            ttnn.copy(self._zero_conv0, cs)
        ttnn.copy(self._zero_rec, self.rec_state)
        # Zero cross-chunk conv carry for new sequence
        ttnn.copy(self._zero_conv_carry, self.conv_carry)
        if self._fused_decode and self._conv_hist_rm is not None and self._owns_conv_hist():
            ttnn.copy(self._hist_zero_source(), self._conv_hist_rm)  # preallocated RM zeros (no allocation here)
            self._conv_fmt = "both"  # conv_states zeroed above too
        # Taps are now the truth (zeros); the window mirror still holds the previous sequence's.
        self._conv_taps_stale = False
        self._conv_win_stale = True

    def _col_proj(self, x, weight, decode_progcfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_cfg=None):
        """Column-parallel qkvz projection; DRAM-sharded decode matmul when enabled.
        out_memory_config: decode result placement (default DRAM; L1 keeps it resident).
        compute_cfg: optional compute config (default self.cfg)."""
        cfg = self.cfg if compute_cfg is None else compute_cfg
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=cfg, memory_config=out_memory_config)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
            decode_out_memory_config=out_memory_config,
        )

    def _ensure_kda_consts(self):
        """Allocate the KDA conv path's constant tensors once. Host writes: reset_state calls this (the model
        runs it before capturing its traces); the lazy call in _kda_conv_prefill covers eager callers."""
        rep_map = ttnn.ReplicateTensorToMesh(self.mesh)
        if self._kda_actual_start is None:
            self._kda_actual_start = ttnn.from_torch(
                torch.tensor([0], dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep_map,
            )
        if self._kda_zero_history is None:
            self._kda_zero_history = ttnn.from_torch(
                torch.zeros(1, self.K - 1, self.qkv_dim_tp, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep_map,
            )

    def _kda_conv_prefill(self, qkv, T, conv_state, conv_sel=None):
        """kda_conv_prefill on this layer's taps and widths. conv_state: the previous chunk's carry
        [1, K-1, C] TILE, or None from scratch. conv_sel: optional persistent one-hot [1, K-1, T] selector of the last
        real rows (traced masked bucket, from-scratch only). Returns (q, k, v, new_state), see kda_conv_prefill."""
        self._ensure_kda_consts()
        history = conv_state if conv_state is not None else self._kda_zero_history
        kd, vd = self.key_dim_tp, self.value_dim_tp
        return kda_conv_prefill(
            qkv, T, history, self.tw["conv_taps"], (kd, kd, vd), self._kda_actual_start, conv_sel=conv_sel
        )

    # ------------------------------------------------------------------ #
    # Fused single-user decode: conv history (_conv_hist_rm) + its handoff from prefill.
    # ------------------------------------------------------------------ #
    def _owns_conv_hist(self):
        """Whether the CURRENT binding is the one that owns _conv_hist_rm: the decode binding. A max_batch_size == 1
        model has a single binding (B == 1 always); in batched mode the B=1 prefill scratch and the B=bg group
        scratch (model.py swaps self.B) must not allocate, zero or write the [_decode_B, K-1, C] history."""
        return self.B == self._decode_B

    def _hist_zero_source(self):
        """Persistent zero source for an in-place history reset (preallocated: no allocation under trace)."""
        if self._decode_B == 1:
            return self._kda_zero_history
        return self._shared_const(("zero_hist", self._decode_B, self.qkv_dim_tp), self._make_zero_hist)

    def _make_zero_hist(self):
        return self._rm_zeros((self._decode_B, self.K - 1, self.qkv_dim_tp))

    def _ensure_conv_hist(self):
        """The persistent [_decode_B, K-1, qkv_dim_tp] ROW_MAJOR bf16 conv history (row u = user u's K-1 previous raw
        conv inputs, oldest first; _decode_B == 1 is the single-user [1, K-1, C] buffer), allocated (zeros) on first
        use. Host write: reset_state calls it before any trace capture; the lazy callers run eagerly."""
        self._ensure_kda_consts()
        if self._conv_hist_rm is None:
            self._conv_hist_rm = self._rm_zeros((self._decode_B, self.K - 1, self.qkv_dim_tp))
        return self._conv_hist_rm

    def _rm_zeros(self, shape):
        """Replicated ROW_MAJOR bf16 DRAM zeros (host write)."""
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _hist_hook_active(self):
        """Writers of the decode conv state refresh the fused history only on the DECODE binding: B == 1 for a
        max_batch_size == 1 model (as before), B == _decode_B for a batched model, so the B=1 prefill scratch
        bindings never write a [1, K-1, C] row into the [_decode_B, K-1, C] history (write_slot does the per-slot
        write instead)."""
        return self._fused_decode and (
            (self._decode_B == 1 and self.B == 1) or (self._decode_B > 1 and self.B == self._decode_B)
        )

    def _write_conv_hist(self, last_inputs):
        """Handoff prefill -> fused decode (max_batch_size == 1): copy the last K-1 conv inputs [1, K-1, C] TILE
        (oldest first, the same tensor that seeds conv_states[1..K-1]) into the RM history, in place.
        Device-only (trace-safe once the history exists)."""
        hist = self._ensure_conv_hist()
        rm = ttnn.to_layout(last_inputs, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(rm, hist)
        ttnn.deallocate(rm)

    def _refresh_packed_single(self):
        """max_batch_size == 1 with the gdn_decode_step op: re-pack the just-written RM history into the packed history
        the op reads, so both formats are current, and tag "packed". Device ops only (the packed buffer and the pack
        tables exist since reset_state), so a traced prefill captures it too: a prefill trace replay, where no Python
        runs to update the tag, still leaves the packed history current for the next decode. No-op otherwise."""
        if self._decode_op and self._decode_B == 1 and self._needed_conv_fmt(1) == "packed":
            self._pack_conv_hist()
            self._conv_fmt = "packed"

    def _rows_to_rm(self, tiles):
        """list of [1, 1, C] TILE conv rows -> one [1, len, C] RM bf16 tensor (consumes nothing)."""
        rows = []
        for t in tiles:
            if t.dtype != ttnn.bfloat16:
                t = ttnn.typecast(t, ttnn.bfloat16)
            rows.append(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        stacked = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for r in rows:
            if r is not stacked:
                ttnn.deallocate(r)
        return stacked

    def _sync_conv_hist_from_states(self):
        """Rebuild the RM history from conv_states[1..K-1].

        max_batch_size == 1: each [1, 1, C] TILE -> the [1, K-1, C] history. Batched: each [1, B, C] TILE (row u =
        user u) -> the full [B, K-1, C] history, a FULL replace (every row), for the writers that produce
        conv_states for the whole batch (batched prefill, group assembly)."""
        hist = self._ensure_conv_hist()
        if self._decode_B == 1:
            stacked = self._rows_to_rm([self.conv_states[m] for m in range(1, self.K)])  # [1, K-1, C]
        else:
            B, C = self.B, self.qkv_dim_tp
            per_tap = []
            for m in range(1, self.K):  # [1, B, C] TILE -> [B, 1, C] RM
                t = self.conv_states[m]
                if t.dtype != ttnn.bfloat16:
                    t = ttnn.typecast(t, ttnn.bfloat16)
                rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                per_tap.append(ttnn.reshape(rm, (B, 1, C)))
            stacked = ttnn.concat(per_tap, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, K-1, C]
            for r in per_tap:
                ttnn.deallocate(r)
        ttnn.copy(stacked, hist)
        ttnn.deallocate(stacked)

    def sync_fused_conv_hist_from_states(self):
        """Public full-replace sync of the fused conv history from conv_states (for model.py writers that write
        dn.conv_states directly, e.g. _assemble_groups_gdn_dev). No-op unless this is the decode binding. Both
        formats are valid afterwards."""
        if self._hist_hook_active():
            self._sync_conv_hist_from_states()
            self._conv_fmt = "both"
            self._refresh_packed_single()

    def _sync_conv_states_from_hist(self):
        """hist -> states: conv_states[j] (j = 1..K-1, FULL Bmax rows) <- hist[:, j-1, :] as [1, Bmax, C] TILE, in
        place into the persistent tensors (conv_states[0] is overwritten by the shift before it is ever read).
        Eager; temporaries are freed immediately."""
        hist = self._ensure_conv_hist()
        B, C = self._decode_B, self.qkv_dim_tp
        for j in range(1, self.K):
            tap = self._slice_along(hist, 1, j - 1, j)  # [B, 1, C] RM
            row = ttnn.reshape(tap, (1, B, C))
            tile = ttnn.to_layout(row, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            src = tile
            if tile.dtype != self.conv_states[j].dtype:
                src = ttnn.typecast(tile, self.conv_states[j].dtype)
            ttnn.copy(src, self.conv_states[j])
            if src is not tile:
                ttnn.deallocate(src)
            ttnn.deallocate(tile)
            ttnn.deallocate(row)  # view of tap: frees tap's buffer

    # ------------------------------------------------------------------ #
    # gdn_decode_step op: packed conv history + on-device conversions to / from the RM history.
    # ------------------------------------------------------------------ #
    def _ensure_conv_hist_packed(self):
        """The persistent [_decode_B, Nv, 4, 32, 32] bf16 TILE packed history the op updates in place (zeros on first
        use; host write -> allocate before any trace capture)."""
        if self._conv_hist_packed is None:
            self._conv_hist_packed = ttnn.from_torch(
                torch.zeros(self._decode_B, self.Nv, 4, 32, 32, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._conv_hist_packed

    def _pack_tables(self):
        """Per-model uint32 index tables (+ a zero row) for the RM <-> packed embedding gathers. Both are slot-position
        aware: packed row (b, h, s, r) with s = 1..3 and r = 2c + (b & 1) holds chunk c of [q_hk | k_hk | v_h] of RM
        history row (b, s - 1); s = 0 / the other parity / r >= 2 * n_chunks are zero. (packed row 2c + (b&1) of tile
        (b, h, s) is the only place the op reads history from.)"""
        key = ("gdn_pack_tables", self._decode_B, self.Nv, self.Nk, self.Dk, self.Dv, self.K)

        def make():
            Bm, Nv, Nk, Dk, Dv, K = self._decode_B, self.Nv, self.Nk, self.Dk, self.Dv, self.K
            rf = Nv // Nk
            qn, vn = Dk // 32, Dv // 32  # 32-channel chunks per q/k head, per v head
            cpr = (2 * Nk * Dk + Nv * Dv) // 32  # RM chunk rows per (b, history row)
            nck = 2 * qn + vn  # chunks per head tile
            n_rm = Bm * (K - 1) * cpr
            zero = n_rm  # index of the appended zero row
            kd32 = Nk * Dk // 32

            def rm_chunk(h, c):
                hk = h // rf
                if c < qn:
                    return hk * qn + c
                if c < 2 * qn:
                    return kd32 + hk * qn + (c - qn)
                return 2 * kd32 + h * vn + (c - 2 * qn)

            to_packed = torch.full((Bm, Nv, 4, 32), zero, dtype=torch.int64)
            for b in range(Bm):
                for h in range(Nv):
                    for sl in range(1, 4):
                        for c in range(nck):
                            to_packed[b, h, sl, 2 * c + (b & 1)] = (b * (K - 1) + sl - 1) * cpr + rm_chunk(h, c)
            to_rm = torch.zeros((Bm, K - 1, cpr), dtype=torch.int64)
            for b in range(Bm):
                for j in range(K - 1):
                    for ch in range(cpr):
                        if ch < kd32:  # q
                            h, c = (ch // qn) * rf, ch % qn
                        elif ch < 2 * kd32:  # k
                            h, c = ((ch - kd32) // qn) * rf, qn + (ch - kd32) % qn
                        else:  # v
                            h, c = (ch - 2 * kd32) // vn, 2 * qn + (ch - 2 * kd32) % vn
                        to_rm[b, j, ch] = ((b * Nv + h) * 4 + j + 1) * 32 + 2 * c + (b & 1)

            def dev(t, dtype, layout):
                return ttnn.from_torch(
                    t,
                    dtype=dtype,
                    layout=layout,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            # Per-parity single-slot gather tables: packed rows (h, s, r) of ONE slot <- rows of a [(K-1)*cpr + 1, 32]
            # RM view of that user's [1, K-1, C] history (+ the appended zero row). Same entries as to_packed[b].
            slot_tabs = []
            for par in (0, 1):
                t = torch.full((Nv, 4, 32), (K - 1) * cpr, dtype=torch.int64)
                for h in range(Nv):
                    for sl in range(1, 4):
                        for c in range(nck):
                            t[h, sl, 2 * c + par] = (sl - 1) * cpr + rm_chunk(h, c)
                slot_tabs.append(dev(t.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT))

            return {
                "slot_to_packed": slot_tabs,
                "to_packed": dev(to_packed.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "to_rm": dev(to_rm.reshape(1, -1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
                "zero_row": dev(torch.zeros(1, 32, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                "cpr": cpr,
            }

        return self._shared_const(key, make)

    def _pack_conv_hist(self):
        """RM history -> packed history (in place into _conv_hist_packed): one embedding gather. Eager only (it must
        not be captured into a decode trace: the op owns the packed history while the trace replays)."""
        tab = self._pack_tables()
        Bm, K, cpr = self._decode_B, self.K, tab["cpr"]
        packed = self._ensure_conv_hist_packed()
        w = ttnn.reshape(self._ensure_conv_hist(), (Bm * (K - 1) * cpr, 32))
        w = ttnn.concat([w, tab["zero_row"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.embedding(tab["to_packed"], w, layout=ttnn.TILE_LAYOUT)  # [1, Bm*Nv*4*32, 32] -> tiles in order
        ttnn.deallocate(w)
        out = ttnn.reshape(out, (Bm, self.Nv, 4, 32, 32))
        ttnn.copy(out, packed)
        ttnn.deallocate(out)

    def _unpack_conv_hist(self):
        """packed history -> RM history (in place into _conv_hist_rm): one embedding gather. Eager only."""
        tab = self._pack_tables()
        Bm, K, C = self._decode_B, self.K, self.qkv_dim_tp
        hist = self._ensure_conv_hist()
        rm = ttnn.to_layout(self._conv_hist_packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.reshape(rm, (Bm * self.Nv * 4 * 32, 32))
        out = ttnn.embedding(tab["to_rm"], w, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, Bm*(K-1)*C/32, 32]
        out = ttnn.reshape(out, (Bm, K - 1, C))
        ttnn.copy(out, hist)
        ttnn.deallocate(out)
        ttnn.deallocate(rm)

    def _write_conv_hist_slot_packed(self, slot, convs):
        """write_slot under the "packed" format: write ONLY slot `slot`'s [1, Nv, 4, 32, 32] tiles of the packed history
        straight from the user's conv taps (convs[1..K-1], [1, 1, C] TILE), bit-identical to what _pack_conv_hist
        produces for that slot (rows 2c + (slot & 1) of tiles s = 1..3; zeros elsewhere). One embedding gather with
        the per-parity table + one in-place slice_write; the other slots' tiles are untouched. Eager only."""
        tab = self._pack_tables()
        packed = self._ensure_conv_hist_packed()
        rows = self._rows_to_rm([convs[m] for m in range(1, self.K)])  # [1, K-1, C] RM bf16
        w = ttnn.reshape(rows, ((self.K - 1) * tab["cpr"], 32))
        w = ttnn.concat([w, tab["zero_row"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self._inplace_slot_write():
            # A DRAM-interleaved TILE source makes slice_write untilize/retilize the whole destination and re-seat it
            # (new address: baked decode traces then read a stale buffer); an L1-sharded source writes tiles in place.
            out = ttnn.embedding(
                tab["slot_to_packed"][slot & 1],
                w,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self._packed_row_shard_memcfg(),
            )  # [1, Nv*4*32, 32]
            ttnn.deallocate(w)
            nr = self.Nv * 4
            out = ttnn.reshape(out, (1, nr, 32, 32))
            # The 5D TILE buffer viewed as [Bm, Nv*4, 32, 32] is the same buffer (a true view).
            ttnn.experimental.slice_write(
                out,
                ttnn.reshape(packed, (self._decode_B, nr, 32, 32)),
                [slot, 0, 0, 0],
                [slot + 1, nr, 32, 32],
                [1, 1, 1, 1],
            )
            ttnn.deallocate(out)
            return
        out = ttnn.embedding(tab["slot_to_packed"][slot & 1], w, layout=ttnn.TILE_LAYOUT)  # [1, Nv*4*32, 32]
        ttnn.deallocate(w)
        out = ttnn.reshape(out, (1, self.Nv, 4, 32, 32))
        ttnn.experimental.slice_write(out, packed, [slot, 0, 0, 0, 0], [slot + 1, self.Nv, 4, 32, 32], [1, 1, 1, 1, 1])
        ttnn.deallocate(out)

    @staticmethod
    def _inplace_slot_write():
        return os.environ.get("QWEN36_GDN_INPLACE_SLOT_WRITE", "1") == "1"

    def _packed_row_shard_memcfg(self):
        """L1 height-sharded TILE config for ONE packed-history row [1, Nv*4, 32, 32] (the sharded source of the in-place
        packed slot write)."""
        cfg = getattr(self, "_packed_row_shard_cfg", None)
        if cfg is None:
            grid_size = self.mesh.compute_with_storage_grid_size()
            n_tiles = self.Nv * 4
            num_cores = max(c for c in range(1, min(48, grid_size.x * grid_size.y) + 1) if n_tiles % c == 0)
            grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)
            cfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (n_tiles * 32 // num_cores, 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self._packed_row_shard_cfg = cfg
        return cfg

    def _remap_packed(self, idx):
        """remap_slots under "packed": ONE gather per layer over the packed tensor viewed as [Bm*Nv*4*32, 32] RM rows.
        Dest row (bd, h, s, r) <- src row (idx[bd], h, s, r') with r' = r - (bd & 1) + (idx[bd] & 1) for the valid
        history rows (s = 1..3, r & 1 == bd & 1, r < 2 * n_chunks), zero for everything else: exactly what
        unpack -> RM gather -> repack yields. Eager only (host index upload once per remap, shared by all layers)."""
        tab = self._pack_tables()
        Bm, Nv = self._decode_B, self.Nv
        key = tuple(idx)
        holder = self.tt_ccl if self.tt_ccl is not None else self
        cache = holder.__dict__.setdefault("_qwen36_gdn_remap_cache", {})
        # ONE persistent device table (allocated at warmup, before any trace capture), refreshed IN PLACE per distinct
        # remap: a table allocated lazily after the traces were captured can land on memory a trace replay overwrites.
        if cache.get("key") != key:
            nck = 2 * (self.Dk // 32) + self.Dv // 32
            zero = Bm * Nv * 4 * 32
            r = torch.arange(32)
            bd = torch.arange(Bm)
            bs = torch.tensor(idx, dtype=torch.int64)
            par_d, par_s = bd & 1, bs & 1
            ok = (r[None, :] < 2 * nck) & ((r[None, :] & 1) == par_d[:, None])  # [Bm, 32]
            src_r = r[None, :] - par_d[:, None] + par_s[:, None]  # [Bm, 32]
            h = torch.arange(Nv)
            sl = torch.arange(4)
            src = ((bs[:, None, None, None] * Nv + h[None, :, None, None]) * 4 + sl[None, None, :, None]) * 32 + src_r[
                :, None, None, :
            ]
            m = ok[:, None, None, :] & (sl[None, None, :, None] >= 1)
            tbl = torch.where(m, src, torch.full_like(src, zero))
            host = ttnn.from_torch(
                tbl.reshape(1, -1).to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            if "dev" not in cache:
                cache["dev"] = ttnn.from_torch(
                    tbl.reshape(1, -1).to(torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                ttnn.copy_host_to_device_tensor(host, cache["dev"])
            cache["key"] = key
        packed = self._ensure_conv_hist_packed()
        rm = ttnn.to_layout(packed, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.reshape(rm, (Bm * Nv * 4 * 32, 32))
        w = ttnn.concat([w, tab["zero_row"]], dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.embedding(cache["dev"], w, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(w)
        out = ttnn.reshape(out, (Bm, Nv, 4, 32, 32))
        ttnn.copy(out, packed)
        ttnn.deallocate(out)
        ttnn.deallocate(rm)

    def warmup_enter_packed(self):
        """Warmup only (before any trace capture): make the packed history the authoritative format so the packed slot
        write / packed remap programs (embedding -> TILE, 5D slice_write, ...) get compiled and their device-side
        program/kernel buffers allocated NOW instead of lazily after the traces were captured. Returns the previous
        conv format tag for warmup_exit_packed, or None when this layer has no packed path."""
        if not (self._hist_hook_active() and self._decode_B > 1 and self._needed_conv_fmt(self._decode_B) == "packed"):
            return None
        prev = self._conv_fmt
        self._pack_conv_hist()
        self._conv_fmt = "packed"
        return prev

    def warmup_exit_packed(self, prev):
        if prev is None:
            return
        self._unpack_conv_hist()
        self._conv_fmt = "both" if prev == "both" else "hist"

    def warmup_packed_sync(self):
        """Warmup only (spec engine, before any trace capture): run the seed's sync_conv_win from the PACKED conv format
        once, so its packed -> RM unpack (untilize, RM embedding gather) and the hist -> conv_states -> window copies are
        compiled now. A conv_states writer leaves that format behind whenever the gdn_decode_step op is the plain decode
        (_refresh_packed_single after a prefill, a packed slot write) -- with QWEN36_VERIFY_SPEC_STEP=1 also under the
        spec engine, whose first real seed then met an uncompiled program behind the post-capture compile guard.
        warmup_enter_packed covers only batched models (_decode_B > 1). The state is warmup state here (prepare_decode
        already seeds it); the round trip is content-neutral and leaves both formats valid ("both")."""
        if not (
            self._decode_op
            and self._conv_win_buf is not None
            and self.conv_states is not None
            and self.B == self._decode_B
            and self._conv_hist_rm is not None
            and self._needed_conv_fmt(self._decode_B) == "packed"
        ):
            return
        if self._conv_fmt != "packed":
            self._pack_conv_hist()
            self._conv_fmt = "packed"
        self.sync_conv_win()

    def _ensure_rm_hist_valid(self):
        """Before any non-decode reader/writer of the RM history / conv_states: when the packed history is the
        authoritative one, convert it back to RM (conv_states stays stale) and tag the format "hist"."""
        if self._conv_fmt == "packed":
            self._unpack_conv_hist()
            self._conv_fmt = "hist"

    def _needed_conv_fmt(self, B):
        if self.use_fused_recurrent_decode:  # spec decode: conv_states stays authoritative (original path)
            return "states"
        if self._decode_op and not self.use_fused_recurrent_decode and B <= 32:
            return "packed"
        return "hist" if B <= self._scan_max_b else "states"

    def _require_conv_fmt(self, need):
        """forward_decode guard (Python only, runs at eager steps / trace capture, not at replay): the format this
        width class reads must be valid. "both" (a writer ran, no prepare_decode_width yet) narrows to `need`,
        since this step makes the other format stale; a valid-other-only state needs the sync hook."""
        if self._conv_fmt == need:
            return
        if need == "packed":
            if self._conv_fmt == "both":  # eager first step after a whole-batch writer: RM -> packed here
                self._pack_conv_hist()
                self._conv_fmt = "packed"
                return
        else:
            self._ensure_rm_hist_valid()  # leaving the packed format: RM becomes the valid one ("hist")
            if self._conv_fmt == need:
                return
        if self._conv_fmt == "both":
            self._conv_fmt = need
            return
        raise RuntimeError(
            f"GDN conv state format is '{self._conv_fmt}' but this decode width needs '{need}': call "
            "prepare_gdn_decode_width(B) / prepare_decode_width(B) before decoding at a new width class"
        )

    def prepare_decode_width(self, B):
        """Make the conv state of the width class of `B` valid before decoding at width B (call once per width
        change, eagerly, outside any trace capture/replay). Returns True when a sync ran. No-op unless the batched
        fused mode is on. hist -> states and states -> hist are in-place full syncs of persistent buffers."""
        if self.B != self._decode_B:
            return False
        if not self._fused_batched:
            # max_batch_size == 1: only the gdn_decode_step op has a conv format to prepare (packed). The RM writers
            # re-pack as they go (_refresh_packed_single); this covers a history still tagged "both" / "hist".
            if not (self._decode_op and self._needed_conv_fmt(B) == "packed") or self._conv_fmt == "packed":
                return False
            self._pack_conv_hist()
            self._conv_fmt = "packed"
            return True
        need = self._needed_conv_fmt(B)
        synced = False
        if need == "packed":
            if self._conv_fmt != "packed":
                if self._conv_fmt == "states":
                    self._sync_conv_hist_from_states()
                self._pack_conv_hist()  # RM ("both" / "hist" / just synced from states) -> packed
                synced = True
            self._conv_fmt = "packed"
            return synced
        if self._conv_fmt == "packed":
            self._unpack_conv_hist()
            self._conv_fmt = "hist"
            synced = True
        if self._conv_fmt not in (need, "both"):
            if need == "hist":
                self._sync_conv_hist_from_states()
            else:
                self._sync_conv_states_from_hist()
            synced = True
        # The coming decode steps update only `need`; the other format goes stale.
        self._conv_fmt = need
        return synced

    def _write_conv_hist_users(self, per_user_last):
        """assemble_batched_state, batched mode: per_user_last[u] = [1, K-1, C] TILE last conv inputs of user u ->
        FULL replace of the [B, K-1, C] history (row u = user u), in place."""
        hist = self._ensure_conv_hist()
        rows = [
            ttnn.to_layout(
                c if c.dtype == ttnn.bfloat16 else ttnn.typecast(c, ttnn.bfloat16),
                ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for c in per_user_last
        ]
        stacked = rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(stacked, hist)
        for r in rows:
            if r is not stacked:
                ttnn.deallocate(r)
        ttnn.deallocate(stacked)

    def _write_conv_hist_slot(self, slot, convs):
        """write_slot, batched mode: write ONLY row `slot` of the history from the user's conv taps (convs[m] =
        [1, 1, C] TILE, m = 1..K-1 are the K-1 previous inputs oldest first). Must run BEFORE the caller consumes
        convs. In-place slice_write of the [1, K-1, C] row; the other (live) users' rows are untouched."""
        hist = self._ensure_conv_hist()
        row = self._rows_to_rm([convs[m] for m in range(1, self.K)])  # [1, K-1, C] RM
        ttnn.experimental.slice_write(row, hist, [slot, 0, 0], [slot + 1, self.K - 1, self.qkv_dim_tp], [1, 1, 1])
        ttnn.deallocate(row)

    # ------------------------------------------------------------------ #
    # Batched fused decode: per-width persistent constants.
    # ------------------------------------------------------------------ #
    def _shared_const(self, key, make):
        """Per-MODEL (not per-layer) constant cache: 48 GDN layers share one set of placement / selection matrices
        and zero blocks. Lives on the shared tt_ccl object (else args, else this layer), so it dies with the model
        instead of outliving the device in a process-level cache. `make` runs on first use (host write)."""
        for holder in (self.tt_ccl, self.args, self):
            if holder is None:
                continue
            try:
                cache = holder.__dict__.setdefault("_qwen36_gdn_batched_consts", {})
            except AttributeError:
                continue
            if key not in cache:
                cache[key] = make()
            return cache[key]
        return make()

    def _replicated(self, t, dtype, layout):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _batched_consts(self, B):
        """Width-B constants (shared across layers), built once (host writes: before any trace capture).

        Every user owns a 32-row block (the chunk size): rows 0..K-2 = its conv history, row K-1 = the live token,
        the rest zero. The live row is _FUSED_LIVE_ROW = K-1 = 3 in every block.
          P    [1, 32B, 32] bf16: placement, P[32u+3, u] = 1. P @ t places row u of a zero-padded [1, 32, N] tensor at
               the live row of block u and zeros elsewhere (exact copy: one nonzero bf16 product, fp32 accumulate).
          Q    [1, B, 32B] bf16: selection, Q[u, 32u+3] = 1. Q @ t compacts the live rows of a [1, 32B, N] tensor.
          ind  [1, 32B, Nv] fp32: 1 at every live row (folded into -exp(A_log) per layer, see _nea_mask_for).
          zeros [B, 32-K, C] bf16 RM: the zero tail of each block of the conv input.
        All four exist only for widths the fused path handles (B <= _scan_max_b); wider widths run the original decode.
        """
        live, blk = self.K - 1, 32

        def make():
            c = {}
            if B <= self._scan_max_b:
                P = torch.zeros(1, blk * B, blk, dtype=torch.bfloat16)
                ind = torch.zeros(1, blk * B, self.Nv, dtype=torch.float32)
                for u in range(B):
                    P[0, blk * u + live, u] = 1.0
                    ind[0, blk * u + live, :] = 1.0
                c["P"] = self._replicated(P, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                c["ind"] = self._replicated(ind, ttnn.float32, ttnn.TILE_LAYOUT)
                Q = torch.zeros(1, B, blk * B, dtype=torch.bfloat16)
                for u in range(B):
                    Q[0, u, blk * u + live] = 1.0
                c["Q"] = self._replicated(Q, ttnn.bfloat16, ttnn.TILE_LAYOUT)
                c["zeros"] = self._rm_zeros((B, blk - self.K, self.qkv_dim_tp))
            return c

        return self._shared_const(("widths", B, self.K, self.Nv, self.qkv_dim_tp, self._scan_max_b), make)

    def _nea_mask_for(self, B):
        """This layer's [1, 32B, Nv] fp32 -exp(A_log) at each user's live row and exactly 0 at every other row, so
        g = nea_mask * softplus(a_placed + dt_bias) is exactly 0 on the padded steps (identity recurrence steps).
        Built once per width (device op: eager, before any trace capture)."""
        if B not in self._nea_mask:
            self._nea_mask[B] = ttnn.mul(
                self._batched_consts(B)["ind"], self.tw["neg_exp_A_fp32"], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        return self._nea_mask[B]

    def _decode_widths(self):
        """Every width a bucketed decode can run: powers of two below the capacity, and the capacity itself."""
        widths, w = [], 1
        while w < self._decode_B:
            widths.append(w)
            w *= 2
        widths.append(self._decode_B)
        return widths

    def _prepare_batched_widths(self):
        """Eagerly build the constants of every decode width (idempotent) so that no width allocates inside a trace
        capture. Other widths (non-bucket callers) are built lazily on first use, which is only safe eagerly."""
        for w in self._decode_widths():
            if w <= self._scan_max_b:
                self._batched_consts(w)
                self._nea_mask_for(w)

    def _matmul_cfg(self):
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _place_live_rows(self, t, B, dtype):
        """[1, B, N] TILE (bf16) -> [1, 32B, N] TILE `dtype` DRAM: row u lands on the live row of user u's 32-row
        block, every other row is exactly zero. CONSUMES t.

        t is first padded to the 32-row tile height with ttnn.pad, which (as in the B=1 path) returns a VIEW of
        the same buffer and zero-fills the implicit tile padding in place, so the matmul's K padding is zero (a
        NaN/Inf in stale tile padding would otherwise poison the one-hot product). The view aliases t: exactly ONE
        of the two handles is deallocated (after the matmul), never both. HiFi4 + fp32 accumulate: the product of a
        bf16 value with 1.0 (plus exact zeros) is an exact copy; fp32 out is exact for a bf16 source."""
        tp = t
        same = True
        if B < 32:
            tp = ttnn.pad(t, [(0, 0), (0, 32 - B), (0, 0)], value=0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Decide aliasing while BOTH handles are alive (a view shares t's buffer).
            same = (tp is t) or (tp.buffer_address() == t.buffer_address())
        out = ttnn.matmul(
            self._batched_consts(B)["P"],
            tp,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._matmul_cfg(),
        )
        ttnn.deallocate(tp)  # frees the shared buffer when tp is a view of t (or is t)
        if not same:
            ttnn.deallocate(t)  # pad returned a fresh buffer (not a view): free the source too
        return out

    def _select_live_rows(self, t, B, dtype, memory_config):
        """[1, 32B, N] TILE -> [1, B, N]: row u = live row of user u's block (exact one-hot select). CONSUMES t."""
        out = ttnn.matmul(
            self._batched_consts(B)["Q"],
            t,
            dtype=dtype,
            memory_config=memory_config,
            compute_kernel_config=self._matmul_cfg(),
        )
        ttnn.deallocate(t)
        return out

    def _fused_conv_batched(self, qkv, B):
        """Causal conv1d + SiLU + q/k/v split for width B on the persistent unified history, ONE kda program.

        qkv [1, B, C] TILE (consumed). Builds the RM input joined[u] = [hist[u] (K-1 rows) | this token | zeros] =
        32 rows per user, flattened to [1, 32B, C], and runs the op with a ZERO history: the live token of every user
        sees its own K-1 previous inputs as ordinary preceding rows, and the rows of one user's block never feed
        another user's live row. Returns q [1, 32B, kd], k, v [1, 32B, vd] bf16 TILE DRAM (live row = 3 of each 32-row
        block) and updates the history IN PLACE (rows [0:B]): joined[:, 1:K] = [hist[1:], this token]. Trace-safe:
        no persistent allocation, the history address is fixed."""
        _DRAM = ttnn.DRAM_MEMORY_CONFIG
        C, kd, vd = self.qkv_dim_tp, self.key_dim_tp, self.value_dim_tp
        Kh = self.K - 1
        hist = self._conv_hist_rm
        assert hist is not None, "fused batched decode needs the conv history (reset_state on the decode binding)"
        zeros = self._batched_consts(B)["zeros"]
        if qkv.dtype != ttnn.bfloat16:
            qkv = ttnn.typecast(qkv, ttnn.bfloat16)
        rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)  # [1, B, C] RM
        ttnn.deallocate(qkv)
        row = ttnn.reshape(rm, (B, 1, C))  # RM view: user u's token as its own row
        hist_b = hist if B == self._decode_B else ttnn.slice(hist, (0, 0, 0), (B, Kh, C), memory_config=_DRAM)
        joined = ttnn.concat([hist_b, row, zeros], dim=1, memory_config=_DRAM)  # [B, 32, C] RM
        if hist_b is not hist:
            ttnn.deallocate(hist_b)
        ttnn.deallocate(row)  # frees rm (view)
        flat = ttnn.reshape(joined, (1, 32 * B, C))  # RM view
        zero_hist = self._kda_zero_history  # [1, K-1, C]
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            flat,
            zero_hist,
            *self.tw["conv_taps"],
            kd,
            kd,
            vd,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=kda_channel_chunk_size(C, cap=256)),
            actual_start=self._kda_actual_start,
            predecessor_carry=zero_hist,  # no sequence parallelism: alias history (see kda_conv_prefill)
            memory_config=_DRAM,
        )
        # Next history = the K-1 newest raw inputs of each user = block rows 1..K-1 (hist[1:], this token).
        new_hist = ttnn.slice(joined, (0, 1, 0), (B, self.K, C), memory_config=_DRAM)  # [B, K-1, C] RM
        ttnn.deallocate(joined)  # frees flat (view)
        if B == self._decode_B:
            ttnn.copy(new_hist, hist)
        else:
            # Prefix write: rows [0:B] only, idle users' history untouched. RM interleaved slice_write, in place.
            ttnn.experimental.slice_write(new_hist, hist, [0, 0, 0], [B, Kh, C], [1, 1, 1])
        ttnn.deallocate(new_hist)
        return q, k, v

    def _forward_decode_fused_batched(self, x, decode_ar=None):
        """Fused decode at width B (1 <= B <= _scan_max_b) of a max_batch_size > 1 model (state rows [0:B]).

        Same kernels and math as _forward_decode_fused, run over B users at once: every user owns a 32-row block
        with its live token at row K-1 = 3 (see _batched_consts); g / beta / z are placed there with exact one-hot
        matmuls and are ZERO on every other row (g = beta = 0 -> identity recurrence steps), so the single chunk_
        gated_delta_rule over [B, 32] tokens updates each user's rec_state row by exactly one real step. The gated
        norm output is compacted back to [1, B, value_dim] with a one-hot select and goes through the unchanged
        out-proj + all-reduce. rec_state is [Bmax, Nv, Dk, Dv] fp32 in the SAME format as the original path (rows
        [0:B] updated in place), so widths can be switched between steps. conv_states is not maintained; the conv
        history is _conv_hist_rm. out-proj input is bf16 (as the B=1 fused path)."""
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE

        tw, Nv = self.tw, self.Nv
        _L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        kd, vd = self.key_dim_tp, self.value_dim_tp
        Bmax = self.B
        B = x.shape[-2]
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        assert 1 <= B <= self._scan_max_b and Bmax == self._decode_B

        qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)

        # ---- z / beta / g placed on the live row of each user's 32-row block, zero elsewhere ----
        if z.dtype != ttnn.bfloat16:  # sigmoid_gated_rms_norm's gate contract
            z = ttnn.typecast(z, ttnn.bfloat16, memory_config=_L1)
        z_p = ttnn.reshape(self._place_live_rows(z, B, ttnn.bfloat16), (B, _FUSED_CHUNK_SIZE, vd))
        beta = ttnn.sigmoid(b, memory_config=_L1)  # sigmoid BEFORE placement: padded rows must stay exactly 0
        ttnn.deallocate(b)
        beta_p = ttnn.reshape(self._place_live_rows(beta, B, ttnn.bfloat16), (B, _FUSED_CHUNK_SIZE, Nv))
        if a.dtype != ttnn.bfloat16:
            a = ttnn.typecast(a, ttnn.bfloat16, memory_config=_L1)
        a_p = self._place_live_rows(a, B, ttnn.float32)  # fp32 [1, 32B, Nv]: exact copy of the bf16 `a`
        a_dt = ttnn.add(a_p, tw["dt_bias_fp32"], memory_config=_DRAM)
        ttnn.deallocate(a_p)
        # nea_mask is -exp(A_log) at live rows and 0 elsewhere: g is exactly 0 on every padded row.
        g = ttnn.mul(
            self._nea_mask_for(B),
            a_dt,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            memory_config=_DRAM,
        )
        ttnn.deallocate(a_dt)
        g_p = ttnn.reshape(g, (B, _FUSED_CHUNK_SIZE, Nv))

        # ---- causal conv1d + SiLU + q/k/v split over the per-user blocks ----
        q, k, v = self._fused_conv_batched(qkv, B)
        q = ttnn.reshape(q, (B, _FUSED_CHUNK_SIZE, kd))
        k = ttnn.reshape(k, (B, _FUSED_CHUNK_SIZE, kd))
        v = ttnn.reshape(v, (B, _FUSED_CHUNK_SIZE, vd))

        # ---- gated delta rule over [B, 32] tokens: fp32 state, rows [0:B] in place ----
        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
        eye, tril, ones, masks = self._fused_const_tiles
        o, new_rec = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g_p,
            beta_p,
            scale=self.scale,
            initial_state=init_state,
            output_final_state=True,
            chunk_size=_FUSED_CHUNK_SIZE,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        if init_state is not self.rec_state:
            ttnn.deallocate(init_state)
        for t in (q, k, v, g_p, beta_p):
            ttnn.deallocate(t)
        if B == Bmax:
            ttnn.copy(new_rec, self.rec_state)
            ttnn.deallocate(new_rec)
        else:
            self._write_recurrent_state_prefix(new_rec, B)

        # ---- gated RMSNorm (no +1) * silu(z) over the blocks, then compact the live rows ----
        normed = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o, z_p, tw["norm_w_1d"], Nv, epsilon=1e-6, output_dtype=ttnn.bfloat16
        )  # [B, 32, vd]
        ttnn.deallocate(o)
        gated = ttnn.mul(normed, z_p, memory_config=_DRAM)
        ttnn.deallocate(normed)
        ttnn.deallocate(z_p)
        gated = self._select_live_rows(ttnn.reshape(gated, (1, _FUSED_CHUNK_SIZE * B, vd)), B, ttnn.bfloat16, _DRAM)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is bf16 here (as the B=1 fused path)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forward_decode_step_op(self, x, decode_ar=None):
        """Decode at width B (1 <= B <= _decode_B <= 32; state rows / slots [0:B]) through ONE op,
        ttnn.experimental.kda.gdn_decode_step: causal conv (packed history, updated in place) + SiLU + q/k/v split +
        gating (beta, g) + recurrence on rec_state rows [0:B] (in place) + gated RMSNorm * silu(z). Then the unchanged
        out-proj + all-reduce. Needs the "packed" conv format (see _require_conv_fmt)."""
        tw = self.tw
        B = x.shape[-2]
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        assert 1 <= B <= self._decode_B == self.B
        # the fused [qkv|z|a|b] decode projection (1D matmul) straight into L1; the op consumes the whole row.
        qkvzab = tpc.matmul_1d_decode(
            x,
            tw["qkvz"],
            self.args.gdn_qkvz_decode_1d_progcfg,
            self.cfg_in,
            out_memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        qkvzab = ttnn.reshape(qkvzab, (1, B, qkvzab.shape[-1]))
        gated = ttnn.experimental.kda.gdn_decode_step(
            qkvzab,
            tw["dt_bias_fp32"],
            tw["neg_exp_A_fp32"],
            self.rec_state,
            tw["norm_w_1d"],
            self.Nv,
            self.Nk,
            self.Dk,
            self.Dv,
            scale=self.scale,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=ttnn.bfloat16,
            conv_hist=self._conv_hist_packed,
            conv_taps=tw["conv_taps_packed"],
            qkvz_dim=self.qkvz_dim_tp,
            fast_mode=True,
            fp32_state=self._decode_step_fp32_state,
        )  # [1, B, Nv*Dv] bf16
        ttnn.deallocate(qkvzab)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is bf16 here (as the fused paths)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def snapshot_fused_decode_state(self):
        """Host copy of the fused-decode conv history (None when the fused path is off). conv_states is NOT the
        decode conv state when the fused path is on, so a caller that snapshots/restores GDN state around a
        throwaway decode run (demo trace capture) must also snapshot/restore this: pair with
        restore_fused_decode_state. Mirrors the demos' rec_state/conv_states host snapshot."""
        if not self._fused_decode or self._conv_hist_rm is None:
            return None
        if self._conv_fmt == "packed":
            self._unpack_conv_hist()  # RM becomes current too (fmt stays "packed")
        return ttnn.to_torch(self._conv_hist_rm, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0))

    def restore_fused_decode_state(self, snap):
        """In-place restore (fixed address, trace-safe) of a snapshot_fused_decode_state result."""
        if snap is None:
            return
        src = ttnn.from_torch(
            snap,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.copy(src, self._conv_hist_rm)
        ttnn.deallocate(src)
        self._conv_fmt = "hist"  # conv_states is not part of the snapshot
        self._refresh_packed_single()  # max_batch_size == 1 + decode op: re-pack, tag "packed"

    def _kda_conv_packed(self, E, T):
        """KDA conv + SiLU + q/k/v split for B independent windows in one op call.

        E: [B, L, C] bf16 TILE, L = K-1+T rows per user ([K-1 carry rows ; T token rows]); read, not
        consumed. The KDA op is batch-1 only, so the B windows go end to end into one [1, B*L, C]
        sequence, zero-padded to a tile multiple, with a zero history. Output row u*L + (K-1) + j reads
        input rows u*L + j .. u*L + j + K-1, all inside user u's window, so every kept row is
        bit-identical to a per-user call. The rows at the carry positions straddle two users; the
        constant one-hot gather (HiFi4, fp32 acc: exact) drops them. The gather is built (host write)
        on the first call per (B, T), so that call must run before trace capture.
        Returns q [1, B*T, key_dim_tp], k [1, B*T, key_dim_tp], v [1, B*T, value_dim_tp], user-major.
        """
        B, L, C = E.shape[0], E.shape[1], self.qkv_dim_tp
        K = self.K
        assert L == K - 1 + T, f"window has {L} rows, expected K-1+T = {K - 1 + T}"
        Lp = -(-(B * L) // tpc.TILE_SIZE) * tpc.TILE_SIZE
        self._ensure_kda_consts()
        mc = ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.to_layout(E, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        x = ttnn.reshape(x, (1, B * L, C))
        if Lp != B * L:
            xp = ttnn.pad(x, [(0, 0), (0, Lp - B * L), (0, 0)], 0.0)
            ttnn.deallocate(x)
            x = xp
        # x is already ROW_MAJOR, so kda_conv_prefill's to_layout is a no-op and it frees x itself.
        q, k, v, _ = kda_conv_prefill(
            x,
            Lp,
            self._kda_zero_history,
            self.tw["conv_taps"],
            (self.key_dim_tp, self.key_dim_tp, self.value_dim_tp),
            self._kda_actual_start,
            emit_state=False,
        )
        G = self._kda_gather.get((B, T))
        if G is None:
            G = ttnn.from_torch(
                kda_pack_gather(B, T, K, Lp),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=mc,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            self._kda_gather[(B, T)] = G
        outs = []
        for t in (q, k, v):
            outs.append(ttnn.matmul(G, t, compute_kernel_config=self._cfg_onehot, memory_config=mc))
            ttnn.deallocate(t)
        return tuple(outs)

    def _row_proj(self, x, weight):
        """Row-parallel out projection: DRAM-sharded decode/prefill matmul (K=gdn_value_dim_tp),
        matching the in-proj. Falls back to plain interleaved on single device (no sharded memcfg)."""
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            return tpc.matmul_1d_decode(
                x, weight, self.args.gdn_out_decode_1d_progcfg, self.cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if not self._out_sharded:
            if x.shape[-2] > tpc.TILE_SIZE:
                # Prefill non-fused arm (single device, or out-sharded): tuned 2D config vs ttnn-auto.
                # fp32 [seq,dim] output too big for L1 (42MB) -> DRAM out; separate tt_all_reduce does the RS.
                # max_cols = device width (11 on BH): wide grid (~10-wide), fp32-neutral.
                pc = tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x, weight, compute_kernel_config=self.cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            self.args.gdn_out_progcfg,
            self.args.act_shard_gdn_value,
            self.args.prefill_progcfg,
            self.args.gdn_value_dim_tp,
        )

    def _project_qkvzab(self, x, S, out_mc=None):
        """Project x → (qkv, z, a, b). Fused path: one [qkv|z|a|b] matmul then slice.
        out_mc: placement of the qkvzab matmul + slices. None → DRAM; prefill+decode now pass L1 to
        keep qkvzab + q/k/v/z/a/b resident (was DRAM to spare NoC traffic — re-measure if reverting)."""
        Nv, qz, az = self.Nv, self.qkv_dim_tp, self.qkvz_dim_tp
        _proj_mc = out_mc if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG
        if self._fuse_ab:
            # Prefill: x is K-sharded (norm skipped its AG) -> fused all-gather + qkvzab matmul.
            if self._fuse_agmm and S > tpc.TILE_SIZE:
                qkvzab = tpc.all_gather_matmul_prefill(
                    x,
                    self.tw["qkvz"],
                    self.tt_ccl,
                    self.cfg,
                    self.args.ccl_topology(),
                    out_memory_config=_proj_mc,
                )
                qkvzab = ttnn.reshape(qkvzab, (1, S, qkvzab.shape[-1]))
            elif getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:
                # Decode: small-grid 1D matmul on the interleaved fused weight (beats the DRAM-sharded grid).
                qkvzab = tpc.matmul_1d_decode(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvz_decode_1d_progcfg,
                    self.cfg_in,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                qkvzab = self._col_proj(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvzab_progcfg,
                    out_memory_config=_proj_mc,
                    compute_cfg=self.cfg_in if S <= tpc.TILE_SIZE else self.cfg,
                )
            qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz), memory_config=out_mc)
            # z (output gate) lives across the chunk kernel (gated = out_f * silu(z)); L1 z (6MB@S=2048)
            # clashes with the scan kernel CBs -> keep DRAM in chunk-prefill; decode (small S) keeps out_mc.
            _z_mc = ttnn.DRAM_MEMORY_CONFIG if (self._fuse_agmm and S > tpc.TILE_SIZE) else out_mc
            z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az), memory_config=_z_mc)
            # a,b end mid-tile; slicing straight from qkvzab untilizes the full 4120-wide tensor.
            # Grab the enclosing tile-aligned block once (no untilize), then split a/b from it (test_gdn_slice_opt).
            _ab_end = min(az + -(-2 * Nv // tpc.TILE_SIZE) * tpc.TILE_SIZE, qkvzab.shape[-1])  # 2*Nv up to a tile
            ab = ttnn.slice(qkvzab, (0, 0, az), (1, S, _ab_end), memory_config=out_mc)
            ttnn.deallocate(qkvzab)
            a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
            b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv), memory_config=out_mc)
            ttnn.deallocate(ab)
            return qkv, z, a, b
        _cfg_in = self.cfg_in if S <= tpc.TILE_SIZE else self.cfg
        qkvz = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvz_progcfg, compute_cfg=_cfg_in)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, S, qz))
        z = ttnn.slice(qkvz, (0, 0, qz), (1, S, az))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, self.tw["ab"], compute_kernel_config=_cfg_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv))
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def forward_prefill(
        self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False, prefill_masks=None
    ):
        """Causal chunk-prefill from scratch. x [1,1,T,dim]: K-sharded (dim/tp per device) when the
        fused in-proj AG-matmul path is active (``_fuse_agmm`` and T>TILE — the norm skips its
        post-AG); replicated otherwise. Output reduce-scattered.

        valid_len: real token count (rest is padding). capture_state: save rec/conv state for decode.
        return_state: when True (per-user batched prefill), return
        ``(output, final_state, conv_new_state)`` for one user's from-scratch B=1
        pass and skip all self.* writeback; the caller stitches per-user states via
        assemble_batched_state(). Single-sequence behavior is unchanged when False.
        prefill_masks: traced masked-bucket prefill (valid_len stays None; the real length lives in PERSISTENT device
        tensors the caller rewrites per request): dict with "bg" [1,T,1] fp32 TILE (1 for t < valid_len; multiplied
        into beta/g), "conv_sel" [1,K-1,T] bf16 TILE one-hot of the last K-1 real conv-input rows (conv_new_state),
        and optionally "qkv" (q/k/v mask, parity only). Requires the KDA conv and a from-scratch (zeroed) carry.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        T = x.shape[1]
        # Pass the RAW valid_len (may be None) to the conv-FIR / seq kernels below — NOT a
        # `valid_len or T` coercion. A full chunk (valid_len is None) must take the kernels'
        # valid_len-None path (a static last-(K-1) slice for the conv state), which is trace-safe;
        # the valid_len-set path builds a one-hot via ttnn.from_torch (a host write) that TT_FATALs
        # ("Writes are not supported during trace capture") inside the captured chunk-outer trace.
        # Masked buckets still pass a real valid_len (< T) so their exact masking is unchanged, and
        # for a full chunk the None slice and the valid_len==T one-hot select the identical rows.

        # Cross-chunk carry (chunk-outer prefill): the recurrent + conv state continue from the
        # persistent buffers (zeroed at sequence start by reset_state_inplace, so a from-scratch
        # single pass reads zeros == None).
        # Per-user prefill (return_state) is always from scratch: must not carry the shared
        # batched buffer (other users' state) as its initial recurrent/conv state.
        carry = not return_state
        if carry and self.conv_carry is None:
            self.reset_state()

        # Prefill qkvzab in L1: keeps proj + q/k/v/z/a/b resident for conv+gate prep.
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=ttnn.L1_MEMORY_CONFIG if tpc.prefill_l1_rows_ok(T) else None)

        # Causal conv + SiLU; conv_state = previous chunk's last K-1 inputs (None/zero from scratch).
        # q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes it.
        _cstate = self.conv_carry if carry else None
        kd = self.key_dim_tp
        if self._gdn_kda_conv and valid_len is None:
            # KDA op: conv + SiLU + q/k/v split in one program; its outputs are already the three
            # token-major tensors the flat-qkv path wants (masked buckets keep the MAC FIR below).
            q, k, v, conv_new_state = self._kda_conv_prefill(
                qkv, T, _cstate, conv_sel=None if prefill_masks is None else prefill_masks["conv_sel"]
            )
            ttnn.deallocate(qkv)
            if self._gdn_flat_qkv:
                _qkv_head_dims = (Nk, Dk, Nv, Dv)
            else:
                q = ttnn.reshape(q, (1, T, Nk, Dk))
                k = ttnn.reshape(k, (1, T, Nk, Dk))
                v = ttnn.reshape(v, (1, T, Nv, Dv))
                _qkv_head_dims = None
        else:
            assert prefill_masks is None, "persistent prefill_masks need the KDA conv (QWEN_GDN_CONV=kda, K=4)"
            # The MAC FIR: masked buckets (their one-hot new_state selection) and QWEN_GDN_CONV=fir.
            conv, conv_new_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                self.K,
                self.mesh,
                # Conv in L1 (output freed before chunk kernel; new_state lands in DRAM internally)
                memory_config=tpc.l1_or_dram(T),
                conv_state=_cstate,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                valid_len=valid_len,
            )
            ttnn.deallocate(qkv)
            if self._gdn_flat_qkv:
                # Flat q/k/v: adapter splits heads inside untilize
                q = ttnn.slice(conv, (0, 0, 0), (1, T, kd))
                k = ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd))
                v = ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp))
                _qkv_head_dims = (Nk, Dk, Nv, Dv)
            else:
                q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
                k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
                v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
                _qkv_head_dims = None
            ttnn.deallocate(conv)
        # GQA late-expand: adapter L2-norms at Nk, expands to Nv after
        beta = ttnn.reshape(ttnn.sigmoid(b), (1, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"])), (1, T, Nv))
        ttnn.deallocate(a)

        # Fused chunk_gated_delta_rule; also used for masked valid_len.
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
            chunk_gated_delta_rule_fused_adapter,
            fused_chunk_enabled,
        )

        _use_fused = fused_chunk_enabled()
        _delta_fn = chunk_gated_delta_rule_fused_adapter if _use_fused else chunk_gated_delta_rule_seq_adapter
        # const_tiles / program_config / wy_inverse only apply to the fused op; the seq adapter has none of them.
        _extra = (
            {
                "const_tiles": self._fused_const_tiles,
                "program_config": self.gdn_program_config,
                "wy_inverse": self.gdn_wy_inverse,
            }
            if _use_fused
            else {}
        )
        if prefill_masks is not None:
            assert _use_fused, "persistent prefill_masks need the fused chunk op"
            _extra["masks"] = (prefill_masks["bg"], prefill_masks.get("qkv"))
        o, final_state = _delta_fn(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_len,
            qkv_head_dims=_qkv_head_dims,
            return_o_bh=self._gdn_fuse_out,
            **_extra,
        )
        B, D = 1, self.qkv_dim_tp
        captured = None
        if return_state:
            # Per-user prefill: return this user's state for assemble_batched_state to stitch
            # into the batched buffers. No self.* writeback; tensors are not deallocated here.
            captured = (final_state, conv_new_state)
        else:
            # ---- Carry recurrent + conv state for the NEXT chunk (chunk-outer prefill). ----
            # In place (ttnn.copy) so the addresses the prefill/decode traces
            # baked in stay valid across execute_trace replays and across sequences.
            ttnn.copy(final_state, self.rec_state)
            ttnn.deallocate(final_state)
            ttnn.copy(conv_new_state, self.conv_carry)  # [1, K-1, D] last-K-1 conv inputs
            # ---- Finalize the decode conv window (last chunk / short prompt). ----
            # conv_states[1..K-1] = the last K-1 real conv inputs; [0] is the (shifted-out) zero.
            # Harmless to refresh every chunk — the last chunk's values are the ones decode reads.
            if capture_state:
                if self.conv_states is None:
                    self.reset_state()
                if self._zero_conv0 is not None:
                    ttnn.copy(self._zero_conv0, self.conv_states[0])
                else:
                    zero = ttnn.from_torch(
                        torch.zeros(1, B, D, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    )
                    ttnn.copy(zero, self.conv_states[0])
                    ttnn.deallocate(zero)
                for j in range(self.K - 1):
                    src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                    ttnn.copy(src, self.conv_states[j + 1])
                if self._hist_hook_active() and self._decode_B == 1:
                    # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
                    # (A batched model prefills per user into the B=1 scratch; write_slot / assemble_batched_state
                    # then move the state into the decode history, so the hook is single-user only.)
                    self._write_conv_hist(conv_new_state)
                    self._refresh_packed_single()  # gdn_decode_step at max_batch_size == 1 reads the packed history
                # Prefill wrote the taps directly, so they are the truth and the [B,K,C] window
                # mirror is behind (_ensure_conv_win re-seeds it at the next eager spec seed).
                self._conv_taps_stale = False
                self._conv_win_stale = True
            ttnn.deallocate(conv_new_state)
        # Gated RMSNorm + SiLU(z); norm/flatten in L1, gated output in DRAM for out-proj
        _L1 = tpc.l1_or_dram(T)
        if self._gdn_fuse_out:
            # Fuse adapter relayout with per-head rms_norm + head-flatten.
            # TILE-native head->token relayout (transpose + fold), dropping the
            # TILE->ROW_MAJOR->TILE round-trip. o is head-major (1,Nv,T,Dv).
            n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            n = ttnn.reshape(n, (1, Nv, T, Dv))
            # Fused head->token relayout: [1,Nv,T,Dv] -> [1,1,T,Nv*Dv].
            n = ttnn.experimental.nlp_concat_heads(n, memory_config=_L1)
            out_f = ttnn.reshape(n, (1, T, self.value_dim_tp))
        else:
            out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp), memory_config=_L1)
            ttnn.deallocate(out_n)
        if self._out_colpar_prefill:
            # Column-parallel out-proj: the gate multiply emits the AGMM input directly as bf16 (the
            # only numerics change vs the fp32 MMRS arm: activation quantized to bf16 before the
            # matmul, as every other projection in the model already does).
            gated = _silu_mul(out_f, z, _L1, dtype=ttnn.bfloat16)
            ttnn.deallocate(out_f)
            ttnn.deallocate(z)
            # TODO(#57458): switch to the op's barrier_semaphore once it is wired up (see tpc.agmm_gather_buffer).
            out = tpc.all_gather_matmul_prefill(
                gated,
                tw["out_colpar"],
                self.tt_ccl,
                self.cfg,
                self.args.ccl_topology(),
                out_memory_config=_L1,
                persistent_output_buffer=tpc.agmm_gather_buffer(self.tt_ccl, gated),
            )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        gated = _silu_mul(out_f, z, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        # Prefill: fused out-proj matmul + reduce-scatter (matmul_reduce_scatter_async), flag-gated.
        if self._fuse_out_mmrs_prefill:
            x_out = ttnn.reshape(gated, (1, 1, T, gated.shape[-1]))
            # fp32 output is load-bearing: o_proj is row-parallel, so the RS SUMS 4 per-device partials
            # across devices — bf16 there tanks PCC to ~0.69 even at ISL 2048 (test_oproj_dtype_isl). Keep fp32.
            out = tpc.matmul_reduce_scatter_prefill(
                x_out, tw["out"], self.tt_ccl, self.cfg, self.args.ccl_topology(), self.args.num_devices, ttnn.float32
            )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if return_state:
            return out, captured[0], captured[1]
        return out

    def forward_prefill_collect(self, x, chunk_size=128, valid_len=None):
        """Per-user prefill that stashes this user's B=1 state for later assembly.

        Called once per user; finalize_pending() then stitches the collected states into the
        batched decode buffers. Returns the user's prefill output (needed for residual + MLP)."""
        out, rec, conv = self.forward_prefill(x, chunk_size=chunk_size, valid_len=valid_len, return_state=True)
        self._pending.append((rec, conv))
        return out

    def finalize_pending(self):
        """Assemble the per-user states collected by forward_prefill_collect into the batched
        decode buffers (row u = user u), then clear the accumulator."""
        assert self._pending, "finalize_pending called with no collected per-user states"
        rec_list = [r for (r, _) in self._pending]
        conv_list = [c for (_, c) in self._pending]
        self.assemble_batched_state(rec_list, conv_list)
        self._pending = []

    def assemble_batched_state(self, rec_list, conv_new_list):
        """Stitch B per-user prefill states (from forward_prefill(return_state=True)) into the
        batched decode buffers.

        rec_list[u]: [1, Nv, Dk, Dv] recurrent state; conv_new_list[u]: [1, K-1, qkv_dim_tp]
        last-(K-1) conv inputs. Row u of rec_state and conv_states[1..K-1] becomes user u's state;
        conv_states[0] is zeroed (shifted-out tap). ttnn has no in-place row write, so buffers are
        built by concat along the batch dim (rec: dim 0; conv: dim 1).

        The result is copied into the fixed-address buffers.
        """
        if self.rec_state is None:
            self.reset_state()
        assert len(rec_list) == self.B and len(conv_new_list) == self.B, "need one state per batch row"
        D = self.qkv_dim_tp
        # B == 1: concat of a single tensor returns its input, which is freed with rec_list below: copy it.
        rec_batched = ttnn.clone(rec_list[0]) if len(rec_list) == 1 else ttnn.concat(rec_list, dim=0)
        conv_states = [
            ttnn.from_torch(
                torch.zeros(1, self.B, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        ]
        for m in range(1, self.K):  # conv_states[m] row u = conv_new_list[u][:, m-1]
            rows = [
                ttnn.reshape(ttnn.slice(conv_new_list[u], (0, m - 1, 0), (1, m, D)), (1, 1, D)) for u in range(self.B)
            ]
            # B == 1: concat of a single tensor returns its input (a view of conv_new_list[u], freed at the end):
            # take an independent copy and leave the row to that free.
            cs = ttnn.clone(rows[0]) if len(rows) == 1 else ttnn.concat(rows, dim=1)  # [1, B, D]
            if len(rows) > 1:
                for r in rows:
                    ttnn.deallocate(r)
            conv_states.append(cs)

        rec_src = (
            rec_batched
            if rec_batched.dtype == self.rec_state.dtype
            else ttnn.typecast(rec_batched, self.rec_state.dtype)
        )
        ttnn.copy(rec_src, self.rec_state)
        if rec_src is not rec_batched:
            ttnn.deallocate(rec_src)
        ttnn.deallocate(rec_batched)
        for m in range(self.K):
            ttnn.copy(conv_states[m], self.conv_states[m])
            ttnn.deallocate(conv_states[m])
        self._conv_taps_stale = False  # taps written from outside: the window mirror is now behind
        self._conv_win_stale = True
        if self._hist_hook_active():
            # Fused decode reads its conv history from the RM buffer, not conv_states.
            if self._decode_B == 1:
                if len(conv_new_list) == 1:
                    self._write_conv_hist(conv_new_list[0])
            else:
                self._write_conv_hist_users(conv_new_list)  # full replace: row u = user u
            self._conv_fmt = "both"  # conv_states[1..K-1] were written above, hist just now
            self._refresh_packed_single()
        for t in rec_list:
            ttnn.deallocate(t)
        for t in conv_new_list:
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ #
    # Per-slot state edits for vLLM continuous batching.
    # ------------------------------------------------------------------ #
    # The demo prefills all B users up front and assembles the whole batch at
    # once (assemble_batched_state). vLLM instead prefills ONE user at a time
    # into its decode slot while the other rows are mid-decode, and condenses
    # the batch when a request finishes. GDN's recurrent+conv state is a fixed
    # [B,...] buffer indexed by physical slot (not paged), so both events need a
    # single-row edit that preserves the other (live) rows. ttnn has no in-place
    # row write, so — exactly like assemble_batched_state — these rebuild the
    # buffer by slice+concat and ttnn.copy the result back (the copy preserves
    # the decode trace's baked buffer address).
    def _slice_along(self, buf, dim, lo, hi):
        """ttnn.slice of buf along `dim` for indices [lo, hi), other dims kept full."""
        start = [0] * len(buf.shape)
        end = list(buf.shape)
        start[dim] = lo
        end[dim] = hi
        return ttnn.slice(buf, tuple(start), tuple(end))

    def _row_shard_memcfg(self, nhw, width):
        """L1 height-shard config for a rank-4 TILE tensor flattened to (nhw, width) rows.

        ``ttnn.experimental.slice_write`` writes directly into an interleaved TILE output only when
        its input is sharded; with an interleaved input it round-trips the output through ROW_MAJOR,
        which mints a new buffer and breaks any address a trace baked in. So every slice_write into
        a persistent buffer shards its source with this config.
        """
        grid_size = self.mesh.compute_with_storage_grid_size()
        assert (
            grid_size.x >= 8 and grid_size.y >= 6
        ), f"GDN state write needs an 8x6 core rectangle, got {grid_size.x}x{grid_size.y}"
        assert nhw % ttnn.TILE_SIZE == 0, f"GDN state write rows {nhw} is not tile-aligned"
        n_tiles = nhw // ttnn.TILE_SIZE

        # Prefer the tuned 8x6=48-core rectangle when the tile count divides by 48, which depends on
        # the per-device head count as well as B (Nv_tp=12: 48*B tiles; Nv_tp=8: 32*B, so only B a
        # multiple of 3; Nv_tp=6: 24*B). Otherwise use the largest core count that divides the tiles.
        if n_tiles % 48 == 0:
            num_cores = 48
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))})
        else:
            num_cores = max(c for c in range(1, min(48, grid_size.x * grid_size.y) + 1) if n_tiles % c == 0)
            grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid,
                (nhw // num_cores, width),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    def _write_recurrent_state_prefix(self, new_rec, B):
        """Write active rows [0:B] without reading or copying idle rows."""
        shard_memcfg = self._row_shard_memcfg(B * self.Nv * self.Dk, self.Dv)
        src = (
            new_rec
            if new_rec.dtype == self.rec_state.dtype
            else ttnn.typecast(new_rec, self.rec_state.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        )
        sharded = ttnn.to_memory_config(src, shard_memcfg)
        ttnn.experimental.slice_write(
            sharded,
            self.rec_state,
            [0, 0, 0, 0],
            [B, self.Nv, self.Dk, self.Dv],
            [1, 1, 1, 1],
        )
        ttnn.deallocate(sharded)
        if src is not new_rec:
            ttnn.deallocate(src)
        ttnn.deallocate(new_rec)

    def _write_index(self, buf, src, idx, dim, consume_src=True):
        """Replace slice `idx` of `buf` along `dim` with `src` (extent 1 along `dim`), preserving
        the other slices, via an in-place copy into `buf`. Consumes `src` (and the temporary
        slices) unless consume_src=False. `src` must already match `buf`'s dtype."""
        n = buf.shape[dim]
        if n == 1:
            ttnn.copy(src, buf)
            if consume_src:
                ttnn.deallocate(src)
            return
        parts = []
        if idx > 0:
            parts.append(self._slice_along(buf, dim, 0, idx))
        parts.append(src)
        if idx < n - 1:
            parts.append(self._slice_along(buf, dim, idx + 1, n))
        new = ttnn.concat(parts, dim=dim)
        ttnn.copy(new, buf)
        ttnn.deallocate(new)
        for p in parts:
            if p is src and not consume_src:
                continue
            ttnn.deallocate(p)

    def write_slot(self, slot, rec, convs, consume=True):
        """Write one user's B=1 prefill state into decode `slot`, preserving every other (live)
        row. The per-slot analogue of assemble_batched_state for vLLM continuous batching.

        rec:   [1, Nv, Dk, Dv] the user's recurrent state.
        convs: list of K [1, 1, qkv_dim_tp] the user's conv taps (conv_states[m] column). Unlike
               assemble_batched_state (which zeroes tap 0), every tap is written straight from the
               user's B=1 prefill state, so decode continues from exactly the produced shift register.
        Consumes rec and convs unless consume=False (device-resident sources that must survive, e.g. the persistent
        B=1 prefill scratch: the slot write then costs no host round trip). Requires the batched buffers
        (allocate_kv_caches(batch_size=B))."""
        assert self.rec_state is not None and self.conv_states is not None, "batched GDN state not allocated"
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        if self._hist_hook_active() and self._decode_B > 1:
            # Fused batched decode reads its conv history from the RM buffer: write ONLY row `slot` (the K-1 newest
            # taps, convs[1..K-1], oldest first) BEFORE the loop below consumes convs. Live rows are untouched.
            if self._conv_fmt == "packed":
                # Direct packed write: the format stays "packed" (RM history / conv_states are stale; every RM reader
                # converts packed -> RM first), so no unpack here and no re-pack at the next decode.
                self._write_conv_hist_slot_packed(slot, convs)
            elif self._conv_fmt in ("both", "hist"):
                self._write_conv_hist_slot(slot, convs)
        self.sync_conv_taps()  # read-modify-write of the taps: they must be current first
        rec_src = rec if rec.dtype == self.rec_state.dtype else ttnn.typecast(rec, self.rec_state.dtype)
        if rec_src is not rec and consume:
            ttnn.deallocate(rec)
        if self.rec_state.shape[0] > 1 and self._inplace_slot_write() and self._rec_row_layout_ok():
            # A DRAM-interleaved TILE source makes slice_write re-tilize and re-seat rec_state (new address, stale for
            # baked traces); the L1-sharded copy of the row is written tile-for-tile in place.
            src = ttnn.slice(rec_src, (0, 0, 0, 0), tuple(rec_src.shape), memory_config=self._rec_row_shard_memcfg())
            ttnn.experimental.slice_write(
                src,
                self.rec_state,
                [slot, 0, 0, 0],
                [slot + 1] + list(self.rec_state.shape)[1:],
                [1, 1, 1, 1],
            )
            ttnn.deallocate(src)
            if consume or rec_src is not rec:
                ttnn.deallocate(rec_src)
        elif self.rec_state.shape[0] > 1:
            # In-place row write (no slice + concat + copy of the whole [B, Nv, Dk, Dv] state; bit-identical).
            ttnn.experimental.slice_write(
                rec_src,
                self.rec_state,
                [slot, 0, 0, 0],
                [slot + 1] + list(self.rec_state.shape)[1:],
                [1, 1, 1, 1],
            )
            if consume or rec_src is not rec:
                ttnn.deallocate(rec_src)
        else:
            self._write_index(self.rec_state, rec_src, slot, dim=0, consume_src=consume or rec_src is not rec)
        # conv_states are stale (and fully rewritten by the hist -> states sync) while the fused history is the
        # authoritative format ("hist" / "packed"): skip the K slice+concat+copy writes then.
        write_states = not (self._hist_hook_active() and self._decode_B > 1 and self._conv_fmt in ("hist", "packed"))
        for m in range(self.K):
            c = convs[m]
            if not write_states:
                if consume:
                    ttnn.deallocate(c)
                continue
            c_src = c if c.dtype == self.conv_states[m].dtype else ttnn.typecast(c, self.conv_states[m].dtype)
            if c_src is not c and consume:
                ttnn.deallocate(c)
            self._write_index(self.conv_states[m], c_src, slot, dim=1, consume_src=consume or c_src is not c)
        self._conv_win_stale = True
        if self._hist_hook_active() and self._decode_B == 1:
            # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
            self._sync_conv_hist_from_states()
            self._conv_fmt = "both"
            self._refresh_packed_single()

    def remap_slots(self, remap, live=None):
        """Reindex the batched decode state after a vLLM batch condense: slot i takes the state
        previously at slot remap[i] (identity entries are no-ops). Mirrors
        seed_manager.apply_slot_remap for GDN's per-slot recurrent+conv state, which the plugin's
        slot_remap does not itself move. In-place copy into the fixed buffers (preserves the decode
        trace's baked addresses).

        Format-aware: only the conv format(s) that are VALID per `_conv_fmt` are gathered. rec_state is always
        gathered. With the fused batched hook active the live history is _conv_hist_rm; under tag "hist" conv_states
        is stale (the full hist->states sync in prepare_decode_width rewrites every row before conv_states is read
        again), under "states" _conv_hist_rm is stale (symmetric), under "both" both are gathered. Without the hook
        only conv_states exists/matters. The tag is left unchanged."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self.sync_conv_taps()  # read-modify-write of the taps: they must be current first
        if self._rec_row_remap_ok():
            self._remap_rec_rows(idx, live)
        else:
            self._gather_indices(self.rec_state, idx, dim=0)
        if self._conv_fmt == "packed" and self._hist_hook_active() and self._decode_B > 1:
            # Parity-aware single gather over the packed rows; RM history / conv_states are stale and stay so.
            self._remap_packed(idx)
            self._conv_win_stale = True
            return
        hist_used = self._hist_hook_active() and self._decode_B > 1 and self._conv_hist_rm is not None
        if not hist_used:
            for m in range(self.K):
                self._gather_indices(self.conv_states[m], idx, dim=1)
        elif self._conv_fmt == "hist":
            self._gather_indices(self._conv_hist_rm, idx, dim=0)
        elif self._conv_fmt == "states":
            for m in range(self.K):
                self._gather_indices(self.conv_states[m], idx, dim=1)
        else:  # "both"
            for m in range(self.K):
                self._gather_indices(self.conv_states[m], idx, dim=1)
            self._gather_indices(self._conv_hist_rm, idx, dim=0)
        self._conv_win_stale = True

    def _rec_row_shard_memcfg(self):
        """L1 height-sharded config for ONE rec_state row [1, Nv, Dk, Dv] (same core choice as the prefix-state write)."""
        cfg = getattr(self, "_rec_row_shard_cfg", None)
        if cfg is None:
            grid_size = self.mesh.compute_with_storage_grid_size()
            nhw = self.Nv * self.Dk
            n_tiles = nhw // ttnn.TILE_SIZE
            num_cores = max(c for c in range(1, min(48, grid_size.x * grid_size.y) + 1) if n_tiles % c == 0)
            grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)
            cfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, (nhw // num_cores, self.Dv), ttnn.ShardOrientation.ROW_MAJOR),
            )
            self._rec_row_shard_cfg = cfg
        return cfg

    def _remap_rec_rows(self, idx, live=None):
        """In place: rec_state row d becomes old row idx[d] (idx is a permutation). Moves only the changed rows, one
        permutation cycle at a time: each row is sliced straight into an L1 height-sharded tile buffer and written back
        with the sharded-input slice_write, which writes tiles into rec_state's own buffer (no full-state concat + copy,
        no layout round trip, address unchanged). SHAPE-STABLE: only 1-row slices at offset s and 1-row writes at
        offset d, all compiled by the full-reversal remap in warmup_gdn_slot_ops (a reversal is all 2-cycles, so it
        reads every row and writes every row). With `live`, rows whose destination holds no live request are left as
        is."""
        buf = self.rec_state
        cfg = self._rec_row_shard_memcfg()
        tail = list(buf.shape)[1:]

        def read(s):
            return ttnn.slice(buf, (s, 0, 0, 0), (s + 1, *tail), memory_config=cfg)

        def write(row, d):
            ttnn.experimental.slice_write(row, buf, [d, 0, 0, 0], [d + 1, *tail], [1, 1, 1, 1])
            ttnn.deallocate(row)

        seen = [False] * len(idx)
        for start in range(len(idx)):
            if seen[start] or idx[start] == start:
                seen[start] = True
                continue
            cyc = []
            d = start
            while not seen[d]:
                seen[d] = True
                cyc.append(d)
                d = idx[d]
            if live is not None:
                if not any(live[c] for c in cyc):
                    continue
                dead = next((j for j, c in enumerate(cyc) if not live[c]), None)
                if dead is not None:
                    # Rows without a live request are don't-care (a prefill rewrites the whole slot before reuse), so
                    # skip writes into them. Rotating a dead row to the head means its old content is still in place
                    # for the cycle's last write, so no temporary is needed (a condense is one read + write per move).
                    cyc = cyc[dead:] + cyc[:dead]
                    for j in range(len(cyc) - 1):
                        if live[cyc[j]]:
                            write(read(cyc[j + 1]), cyc[j])
                    if live[cyc[-1]]:
                        write(read(cyc[0]), cyc[-1])
                    continue
            # cyc[j] <- old cyc[j + 1]; read each source just before its own row is overwritten, the head via tmp.
            tmp = read(cyc[0])
            for j in range(len(cyc) - 1):
                write(read(cyc[j + 1]), cyc[j])
            write(tmp, cyc[-1])

    def _rec_row_remap_ok(self):
        return os.environ.get("QWEN36_GDN_REC_ROW_REMAP", "1") == "1" and self._rec_row_layout_ok()

    def _rec_row_layout_ok(self):
        return (
            len(self.rec_state.shape) == 4
            and self.rec_state.layout == ttnn.TILE_LAYOUT
            and not self.rec_state.is_sharded()
            and (self.Nv * self.Dk) % ttnn.TILE_SIZE == 0
        )

    def remap_spec_slots(self, remap):
        """remap_slots plus the spec state: each user's ring blocks (every token slot) and E_prev row move too."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self.remap_slots(idx)
        if self._spec_ring is None:
            return
        B, Nv, Dk, Dv = self.B, self.Nv, self.Dk, self.Dv
        T = self._spec_shape[1]
        r4 = ttnn.reshape(self._spec_ring, (T, B * Nv, Dk, Dv))
        assert (
            r4.buffer_address() == self._spec_ring.buffer_address()
        ), "the rank-4 ring view copied instead of aliasing; the copy would land in a temporary"
        parts = [ttnn.slice(r4, (0, j * Nv, 0, 0), (T, (j + 1) * Nv, Dk, Dv)) for j in idx]
        new = ttnn.concat(parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for p in parts:
            ttnn.deallocate(p)
        ttnn.copy(new, r4)
        ttnn.deallocate(new)
        self._gather_indices(self.verify_win_cur(), idx, dim=0)

    def _gather_indices(self, buf, idx, dim):
        """In place: slice i along `dim` becomes old slice idx[i] (idx is a permutation; identity rows untouched,
        identity overall => no-op). Reads every changed source row from the old buffer first, then writes each into
        its destination, so gathering from `buf` into itself is safe."""
        changed = [(d, s) for d, s in enumerate(idx) if d != s]
        if not changed:
            return
        # SHAPE-STABLE: only ever 1-row slices and the single-row _write_index (whose [0,i) / (i,n) slice shapes are
        # all compiled by warmup_gdn_slot_ops via the per-slot writes). The previous maximal-run decomposition sliced
        # runs of arbitrary length, so a real condense compiled NEW slice/concat programs after the traces were
        # captured (a captured trace replay can then clobber their binaries / buffers -> non-deterministic output).
        # Every source row is read from the OLD buffer before any row is overwritten (a remap is a permutation).
        rows = [(d, self._slice_along(buf, dim, s, s + 1)) for d, s in changed]
        for d, row in rows:
            self._write_index(buf, row, d, dim, consume_src=True)

    def forward_prefill_batched(self, x, chunk_size=128, valid_lens=None, carry=False):
        """Batched prefill: all B users in one pass (no per-user Python loop).

        The chunk-seq GDN kernel scans a leading BH = B*H batch dim, each (user, head) row an
        independent causal scan, so B is a true batch dim (not a time concat). Runs projection /
        conv-FIR / chunk-parallel recurrence over [B, T, *] and writes straight into the batched
        decode buffers (rec_state[B,Nv,Dk,Dv], conv_states[*][1,B,D]); row u == user u.

        x:          [B, T, dim] replicated (all users padded to a common bucket length T).
        valid_lens: optional list of B real token counts (< T => right-padding masked per row);
                    None => every row is full length T.
        carry:      False (default) => from scratch (single-shot). True => CHUNK-OUTER carry: read
                    the recurrent state (self.rec_state) and conv left-context (self._batched_conv_carry)
                    from the previous chunk and write the updated ones back, so a long prompt can be
                    prefilled chunk-by-chunk over the batch. Mirrors the B=1 forward_prefill carry;
                    the caller zeroes rec_state (reset_state_inplace) + _batched_conv_carry at
                    sequence start, so the first chunk reads zeros (== from scratch).

        KERNEL CAP: gated_delta_attn_seq maps one BH = B*Nv_tp row per core and is L1-bound, so BH
        must stay <= ~32 (at TP=4, Nv_tp=8 => B <= 4). Larger B trips an L1 clash (B=8) or the
        kernel's `BH <= compute_grid` assert (B=32); B>4 would need grouped launches (groups <=4).
        The model currently prefills per-user instead (see prefill_paged_peruser).
        """
        if self.rec_state is None:
            self.reset_state()
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[-3], x.shape[-2], x.shape[-1]))  # [.,B,T,dim] -> [B,T,dim]
        B, T = x.shape[0], x.shape[1]
        D = self.qkv_dim_tp

        # Route through the shared per-token projection (handles _fuse_ab/_fuse_agmm — required
        # when the caller's norm skipped its post-AG and x arrives K-sharded, e.g. prefill_paged_
        # grouped). A plain ttnn.linear(x, tw["qkvz"]) here would (a) mismatch the K-sharded width
        # against the fused-weight's full-K height, and (b) KeyError on tw["ab"], which doesn't
        # exist when _fuse_ab folds a/b into tw["qkvz"]. Flatten the batch dim into the token dim
        # (the projection is per-token; user boundaries don't matter to a linear layer) since
        # _project_qkvzab's slicing assumes a leading dim of 1.
        x_flat = ttnn.reshape(x, (1, B * T, x.shape[-1]))
        qkv_flat, z_flat, a_flat, b_flat = self._project_qkvzab(x_flat, B * T, out_mc=ttnn.DRAM_MEMORY_CONFIG)
        qkv = ttnn.reshape(qkv_flat, (B, T, D))
        z = ttnn.reshape(z_flat, (B, T, self.qkvz_dim_tp - D))
        a = ttnn.reshape(a_flat, (B, T, Nv))
        b = ttnn.reshape(b_flat, (B, T, Nv))

        # FIR causal conv1d + SiLU over each user's sequence (per-row valid_len picks each user's
        # decode conv window). Chunk-outer carry: left-context = previous chunk's last K-1 inputs.
        if carry and getattr(self, "_batched_conv_carry", None) is None:
            # First chunk of a chunk-outer prefill: zeroed left-context (== from scratch).
            self._batched_conv_carry = ttnn.from_torch(
                torch.zeros(B, self.K - 1, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        conv_carry_in = self._batched_conv_carry if carry else None
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=conv_carry_in,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=valid_lens,
        )
        ttnn.deallocate(qkv)

        kd = self.key_dim_tp
        # Flat token-major q/k/v (no host head-split / GQA): the fused op does in-kernel L2-norm and
        # GQA (Nk->Nv) from qkv_head_dims, matching the single-user forward_prefill fused path.
        q = ttnn.slice(conv, (0, 0, 0), (B, T, kd))
        k = ttnn.slice(conv, (0, 0, kd), (B, T, 2 * kd))
        v = ttnn.slice(conv, (0, 0, 2 * kd), (B, T, D))
        ttnn.deallocate(conv)

        beta = ttnn.reshape(ttnn.sigmoid(b), (B, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"])), (B, T, Nv))
        ttnn.deallocate(a)

        # Chunk-parallel recurrence over the BH = B*Nv batch (each row an independent scan). Fused
        # chunk_gated_delta_rule (same op as single-user prefill); per-row valid_lens mask each user.
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
            chunk_gated_delta_rule_fused_adapter,
            fused_chunk_enabled,
        )

        _use_fused = fused_chunk_enabled()
        _delta_fn = chunk_gated_delta_rule_fused_adapter if _use_fused else chunk_gated_delta_rule_seq_adapter
        _extra = (
            {
                "const_tiles": self._fused_const_tiles,
                "program_config": self.gdn_program_config,
                "wy_inverse": self.gdn_wy_inverse,
            }
            if _use_fused
            else {}
        )
        o, final_state = _delta_fn(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_lens,
            qkv_head_dims=(Nk, Dk, Nv, Dv),
            **_extra,
        )

        # ---- write the batched decode state directly (row u == user u) ----
        rec_src = (
            final_state
            if final_state.dtype == self.rec_state.dtype
            else ttnn.typecast(final_state, self.rec_state.dtype)
        )
        ttnn.copy(rec_src, self.rec_state)
        if rec_src is not final_state:
            ttnn.deallocate(rec_src)
        ttnn.deallocate(final_state)
        # conv_states[0] = shifted-out zero; conv_states[m] row u = conv_new_state[u, m-1].
        zero0 = ttnn.from_torch(
            torch.zeros(1, B, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        new_conv = [zero0]
        for m in range(1, self.K):
            cs = ttnn.reshape(ttnn.slice(conv_new_state, (0, m - 1, 0), (B, m, D)), (1, B, D))  # [1,B,D]
            new_conv.append(cs)
        if carry:
            # Preserve this chunk's last K-1 inputs as the next chunk's left-context (replace the
            # buffer just consumed by the FIR above).
            if conv_carry_in is not None:
                ttnn.deallocate(conv_carry_in)
            self._batched_conv_carry = conv_new_state  # [B, K-1, D]
        else:
            ttnn.deallocate(conv_new_state)
        for m in range(self.K):
            ttnn.copy(new_conv[m], self.conv_states[m])
            ttnn.deallocate(new_conv[m])
        self._conv_taps_stale = False  # taps written from outside: the window mirror is now behind
        self._conv_win_stale = True
        if self._hist_hook_active() and B == self._decode_B:
            # Fused decode reads its conv history from the RM buffer, not conv_states: full replace from the
            # just-written conv_states (single-user: the [1, K-1, C] row; batched: every user's row).
            self._sync_conv_hist_from_states()
            self._conv_fmt = "both"
            self._refresh_packed_single()

        # ---- output (gated RMSNorm + SiLU(z) gate + row-parallel out proj + all-reduce) ----
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (B, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, B, T, partial.shape[-1]))
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, x, decode_ar=None):
        """decode_ar: tp_common.DecodeResidualAllReduce (QWEN36_DECODE_ALLREDUCE=1) -> x is the replicated
        residual-norm output and the out-proj partial is all-reduced (replicated RES40, bf16 unless
        QWEN36_DECODE_AR_GDN_FP32=1) instead of reduce-scattered."""
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        Bmax = self.B
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode conv→recurrence→norm/gate chain L1-resident
        if self.conv_states is None:
            self.reset_state()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))

        # Active decode width, taken from the input. Normally == Bmax. BUCKETED decode: a request
        # feeds B<Bmax tokens and the whole step runs on state rows [0:B]; idle rows [B:Bmax] are
        # preserved. Conv taps are per-channel (broadcast over batch), so the conv weighted-sum
        # works at any width. The B==Bmax path is byte-identical to before.
        B = x.shape[-2]
        if self._decode_op and not self.use_fused_recurrent_decode and B <= 32 and Bmax == self._decode_B:
            # gdn_decode_step op (QWEN36_GDN_DECODE_STEP_OP): every width 1..Bmax on the packed conv history.
            self._require_conv_fmt("packed")
            return self._forward_decode_step_op(x, decode_ar)
        if self._fused_decode and Bmax == 1 and B == 1 and not self.use_fused_recurrent_decode:
            # (Spec decode sets use_fused_recurrent_decode: its verify advances conv_states + rec_state with the fused
            # recurrent op, so decode must stay on that path too -- the fused-conv decode below keeps its history in
            # _conv_hist_rm instead.)
            # QWEN36_GDN_FUSED_DECODE (default on), single-user: fused conv / chunk-delta-rule / gated-norm kernels.
            return self._forward_decode_fused(x, decode_ar)
        # Spec decode (use_fused_recurrent_decode) keeps conv_states authoritative, so it skips the fused-batched path
        # (which keeps its history in _conv_hist_rm) and takes the original path below, like the B == 1 guard above.
        if self._fused_batched and not self.use_fused_recurrent_decode:
            # Batched fused decode (max_batch_size > 1, every width 1..Bmax). The conv history is the unified
            # _conv_hist_rm [Bmax, K-1, C]; conv_states is NOT maintained in this mode.
            assert Bmax == self._decode_B, (
                f"fused batched GDN decode needs the decode binding (B={Bmax}, max_batch_size={self._decode_B}); "
                "set QWEN36_GDN_FUSED_DECODE=0 to run another batch size"
            )
            if B <= self._scan_max_b:
                self._require_conv_fmt("hist")
                return self._forward_decode_fused_batched(x, decode_ar)
            # Wide widths (B > SCAN_MAX): the ENTIRE original path below (shift-register conv on conv_states +
            # original recurrence on the shared rec_state); conv_states must be the valid format.
            self._require_conv_fmt("states")

        qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)
        q, k, v = self._decode_conv_original(qkv, B, Bmax)

        # GQA expand Q/K Nk→Nv; recurrence L2-norms + scales internally
        rf = Nv // Nk
        q = ttnn.repeat_interleave(q, rf, dim=1)
        k = ttnn.repeat_interleave(k, rf, dim=1)
        # Decode: hand q/k/v to the recurrent kernel in L1. The kernel typecasts + does a LOCAL
        # l2-norm (no cross-device gather), so placement is output-neutral here (unlike SDPA-q,
        # which hard-requires DRAM, and unlike the residual→DistributedNorm all-gather).
        q = ttnn.reshape(q, (B, 1, Nv, Dk), memory_config=_L1)
        k = ttnn.reshape(k, (B, 1, Nv, Dk), memory_config=_L1)
        v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)

        beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
        ttnn.deallocate(b)
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (B, 1, Nv))

        # fp32 decode step by default (QWEN35_GDN_DECODE_BF16=1 reverts)
        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
        if self.use_fused_recurrent_decode:
            # Spec-decode path. The whole recurrence (decay->k.S->delta->outer->q.S) is ONE fused
            # device op rather than the ~13-op composite: 0.345 vs 0.541 ms, and closer to the FLA
            # reference (PCC 0.999991 vs 0.999981). Decode runs under trace, so the dispatch saving
            # is small — the reason spec decode selects it is CONSISTENCY. Spec verify advances GDN
            # with this same op, so decode and verify must use identical math or every greedy
            # near-tie flips between them and acceptance drops (measured 2.82 -> 2.00 /3 when the
            # two paths disagreed at ~1e-5).
            assert B == Bmax, "fused recurrent decode path supports full-batch only (spec decode)"
            o, new_rec = fused_recurrent_gated_delta_rule_ttnn(
                q,
                k,
                v,
                beta,
                g,
                scale=self.scale,
                initial_state=self.rec_state,
                device=self.mesh,
                high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
            )
        else:
            o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
                q,
                k,
                v,
                beta,
                g,
                scale=self.scale,
                initial_state=init_state,
                device=self.mesh,
                high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
            )
        if init_state is not self.rec_state:
            ttnn.deallocate(init_state)
        # In-place update preserves rec_state address for decode trace replay
        if B == Bmax:
            ttnn.copy(new_rec, self.rec_state)
            ttnn.deallocate(new_rec)
        else:
            self._write_recurrent_state_prefix(new_rec, B)

        out_r = ttnn.reshape(o, (B, Nv, Dv))
        out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)  # gated norm (no +1)
        ttnn.deallocate(out_r)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f, z, _L1)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is fp32 here
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return out

    def _decode_conv_original(self, qkv, B, Bmax):
        """The original decode conv: shift-register conv_states + weighted sum + SiLU, then the q/k/v split.
        Returns q [B, Nk, Dk], k [B, Nk, Dk], v [B, Nv, Dv]. Consumes qkv. Unchanged from the pre-batched code."""
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1 = ttnn.L1_MEMORY_CONFIG
        # Conv1d shift-register + weighted sum + SiLU.
        # The K taps below ARE the shift register, so they must be current: a preceding fullbatch
        # verify advanced only the [B,K,C] window mirror. No-op (zero device ops, so trace-safe)
        # unless a spec verify actually ran — see sync_conv_taps.
        self.sync_conv_taps()
        st = self.conv_states
        if B < Bmax:
            # Bucketed decode: active requests occupy a contiguous prefix [0:B]; idle rows [B:Bmax]
            # hold no live request (a slot is re-initialized by prefill/write_slot when reused), so
            # they are don't-care. Pad the width-B new input up to Bmax and run the SAME full-width
            # shift-register as below -- the conv sum's active rows [0:B] are exact and the downstream
            # q/k/v slices take [0:B]. This keeps the op COUNT identical to the baseline path (just a
            # single pad), vs a per-row slice/concat that added ~4*K ops/layer and erased the width win.
            qkv_p = ttnn.pad(qkv, [(0, 0), (0, Bmax - B), (0, 0)], value=0.0, memory_config=_L1)
            ttnn.deallocate(qkv)
            qkv = qkv_p
        for j in range(self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        self._conv_win_stale = True  # decode shifted the taps; the window mirror is now behind
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
        ttnn.deallocate(conv)

        return q, k, v

    def _forward_decode_fused(self, x, decode_ar=None):
        """Single-user (max_batch_size == 1, active width 1) decode through the fused kernels (see
        QWEN36_GDN_FUSED_DECODE). Same math as the original path, fewer ops:
          * ttnn.experimental.kda.qkv_causal_conv1d_silu: 4-tap causal conv + SiLU + q/k/v split in one program,
            history = the persistent RM buffer _conv_hist_rm (the last K-1 raw conv inputs, oldest first);
          * ttnn.transformer.chunk_gated_delta_rule with chunk_size=32 over ONE live token padded to a 32-row tile:
            beta and g are zero-padded, so the 31 padded steps are identity state updates (the qkv padding rows
            only feed those identity steps), and it does the q/k L2-norm, GQA expansion and the fp32 state update;
          * ttnn.experimental.kda.sigmoid_gated_rms_norm (= norm * w * sigmoid(z)) then * z -> norm * w * silu(z).
        Differences from the original path (all follow the validated fused decoder, models/demos/qwen38_27b_qb2):
        a and g stay fp32 (dt_bias / -exp(A_log) are fp32 copies), and the out-proj input is bf16 (the original
        feeds fp32; the out-proj partial is therefore bf16 too). conv_states is not updated here -- only
        _conv_hist_rm and rec_state are. conv input / q / k / v are DRAM; the padded z_p / g_p / beta_p are views of
        their (L1) sources and stay in that memory space until their last consumer. No allocation of persistent
        state: trace-safe.
        """
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE

        tw, Nv = self.tw, self.Nv
        _L1, _DRAM = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        C, kd, vd = self.qkv_dim_tp, self.key_dim_tp, self.value_dim_tp
        # The kernels run on one 32-row tile; row 0 is the live token. Zero rows: identity recurrence steps.
        # NOTE: ttnn.pad of a [1,1,C] TILE tensor up to the 32-row tile height returns a VIEW of the SAME device
        # buffer (the tile already holds the 32 rows; pad only zero-fills the implicit tile padding in place and
        # ignores memory_config). So the padded tensors (z_p, g_p, beta_p, qkv_p) must NEVER be freed by
        # deallocating their SOURCE (that frees the shared buffer under the live view -> later allocations
        # overwrite it -> garbage): the source is left alone and exactly ONE handle (the padded one) is
        # deallocated, after the padded tensor's last consumer.
        pad_rows = [(0, 0), (0, _FUSED_CHUNK_SIZE - 1), (0, 0)]

        qkv, z, a, b = self._project_qkvzab(x, 1, out_mc=_L1)

        # ---- z (the out gate) and the gates, ahead of the kernels: DRAM, 32 rows, zero padded ----
        if z.dtype != ttnn.bfloat16:  # sigmoid_gated_rms_norm's gate contract
            z = ttnn.typecast(z, ttnn.bfloat16, memory_config=_L1)
        z_p = ttnn.pad(z, pad_rows, value=0.0, memory_config=_DRAM)  # view of z; freed once, after the gate mul
        # beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias), a/g in fp32
        beta = ttnn.sigmoid(b, memory_config=_L1)
        ttnn.deallocate(b)
        a32 = ttnn.typecast(a, ttnn.float32, memory_config=_L1)
        ttnn.deallocate(a)
        a_dt = ttnn.add(a32, tw["dt_bias_fp32"], memory_config=_L1)
        ttnn.deallocate(a32)
        g = ttnn.mul(
            tw["neg_exp_A_fp32"],
            a_dt,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
            memory_config=_L1,
        )
        ttnn.deallocate(a_dt)
        g_p = ttnn.pad(g, pad_rows, value=0.0, memory_config=_DRAM)  # view of g; freed once, after the chunk op
        beta_p = ttnn.pad(beta, pad_rows, value=0.0, memory_config=_DRAM)  # view of beta; freed once after chunk

        # ---- causal conv1d + SiLU + q/k/v split: one program on the persistent RM history ----
        hist = self._ensure_conv_hist()
        qkv_p = ttnn.pad(qkv, pad_rows, value=0.0, memory_config=_DRAM)  # view of qkv
        row_qkv = ttnn.to_layout(qkv_p, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)  # new (RM, DRAM) buffer
        ttnn.deallocate(qkv_p)  # last consumer of the tile view done: frees the shared qkv buffer, once
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            row_qkv,
            hist,
            *tw["conv_taps"],
            kd,
            kd,
            vd,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=kda_channel_chunk_size(C, cap=256)),
            actual_start=self._kda_actual_start,
            # No sequence parallelism on the TP mesh: alias history (see kda_conv_prefill).
            predecessor_carry=hist,
            memory_config=_DRAM,
        )
        # History for the next token: [hist[1:], this token's raw conv input], updated in place (fixed address).
        keep = ttnn.slice(hist, (0, 1, 0), (1, self.K - 1, C), memory_config=_DRAM)
        cur = ttnn.slice(row_qkv, (0, 0, 0), (1, 1, C), memory_config=_DRAM)
        ttnn.deallocate(row_qkv)
        new_hist = ttnn.concat([keep, cur], dim=1, memory_config=_DRAM)
        ttnn.deallocate(keep)
        ttnn.deallocate(cur)
        ttnn.copy(new_hist, hist)
        ttnn.deallocate(new_hist)

        # ---- gated delta rule: one op, fp32 state in place (decode/prefill traces keep rec_state's address) ----
        # No program_config: gdn_program_config is a prefill-geometry knob; the op picks its own for this shape.
        eye, tril, ones, masks = self._fused_const_tiles
        o, new_rec = ttnn.transformer.chunk_gated_delta_rule(
            q,
            k,
            v,
            g_p,
            beta_p,
            scale=self.scale,
            initial_state=self.rec_state,
            output_final_state=True,
            chunk_size=_FUSED_CHUNK_SIZE,
            output_head_major=True,
            eye=eye,
            tril=tril,
            ones=ones,
            masks=masks,
        )
        for t in (q, k, v, g_p, beta_p):
            ttnn.deallocate(t)
        ttnn.copy(new_rec, self.rec_state)
        ttnn.deallocate(new_rec)

        # ---- gated RMSNorm (no +1) * silu(z) = (norm * w * sigmoid(z)) * z, bf16 into the out-proj ----
        normed = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            o, z_p, tw["norm_w_1d"], Nv, epsilon=1e-6, output_dtype=ttnn.bfloat16
        )
        ttnn.deallocate(o)
        gated = ttnn.mul(normed, z_p, memory_config=_DRAM)
        ttnn.deallocate(normed)
        ttnn.deallocate(z_p)
        # Keep only the live row logically; the physical tile geometry is unchanged (a free view).
        gated = ttnn.reshape(gated, ttnn.Shape([1, 1, vd]), gated.padded_shape)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, 1, partial.shape[-1]))
        if decode_ar is not None:
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is bf16 here (fp32 cast only if asked)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------ #
    # Batched (multi-user) speculative verify
    # ------------------------------------------------------------------ #
    # One verify advances B users x T = K+1 candidate rows in one pass over the 32-row decode tile
    # (user-major: row u*T + j). Nothing commits here: the recurrent op writes every token's state
    # into a persistent ring and the conv window keeps every candidate's shift register, so the host
    # commits by handing the next replay selectors (state_blk_idx, conv_sel).
    #
    # Everything the body touches is persistent and allocated by prepare_spec_verify: the body runs
    # under a captured trace, where a first-time allocation or host write would either TT_FATAL at
    # capture or bake a throwaway address into the replay.

    def prepare_spec_verify(self, n_users, T):
        """Allocate the persistent buffers a batched spec verify reads and writes.

        n_users must be the full decode batch (the ring, the window and the row packing are all
        sized by it) and n_users * T must fit the one 32-row decode tile every decode matmul,
        head-split and norm config in this stack assumes.

        Allocates (idempotent for the same (n_users, T); a different shape frees and reallocates):
          _spec_ring      fp32 TILE DRAM [T*B*Nv, Dk, Dv]  per-token state ring (op writes in place)
          _verify_win_buf bf16 TILE DRAM [B, K-1+T, C]     E_prev, the conv window
          _conv_win_buf   bf16 TILE DRAM [B, K, C]         the durable shift register
          _spec_win_pad   bf16 TILE DRAM [B, T-1, C]       zero tail for the seed (T > 1 only)

        Call it eagerly, before the verify warmup pass, so the warmup compiles every program the
        captured body replays.
        """
        assert 1 <= n_users <= self.B, f"spec verify rows must be in 1..B: n_users={n_users}, B={self.B}"
        n_users = self.B  # the state buffers are always B wide; a b-row verify (b < B) runs on them
        assert T >= 1, f"T must be at least 1, got {T}"
        assert (
            n_users * T <= tpc.TILE_SIZE
        ), f"{n_users} users x {T} rows = {n_users * T} exceeds the {tpc.TILE_SIZE}-row decode tile"
        if self.conv_states is None:
            self.reset_state()
        if self._spec_shape == (n_users, T):
            self._ensure_conv_win()
            return
        self._release_spec_bufs()
        B, Nv, Dk, Dv, K, C = self.B, self.Nv, self.Dk, self.Dv, self.K, self.qkv_dim_tp
        mc, cdt = ttnn.DRAM_MEMORY_CONFIG, self.conv_states[0].dtype
        # fp32 unconditionally: the fused op's ring mode reads and writes fp32 state blocks.
        self._spec_ring = ttnn.zeros(
            [T * B * Nv, Dk, Dv], device=self.mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_config=mc
        )
        if self._spec_fused:
            # QWEN36_VERIFY_SPEC_STEP=1: the window ping-pong pair (see __init__) instead of the single buffer, at
            # parity 0; plus the fused seed's constants, so no later seed allocates under a parked trace.
            self._verify_win_buf = None
            self._verify_win_pair = [
                ttnn.zeros([B, K - 1 + T, C], device=self.mesh, dtype=cdt, layout=ttnn.TILE_LAYOUT, memory_config=mc)
                for _ in range(2)
            ]
            self._spec_win_par = 0
            self._ensure_spec_fused_seed()
            _log_spec_step_once(
                ("buffers", B, T),
                f"[gdn-spec-step] QWEN36_VERIFY_SPEC_STEP=1: gdn_spec_step verify at (B={B}, T={T}), "
                f"hnew_depth={self._spec_hnew_depth(B)}, window pair [{B}, {K - 1 + T}, {C}] x2, ctrl page "
                f"{spec_ctrl_words(B, Nv)} words",
            )
        else:
            self._verify_win_buf = ttnn.zeros(
                [B, K - 1 + T, C], device=self.mesh, dtype=cdt, layout=ttnn.TILE_LAYOUT, memory_config=mc
            )
        self._spec_win_pad = (
            None
            if T == 1
            else ttnn.zeros([B, T - 1, C], device=self.mesh, dtype=cdt, layout=ttnn.TILE_LAYOUT, memory_config=mc)
        )
        self._spec_shape = (n_users, T)
        self._ensure_conv_win()

    def verify_win_cur(self):
        """The conv window E_prev the NEXT verify reads -- what the seed / park / restore / remap paths read and
        write: _verify_win_buf, or under QWEN36_VERIFY_SPEC_STEP=1 the current half of the window ping-pong pair."""
        if self._spec_fused:
            assert self._verify_win_pair is not None, "spec window pair before prepare_spec_verify"
            return self._verify_win_pair[self._spec_win_par]
        return self._verify_win_buf

    # ---- QWEN36_VERIFY_SPEC_STEP=1 helpers (gdn_spec_step) ----
    @staticmethod
    def _spec_hnew_depth(B):
        """States buffered between the op's compute and its ring writer: 2 (QWEN36_SPEC_STEP_HNEW_DEPTH=4 opts into
        the deeper buffering). Depth 4 costs 128 KB of L1 per core, and at B = 4 on the P150x8 column its static
        dataflow buffers clash with the verify bucket's L1 buffers (TT_THROW at launch, T = 4 and T = 8), while at
        B = 8 K = 3 it measured the same verify time as depth 2 (37.2 ms both)."""
        env = os.environ.get("QWEN36_SPEC_STEP_HNEW_DEPTH")
        if env is not None:
            d = int(env)
            assert d in (2, 4), f"QWEN36_SPEC_STEP_HNEW_DEPTH must be 2 or 4, got {env}"
            return d
        return 2

    def _ensure_spec_fused_seed(self):
        """Allocate (once per B) the fused seed's constants: the scratch half of its [B, K, C] window pair and its
        identity ctrl page (parity 0, mi 0, ring block (u, h) = u*Nv + h = rec_state's own block, no holds). EAGER:
        prepare_spec_verify forces it before any capture; the call from _seed_fullbatch_fused only covers a seed
        that runs before prepare_spec_verify. Re-allocated if B changed under us."""
        B, Nv, K, C = self.B, self.Nv, self.K, self.qkv_dim_tp
        if self._seed_win_scratch is not None and tuple(self._seed_win_scratch.shape) != (B, K, C):
            ttnn.deallocate(self._seed_win_scratch)
            self._seed_win_scratch = None
        if self._seed_win_scratch is None:
            self._seed_win_scratch = ttnn.zeros(
                [B, K, C],
                device=self.mesh,
                dtype=self.conv_states[0].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if self._seed_ctrl is not None and int(self._seed_ctrl.shape[-1]) != spec_ctrl_words(B, Nv):
            ttnn.deallocate(self._seed_ctrl)
            self._seed_ctrl = None
        if self._seed_ctrl is None:
            self._seed_ctrl = ttnn.from_torch(
                spec_ctrl_page([0] * B, B, Nv, parity=0),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

    def _gdn_spec_step(self, qkvzab, win_a, win_b, ring, ctrl, T, B, max_users=None):
        """The one fused dispatch (QWEN36_VERIFY_SPEC_STEP=1): raw [1, B*T, W] projection rows in, gated
        [1, B*T, Nv*Dv] bf16 (L1) out; ring / win[1-par] updated in place. Same call for the verify (ring =
        _spec_ring, the staged ctrl page) and the seed (ring = rec_state, the identity page, T = 1). dt_bias /
        neg_exp_A / norm weight / fast_mode / fp32_state and the default HiFi4 + fp32-accumulate kernel config are
        exactly what _forward_decode_step_op hands gdn_decode_step, so one verify token is one plain decode step."""
        tw = self.tw
        extra = {}
        _mu = max_users if max_users is not None else getattr(self, "_spec_max_users", 0)
        if _mu:
            extra["max_users"] = int(_mu)
        _vb = gdn_spec_step_value_blocks()
        if _vb > 1 and T > 1:
            # the T = 1 seed keeps one core per (user, head): with a single token the column split only adds bulk-read
            # contention and the exchanges (B = 8, device: 81 -> 90 us per layer); every T > 1 verify splits
            extra["value_blocks"] = _vb
            _log_spec_step_once(
                ("value_blocks", _vb),
                f"[gdn-spec-step] QWEN36_SPEC_STEP_VALUE_BLOCKS={_vb}: state columns split over up to {_vb} cores "
                "per (user, head) (bit-identical; the op clamps to the grid per (B, T))",
            )
        _opts = gdn_spec_step_kernel_opts()
        if _opts and self._decode_step_fp32_state:  # the restructurings exist for the fast_mode + fp32_state program
            extra["kernel_opts"] = _opts
            _log_spec_step_once(
                ("kernel_opts", _opts),
                f"[gdn-spec-step] QWEN36_SPEC_STEP_OPT={_opts:#x}: bit-identical kernel restructurings on",
            )
        return ttnn.experimental.kda.gdn_spec_step(
            qkvzab,
            win_a,
            win_b,
            ring,
            ctrl,
            tw["conv_taps_rows"],
            tw["dt_bias_fp32"],
            tw["neg_exp_A_fp32"],
            tw["norm_w_1d"],
            self.Nv,
            self.Nk,
            self.Dk,
            self.Dv,
            T,
            B,
            self.qkvz_dim_tp,
            conv_kernel=self.K,
            scale=self.scale,
            hnew_depth=self._spec_hnew_depth(B),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            output_dtype=ttnn.bfloat16,
            fast_mode=True,
            fp32_state=self._decode_step_fp32_state,
            **extra,
        )

    def _project_qkvzab_raw(self, x, S, out_mc=None):
        """The decode qkvzab projection WITHOUT the four slices: [1, S, W] with q|k|v in columns [0, qkv_dim_tp), z in
        [qkv_dim_tp, qkvz_dim_tp) and a|b in [qkvz_dim_tp, qkvz_dim_tp + 2*Nv) -- the tile gdn_decode_step (plain
        decode, _forward_decode_step_op) and gdn_spec_step (fused spec verify / seed) consume directly. The very
        matmul_1d_decode call (same weight, program config and compute config cfg_in) of _project_qkvzab's decode
        branch and of _forward_decode_step_op, so the projected values are the plain decode step's."""
        assert self._fuse_ab and getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE, (
            "the fused GDN spec op needs the fused [qkv|z|a|b] weight and the 1D decode projection "
            f"(fuse_ab={self._fuse_ab}, proj_1d_decode={getattr(self.args, 'proj_1d_decode', False)}, S={S})"
        )
        qkvzab = tpc.matmul_1d_decode(
            x,
            self.tw["qkvz"],
            self.args.gdn_qkvz_decode_1d_progcfg,
            self.cfg_in,
            out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(qkvzab, (1, S, qkvzab.shape[-1]))

    def _forward_verify_recurrent_fused(self, x, valid_len, T, n_users, spec_ctrl, pre_gathered, decode_ar=None):
        """forward_verify_recurrent under QWEN36_VERIFY_SPEC_STEP=1: the same gather + ONE decode projection, handed
        to gdn_spec_step UN-SLICED (the op reads q|k|v|z|a|b out of the projection tile itself)."""
        qkvzab, _, _, _, bucket = self._verify_project(x, valid_len, pre_gathered, raw=True)
        return self._verify_fullbatch_fused(qkvzab, T, n_users, bucket, spec_ctrl, decode_ar=decode_ar)

    def _verify_fullbatch_fused(self, qkvzab, T, n_users, bucket, ctrl, decode_ar=None):
        """``_verify_fullbatch`` under QWEN36_VERIFY_SPEC_STEP=1: ONE gdn_spec_step dispatch per layer.

        The op rebuilds every user's conv window from win[par] (rows mi+1 .. mi+K-1) + its T new projection rows,
        runs the depthwise conv + SiLU, the l2norms, the beta / decay gates, the T-step gated delta rule with each
        token's state written to ring block (t*B + u)*Nv + h (initial block per (u, h) from the ctrl page; HOLD = no
        ring writes, window copied through), the gated RMSNorm and the silu(z) gate, and writes the new window into
        win[1-par]. Everything it touches is persistent (prepare_spec_verify) or produced in-trace (qkvzab), so the
        body is trace-capturable; ctrl is DATA the host stages per replay (model.verify_traced), which also flips
        _spec_win_par afterwards. The staleness invariant of _verify_fullbatch holds unchanged: for the whole spec loop
        the durable conv truth is win[cur] plus the host's mi; the K taps and _conv_win_buf are behind it.
        """
        B, R = n_users, n_users * T
        win = self._verify_win_pair
        # B rows (a bucket b <= self.B) run on the shared self.B-wide ring / ctrl page: max_users = the state width.
        gated = self._gdn_spec_step(qkvzab, win[0], win[1], self._spec_ring, ctrl, T, B, max_users=self.B)
        ttnn.deallocate(qkvzab)
        return self._out_proj_tail(gated, R, bucket, decode_ar=decode_ar)

    def _seed_fullbatch_fused(self, qkvzab, n_users, bucket):
        """``_seed_fullbatch`` under QWEN36_VERIFY_SPEC_STEP=1: gdn_spec_step at T = 1 against the DURABLE state.

        ring := rec_state itself ([B, Nv, Dk, Dv] = B*Nv blocks of [Dk, Dv]; the identity ctrl page reads block
        u*Nv + h and the op writes block 0*B*Nv + u*Nv + h -- the same block -- so rec_state is updated IN PLACE).
        Window pair := (_conv_win_buf, a scratch) at parity 0: with mi = 0 the op reads the shift register's rows
        1..K-1 (the carry) + the new row, and writes the new register into the scratch, which is copied back so
        _conv_win_buf keeps the single-mirror semantics every other conv-state path relies on (sync_conv_taps /
        sync_conv_win untouched). Same kernel and arithmetic as the verify and as one plain gdn_decode_step.
        """
        B = R = n_users
        assert self.rec_state.dtype == ttnn.float32, "the fused GDN spec op needs the fp32 recurrent state"
        self._ensure_conv_win()  # the mirror current from the live taps (prefill / reset / plain decode move them)
        self._ensure_spec_fused_seed()
        gated = self._gdn_spec_step(
            qkvzab, self._conv_win_buf, self._seed_win_scratch, self.rec_state, self._seed_ctrl, 1, B
        )
        ttnn.deallocate(qkvzab)
        ttnn.copy(self._seed_win_scratch, self._conv_win_buf)  # the new shift register, in place
        # Same flag handling as _seed_fullbatch: push the window back out to the K taps now (both mirrors current).
        self._conv_taps_stale, self._conv_win_stale = True, False
        self.sync_conv_taps()
        return self._out_proj_tail(gated, R, bucket)

    def seed_spec_state(self):
        """Load the live decode state into the spec buffers so the first replay can run with mi = 0.

        rec_state -> ring token slot 0 (blocks [0, B*Nv), where a mi = 0 state_blk_idx points); the
        conv shift register -> the first K rows of E_prev (a mi = 0 conv_sel reads them as concat
        rows 1 .. K-1). The window's remaining T-1 rows are zero padding no mi = 0 selector reaches.
        """
        assert self._spec_ring is not None, "seed_spec_state before prepare_spec_verify"
        self._write_ring_slot0_all()
        self.sync_conv_win()  # taps are the truth outside the spec loop; mirror them into the window
        full = (
            self._conv_win_buf
            if self._spec_win_pad is None
            else ttnn.concat([self._conv_win_buf, self._spec_win_pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        ttnn.copy(full, self.verify_win_cur())  # full-shape copy: the window buffer keeps its address
        if full is not self._conv_win_buf:
            ttnn.deallocate(full)
        self._conv_taps_stale, self._conv_win_stale = False, False

    def _write_ring_slot0_all(self):
        """rec_state (every user) -> ring token slot 0 (blocks [0, B*Nv)), in place: the ring half of seed_spec_state.
        Two device ops (sharded copy of the state + slice_write into the rank-4 ring view)."""
        B, Nv, Dk, Dv = self.B, self.Nv, self.Dk, self.Dv
        T = self._spec_shape[1]
        # 4D views: slice_write needs input/output/bounds at equal rank and a rank-4 sharded input,
        # and adding a leading unit dim to a TILE tensor is a pure view (same buffer, so the write
        # lands in the ring itself).
        src = ttnn.reshape(self.rec_state, (1, B * Nv, Dk, Dv))
        cast = None if src.dtype == ttnn.float32 else ttnn.typecast(src, ttnn.float32)
        sharded = ttnn.to_memory_config(src if cast is None else cast, self._row_shard_memcfg(B * Nv * Dk, Dv))
        if cast is not None:
            ttnn.deallocate(cast)
        ring4 = ttnn.reshape(self._spec_ring, (1, T * B * Nv, Dk, Dv))
        assert (
            ring4.buffer_address() == self._spec_ring.buffer_address()
        ), "the rank-4 ring view copied instead of aliasing; slice_write would write to a temporary"
        ttnn.experimental.slice_write(sharded, ring4, [0, 0, 0, 0], [1, B * Nv, Dk, Dv], [1, 1, 1, 1])
        ttnn.deallocate(sharded)

    # ---- QWEN36_SPEC_FAST_JOIN=1: per-row ring / E_prev seeding without the eager T = 1 seed forward ----
    def _spec_ring_rows4(self):
        """The ring as [T*B, Nv, Dk, Dv]: row (t*B + u) is user u's token-slot-t state block group, so a per-user
        write into token slot 0 is the one-row slice_write of _remap_rec_rows at offset u. A pure view of _spec_ring
        (same buffer), cached per ring allocation."""
        cached = getattr(self, "_spec_ring_rows4_cache", None)
        if cached is not None and cached[0] is self._spec_ring:
            return cached[1]
        B, Nv, Dk, Dv = self.B, self.Nv, self.Dk, self.Dv
        T = self._spec_shape[1]
        r4 = ttnn.reshape(self._spec_ring, (T * B, Nv, Dk, Dv))
        assert (
            r4.buffer_address() == self._spec_ring.buffer_address()
        ), "the rank-4 ring view copied instead of aliasing; slice_write would write to a temporary"
        self._spec_ring_rows4_cache = (self._spec_ring, r4)
        return r4

    def _fresh_taps_tile(self, route=None):
        """[B, K-1, C] bf16 TILE: every user's K-1 most recent raw conv inputs, oldest first (conv taps 1 .. K-1,
        what a mi = 0 verify reads as E_prev rows 1 .. K-1). The source is whichever conv-state format is valid:
          "hist"  the fused decode's RM history _conv_hist_rm (row u = user u; written per slot by write_slot, unpacked
                  first when the gdn_decode_step op left it "packed"): ONE tilize, no conv_states traffic;
          "win"   conv_states -> _conv_win_buf (sync_conv_win) and a slice of its rows 1 .. K-1 (the original decode
                  path's "states" format, or a model without the fused history).
        ``route`` forces one (warmup compiles both); None picks by the current format. Returns None when the forced
        "hist" route has no history buffer."""
        B, K, C = self.B, self.K, self.qkv_dim_tp
        mc = ttnn.DRAM_MEMORY_CONFIG
        hist_ok = self._conv_hist_rm is not None and self.B == self._decode_B and self._hist_hook_active()
        if route is None:
            if hist_ok and self._conv_fmt == "packed":
                self._ensure_rm_hist_valid()  # packed -> hist (the op's format after a plain decode / packed slot write)
            route = "hist" if hist_ok and self._conv_fmt in ("both", "hist") else "win"
        if route == "hist":
            if self._conv_hist_rm is None:
                return None
            return ttnn.to_layout(self._conv_hist_rm, ttnn.TILE_LAYOUT, memory_config=mc)
        assert not self._conv_taps_stale, "fast seed: the conv taps are stale (a seed left them behind the window)"
        self.sync_conv_win()  # taps -> window (a traced prefill writes the taps without flagging the window stale)
        return ttnn.slice(self._conv_win_buf, (0, 1, 0), (B, K, C))

    def seed_spec_rows_select(self, rows, sel):
        """QWEN36_SPEC_FAST_JOIN=1: seed ring token slot 0 and the conv window E_prev of the users in ``rows`` from
        their durable prefill state (rec_state, conv taps), keeping every other user's ring blocks and window row,
        with a handful of exact data-movement ops per layer instead of seed_spec_rows' whole-slot rebuild:
          ring   per user: an L1-sharded slice of rec_state[u] + a one-row slice_write into token slot 0 (the
                 _remap_rec_rows pattern), or the whole-slot write of seed_spec_state when ``rows`` is every user;
          E_prev E_new = sel @ [E_prev ; taps(1..K-1)] (one-hot bf16, HiFi4 + fp32 accumulate: exact), copied back in
                 place; ``sel`` is the device tensor of fresh_row_select(rows, B, T, K), shared by every GDN layer.
        The ring / E_prev the next verify reads are then those of seed_spec_rows(rows); rec_state, the taps and the
        staleness flags are untouched (nothing advanced them: the verify itself consumes the first token).
        Eager; every program it can run is compiled by warmup_fast_seed before capture."""
        assert self._spec_ring is not None, "seed_spec_rows_select before prepare_spec_verify"
        B, Nv, Dk, Dv = self.B, self.Nv, self.Dk, self.Dv
        rows = sorted({int(u) for u in rows})
        assert rows and all(0 <= u < B for u in rows), f"rows {rows} outside range({B})"
        if len(rows) == B:
            self._write_ring_slot0_all()
        else:
            assert self._rec_row_layout_ok(), "fast seed needs a rank-4 TILE interleaved rec_state"
            r4 = self._spec_ring_rows4()
            cfg = self._rec_row_shard_memcfg()
            for u in rows:
                if self.rec_state.dtype == ttnn.float32:
                    row = ttnn.slice(self.rec_state, (u, 0, 0, 0), (u + 1, Nv, Dk, Dv), memory_config=cfg)
                else:
                    s = ttnn.slice(self.rec_state, (u, 0, 0, 0), (u + 1, Nv, Dk, Dv))
                    c = ttnn.typecast(s, ttnn.float32)
                    ttnn.deallocate(s)
                    row = ttnn.to_memory_config(c, cfg)
                    ttnn.deallocate(c)
                ttnn.experimental.slice_write(row, r4, [u, 0, 0, 0], [u + 1, Nv, Dk, Dv], [1, 1, 1, 1])
                ttnn.deallocate(row)
        taps = self._fresh_taps_tile()  # [B, K-1, C]
        win = self.verify_win_cur()
        cat = ttnn.concat([win, taps], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, K-1+T + K-1, C]
        ttnn.deallocate(taps)
        new = ttnn.matmul(sel, cat, compute_kernel_config=self._cfg_onehot, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cat)
        ttnn.copy(new, win)  # in place: the window buffer keeps its address
        ttnn.deallocate(new)

    def seed_masked_ok(self):
        """Host predicate for seed_spec_rows_masked: True only when the taps would come from the "hist" route and
        nothing eager has to run first (no unpack of a packed history, no stale taps, the decode binding). The engine
        calls seed_spec_rows_select when this is False."""
        return (
            self._spec_ring is not None
            and self._rec_row_layout_ok()
            and self._conv_hist_rm is not None
            and self.B == self._decode_B
            and self._hist_hook_active()
            and self._conv_fmt in ("both", "hist")
            and not self._conv_taps_stale
        )

    def seed_spec_rows_masked(self, mask, sel):
        """seed_spec_rows_select with a FIXED op sequence (no host branch on which users are fresh), so the 48-layer
        seed can be captured in a trace. ``mask`` is a persistent device tensor, fp32 TILE, shape [1, B*Nv, Dk, 1]
        (the column-broadcast shape of ttnn.where against the slot-0 image): 1.0 for the Nv head blocks of every fresh
        user, 0.0 elsewhere. ``sel`` is the same one-hot fresh_row_select tensor as seed_spec_rows_select takes.
          ring   new = where(mask, rec_state, ring slot 0) over every user (an exact select: a non-fresh user's blocks
                 are bit-identical to before, nan / inf included), then ONE slice_write of the whole slot 0;
          E_prev the "hist" taps route + concat + matmul(sel) + copy, as in seed_spec_rows_select.
        Needs seed_masked_ok(). Writes verify_win_cur(): the window parity is host state baked into a trace, so the
        engine captures one trace per parity. Every temporary is freed, so each call allocates the same way."""
        assert (
            self.seed_masked_ok()
        ), "seed_spec_rows_masked: the hist taps route is not ready (use seed_spec_rows_select)"
        self._seed_masked_impl(mask, sel)

    def _seed_masked_impl(self, mask, sel):
        """The op sequence of seed_spec_rows_masked, without the host predicate (warmup compiles it with it unmet).
        Per layer, fp32 rec_state: slice, where, to_memory_config, slice_write, to_layout (taps), concat, matmul,
        copy = 8 ops (bf16 rec_state adds one typecast)."""
        assert self._spec_ring is not None, "seed_spec_rows_masked before prepare_spec_verify"
        B, Nv, Dk, Dv = self.B, self.Nv, self.Dk, self.Dv
        T = self._spec_shape[1]
        mc = ttnn.DRAM_MEMORY_CONFIG
        ring4 = ttnn.reshape(self._spec_ring, (1, T * B * Nv, Dk, Dv))  # view, as in _write_ring_slot0_all
        assert (
            ring4.buffer_address() == self._spec_ring.buffer_address()
        ), "the rank-4 ring view copied instead of aliasing; slice_write would write to a temporary"
        old = ttnn.slice(ring4, (0, 0, 0, 0), (1, B * Nv, Dk, Dv), memory_config=mc)
        src = ttnn.reshape(self.rec_state, (1, B * Nv, Dk, Dv))
        cast = None if src.dtype == ttnn.float32 else ttnn.typecast(src, ttnn.float32)
        new = ttnn.where(mask, src if cast is None else cast, old, memory_config=mc)
        if cast is not None:
            ttnn.deallocate(cast)
        ttnn.deallocate(old)
        sharded = ttnn.to_memory_config(new, self._row_shard_memcfg(B * Nv * Dk, Dv))
        ttnn.deallocate(new)
        ttnn.experimental.slice_write(sharded, ring4, [0, 0, 0, 0], [1, B * Nv, Dk, Dv], [1, 1, 1, 1])
        ttnn.deallocate(sharded)
        taps = self._fresh_taps_tile(route="hist")  # [B, K-1, C]
        win = self.verify_win_cur()
        cat = ttnn.concat([win, taps], dim=1, memory_config=mc)
        ttnn.deallocate(taps)
        new_win = ttnn.matmul(sel, cat, compute_kernel_config=self._cfg_onehot, memory_config=mc)
        ttnn.deallocate(cat)
        ttnn.copy(new_win, win)  # in place: the window buffer keeps its address
        ttnn.deallocate(new_win)

    def warmup_fast_seed(self, sel, mask=None):
        """Warmup only (spec engine, after prepare_spec_verify, before any capture): run seed_spec_rows_select for every
        single user and for all users at once, and build the taps tile through BOTH routes, so every program the fast
        seed can run after capture is compiled now. The state is warmup state here; the ring and window contents are
        overwritten by the real seed. With ``mask`` (see seed_spec_rows_masked) it also runs the masked seed once per
        window parity (the parity is restored), so its programs and both window buffers are touched before capture."""
        for u in range(self.B):
            self.seed_spec_rows_select([u], sel)
        self.seed_spec_rows_select(range(self.B), sel)
        if mask is not None and self._conv_hist_rm is not None:
            par = self._spec_win_par
            try:
                for p in (0, 1) if self._spec_fused else (par,):
                    self._spec_win_par = p
                    self._seed_masked_impl(mask, sel)
            finally:
                self._spec_win_par = par
        for route in ("hist", "win"):
            t = self._fresh_taps_tile(route=route)
            if t is not None:
                ttnn.deallocate(t)

    def seed_spec_rows(self, rows):
        """seed_spec_state for the users in ``rows`` only; every other user's ring blocks and E_prev row are kept.
        One fixed program per user offset: warm every user before trace capture."""
        assert self._spec_ring is not None, "seed_spec_rows before prepare_spec_verify"
        Nv, Dk, Dv, K, C = self.Nv, self.Dk, self.Dv, self.K, self.qkv_dim_tp
        T = self._spec_shape[1]
        rows = {int(u) for u in rows}
        if self.B == 1:
            # B == 1: a ttnn.slice over the whole rec_state / window is a no-op that returns the buffer itself, so the
            # per-row path below would hand the durable state to _write_spec_rows, which frees its parts. The one
            # user is every user: the whole-slot seed is the same write.
            assert rows == {0}, f"rows {sorted(rows)} outside range(1)"
            self.seed_spec_state()
            return
        self.sync_conv_win()
        ring_parts = {}
        for u in sorted(rows):
            s = ttnn.slice(self.rec_state, (u, 0, 0, 0), (u + 1, Nv, Dk, Dv))
            if s.dtype != ttnn.float32:
                c = ttnn.typecast(s, ttnn.float32)
                ttnn.deallocate(s)
                s = c
            ring_parts[u] = s
        full = (
            self._conv_win_buf
            if self._spec_win_pad is None
            else ttnn.concat([self._conv_win_buf, self._spec_win_pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        e_parts = {u: ttnn.slice(full, (u, 0, 0), (u + 1, K - 1 + T, C)) for u in sorted(rows)}
        self._write_spec_rows(ring_parts, e_parts)
        if full is not self._conv_win_buf:
            ttnn.deallocate(full)
        self._conv_taps_stale, self._conv_win_stale = False, False

    def _write_spec_rows(self, ring_parts, e_parts):
        """Write per-user device tensors ([1, Nv, Dk, Dv] fp32 / [1, K-1+T, C]) into ring slot 0 / the E_prev row;
        users not given keep theirs. Deallocates the given tensors."""
        assert self._spec_ring is not None, "_write_spec_rows before prepare_spec_verify"
        B, Nv, Dk, Dv, K, C = self.B, self.Nv, self.Dk, self.Dv, self.K, self.qkv_dim_tp
        T = self._spec_shape[1]
        ring4 = ttnn.reshape(self._spec_ring, (1, T * B * Nv, Dk, Dv))
        assert (
            ring4.buffer_address() == self._spec_ring.buffer_address()
        ), "the rank-4 ring view copied instead of aliasing; slice_write would write to a temporary"
        parts = [
            ring_parts[u] if u in ring_parts else ttnn.slice(ring4, (0, u * Nv, 0, 0), (1, (u + 1) * Nv, Dk, Dv))
            for u in range(B)
        ]
        # B == 1: ttnn.concat of ONE tensor returns that tensor (no copy), so the part must not be freed here.
        blk0 = parts[0] if B == 1 else ttnn.concat(parts, dim=1)
        if B > 1:
            for s in parts:
                ttnn.deallocate(s)
        sharded = ttnn.to_memory_config(blk0, self._row_shard_memcfg(B * Nv * Dk, Dv))
        ttnn.deallocate(blk0)
        ttnn.experimental.slice_write(sharded, ring4, [0, 0, 0, 0], [1, B * Nv, Dk, Dv], [1, 1, 1, 1])
        ttnn.deallocate(sharded)
        win = self.verify_win_cur()
        eparts = [e_parts[u] if u in e_parts else ttnn.slice(win, (u, 0, 0), (u + 1, K - 1 + T, C)) for u in range(B)]
        new = eparts[0] if B == 1 else ttnn.concat(eparts, dim=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if B > 1:
            for e in eparts:
                ttnn.deallocate(e)
        ttnn.copy(new, win)
        ttnn.deallocate(new)

    def spec_users_save_enqueue(self, users):
        """Phase 1 of a batched park: slice the ring blocks (every slot) and the E_prev row of each user and enqueue their
        D2H copies NON-blocking. The host tensors are filled only by a synchronize that comes after this call; hand the
        result to spec_users_save_finish then. Many layers can enqueue before the one synchronize."""
        assert self._spec_ring is not None, "save_spec_user before prepare_spec_verify"
        B, Nv, Dk, Dv, K, C = self.B, self.Nv, self.Dk, self.Dv, self.K, self.qkv_dim_tp
        T = self._spec_shape[1]
        r4 = ttnn.reshape(self._spec_ring, (T, B * Nv, Dk, Dv))
        win = self.verify_win_cur()
        pend = []
        for u in users:
            ring_dev = ttnn.slice(r4, (0, u * Nv, 0, 0), (T, (u + 1) * Nv, Dk, Dv))
            e_dev = ttnn.slice(win, (u, 0, 0), (u + 1, K - 1 + T, C))
            pend.append(
                (
                    u,
                    ring_dev,
                    e_dev,
                    ttnn.from_device(ring_dev, blocking=False),
                    ttnn.from_device(e_dev, blocking=False),
                )
            )
        return pend

    def spec_users_save_finish(self, pend):
        """Phase 2 (after the synchronize): {u: (ring [D*T, Nv, Dk, Dv], E_prev row [D, K-1+T, C])} host tensors."""
        comp = ttnn.ConcatMeshToTensor(self.mesh, dim=0)
        out = {}
        for u, ring_dev, e_dev, ring_h, e_h in pend:
            out[u] = (ttnn.to_torch(ring_h, mesh_composer=comp), ttnn.to_torch(e_h, mesh_composer=comp))
            ttnn.deallocate(ring_dev)
            ttnn.deallocate(e_dev)
        return out

    def save_spec_user(self, u):
        """Host copy of user u's ring blocks (every slot) and E_prev row.

        Returns ([D*T, Nv, Dk, Dv], [D, K-1+T, C]); D = devices. One synchronize (the engine's batched park calls
        spec_users_save_enqueue for all layers and rows instead, and synchronizes once for all of them).
        """
        pend = self.spec_users_save_enqueue([u])
        ttnn.synchronize_device(self.mesh)
        return self.spec_users_save_finish(pend)[u]

    def spec_users_load_enqueue(self, states):
        """Write host states {u: (ring slot [D, Nv, Dk, Dv], E_prev row [D, K-1+T, C])} into ring slot 0 / the E_prev row of
        each u (other users kept) without synchronizing; the caller synchronizes (and keeps ``states`` alive until then).
        """
        mapper = ttnn.ShardTensorToMesh(self.mesh, dim=0)

        def up(x, dtype):
            return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, mesh_mapper=mapper)

        ring_parts = {u: up(r, ttnn.float32) for u, (r, _) in states.items()}
        e_parts = {u: up(e, self.verify_win_cur().dtype) for u, (_, e) in states.items()}
        self._write_spec_rows(ring_parts, e_parts)

    def load_spec_users(self, states):
        """Write host states {u: (ring slot [D, Nv, Dk, Dv], E_prev row [D, K-1+T, C])}.

        Goes into ring slot 0 / the E_prev row of each u; other users kept.
        """
        self.spec_users_load_enqueue(states)
        ttnn.synchronize_device(self.mesh)

    def forward_verify_recurrent(
        self,
        x,
        valid_len,
        pre_gathered=False,
        n_users=1,
        state_blk_idx=None,
        conv_sel=None,
        spec_ctrl=None,
        decode_ar=None,
    ):
        """Batched hybrid spec-decode verify for GDN: advance B users x T candidate rows at once.

        x : [1, 1, bucket, dim] prefill-normed input; the first ``valid_len`` rows are real and
        user-major (row u*T + j == user u, candidate j). Returns [1, 1, bucket, dim/tp], zero-padded
        past ``valid_len`` (nothing downstream reads those rows: attention is causal and verify only
        row-selects real rows).

        valid_len  : n_users * T real rows; must fit one 32-row decode tile.
        pre_gathered: x is already full-dim (decode-config verify runs the layer norms in
                     Mode.DECODE, which gathers pre-norm), so skip the all-gather.
        n_users    : number of real users packed into the tile; 1..self.B (a bucket b <= B runs on the B-wide state).
        state_blk_idx: device uint32 ROW_MAJOR [B*Nv], from spec_state_blk_idx(mi, B, Nv). Selects
                     each (user, head)'s initial state block inside the ring.
        conv_sel   : device bf16 TILE [B, K-1+T, K-1+2T] one-hot, from spec_conv_sel(mi, B, T, K).
                     Rebuilds each user's conv window from the accepted prefix + the new rows.
        spec_ctrl  : QWEN36_VERIFY_SPEC_STEP=1 only: device uint32 ROW_MAJOR [1, N] page from spec_ctrl_page
                     ({parity, mi, ring block | HOLD}); replaces the two selectors above (both ignored).
        decode_ar  : QWEN36_VERIFY_DAR=1 only (fused spec op only): tp_common.DecodeResidualAllReduce. x is then the local
                     sharded norm of the replicated RES40 residual and the out-projection is all-reduced back into RES40,
                     exactly like _forward_decode_step_op, instead of reduce-scattered into the fractured residual.

        Side effects: the ring is updated in place (every candidate's state, for every user) and
        _verify_win_buf (fused: win[1-par]) becomes the new E_prev. rec_state and conv_states are untouched; the
        host commits later by index.
        """
        assert valid_len <= tpc.TILE_SIZE, f"verify bucket {valid_len} exceeds one tile"
        assert 1 <= n_users <= self.B, f"spec verify rows must be in 1..B: n_users={n_users}, B={self.B}"
        assert valid_len % n_users == 0, f"valid_len {valid_len} is not {n_users} whole user row-groups"
        T = valid_len // n_users
        assert self._spec_shape == (self.B, T), (
            f"spec buffers are sized for {self._spec_shape}, verify asked for {(self.B, T)}; "
            "call prepare_spec_verify(n_users, T) before capture"
        )
        if self._spec_fused:
            if spec_ctrl is None:
                raise ValueError(
                    "QWEN36_VERIFY_SPEC_STEP=1: forward_verify_recurrent needs spec_ctrl, the gdn_spec_step ctrl page "
                    "(spec_ctrl_page); call prepare_spec_verify + seed_spec_state first"
                )
            return self._forward_verify_recurrent_fused(
                x, valid_len, T, n_users, spec_ctrl, pre_gathered, decode_ar=decode_ar
            )
        if decode_ar is not None:
            raise ValueError("QWEN36_VERIFY_DAR needs the fused gdn_spec_step verify (QWEN36_VERIFY_SPEC_STEP=1)")
        if state_blk_idx is None or conv_sel is None:
            raise ValueError("forward_verify_recurrent needs state_blk_idx and conv_sel (the deferred-select verify)")
        qkv_all, z_all, a_all, b_all, bucket = self._verify_project(x, valid_len, pre_gathered)
        return self._verify_fullbatch(qkv_all, z_all, a_all, b_all, T, n_users, bucket, state_blk_idx, conv_sel)

    def _verify_project(self, x, valid_len, pre_gathered, raw=False):
        """Gather the real rows to full dim and run one decode qkvzab projection over them.

        Used by the batched verify. ``_project_qkvzab`` at S = valid_len <= TILE_SIZE
        routes through matmul_1d_decode, the projection a plain decode step runs. That matmul is
        row-independent and processes a full 32-row M-tile however many rows are real, so one
        projection over every user's rows is per-row identical to projecting them one at a time. The
        AGMM prefill projection rounds differently and would drift the state away from plain decode.
        Returns (qkv, z, a, b, bucket).
        ``raw`` (QWEN36_VERIFY_SPEC_STEP=1): the un-sliced [1, R, W] projection instead -> (qkvzab, None, None, None,
        bucket).
        """
        mc = ttnn.DRAM_MEMORY_CONFIG
        if self.conv_states is None:
            self.reset_state()
        # Decode-config verify hands us the DECODE attn-norm output, which is L1 WIDTH-SHARDED.
        # We slice valid rows out of it below, so interleave first (row-slicing a width-shard is not
        # supported). We then own that copy and must free it.
        _x_owned = False
        # QWEN36_VERIFY_LAYOUT_CUTS: when every row is real, the sharded norm output goes straight to matmul_1d_decode, which
        # interleaves it into L1 itself (what _forward_decode_step_op does). The old sharded -> DRAM -> L1 round trip
        # was one extra ShardedToInterleaved + one Copy per GDN layer. Only the row slice below needs interleaved input.
        _direct = (
            pre_gathered
            and x.is_sharded()
            and x.shape[-2] == valid_len
            and tpc.verify_layout_cuts_enabled()
            and valid_len <= tpc.TILE_SIZE
        )
        if pre_gathered and x.is_sharded() and not _direct:
            x = ttnn.to_memory_config(x, mc)
            _x_owned = True
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        bucket = x.shape[-2]  # x is K-sharded [1, bucket, dim/tp] (full-dim when pre_gathered)
        R = valid_len  # real rows, user-major

        x_valid = x if R == bucket else ttnn.slice(x, (0, 0, 0), (1, R, x.shape[-1]))
        if pre_gathered:
            # Already full-dim [1, bucket, dim]: gathering again would quadruple the feature dim.
            xg = x_valid
        else:
            # x is 3D [1, R, dim/tp] here (reshaped above), so gather the last (feature) dim.
            xg = tt_all_gather(
                x_valid,
                self.mesh,
                self.tt_ccl,
                cluster_axis=None,
                dim=-1,
                topology=self.args.ccl_topology(),
                memory_config=mc,
            )
            if x_valid is not x:
                ttnn.deallocate(x_valid)
        if raw:
            qkv_all, z_all, a_all, b_all = self._project_qkvzab_raw(xg, R, out_mc=mc), None, None, None
        else:
            qkv_all, z_all, a_all, b_all = self._project_qkvzab(xg, R, out_mc=mc)
        if xg is not x:
            ttnn.deallocate(xg)
        if _x_owned:
            ttnn.deallocate(x)
        return qkv_all, z_all, a_all, b_all, bucket

    def forward_seed_recurrent(self, x, valid_len, pre_gathered=False, n_users=1):
        """The spec loop's seed: advance GDN by one row per user with the verify's arithmetic.

        Same shape as a plain decode step (B users x 1 token) but built from the verify's pieces: the
        packed KDA conv over each user's [K, C] window (``_kda_conv_packed``) instead of the K-tap
        shift-register MAC. The two are not bit-equal, and the state the seed leaves is what every
        later verify replay resumes from; seeding with the decode formulation would fork the greedy
        trajectory at a near tie.

        Non-ring, unlike ``forward_verify_recurrent``: the spec ring and E_prev do not exist yet
        (``prepare_spec_verify`` runs later, inside ``capture_verify_trace``). It advances the
        durable state (``rec_state`` in place; ``_conv_win_buf`` and the K taps to the new window),
        which ``seed_spec_state`` later copies into the ring and E_prev.

        x : [1, 1, B, dim] (full-dim when ``pre_gathered``). Returns [1, 1, B, dim/tp].
        """
        assert n_users == self.B, f"the seed runs the whole decode batch: n_users={n_users}, B={self.B}"
        assert valid_len == n_users, f"the seed is one row per user: valid_len={valid_len}, n_users={n_users}"
        assert n_users <= tpc.TILE_SIZE, f"{n_users} users exceeds the {tpc.TILE_SIZE}-row decode tile"
        if self._spec_fused:
            qkvzab, _, _, _, bucket = self._verify_project(x, valid_len, pre_gathered, raw=True)
            return self._seed_fullbatch_fused(qkvzab, n_users, bucket)
        qkv_all, z_all, a_all, b_all, bucket = self._verify_project(x, valid_len, pre_gathered)
        return self._seed_fullbatch(qkv_all, z_all, a_all, b_all, n_users, bucket)

    def _seed_fullbatch(self, qkv_all, z_all, a_all, b_all, n_users, bucket):
        """``_verify_fullbatch`` at T = 1, run against the durable state instead of the ring."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        C, K = self.qkv_dim_tp, self.K
        B = n_users

        # The carry is taps [1, K) of the [B, K, C] mirror; E = [carry(K-1) ; this row(1)] is both the
        # conv input and the new shift register. _ensure_conv_win() first makes the mirror current
        # (prefill, reset and plain decode steps move the taps under it).
        self._ensure_conv_win()
        qkv_new = self._rows_to_users(qkv_all, B, 1, C)  # [B, 1, C]
        carry = ttnn.slice(self._conv_win_buf, (0, 1, 0), (B, K, C))  # [B, K-1, C]
        E = ttnn.concat([carry, qkv_new], dim=1, memory_config=mc)  # [B, K, C]
        ttnn.deallocate(carry)
        ttnn.deallocate(qkv_new)
        q_all, k_all, v_all = self._kda_conv_packed(E, 1)  # each [1, B, width], SiLU applied

        q_all, k_all, v_all, beta_all, g_all = self._heads_and_gates(q_all, k_all, v_all, a_all, b_all, B, 1)

        # Plain mode: the initial state is the durable rec_state and the op returns the final state
        # only (no per-token ring exists yet, and at T = 1 the two coincide). The call and its
        # [B,1,Nv,*] shapes match the fused recurrent decode step, so it adds no new program.
        o_all, states = fused_recurrent_gated_delta_rule_ttnn(
            q_all,
            k_all,
            v_all,
            beta_all,
            g_all,
            scale=self.scale,
            initial_state=self.rec_state,
            device=self.mesh,
            output_per_token_state=False,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        for t in (q_all, k_all, v_all, beta_all, g_all):
            ttnn.deallocate(t)

        # rec_state is written in place (decode traces bake its address); the new shift register is E
        # itself, which already has exactly K rows.
        ttnn.copy(states, self.rec_state)
        ttnn.deallocate(states)
        ttnn.copy(E, self._conv_win_buf)
        ttnn.deallocate(E)
        # Push the window back out to the K taps rather than leaving them stale. sync_conv_win()
        # branches on _conv_taps_stale, and the throwaway seed_spec_state() inside
        # capture_verify_trace exists only to compile the branch the real post-capture
        # seed_spec_state() takes, so both calls must see the same flags. A stale-tap window would
        # also make the next plain decode step (sync_conv_taps) pay K slice+copy per layer while the
        # verify trace is parked.
        self._conv_taps_stale, self._conv_win_stale = True, False
        self.sync_conv_taps()  # clears _conv_taps_stale; both mirrors now hold the seeded window

        return self._out_tail(o_all, z_all, B, bucket)

    def _rows_to_users(self, t, B, T, width):
        """[1, B*T, width] -> [B, T, width]. Rows are user-major, so this is a pure regroup.

        B == 1 is the identity (the same buffer is returned, so free only one handle). For B > 1 the
        reshape allocates, because each user's T rows have to start a fresh tile row; the source is
        freed here. The tiled reshape builds a host page-map on a program-cache miss, so a traced
        body must have been warmed up at this exact shape.
        """
        if B == 1:
            return t
        out = ttnn.reshape(t, (B, T, width))
        ttnn.deallocate(t)
        return out

    def _heads_and_gates(self, q_all, k_all, v_all, a_all, b_all, B, T):
        """Packed-conv q/k/v and the raw a/b rows -> the fused recurrent op's inputs.

        q/k/v become [B, T, heads, D] (q/k repeated up to the value-head count); beta and g are
        [B, T, Nv]. Consumes every input."""
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1 = ttnn.L1_MEMORY_CONFIG
        q_all = _replace(q_all, ttnn.reshape(q_all, (B, T, Nk, Dk)))
        k_all = _replace(k_all, ttnn.reshape(k_all, (B, T, Nk, Dk)))
        v_all = _replace(v_all, ttnn.reshape(v_all, (B, T, Nv, Dv)))
        rf = Nv // Nk
        if rf != 1:
            q_all = _replace(q_all, ttnn.repeat_interleave(q_all, rf, dim=2))
            k_all = _replace(k_all, ttnn.repeat_interleave(k_all, rf, dim=2))
        a_new = self._rows_to_users(a_all, B, T, Nv)
        b_new = self._rows_to_users(b_all, B, T, Nv)
        beta_all = ttnn.sigmoid(b_new, memory_config=_L1)
        ttnn.deallocate(b_new)
        g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_new, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a_new)
        return q_all, k_all, v_all, beta_all, g_all

    def _out_tail(self, o_all, z_all, R, bucket):
        """Gated norm, out-proj and all-reduce over R rows, zero-padded to the bucket.

        rms_norm normalises over the last dim, so one call covers every row. Consumes o_all, z_all."""
        tw, Nv, Dv = self.tw, self.Nv, self.Dv
        _L1, mc = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        out_n = ttnn.rms_norm(ttnn.reshape(o_all, (R, Nv, Dv)), weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
        ttnn.deallocate(o_all)
        out_f_b = ttnn.reshape(out_n, (1, R, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f_b, z_all, mc)
        ttnn.deallocate(out_f_b)
        ttnn.deallocate(z_all)
        return self._out_proj_tail(gated, R, bucket)

    def _out_proj_tail(self, gated, R, bucket, decode_ar=None):
        """Out-proj + all-reduce (+ the R < bucket zero pad) over gated [1, R, value_dim_tp] rows; consumes gated.
        The tail of _out_tail, shared with the gdn_spec_step paths (whose op emits the gated rows itself).
        decode_ar (QWEN36_VERIFY_DAR): all-reduce into the replicated RES40 residual (no zero pad: bucket == R)."""
        tw = self.tw
        mc, rm = ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
        if decode_ar is not None:
            assert R == bucket, f"QWEN36_VERIFY_DAR needs a bucket of exactly the real rows (R={R}, bucket={bucket})"
            return decode_ar.all_reduce(partial, keep_fp32=True)  # partial is bf16 here, as _forward_decode_step_op
        o_red = tt_all_reduce(
            partial, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=self.args.ccl_topology(), memory_config=mc
        )
        if R < bucket:
            o_rm = ttnn.to_layout(o_red, rm)
            ttnn.deallocate(o_red)
            pad = self._verify_pad_buf(bucket - R, o_rm.shape[-1], o_rm.dtype, rm, mc)
            o_full = ttnn.concat([o_rm, pad], dim=2, memory_config=mc)
            ttnn.deallocate(o_rm)
            o_red = ttnn.to_memory_config(ttnn.to_layout(o_full, ttnn.TILE_LAYOUT), mc)
            ttnn.deallocate(o_full)
        return o_red

    def _verify_fullbatch(self, qkv_all, z_all, a_all, b_all, T, n_users, bucket, state_blk_idx, conv_sel):
        """Device body of a batched verify over already-projected [1, B*T, *] rows; no per-token loop."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        C = self.qkv_dim_tp
        B = n_users

        # cat = [E_prev(K-1+T) ; new qkv(T)] per user; conv_sel is one-hot, so the matmul is an exact
        # row gather (bf16 in, HiFi4 + fp32 accumulate) that drops the rows the last commit rejected
        # and appends the new inputs. E_new is both the next E_prev and the conv input.
        qkv_new = self._rows_to_users(qkv_all, B, T, C)
        cat = ttnn.concat([self._verify_win_buf, qkv_new], dim=1, memory_config=mc)  # [B, K-1+2T, C]
        ttnn.deallocate(qkv_new)
        E_new = ttnn.matmul(conv_sel, cat, compute_kernel_config=self._cfg_onehot, memory_config=mc)
        ttnn.deallocate(cat)
        # In place, so the trace's baked address survives. The staleness flags are deliberately not
        # touched: for the whole spec loop the durable conv truth is _verify_win_buf plus the host's
        # mi, so both the K conv_states taps and _conv_win_buf are behind it, and marking one stale
        # would claim the other is current. No in-loop consumer may read either; the loop's own carry
        # is this buffer and conv_sel.
        ttnn.copy(E_new, self._verify_win_buf)
        q_all, k_all, v_all = self._kda_conv_packed(E_new, T)  # each [1, B*T, width], SiLU applied
        ttnn.deallocate(E_new)

        q_all, k_all, v_all, beta_all, g_all = self._heads_and_gates(q_all, k_all, v_all, a_all, b_all, B, T)

        # Ring mode: each (user, head)'s initial state is the ring block state_blk_idx names, and every
        # token's state is written back in place. The returned state is self._spec_ring itself, so
        # there is nothing to copy or free.
        o_all, _ring = fused_recurrent_gated_delta_rule_ttnn(
            q_all,
            k_all,
            v_all,
            beta_all,
            g_all,
            scale=self.scale,
            initial_state=self._spec_ring,
            device=self.mesh,
            output_per_token_state=True,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
            initial_state_block_idx=state_blk_idx,
        )
        for t in (q_all, k_all, v_all, beta_all, g_all):
            ttnn.deallocate(t)

        return self._out_tail(o_all, z_all, B * T, bucket)

    def _verify_pad_buf(self, rows, width, dtype, layout, mc):
        """Persistent zero pad for the trace-safe verify output (bucket - valid_len rows). Allocated
        once per (rows,width,dtype) at a fixed address so the pad concat is trace-capturable; ttnn.zeros
        allocates a fresh buffer each call, which is a host write that TT_FATALs inside a captured trace.
        First call (verify warmup, before begin_trace_capture) allocates; later calls reuse the buffer."""
        cache = getattr(self, "_verify_pad_cache", None)
        if cache is None:
            cache = self._verify_pad_cache = {}
        key = (rows, width, dtype, layout)
        buf = cache.get(key)
        if buf is None:
            buf = ttnn.zeros([1, 1, rows, width], device=self.mesh, dtype=dtype, layout=layout, memory_config=mc)
            cache[key] = buf
        return buf

    def _ensure_conv_win(self):
        """Allocate the persistent [B, K, qkv_dim_tp] shift register once, seeded from conv_states.

        Allocate-and-seed happens on an eager call (prepare_spec_verify, the spec loop's seed),
        never lazily inside a captured trace: by capture time the buffer exists and the trace body
        only ever reads/writes it at fixed offsets, which the warmup pass has already compiled.

        Re-seeds an EXISTING buffer whose taps moved underneath it (`_conv_win_stale`: a new prompt's
        prefill, a reset, a plain decode step)."""
        if self._conv_win_buf is None:
            self._conv_win_buf = ttnn.zeros(
                [self.B, self.K, self.qkv_dim_tp],
                device=self.mesh,
                dtype=self.conv_states[0].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._conv_win_stale = True
        if self._conv_win_stale:
            self.sync_conv_win()

    def sync_conv_taps(self):
        """Rebuild conv_states[0..K-1] from the persistent window — the inverse of sync_conv_win.

        A batched verify advances only _conv_win_buf / _verify_win_buf (see _verify_fullbatch), so
        the K tap buffers go stale while nothing reads them. Every tap consumer calls this first; it
        is a no-op (zero device ops, so safe in a traced body) unless a verify ran since the taps
        were last written."""
        if not self._conv_taps_stale:
            return
        self._conv_taps_stale = False
        if self._conv_win_buf is None or self.conv_states is None:
            return
        B, C = self.B, self.qkv_dim_tp
        for j in range(self.K):
            row = ttnn.slice(self._conv_win_buf, (0, j, 0), (B, j + 1, C))  # [B, 1, C]
            src = ttnn.reshape(row, (1, B, C))
            ttnn.copy(src, self.conv_states[j])
            ttnn.deallocate(src)
            if B > 1:  # B == 1 makes the reshape an identity (src is row)
                ttnn.deallocate(row)

    def sync_conv_win(self):
        """Mirror conv_states[0..K-1] into the persistent [B, K, qkv_dim_tp] shift-register buffer.

        conv_states stay the source of truth outside the spec loop (prefill fills them, the decode
        path shifts them, slot edits rewrite them); this buffer is the copy the batched verify seeds
        E_prev from, so it has to be re-seeded whenever conv_states are set from outside."""
        if self._conv_win_buf is None or self.conv_states is None:
            return
        # A non-spec fused/packed decode leaves conv_states stale (_conv_fmt "hist"/"packed"); the spec seed reads them as the
        # truth, so bring them forward first. Only the decode binding ever holds those formats.
        if self._conv_fmt in ("packed", "hist") and self.B == self._decode_B and self._conv_hist_rm is not None:
            self._ensure_rm_hist_valid()  # packed -> hist
            self._sync_conv_states_from_hist()
            self._conv_fmt = "both"
        if self._conv_taps_stale:
            # The window is the truth here (a verify advanced it and left the taps behind), so copying
            # the taps over it would undo the verify. Bring the taps forward instead; both mirrors
            # then agree and there is nothing left to copy.
            self.sync_conv_taps()
            self._conv_win_stale = False
            return
        B, C = self.B, self.qkv_dim_tp
        rows = [ttnn.reshape(c, (B, 1, C)) for c in self.conv_states]  # [1,B,C] -> [B,1,C]
        w = ttnn.concat(rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, K, C]
        ttnn.copy(w, self._conv_win_buf)
        ttnn.deallocate(w)
        if B > 1:  # B == 1 makes the reshape an identity (rows[j] is conv_states[j])
            for r in rows:
                ttnn.deallocate(r)
        self._conv_win_stale = False
