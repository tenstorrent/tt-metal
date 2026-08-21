# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel Gated DeltaNet for Qwen3.5.

Recurrence is per value-head (no cross-device comms inside); all-reduce after row-parallel out.
Reuses `recurrent_gated_delta_rule_decode_ttnn`; weights interleaved. GDN norm uses raw weight
(no +1) + SiLU(z) gate — distinct from QK/layer norms.
"""

import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.gdn.conv_fir_wh import (
    causal_conv1d_fir_dispatch as _causal_conv1d_fir,  # Upstream _causal_conv1d_fir on Blackhole; on Wormhole a local fork that builds the padded; input in ROW_MAJOR so the K shifted taps stop untilizing the whole tensor each; (UntilizeWithUnpadding 1,033us -> 15us at seq 2048). See conv_fir_wh.py.
)
from models.demos.blackhole.qwen36.tt.gdn.recurrent_decode_wh import (
    recurrent_gated_delta_rule_decode_dispatch as recurrent_gated_delta_rule_decode_ttnn,  # Upstream on Blackhole; on Wormhole a local variant that skips q's dead-weight fp32 promotion (q never feeds the state write). See recurrent_decode_wh.py.
)
from models.demos.blackhole.qwen36.tt.wh_compat import apply as _apply_wh_compat
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)

_apply_wh_compat()  # Wormhole GDN L1 adjustments (see tt/wh_compat.py)
from models.tt_transformers.tt.ccl import tt_all_reduce

# Wormhole prefill conv1d: replace the K-1 carry concat with a native-pad big conv plus a small
# patch conv. Implemented and verified BIT-EXACT (test_gdn_conv1d_splice_bitexact, 9/9 cases,
# max|diff| == 0.0), but measured NET NEUTRAL in the real layer -- the patch conv costs ~160us on one
# core no matter how few rows it emits, which eats the 205us concat saving. Full measurements are in
# _conv1d_prefill, right after the conv call -- including why the MAC FIR patch does not rescue it.
# Flip to True to re-test if slice_write or the patch step ever get cheaper.
_SPLICE_CARRY = False


def _softplus_add(a, bias):
    """g-gate: softplus(a + bias) fused into one op (softplus as a post-activation on the add)."""
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def _silu_mul(x, z, memory_config):
    """out-gate: x * silu(z). NOT fused into one op: fusing silu via input_tensor_b_activations
    overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small inputs)."""
    return ttnn.multiply(x, ttnn.silu(z, memory_config=memory_config), memory_config=memory_config)


def _gqa_expand_heads(t, repeats, batch, n_heads, head_dim, memory_config):
    """GQA Nk->Nv expand without a TILE->ROW_MAJOR->TILE round-trip.

    ``ttnn.repeat_interleave`` always untilizes TILE inputs so its concat is not subject to
    32-padding (Nk is padded to 32). ``ttnn.concat`` only untilizes when the *concat dim itself* is
    padded; Dk is already tile-aligned (128), so concatenating ``repeats`` copies along dim=-1 stays
    TILE. Reshape ``[B, Nk, rf*Dk] -> [B, 1, Nv, Dk]`` is logical repeat_interleave (each head row
    becomes ``[h_i | h_i | ...]`` then folds into ``rf`` rows).

    Lands ``[B, 1, Nv, Dk]`` (not ``[B, Nv, 1, Dk]``): last-two ``(Nv, Dk)`` keeps rms_norm on a
    packed tile instead of padding every head into its own 32-row tile."""
    out_shape = (batch, 1, n_heads, head_dim)
    if repeats == 1:
        return ttnn.reshape(t, out_shape, memory_config=memory_config)
    return ttnn.reshape(
        ttnn.concat([t] * repeats, dim=-1, memory_config=memory_config),
        out_shape,
        memory_config=memory_config,
    )


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's linear_attn.* weights across the mesh."""
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp

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
        # Optional trailing zero rows per device so the fused width's TILE COUNT factors well for the
        # prefill matmul (see gdn_qkvzab_pad_tiles in model_config.py). The pad sits AFTER b, past every
        # slice offset in _project_qkvzab, so it is never read — it only changes the matmul's N.
        _pad_rows = getattr(args, "gdn_qkvzab_pad_tiles", 0) * 32
        # ab_gap: zero rows between a and b so b starts on a tile boundary (see gdn_ab_gap in
        # model_config.py) — removes _project_qkvzab's a/b-split untilize/retilize round-trip.
        _ab_gap = getattr(args, "gdn_ab_gap", 0)
        _blocks = []
        for d in range(tp):
            _parts = [
                qkv_re[d * qkv_per : (d + 1) * qkv_per],
                z_w[d * z_per : (d + 1) * z_per],
                a_w[d * nv_per : (d + 1) * nv_per],
            ]
            if _ab_gap:
                _parts.append(torch.zeros(_ab_gap, qkv_re.shape[-1], dtype=qkv_re.dtype))
            _parts.append(b_w[d * nv_per : (d + 1) * nv_per])
            if _pad_rows:
                _parts.append(torch.zeros(_pad_rows, qkv_re.shape[-1], dtype=qkv_re.dtype))
            _blocks.append(torch.cat(_parts, dim=0))
        fused = torch.cat(_blocks, dim=0)
        # proj_1d_decode: interleaved weight (fast small-grid 1D decode matmul; prefill AGMM verified
        # bit-identical on interleaved). Distinct cache suffix.
        _proj1d = getattr(args, "proj_1d_decode", False)
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.gdn_qkvzab_weight_memcfg,
            # Padded/gapped and plain weights have different shapes, and as_tensor reloads a cache
            # file as-is — so both the pad AND the gap must qualify the cache key or a stale file
            # silently wins.
            cache_path=c(
                "qkvzab"
                + (".il" if _proj1d else ".dramshard")
                + (f".pad{_pad_rows}" if _pad_rows else "")
                + (f".abgap{_ab_gap}" if _ab_gap else "")
            ),
            dtype=ttnn.bfloat8_b,
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
            dtype=ttnn.bfloat8_b,
        )
        # Separate A+B projection (column-parallel fallback). Same ab_gap as the fused path (see
        # gdn_ab_gap in model_config.py) so the a/b split in _project_qkvzab stays tile-native.
        _ab_gap = getattr(args, "gdn_ab_gap", 0)
        _ab_parts_per_device = lambda d: (  # noqa: E731
            [a_w[d * nv_per : (d + 1) * nv_per]]
            + ([torch.zeros(_ab_gap, a_w.shape[-1], dtype=a_w.dtype)] if _ab_gap else [])
            + [b_w[d * nv_per : (d + 1) * nv_per]]
        )
        ab = torch.cat(
            [torch.cat(_ab_parts_per_device(d), dim=0) for d in range(tp)],
            dim=0,
        )
        tw["ab"] = tpc.shard_w(
            ab,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=c("ab" + (f".abgap{_ab_gap}" if _ab_gap else "")),
            dtype=ttnn.bfloat8_b,
        )
    # Row-parallel out projection: DRAM-width-sharded (like the in-proj) — decode tput win.
    _out_sharded = getattr(args, "gdn_out_weight_memcfg", None) is not None
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.gdn_out_weight_memcfg if _out_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out.dramshard" if _out_sharded else "out"),
        dtype=ttnn.bfloat8_b,
    )
    # Per-head params
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    # Conv taps (4), sharded per Q/K/V head grouping
    taps = tpc.prepare_conv_taps(conv1d_w, key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    # Same taps stacked to [K, 1, qkv_dim] (-> [K,1,qkv_dim_tp] per device) for forward_decode's
    # one-shot FIR: multiply([K,B,C],[K,1,C]) + sum(dim=0) replaces the K-step multiply/mac chain.
    # Kept ALONGSIDE the per-tap list, which the prefill FIR paths (_causal_conv1d_fir) still use.
    tw["conv_taps_stack"] = tpc.shard_small(torch.stack([t.reshape(1, -1) for t in taps], dim=0), mesh, c("tap_stack"))
    # Depthwise conv1d weight [qkv_dim, 1, K], host-held mesh-sharded (dim=0) for prepare_conv_weights / _conv1d_prefill.
    W1d = torch.stack(taps, dim=-1).reshape(args.gdn_qkv_dim, 1, args.gdn_conv_kernel_size).contiguous()
    tw["conv_w1d"] = ttnn.from_torch(
        W1d,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
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
        # Zero-column gap between a and b in the fused/ab weight so b starts on a tile boundary —
        # see gdn_ab_gap in model_config.py and _project_qkvzab's a/b slice below. 0 on every config
        # except Wormhole 9B (model_config scopes it), which restores the pre-gap slice dance below.
        self._ab_gap = getattr(args, "gdn_ab_gap", 0)
        # Wormhole 9B ONLY: the TILE-preserving decode rework (concat GQA expand instead of
        # repeat_interleave, matmul-native recurrence output, L1-resident rec_state, beta/g left at
        # their natural rank). MEASURED and validated on N300 + Qwen3.5-9B only, so Blackhole keeps
        # its tuned path byte-for-byte and the 27B (dim 5120) keeps the geometry it was validated
        # with. Same dim-based gate as _qkv_l1_tuned_for_this_model in forward_prefill (HF_MODEL is
        # often a hashed snapshot dir, so name-based checks are unreliable).
        self._decode_tile_opt = (not tpc.is_blackhole()) and args.dim <= 4096
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
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
        self.cfg = tpc.COMPUTE_HIFI2
        # Must match load_gdn_weights_tp gates
        self._dram_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
        self._fuse_ab = self._dram_sharded
        # Fuse prefill norm-allgather + qkvzab in-proj into all_gather_minimal_matmul_async.
        # Requires the folded qkvzab weight; norm's post-AG is disabled in layer.py (GDN, prefill).
        # BH-only: all_gather_matmul_prefill's grid assumes BH's taller (9-10 row) compute grid; WH
        # tops out at 8 rows, so this fusion is unvalidated there. Must match layer.py's
        # _fuse_norm_agmm gate. Falls back to the unfused AG + matmul path on WH.
        self._fuse_agmm = self._fuse_ab and tpc.is_blackhole()
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        # BH-only: matmul_reduce_scatter_prefill's default grid=(8,8)/rs_offset=(0,8) needs rows 8-9,
        # which don't exist on WH's 8-row grid. Falls back to the unfused matmul + all_reduce path.
        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1 and tpc.is_blackhole()
        # Pre-build chunk masks once (trace-safe; avoids from_torch inside captured trace)
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        # Prefill fused-op constant tiles, owned by this layer (avoids process-lifetime C++ cache vs device lifetime).
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

        self._fused_const_tiles = build_fused_const_tiles(mesh, _FUSED_CHUNK_SIZE)
        self.conv_states = None
        self.rec_state = None
        # In-place state updates for decode/prefill traces (set by model allocate_kv_caches)
        self._stable_state = False
        self.conv_carry = None  # cross-chunk prefill conv carry [1, K-1, qkv_dim_tp]
        # Native ttnn.conv1d depthwise prefill; L1_FULL slice keeps it trace-safe.
        # Only used when valid_len is None (masked buckets keep the MAC FIR).
        # Native depthwise ttnn.conv1d vs the MAC FIR fallback. The native path is pinned to L1 by
        # slice_config=Conv2dL1FullSliceConfig (see _conv1d_prefill) — it deliberately avoids the
        # DRAM-slicing path because that does host reads a trace capture rejects. On Wormhole that L1
        # pinning is exactly what breaks: the conv's statically-allocated circular buffers collide with
        # the L1 tensors already resident ("clash with L1 buffers"; CBs are L1-only, so the conv itself
        # has to move, not just its inputs). The MAC FIR is the reference implementation — already the
        # path taken for masked buckets — and runs happily from DRAM. Measured on N300: switching to it
        # clears every CB clash in the GDN TP suite (11/11). Blackhole keeps the tuned native conv1d.
        # Native on BOTH arches now. MEASURED (N300, seq 2048, single-layer profile): the 16-op MAC FIR
        # (2,833us) collapses to ~970us of which the conv2d kernel itself is only 200us —
        # 17,907us -> ~16,4xxus for the layer.
        #
        # This was long believed unshippable on WH: at B=8/B=32 the per-user prefill path died with
        #   "clash with L1 buffers ... L1 buffer allocated at 809856 and
        #    static circular buffer region ends at 892160"
        # ROOT CAUSE (found 2026-08, fixed in forward_decode): the conv runs with only ~2% L1 headroom
        # (its own buffers ~443KB/core + an ~871KB/core CB region against 1,337KB/core usable), so ANY
        # sizeable L1-resident tensor tips it over. The culprit was
        # recurrent_gated_delta_rule_decode_ttnn handing the recurrent state back L1-RESIDENT: the
        # un-split decode branch assigned it straight to self.rec_state, parking Nv*Dk*Dv*4 = 1MB
        # (16KB/bank) in L1 across the whole of the NEXT prefill call. It presented as
        # nondeterministic — failing on the 5th call, not the 2nd — only because it raced CPython GC of
        # other shadowed tensors. See the spill in forward_decode.
        #
        # Dead ends, for anyone re-tuning this (all measured):
        #   * act_block_h_override=32/64  -> CB region end byte-IDENTICAL (892160).
        #   * core_grid forced 8x8 / 8x2  -> same 892160 CB end AND same 809856 L1 address, though the
        #                                    reported core range does change.
        #   * config_tensors_in_dram=True -> HANGS the op and wedges the ETH cores.
        #   * The conv alone on a bare device is FINE at T=128..2048, which is what proved the problem
        #     was resident L1 rather than anything shape-internal to the conv.
        #
        # CAUTION when experimenting here: a mid-flight program failure inside the per-user prefill
        # loop leaves the CCL fabric inconsistent, so the NEXT process hangs on its first collective and
        # reports a misleading result. Recover with `tt-topology -l mesh` (this host is MESH; the tool's
        # default is linear and flashing it breaks device discovery) and test one variant per reset.
        # QWEN35_GDN_CONV1D=0 falls back to the MAC FIR (which masked buckets use regardless).
        _conv1d_env = os.environ.get("QWEN35_GDN_CONV1D")
        self._gdn_conv1d = True if _conv1d_env is None else (_conv1d_env == "1")
        # Prepared depthwise conv weights, keyed by (input_width, padding) — the padded (from-scratch)
        # and concat (carried) forms are two different conv geometries. Populated on first prefill call.
        self._conv1d_wprep = None
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill

    # Usable L1 (bytes, whole device) left for the recurrent state once the decode kernel's own
    # allocations are in place. Measured on an 8x8 Wormhole grid at B=32: before requesting the state
    # the kernel already holds ~864KB of the ~1336KB usable per bank, leaving ~485KB/bank -> ~31MB
    # across 64 banks. Blackhole has both a larger L1 and (at TP=4) a smaller per-device Nv, and its
    # batched decode is already validated there, so the split is Wormhole-only.
    _DECODE_STATE_L1_BUDGET = 31 * (1 << 20)
    # Approximate on-device bytes/element by rec_state dtype (bfloat8_b packs ~1 B/elem plus a small
    # shared-exponent overhead, rounded down here since this budget is already a conservative
    # estimate). Used by _decode_batch_split so the split threshold tracks whatever dtype
    # reset_state actually allocated instead of assuming fp32.
    _STATE_BYTES_PER_ELEM = {ttnn.bfloat8_b: 1, ttnn.bfloat16: 2, ttnn.float32: 4}

    def _decode_batch_split(self, B):
        """Largest batch slice whose recurrent state fits the decode kernel's spare L1.

        Returns B itself (no splitting, byte-identical to the validated path) whenever the whole
        batch fits — which is every case on Blackhole, and B<=16 on a Wormhole N300 at fp32 state
        (more at bf16/bf8 -- see _STATE_BYTES_PER_ELEM).
        """
        if tpc.is_blackhole():
            return B
        elem_bytes = self._STATE_BYTES_PER_ELEM.get(self.rec_state.dtype, 4)
        per_user = self.Nv * self.Dk * self.Dv * elem_bytes  # one user's state, actual dtype
        # x2: the decode kernel holds the pre-decay state AND the freshly-decayed copy of it live at
        # once (recurrent_gated_delta_rule_decode_ttnn's `h = ttnn.multiply(h, decay_bhkv, ...)` does
        # not free the original `h` first) -- the budget must cover both, not just one copy.
        max_b = max(1, self._DECODE_STATE_L1_BUDGET // (2 * max(1, per_user)))
        if max_b >= B:
            return B
        # Prefer an even split into equal power-of-2-friendly slices (B is always a power of 2 here).
        step = 1
        while step * 2 <= max_b:
            step *= 2
        return step

    def _spill_rec_state_to_dram(self):
        """Move rec_state out of L1 before prefill. Native conv1d has ~2% L1 headroom; a resident
        [B,Nv,Dk,Dv] state is the clash documented on this class (see the _gdn_conv1d comment).
        No-op when the buffer is already DRAM or unset. Must not run under _stable_state: that
        path bakes rec_state's address into prefill/decode traces."""
        if self.rec_state is None or self._stable_state:
            return
        if self.rec_state.memory_config().buffer_type == ttnn.BufferType.DRAM:
            return
        spilled = ttnn.to_memory_config(self.rec_state, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(self.rec_state)
        self.rec_state = spilled

    def _promote_rec_state_to_l1(self):
        """Eager-decode only: hoist rec_state into L1 so the recurrent kernel's DRAM->L1 copy
        (to_memory_config) and the post-step L1->DRAM spill are both skipped. Not used under
        _stable_state — those traces bake the DRAM address allocated in reset_state. Also skipped
        when the batch does not fit L1 (the existing _decode_batch_split path), and on any config
        outside the Wormhole-9B scope this rework was measured on (_decode_tile_opt) -- Blackhole in
        particular never splits, so without that gate it would newly park a multi-MB fp32 state in
        L1 for the whole decode."""
        if not self._decode_tile_opt or self.rec_state is None or self._stable_state:
            return
        if self.rec_state.memory_config().buffer_type == ttnn.BufferType.L1:
            return
        B = int(self.rec_state.shape[0])
        if self._decode_batch_split(B) < B:
            return
        promoted = ttnn.to_memory_config(self.rec_state, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(self.rec_state)
        self.rec_state = promoted

    def reset_state(self):
        # Allocated directly on device. These are pure zeros, so the host torch.zeros + PCIe upload
        # they replace bought nothing — and the recurrent state is the model's largest such buffer
        # ([B, Nv, Dk, Dv] fp32 is tens of MB at B=32), re-materialized on every sequence reset.
        def z(shape, dtype=ttnn.bfloat16):
            return ttnn.zeros(ttnn.Shape(list(shape)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh)

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        # fp32 recurrent state on Blackhole (QWEN35_GDN_STATE_BF16=1 reverts to bf16 there too).
        # Wormhole defaults to bfloat16 state.
        #
        # A bfloat8_b state was tried here (smaller footprint, eases _decode_batch_split pressure, and
        # a faster read/query matmul -- BF16 x BF8 -> BF16 measured ~44% faster than BF16 x BF16 at the
        # real production shape). It was reverted: the full test suite showed real accumulation-drift
        # failures directly attributable to it (confirmed by rerunning against the pre-bf8 commit) --
        # test_gdn_tp_fused_chunk_prefill (fused-chunk prefill vs 256-step decode PCC 0.9776 < 0.99) and
        # test_model_tp_decode_batched B8/B32 (batched decode logits PCC 0.9519 < 0.97 for one user at
        # len=128). bf16 state does not reproduce either failure.
        # RE-VERIFIED (after the TILE-preserving GQA-expand/ab_gap decode rework) that the bf8-state
        # revert above still stands, so do not retry it: test_gdn_tp.py passes clean with bf8 state
        # (single-layer, <=2 decode steps is too short a window for the drift to show), but
        # test_model_tp_decode_batched still fails at BOTH B8 and B32 with the same signature as the
        # original revert -- user 1 step 3 logits PCC 0.9660 < 0.97. It is not even a perf win at
        # B=32: device time 2,917->1,933us (-33.7%, ops 111->66, since bf8's halved per-user bytes
        # put max_b at 62 >= 32 so _decode_batch_split stops splitting) but op-to-op gap
        # 5,473->7,903us (+44.4%), for a WORSE effective total (~9,836 vs ~8,390us) -- this step is
        # dispatch-bound, so the split path's more-numerous launches actually schedule tighter.
        if tpc.is_blackhole() and os.environ.get("QWEN35_GDN_STATE_BF16") != "1":
            self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv), dtype=ttnn.float32)
        else:
            self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv))
        # Cross-chunk conv carry + persistent zero sources (created before any trace)
        self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        self._zero_conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        # _zero_rec must match self.rec_state's dtype for reset_state_inplace's ttnn.copy to work.
        self._zero_rec = z((self.B, self.Nv, self.Dk, self.Dv), dtype=self.rec_state.dtype)
        # Chunk-outer batched-prefill conv left-context (allocated lazily by forward_prefill_batched).
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
        self._batched_conv_carry = None

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

    def _col_proj(self, x, weight, decode_progcfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG, prefill_out_dtype=None):
        """Column-parallel qkvz projection; DRAM-sharded decode matmul when enabled.
        out_memory_config: decode result placement (default DRAM; L1 keeps it resident).
        prefill_out_dtype: pin the PREFILL result dtype (see sharded_decode_matmul); needed because
        ttnn.linear otherwise inherits in0's dtype, and in0 here can be a bf8 norm output. Callers
        only ever pass it when in0 IS narrowed, which never happens in decode, so the non-sharded
        limb below can apply it unconditionally."""
        if not self._dram_sharded:
            return ttnn.linear(
                x,
                weight,
                compute_kernel_config=self.cfg,
                memory_config=out_memory_config,
                **({"dtype": prefill_out_dtype} if prefill_out_dtype is not None else {}),
            )
        _kpass1 = getattr(self.args, "gdn_qkvzab_prefill_progcfg", None)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            # PREFILL BLOCKING. This used to be the layer's worst matmul (2,151us at 36.6% of peak,
            # output subblock 1x1): the folded [qkv|z|a|b] width is 6176 = 193 tiles, 193 is PRIME, so
            # at 8 columns per_core_N=25 whose only divisors are 1/5/25. Under COMPUTE_HIFI2 the output
            # subblock is capped at 4, which 25 does not divide, so out_subblock_w collapsed to 1 and
            # (out_block_w being a multiple of it) _safe_half_out_block_w could only reach 5 — five K
            # passes, each re-reading the 16MB DRAM-resident in0. out_block_w=25 (one pass) overflowed
            # L1: "circular buffers ... grow to 1683136 B which is beyond max L1 size of 1499136 B".
            #
            # Both of those limits come from fp32_dest_acc_en, not from the shape. Off, the subblock
            # ceiling rises 4 -> 8 (so 25 -> out_subblock_w=5) and the intermediate CB halves (so
            # out_block_w=25 fits). Hence the pair below: the one-K-pass progcfg AND the matching
            # no-fp32-acc compute config, prefill only — decode keeps self.cfg. They are coupled and
            # must be passed together; the progcfg's blocking is illegal with fp32 dest acc on.
            #
            # MEASURED, device kernel duration, N150, M=2048 K=4096 (tests/perf/test_gdn_inproj_sweep.py
            # sweeps ~200 points across blocking x compute cfg x N padding x grid width):
            #     N=6912  sub_w=3 blk_w=9   fp32_acc ON   3 passes  1493us   <- previous config
            #     N=6176  sub_w=5 blk_w=25  fp32_acc OFF  1 pass    1255us   -16.0%
            # This also let the 6176 -> 6912 pad be reverted (see gdn_qkvzab_pad_tiles in
            # model_config.py), so the shape is back to 11.9% fewer FLOPs as well. Costs PCC
            # 0.99997 -> 0.99992 vs an fp32 reference.
            #
            # Also measured and rejected: routing this through the subblock-maximizing
            # create_prefill_mlp_matmul_program_config picks 7 columns -> per_core_N=28, and even with
            # one K pass that is slower (1,309us) — losing 8 of 64 cores costs more than the wider
            # subblock wins. Moving the output to L1 is impossible at this blocking: the full
            # per_core_N-wide CBs leave no room ("statically allocated circular buffers clash with L1
            # buffers ... L1 buffer allocated at 684032, static CB region ends at 939200").
            #
            # gdn_qkvzab_prefill_progcfg is None on Blackhole (it takes the fused AGMM path instead),
            # so fall back to the shared factory + self.cfg there.
            _kpass1 or self.args.prefill_progcfg,
            self.args.dim,
            decode_out_memory_config=out_memory_config,
            prefill_compute_cfg=tpc.COMPUTE_HIFI2_NO_FP32_ACC if _kpass1 is not None else None,
            prefill_out_dtype=prefill_out_dtype,
        )

    def _normalize_valid_len(self, valid_len, T):
        """``valid_len >= T`` means "no padding in this chunk" == ``None``. See forward_prefill for
        why the None form is the one to take (cheaper, trace-safe, better tested) and why the
        normalization is Wormhole-only."""
        if (
            not tpc.is_blackhole()
            and valid_len is not None
            and not isinstance(valid_len, (list, tuple))
            and valid_len >= T
        ):
            return None
        return valid_len

    def prefill_uses_native_conv1d(self, T, valid_len=None):
        """Will a T-token prefill chunk with this ``valid_len`` take the native ttnn.conv1d depthwise
        path (True), or the MAC FIR fallback (False)?

        Public because the CALLER needs it before the call: layer.py decides attention_norm's prefill
        gather dtype from it (the FIR's ttnn.addcmul requires all three operands to share a dtype, so
        the FIR path forces bf16 -- see layer.py). This is the same expression forward_prefill uses
        for ``_use_native_conv1d``, exported rather than duplicated so the two cannot drift: a caller
        that narrowed the gather while the conv silently fell back to the FIR would crash in
        ternary.cpp, and only on the masked tail chunk that no single-layer perf test exercises."""
        return self._gdn_conv1d and self._normalize_valid_len(valid_len, T) is None and self._conv1d_native_fits_l1(T)

    def _conv1d_native_fits_l1(self, T):
        """Can ttnn.conv1d's statically-allocated CBs fit L1 for a T-token prefill?

        The height-shard grid is chosen from the OUTPUT tile count:
        determine_parallel_config() calls find_closest_largest_divisor_with_num_padding_and_mult(
        out_nhw_ntiles, max_num_cores, ...), so with out_ntiles = ceil(T/32):

            out_ntiles <= num_cores  ->  1 output tile per core
            out_ntiles >  num_cores  ->  >=2 tiles per core

        and the activation CBs scale with that tile count. They are already near the limit at one
        tile because this is a COALESCED 1D depthwise: act_block_w is
        round_up(in_channels * kernel_w, TILE_WIDTH) = qkv_dim_tp * K -- 512 tiles at C=4096, K=4,
        i.e. ~1MB of tilized activation per core on its own. At two tiles per core it overflows:

            T=2048 -> 64 out tiles -> 64 cores x 1 tile -> fits (this is the profiled path)
            T=2304 -> 72 out tiles -> no divisor <=64, so 40 cores x 2 tiles
                      -> "Statically allocated circular buffers ... grow to 1678592 B which is
                          beyond max L1 size of 1499136 B" (test_model_tp_long_prefill)

        Neither act_block_h_override=32 nor config_tensors_in_dram fixes this (MEASURED: 1,940,736 B
        and 1,678,624 B respectively -- the first is worse because forcing the block height also
        re-picks the grid, the second because conv config tensors live in L1_SMALL, not the CB pool).
        act_block_w is not user-controllable: coalescing is auto-selected by
        should_coalesce_1d_depthwise_conv_reads() and turning it off is not exposed through
        Conv2dConfig. So the only lever is to use this path only where it fits, and fall back to the
        MAC FIR (which is what Wormhole did unconditionally before the native conv1d was enabled).

        Blackhole has a larger L1 and was validated on this path, so it always qualifies.

        CHANNEL WIDTH is the other axis the CB size depends on (act_block_w = qkv_dim_tp * K,
        per the docstring above) and this function used to ignore it, silently assuming the
        TP=4 reference width (qkv_dim_tp=4096, "already near the limit at one tile"). At TP=2
        (N300) qkv_dim_tp DOUBLES to 8192 since fewer devices share the same total channel
        width -- same one-tile-per-core grid, but act_block_w now overflows L1 the same way
        the docstring's "2 tiles per core" example does, just via width instead of tile count.
        MEASURED: test_gdn_tp_prefill (T=128, N300) hit exactly this -- "Statically allocated
        circular buffers ... clash with L1 buffers" -- because qkv_dim_tp*K (8192*4=32768) is
        2x the validated 4096*4=16384 reference point. Cap on that reference product directly
        rather than re-deriving a new threshold: it's the same one tile/core CB budget that
        was already shown to fit at 4096*4 and overflow at higher per-core activation width.
        """
        if tpc.is_blackhole():
            return True
        grid = self.mesh.compute_with_storage_grid_size()
        _fits_grid = -(-T // tpc.TILE_SIZE) <= grid.x * grid.y
        _fits_channel_width = self.qkv_dim_tp * self.K <= 4096 * 4
        return _fits_grid and _fits_channel_width

    def _conv1d_prefill(self, qkv, T, conv_state, _force_splice=False):
        """Depthwise causal conv1d + SiLU via ttnn.conv1d. Returns (out [1,T,C], new_state [1,K-1,C]) DRAM TILE.

        Prepends K-1 carry rows with padding=0 so one program serves every chunk (native pad only zeros,
        so it can't inject cross-chunk carry into a shared trace).
        """
        dev, K, C = self.mesh, self.K, self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        Lin = (K - 1) + T
        # From-scratch chunk: let ttnn.conv1d apply the K-1 causal zero pad itself instead of
        # materialising it with ttnn.concat (197us at T=2048, a full 16MB pass). conv1d takes
        # padding=[pad_left, pad_right] and prepare_conv_weights the 2D 4-tuple
        # (top, bottom, left, right) -- both must describe the SAME asymmetric pad, and a symmetric
        # prep pad silently produces wrong values (verified: max|diff| 26.9 vs 0 for the correct pair).
        #
        # Applies to the FROM-SCRATCH chunk only (conv_state is None): per-user prefill
        # (forward_prefill(return_state=True) <- prefill_chunked_peruser) and the demo path. A carried
        # chunk still needs the concat, since native padding can only ever fill zeros -- which is why
        # upstream used the concat unconditionally to keep ONE conv program per chunk. We now get two
        # (padded / concat), so _conv1d_wprep is keyed by geometry below.
        #
        # NOT extended to the _stable_state path, though it would apply there on chunk 1: after
        # reset_state_inplace the carry buffer provably holds zeros, so a `_carry_is_zero` flag would
        # let chunk 1 take this form too. Tried, works, 11/11 tests pass -- and REVERTED, because the
        # flag has to be cleared at every site that writes conv_carry and getting that wrong makes the
        # conv silently substitute zeros for a real carry (wrong output, no error), which is a bad
        # trade for a gain the perf harness cannot even show: test_profile_single_layer_prefill runs a
        # warmup forward first, so by the measured iteration the carry is already non-zero.
        #
        # Wormhole only: this function is Blackhole's tuned default conv path, so BH keeps the
        # single-program concat form byte-identical.
        #
        # _conv_len also decides how many CORES the conv gets, and the two forms are not equal.
        # determine_parallel_config() picks the height-shard grid with
        # find_closest_largest_divisor_with_num_padding_and_mult(nhw_ntiles, max_num_cores, ...):
        #   _native_pad: _conv_len = T = 2048 = 64 tiles -> 64 divides 64 -> 64 cores x 1 tile.
        #   concat form: _conv_len = Lin = 2051 -> 65 tiles -> no divisor of 65 is <= 64, so it pads
        #                to 66 and lands on 33 cores x 2 tiles -- HALF the grid idle.
        # That matches the profile (InterleavedToShardedDeviceOperation reports 33 cores). So the
        # carry costs more than the 210us concat: it also halves the conv's input-side parallelism.
        # Removing both together is possible but needs an op-level splice -- see the note at the end
        # of this function.
        # CARRY SPLICE (Wormhole, currently OFF -- see the measured negative result after the conv
        # call below). When on, the big conv takes the native-pad form even when there IS a carry,
        # and the K-1 output rows the carry would have changed are patched afterwards by a second tiny
        # conv. That buys BOTH wins above at once -- no 205us full-tensor concat, and _conv_len = T so
        # the conv gets all 64 cores instead of 33. See the splice block after the conv call for the
        # correctness argument and the measurements. _force_concat is for
        # test_gdn_conv1d_splice_bitexact, which runs both forms in one process and compares.
        # _force_splice turns the splice on regardless of _SPLICE_CARRY so that test keeps covering it.
        # T > TILE_SIZE is a correctness guard, not an optimization gate: at T <= TILE_SIZE the
        # [0:TILE) head slice below covers the WHOLE of qkv_rm, and a full-coverage ttnn.slice returns
        # an ALIAS rather than a copy -- deallocating it then frees the big conv's own input
        # ("Tensor is not allocated", caught by test_gdn_conv1d_splice_bitexact at T=32). Falling back
        # to the concat form there costs nothing: the splice exists to kill a full-tensor concat and
        # restore the 64-core grid, and at T <= 32 the concat is one tile and the grid is not the
        # bottleneck. Prefill chunks are 128 tokens, so production never takes this branch anyway.
        _splice = (
            (_SPLICE_CARRY or _force_splice) and conv_state is not None and not tpc.is_blackhole() and T > tpc.TILE_SIZE
        )
        _native_pad = (conv_state is None or _splice) and not tpc.is_blackhole()
        _conv_len = T if _native_pad else Lin
        _conv_pad = [K - 1, 0] if _native_pad else 0
        _prep_pad = (0, 0, K - 1, 0) if _native_pad else (0, 0)
        _xfix = None  # splice input; stays None on Blackhole and on the from-scratch/concat paths
        if tpc.is_blackhole():
            # ---- Blackhole: the validated TILE prologue, byte-for-byte unchanged. ----
            # new_state: last K-1 real input tokens (for the next chunk's carry), TILE/DRAM.
            new_state = ttnn.slice(qkv, (0, T - (K - 1), 0), (1, T, C))
            new_state = ttnn.to_memory_config(ttnn.to_layout(new_state, ttnn.TILE_LAYOUT), _dram)
            if conv_state is None:
                pad = ttnn.zeros(
                    [1, K - 1, C], device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_dram
                )
                xin = ttnn.concat([pad, qkv], dim=1, memory_config=_dram)
                ttnn.deallocate(pad)
            else:
                xin = ttnn.concat([conv_state, qkv], dim=1, memory_config=_dram)
            xin = ttnn.to_layout(xin, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        else:
            # ---- Wormhole: same three tensors, built off ONE untilize of qkv. ----
            # ttnn.conv1d consumes ROW_MAJOR and the [carry|qkv] concat starts at row K-1=3 (off-tile,
            # so a TILE concat relayouts internally regardless), which made the BH prologue above pay,
            # at T=2048 on N300: a full-tensor tilize in the concat immediately undone by the
            # to_layout(ROW_MAJOR) untilize (246us + 261us), plus a THIRD full-tensor untilize (213us)
            # for a new_state slice that reads 3 rows. Doing it all in ROW_MAJOR is ~720us/layer
            # cheaper for identical output. Kept off Blackhole: that arch's larger L1 was tuned around
            # the TILE prologue and this is a WH-measured change only.
            _rm = ttnn.ROW_MAJOR_LAYOUT
            # qkv normally arrives already ROW_MAJOR: forward_prefill asks _project_qkvzab to fold
            # this untilize into the qkvzab slice (qkv_row_major=True). Keep the to_layout for the
            # callers that still hand over TILE (per-user prefill, tests calling this directly).
            qkv_rm = qkv if qkv.layout == _rm else ttnn.to_layout(qkv, _rm, memory_config=_dram)
            # Only K-1 rows, so tilizing the carry back is ~12us, not a full-tensor relayout.
            new_state = ttnn.slice(qkv_rm, (0, T - (K - 1), 0), (1, T, C))
            new_state = ttnn.to_memory_config(ttnn.to_layout(new_state, ttnn.TILE_LAYOUT), _dram)
            # Splice: build the fix conv's input HERE, while qkv_rm and conv_state are both certainly
            # alive. Doing it after the big conv would race the `deallocate(xin)` below, which frees
            # qkv_rm on the production path (qkv arrives TILE, so qkv_rm is a fresh tensor).
            # [carry | qkv[0:TILE]] = (K-1)+TILE rows -> the conv emits (K-1)+TILE-K+1 = TILE rows,
            # every one of them seeing the real carry. Measured 5us vs the 205us full-tensor concat.
            if _splice:
                _cs_rm = ttnn.to_layout(conv_state, _rm, memory_config=_dram)
                _head = ttnn.slice(qkv_rm, (0, 0, 0), (1, tpc.TILE_SIZE, C), memory_config=_dram)
                _xfix = ttnn.concat([_cs_rm, _head], dim=1, memory_config=_dram)
                ttnn.deallocate(_head)
                if _cs_rm is not conv_state:
                    ttnn.deallocate(_cs_rm)
            if _native_pad:
                # No concat at all: ttnn.conv1d applies the K-1 causal zero pad itself (padding
                # [left, right] = [K-1, 0]). VERIFIED bit-identical to the concat form at T=2048.
                xin = qkv_rm
            elif conv_state is None:
                pad = ttnn.zeros([1, K - 1, C], device=dev, dtype=ttnn.bfloat16, layout=_rm, memory_config=_dram)
                xin = ttnn.concat([pad, qkv_rm], dim=1, memory_config=_dram)
                ttnn.deallocate(pad)
            else:
                # conv_state arrives TILE (previous chunk's new_state); K-1 rows, cheap to convert.
                cs_rm = ttnn.to_layout(conv_state, _rm, memory_config=_dram)
                xin = ttnn.concat([cs_rm, qkv_rm], dim=1, memory_config=_dram)
                if cs_rm is not conv_state:
                    ttnn.deallocate(cs_rm)
            # NB: when _native_pad, xin IS qkv_rm -- freeing it here would free the conv's input.
            if qkv_rm is not qkv and xin is not qkv_rm:
                ttnn.deallocate(qkv_rm)
        # Latch this BEFORE the reshape: reshape returns a NEW Tensor object that shares xin's
        # buffer, so `xin is qkv` stops being true afterwards even though deallocating it would
        # still free qkv's buffer (and forward_prefill deallocates qkv itself).
        _xin_aliases_qkv = xin is qkv
        xin = ttnn.reshape(xin, (1, _conv_len, 1, C))
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # Needs l1_small_size on the device (prefill/demo set 24576); matches the validated A/B config.
        #
        # HEIGHT_SHARDED is not just the default here, it is the only layout that reaches the
        # specialized 1D-depthwise kernel: should_coalesce_1d_depthwise_conv_reads() bails out for
        # anything else (conv2d_utils.cpp), and BLOCK/WIDTH_SHARDED would fall back to the generic
        # im2col path. Coalescing is already active -- it lays all K taps out as one contiguous
        # activation block, which is what makes the profiled matmul shape 2048 x 16384 x 4096
        # (16384 = qkv_dim_tp * K). Nothing to turn on; it is auto-selected and already the fast path.
        #
        # Knobs checked and deliberately NOT set (Conv2dConfig, conv2d_device_operation_types.hpp):
        #   activation           -- SiLU folding drops PCC to ~0.84 on this depthwise (see below).
        #   enable_activation_reuse -- reuses data across consecutive image ROWS; input_height=1 here.
        #   force_split_reader   -- ignored unless the per-core act block height exceeds one tile,
        #                           and it is exactly one tile at T=2048 on 64 cores.
        #   deallocate_activation -- documented no-op when the input is in DRAM, which xin always is.
        #   act_block_w_div      -- ignored for HEIGHT_SHARDED.
        #   enable_act_double_buffer -- would double a 32x4096 bf16 activation CB (256KB -> 512KB per
        #                           core). This path already runs within ~2% of WH L1 (see the
        #                           nlp_concat_heads leak note in forward_prefill); not worth the cliff.
        #   enable_weights_double_buffer -- the only remaining candidate: the depthwise weight is tiny
        #                           (C x 1 x K), so the extra L1 is small. Unmeasured; A/B it before
        #                           enabling.
        # The real conv-side lever is not a config field but _conv_len -- see the note there.
        conv_cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        # Prepare conv weight once per GEOMETRY (warmup); avoids host reprocess + keeps traced replay
        # device-only. Keyed by (input_width, padding) because the padded and concat forms are two
        # different conv programs -- a single cached weight would be prepared for the wrong geometry
        # on whichever form ran second.
        if self._conv1d_wprep is None:
            self._conv1d_wprep = {}

        def _conv(x, clen, cpad, ppad):
            """One depthwise conv1d over a clen-row ROW_MAJOR input. Weight prep cached by geometry."""
            wkey = (clen, ppad)
            if wkey not in self._conv1d_wprep:
                self._conv1d_wprep[wkey] = ttnn.prepare_conv_weights(
                    weight_tensor=self.tw["conv_w1d"],
                    input_memory_config=_dram,
                    input_layout=ttnn.ROW_MAJOR_LAYOUT,
                    weights_format="OIHW",
                    in_channels=C,
                    out_channels=C,
                    batch_size=1,
                    input_height=1,
                    input_width=clen,
                    kernel_size=(1, K),
                    stride=(1, 1),
                    padding=ppad,
                    dilation=(1, 1),
                    has_bias=False,
                    groups=C,
                    device=dev,
                    input_dtype=ttnn.bfloat16,
                    conv_config=conv_cfg,
                    compute_config=cc,
                )
            return ttnn.conv1d(
                input_tensor=x,
                weight_tensor=self._conv1d_wprep[wkey],
                device=dev,
                in_channels=C,
                out_channels=C,
                batch_size=1,
                input_length=clen,
                kernel_size=K,
                stride=1,
                padding=cpad,
                dilation=1,
                groups=C,
                dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
                # L1_FULL slice: keep the conv in L1 instead of DRAM-width-slicing. The DRAM-slice path does
                # host reads that begin_trace_capture rejects (see uniad); L1_FULL is trace-safe (as UNet).
                slice_config=ttnn.Conv2dL1FullSliceConfig,
                return_output_dim=False,
                return_weights_and_bias=False,
            )

        out = _conv(xin, _conv_len, _conv_pad, _prep_pad)
        # ---- The carry splice: IMPLEMENTED, VERIFIED BIT-EXACT, and OFF because it does not pay. ----
        # Correctness argument (this part all held up): the conv is linear and causal with kernel K, so
        # output row t reads input rows [t-K+1, t]. Only rows t < K-1 reach back past row 0 and thus
        # depend on conv_state; under the native pad those K-1 rows see zeros instead of the carry and
        # EVERY row t >= K-1 is already exactly right. So the big conv can always take the native-pad
        # form (no concat, _conv_len = T, all 64 cores) and the first rows be overwritten:
        #   1. big = conv1d(qkv_rm, padding=[K-1, 0])                    -> [1,T,C], rows 0..K-2 wrong
        #   2. fix = conv1d(concat([carry, qkv_rm[0:TILE]]), padding=0)  -> TILE fully-correct rows
        #   3. silu both, then slice_write(fix, big, [0,0,0,0], [1,1,TILE,C], [1,1,1,1])
        # test_gdn_conv1d_splice_bitexact confirms torch.equal on `out` AND `new_state` for 9 cases
        # (T=32/128/2048 x nonzero/zero/no carry), max|diff| exactly 0.0. The splice is CORRECT.
        #
        # Two notes where the original design sketch was wrong:
        #  * slice_write is ROW_MAJOR-only (output interleaved, input interleaved or height/block
        #    sharded, rank 4, bf16) -- not the tiled path the sketch assumed. That is easier, not
        #    harder: the conv output is already ROW_MAJOR right after sharded_to_interleaved.
        #  * "at the cost of one extra tiny conv program" -- a tiny depthwise conv is NOT tiny here.
        #
        # MEASURED IN THE REAL LAYER (Tracy, T=2048, N300, Qwen3.5-9B), splice ON vs the concat form:
        #     concat                     205us  ->    5us     (the win, as predicted)
        #     big conv chain (i2s/halo/move/conv)
        #       33 cores  100+78+12+202 = 392us
        #       64 cores   96+11+12+202 = 321us              (the other win, as predicted)
        #     fix chain  i2s 10 + halo 10 + move 10 + conv 163 + slice_write 67 = +260us   <-- the killer
        #     TOTAL                      597us  ->  586us    (-11us: noise, plus 5 more op-to-op gaps)
        #
        # The fix conv is 163us on ONE core, and shrinking the patch does not help: measured 161.4us for
        # 3 output rows, 166.9 for 8, 176.0 for 16, 194.0 for 32. It is fixed overhead, because the
        # coalesced 1D depthwise sets act_block_w = in_channels * K = 16384 regardless of row count, so
        # a 3-row conv streams almost the same activation block as a 2048-row one -- onto a single core,
        # since 3 or 32 output rows is one tile and determine_parallel_config gives it one core.
        #
        # So the splice trades a 205us concat plus 71us of lost parallelism for a ~260us patch.
        #
        # The MAC FIR (_causal_conv1d_fir_wh) as the patch instead of ttnn.conv1d: MEASURED, and it is
        # not enough either. 141.8us for a 32-row patch plus 16.2us to untilize its TILE output for
        # slice_write (which is ROW_MAJOR-only) = 158us, vs 191us for the conv1d patch chain. The FIR
        # runs ~12 small programs (K taps x slice/tilize/multiply/addcmul) and at 32 rows every one of
        # them is fixed-overhead-bound at ~12us, so there is no shape of this patch that gets cheap.
        # Best case for the whole splice:
        #     5 (concat) + 321 (big conv, 64 cores) + 141.8 + 16.2 (FIR patch) + 67 (slice_write) = 551us
        #     vs 597us today  ->  -46us, ~7%
        # and slice_write alone is 67us for 256KB on one core, which is the next floor under it.
        #
        # NOT taken: -46us is small next to the -249us the in-proj rework banked, it needs 5 extra
        # programs per layer, and the FIR patch would make K-1 of the T output rows come from a
        # DIFFERENT kernel than the rest -- so it is no longer bit-exact with the concat form, only
        # PCC-equal, and test_gdn_conv1d_splice_bitexact's torch.equal gate would have to be weakened
        # to a tolerance on exactly the rows most likely to hide an indexing bug. Bad trade.
        #
        # Left in place but OFF (see _SPLICE_CARRY) rather than deleted, because the correctness half is
        # done and bit-exact with the conv1d patch -- if slice_write and the patch ever get cheap,
        # flipping the flag is the whole change.
        #
        # Verified bit-identical (torch.equal on both `out` and `new_state`) against the concat form by
        # tests/perf/test_gdn_conv1d_splice_bitexact.py, which runs both arms in one process via
        # _force_concat. That test is the gate this optimization was waiting on -- it is a
        # silent-wrongness risk (a bad splice produces plausible numbers, not an error), so if you
        # touch this block, re-run it.
        #
        # xin aliases qkv when the caller handed over ROW_MAJOR qkv and _native_pad skipped the
        # concat: freeing it here would free forward_prefill's qkv out from under its own
        # deallocate(). Every other path built xin as a fresh tensor, so it is ours to free.
        if not _xin_aliases_qkv:
            ttnn.deallocate(xin)
        # SiLU stays a separate op (folding it via conv_config.activation drops PCC to ~0.84 on this
        # depthwise), but it runs on the conv's SHARDED L1 output, before sharded_to_interleaved.
        # Same elementwise work either way; the difference is bandwidth. On the interleaved DRAM
        # tensor it is a 16MB read + 16MB write over the DRAM bus; on the sharded output every core
        # touches only its own L1 shard.
        #
        # Measured A/B, same session, tt-smi reset between every run, T=2048 on N300
        # (test_profile_single_layer_prefill, layer0_gdn), UnaryDeviceOperation for this SiLU:
        #     interleaved (before): 286us, 266us
        #     sharded    (after):   207.6us, 207.7us
        # ~68us/layer. Note the sharded arm is stable to 0.1us across runs while the interleaved arm
        # swings ~20us -- consistent with DRAM contention being the variable part, which is the
        # mechanism this change removes. Do NOT read the report's TOTAL device time to judge this:
        # ReduceScatterMinimalAsync alone varies 1,982/2,079/2,452us run to run, swamping the delta.
        #
        # WORMHOLE ONLY. This tail is shared with Blackhole -- for which _conv1d_prefill is the tuned
        # default conv path -- and the numbers above are N300-measured. Same reasoning as the TILE
        # prologue above: BH keeps its validated ordering until someone measures it there.
        if not tpc.is_blackhole() and out.is_sharded():
            _pre_silu = out
            out = ttnn.silu(out, memory_config=out.memory_config())
            ttnn.deallocate(_pre_silu)
            out = ttnn.sharded_to_interleaved(out, _dram)
            if _xfix is not None:
                # Patch the K-1 carry-dependent rows (see the splice note above). Runs on the
                # ROW_MAJOR interleaved `out`, which is exactly what slice_write requires, and before
                # the to_layout(TILE) so no extra relayout is introduced.
                _fix = _conv(
                    ttnn.reshape(_xfix, (1, (K - 1) + tpc.TILE_SIZE, 1, C)), (K - 1) + tpc.TILE_SIZE, 0, (0, 0)
                )
                ttnn.deallocate(_xfix)
                _fix_pre = _fix
                _fix = ttnn.silu(_fix, memory_config=_fix.memory_config())
                ttnn.deallocate(_fix_pre)
                out = ttnn.experimental.slice_write(
                    ttnn.reshape(_fix, (1, 1, tpc.TILE_SIZE, C)),
                    ttnn.reshape(out, (1, 1, T, C)),
                    [0, 0, 0, 0],
                    [1, 1, tpc.TILE_SIZE, C],
                    [1, 1, 1, 1],  # step is positional-required in this build, not defaulted
                )
                ttnn.deallocate(_fix)
            out = ttnn.reshape(out, (1, T, C))
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
            return out, new_state
        out = ttnn.sharded_to_interleaved(out, _dram)
        out = ttnn.reshape(out, (1, T, C))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
        return ttnn.silu(out, memory_config=_dram), new_state

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
                # MODEL-GATED (27B on Wormhole). Emit bf8 so the row-parallel reduce-scatter that
                # consumes this carries half the bytes. As the matmul's OUTPUT DTYPE, not a typecast
                # afterwards -- a separate pass over this [S,dim] tensor costs ~175us and would eat
                # most of the win.
                #
                # THIS OP HAS A DOCUMENTED DTYPE CLIFF (see forward_prefill's fp32/bf16 notes): the
                # RS sums num_devices row-parallel partials, and dropping fp32->bf16 measured PCC
                # ~0.69 at TP=4 on Blackhole. So this was measured, not reasoned by analogy with the
                # MLP's down-proj:
                #     MEASURED (T3K TP=8, 27B, T=128, test_gdn_tp_prefill)
                #         bf16  PCC 0.9994457
                #         bf8   PCC 0.9992864     <- 4th decimal, nothing like the TP=4 cliff
                #     MEASURED (T3K TP=8, 27B, seq 2048, single-layer profile, device kernel time)
                #         out-proj matmul 2048x768x5120   454 -> 315us  (-139, writes half the bytes)
                #         reduce-scatter  [1,1,2048,5120] 1029 -> 553us  (-476)
                #                                                       = -615us/layer
                # 553us also matches the MLP's already-bf8 RS at the identical shape (581-587us),
                # which is the cross-check that this is the expected floor and not a fluke.
                #
                # Gated to the 27B because the cliff above proves the safe dtype here is TP- and
                # model-dependent, and TP=8/dim=5120 is the only configuration measured. Blackhole
                # does not reach this arm at all (_fuse_out_mmrs_prefill takes the fused path there),
                # but the 9B on Wormhole DOES, and it is deliberately left on bf16.
                _rs_bf8 = self.args.dim > 4096 and not tpc.is_blackhole()
                _dt = {"dtype": ttnn.bfloat8_b} if _rs_bf8 else {}
                return ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=self.cfg,
                    program_config=pc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    **_dt,
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

    def _project_qkvzab(self, x, S, out_mc=None, qkv_row_major=False):
        """Project x → (qkv, z, a, b). Fused path: one [qkv|z|a|b] matmul then slice.
        out_mc: placement of the qkvzab matmul + slices. None → DRAM; prefill+decode now pass L1 to
        keep qkvzab + q/k/v/z/a/b resident (was DRAM to spare NoC traffic — re-measure if reverting).

        qkv_row_major: return qkv already ROW_MAJOR (DRAM). The native ttnn.conv1d prefill path
        consumes ROW_MAJOR, and qkv is the FIRST slice of qkvzab (offset 0 in the last dim), so the
        slice and the relayout are the same pass: untilize_with_unpadding does both. Slicing in TILE
        and then untilizing is two full passes over the same 16MB at S=2048 (SliceDeviceOperation
        198us + UntilizeDeviceOperation 220us on N300). Only qkv can be fused this way -- z/a/b start
        at non-zero offsets, which untilize_with_unpadding cannot express."""
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
                    self.cfg,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                # PIN THE RESULT TO in0's WIDEST FORM, NOT in0's OWN dtype. On the unfused prefill
                # path in0 is attention_norm's gathered output, which layer.py narrows to bf8 on the
                # native-conv chunks. ttnn.linear would then emit a bf8 qkvzab (matmul.cpp:72
                # defaults the output dtype to in0's) and every downstream consumer would silently
                # change with it: qkv feeds the depthwise conv, and a/b feed sigmoid/softplus ->
                # beta/g, i.e. the RECURRENT decay, where error compounds across chunks rather than
                # dying with the token. The gather win is a read-side one (half the bytes over the
                # ring, half the in0 CB) and survives pinning the output; the output narrowing is a
                # separate, unmeasured accuracy trade, so it is deliberately not taken here.
                _qkvzab_dt = ttnn.bfloat16 if x.dtype != ttnn.bfloat16 else None
                qkvzab = self._col_proj(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvzab_progcfg,
                    out_memory_config=_proj_mc,
                    prefill_out_dtype=_qkvzab_dt,
                )
            if qkv_row_major:
                # end index is INCLUSIVE; qkvzab is (1,S,W) so the slice is [0:1, 0:S, 0:qz].
                qkv = ttnn.untilize_with_unpadding(qkvzab, (0, S - 1, qz - 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz), memory_config=out_mc)
            # z (output gate) lives across the chunk kernel (gated = out_f * silu(z)); L1 z (8MB@S=2048)
            # clashes with the scan kernel CBs -> keep DRAM in chunk-prefill; decode (small S) keeps out_mc.
            # RE-TESTED after the in-proj rework freed 3MB of qkvzab (z in L1 measures 65.1us vs 92.7us
            # in isolation, tests/perf/test_gdn_conv1d_sweep.py) and it STILL does not fit: forcing
            # _z_mc = L1 passes test_gdn_tp and test_prefill but fails the full layer with "statically
            # allocated circular buffers in program 84 clash with L1 buffers ... L1 buffer allocated at
            # 793472 and static circular buffer region ends at 892160". z is 8MB = 128KB/core and stays
            # live across the scan kernel, which is exactly the collision this line already avoided.
            # Note which tests caught it: only test_profile_single_layer_prefill, because it is the only
            # one where attention + MLP + GDN contend for L1 in one layer.
            _z_mc = ttnn.DRAM_MEMORY_CONFIG if (self._fuse_agmm and S > tpc.TILE_SIZE) else out_mc
            z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az), memory_config=_z_mc)
            # b starts at a tile boundary (Nv + ab_gap) instead of right after a's real Nv columns —
            # see gdn_ab_gap in model_config.py. A ttnn.slice's STARTING offset must be tile-aligned
            # to stay tile-native (a non-tile-aligned END is free); az (a's start within qkvzab) is
            # itself already tile-aligned, so a and b can be sliced directly from qkvzab with no
            # intermediate "grab the enclosing tile-aligned ab block first" step and no untilize.
            if self._ab_gap:
                b_start = Nv + self._ab_gap
                a = ttnn.slice(qkvzab, (0, 0, az), (1, S, az + Nv), memory_config=out_mc)
                b = ttnn.slice(qkvzab, (0, 0, az + b_start), (1, S, az + b_start + Nv), memory_config=out_mc)
                ttnn.deallocate(qkvzab)
            else:
                # No gap (Blackhole / 27B): b's start is NOT tile-aligned, so slicing it straight
                # from the wide qkvzab would untilize the whole tensor. Grab the enclosing
                # tile-aligned block once (no untilize), then split a/b from it (test_gdn_slice_opt).
                _ab_end = min(az + -(-2 * Nv // tpc.TILE_SIZE) * tpc.TILE_SIZE, qkvzab.shape[-1])
                ab = ttnn.slice(qkvzab, (0, 0, az), (1, S, _ab_end), memory_config=out_mc)
                ttnn.deallocate(qkvzab)
                a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
                b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv), memory_config=out_mc)
                ttnn.deallocate(ab)
            return qkv, z, a, b
        qkvz = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvz_progcfg)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, S, qz))
        z = ttnn.slice(qkvz, (0, 0, qz), (1, S, az))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, self.tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        b_start = Nv + self._ab_gap
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, b_start), (1, S, b_start + Nv))
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def forward_prefill(self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False):
        """Causal chunk-prefill from scratch. x [1,1,T,dim]: K-sharded (dim/tp per device) when the
        fused in-proj AG-matmul path is active (``_fuse_agmm`` and T>TILE — the norm skips its
        post-AG); replicated otherwise. Output reduce-scattered.

        valid_len: real token count (rest is padding). capture_state: save rec/conv state for decode.
        return_state: when True (per-user batched prefill), return
        ``(output, final_state, conv_new_state)`` for one user's from-scratch B=1
        pass and skip all self.* writeback; the caller stitches per-user states via
        assemble_batched_state(). Single-sequence behavior is unchanged when False.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        self._spill_rec_state_to_dram()
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
        #
        # ... and because those rows ARE identical, normalize the *unpadded* case to None here.
        # valid_len >= T means "no padding in this chunk", which is what callers like model.py's
        # prefill_tp (`valid_len = valid_len or T`) and the single-layer perf test pass. Taking the
        # masked path for it is pure overhead with no effect on the result:
        #   * conv FIR new_state: the one-hot picks x_padded rows [valid_len, valid_len+K-1) =
        #     [T, T+K-1); the None path statically slices [total_len-(K-1), total_len) with
        #     total_len = (K-1)+T — the same rows. Costs a [K-1,total_len]x[total_len,C] one-hot
        #     matmul (95us of 21.7ms at T=2048) plus a host from_torch that blocks trace capture.
        #     VERIFIED bit-identical (conv output AND new_state) at T=128/256/2048 on N300.
        #   * fused chunk adapter: its mask is gated on `valid_len < T`, so it was already a no-op.
        # Net effect is the cheaper, trace-safe, better-tested path (every test_gdn_tp prefill test
        # exercises valid_len=None) for byte-identical output.
        #
        # WORMHOLE ONLY, deliberately. On Blackhole `_gdn_conv1d` is True, so the gate below
        # (`self._gdn_conv1d and valid_len is None`) would additionally switch the conv from the MAC
        # FIR to the native ttnn.conv1d — a different kernel with different rounding. That may well be
        # the better path there, but it is a Blackhole retune and needs Blackhole measurement, so this
        # normalization stays off BH until then.
        # (_normalize_valid_len / prefill_uses_native_conv1d implement this and the _use_native_conv1d
        # gate below; layer.py reads the same predicate to pick attention_norm's gather dtype.)
        valid_len = self._normalize_valid_len(valid_len, T)

        # Cross-chunk carry (chunk-outer prefill): when _stable_state, the recurrent + conv
        # state continue from the persistent buffers (zeroed at sequence start by
        # reset_state_inplace, so a from-scratch single pass reads zeros == None). The demo
        # path (_stable_state False) is unchanged: no carry, reassign state.
        # Per-user prefill (return_state) is always from scratch: must not carry the shared
        # batched buffer (other users' state) as its initial recurrent/conv state.
        carry = self._stable_state and not return_state
        if carry and self.conv_carry is None:
            self.reset_state()

        # Prefill qkvzab placement. These are [1,T,qkv_dim_tp] and grow with the chunk length — at
        # T=2048 (qkv_dim_tp=4096, bf16) that is 16MB per tensor, and the FIR keeps several live at
        # once. Blackhole's larger L1 absorbed this at the chunk sizes it was tuned for; on WH it both
        # OOMs the allocator at long chunks AND, at short chunks, leaves enough L1 resident that the
        # downstream conv1d/chunk kernels' statically-allocated circular buffers collide with it
        # ("clash with L1 buffers" — CBs are L1-only, so only the tensors can move). Measured on WH:
        # keeping these in DRAM unconditionally clears the OOMs and the 8-core CB clash. Costs a DRAM
        # round-trip vs the L1 fast path, so it stays Blackhole-only-off: BH keeps its tuned behaviour.
        _big_prefill = not tpc.is_blackhole()
        _proj_mc = ttnn.DRAM_MEMORY_CONFIG if _big_prefill else ttnn.L1_MEMORY_CONFIG
        # Native ttnn.conv1d only where its CBs fit L1 (see _conv1d_native_fits_l1); otherwise the
        # MAC FIR, exactly as Wormhole did before the native path was enabled.
        _use_native_conv1d = self.prefill_uses_native_conv1d(T, valid_len)
        # qkv_row_major would fold the qkv slice into the ROW_MAJOR relayout that the native conv1d
        # needs (worth ~140us/layer at T=2048: SliceDeviceOperation 198us + UntilizeDeviceOperation
        # 220us become one pass). It is OFF because ttnn.untilize_with_unpadding HANGS on Wormhole
        # for this shape. Minimal repro -- no CCL, no model, one opened device:
        #     t = <[1, 128, 6912] bf16 TILE DRAM>
        #     ttnn.untilize_with_unpadding(t, (0, 127, 4095), memory_config=DRAM)
        #     ttnn.synchronize_device(dev)   # <-- never returns
        # The tensor it returns has logical shape, padded_shape, layout and memory_config IDENTICAL
        # to the slice+to_layout it replaces, so the op is dispatched correctly and simply never
        # completes. ttnn dispatch is async, so the hang surfaces at the next blocking call -- in the
        # full model that is the fused chunk kernel, which makes the stack trace point at the wrong
        # op entirely. Flip to True to re-test after a ttnn uplift.
        _qkv_rm = False
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=_proj_mc, qkv_row_major=_qkv_rm)

        # FIR conv1d; conv_state = previous chunk's last K-1 inputs (None/zero from scratch)
        _cstate = self.conv_carry if carry else None
        if _use_native_conv1d:
            # Native depthwise ttnn.conv1d (masked buckets keep the MAC FIR: valid_len new_state differs)
            conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)
        else:
            conv, conv_new_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                self.K,
                self.mesh,
                # Conv in L1 (output freed before chunk kernel; new_state lands in DRAM internally);
                # DRAM at long chunks, where the FIR's [1,T,qkv_dim_tp] working set overruns WH L1.
                memory_config=_proj_mc,
                conv_state=_cstate,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                valid_len=valid_len,
            )
        ttnn.deallocate(qkv)

        # q/k/v: L1 on Wormhole, but only up to T=2048 -- beyond that, DRAM. beta/g stay DRAM always.
        #
        # The note this replaces said "q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes
        # it". That was true, and it went stale when the in-proj rework dropped the 6912 pad: qkvzab is
        # 3MB smaller, which is enough headroom for q+k+v (4+4+8MB = 256KB/core total) AT T=2048.
        # MEASURED in the real layer (Tracy, T=2048, N300 — perf14 -> perf16):
        #     q  47 -> 34us     k  59 -> 33us     v  94 -> 65us      = -68us/layer
        # matching the isolated sweep (192 -> 132us).
        #
        # The T=2048 headroom does NOT extrapolate to longer prompts: q/k/v scale linearly with T, and
        # test_model_tp_long_prefill (T=2304, full 8-layer model) hit "Statically allocated circular
        # buffers in program 4416 clash with L1 buffers ... L1 buffer allocated at 1178496 and static
        # circular buffer region ends at 1277120" in _row_proj's matmul -- the extra ~288KB/core of q/k/v
        # at T=2304 (vs 256KB/core at T=2048) left too little L1 for that matmul's circular buffers.
        # Gating on T<=2048 (the only size this was ever measured safe at) avoids the crash for any
        # longer single-pass prefill while keeping the win for T<=2048 chunks/prompts.
        #
        # z is NOT included and must not be: same experiment, z in L1 fails the full layer with a
        # scan-kernel CB clash (see the _z_mc note in _project_qkvzab). q/k/v are consumed by the chunk
        # kernel's reader; z is multiplied in at the very end, so it stays live strictly longer.
        #
        # If you widen this (raise the T threshold or re-include z), gate it on
        # test_profile_single_layer_prefill AND test_model_tp_long_prefill, NOT on test_gdn_tp or
        # test_prefill -- those pass with z in L1 / no T cap and only the full multi-layer model at
        # realistic T catches the clash.
        # Wormhole only: Blackhole's placement was tuned separately and is left byte-identical.
        kd = self.key_dim_tp
        # MODEL-GATED. The Wormhole L1 placement below was tuned for the 9B (dim 4096); it is NOT
        # safe for the 27B (dim 5120), whose chunked prefill dies inside chunk_gated_delta_rule with
        #   "circular buffers in program N clash with L1 buffers on core range [0-0 - 0-0]"
        # Note the trigger is NOT the resident slice size: at T=2048 the 27B's per-device q+k+v is
        # 5.2MB vs the 9B's 16.8MB, and it has fewer heads per device (nv 6 vs 16). So it is the op's
        # own CB layout at these head dims, and shrinking the slices would not help. The 27B path
        # never placed these in L1 (see gdn/tp.py on the 27B branch), so give it that behaviour
        # verbatim rather than retuning the 9B's win. Gate on dim, not model_name: HF_MODEL is often
        # a hashed snapshot directory, which makes name-based checks unreliable.
        _qkv_l1_tuned_for_this_model = self.args.dim <= 4096
        _qkv_mc = (
            None
            if (tpc.is_blackhole() or not _qkv_l1_tuned_for_this_model)
            else (ttnn.L1_MEMORY_CONFIG if T <= 2048 else ttnn.DRAM_MEMORY_CONFIG)
        )
        if self._gdn_flat_qkv:
            # Flat q/k/v: adapter splits heads inside untilize
            q = ttnn.slice(conv, (0, 0, 0), (1, T, kd), memory_config=_qkv_mc)
            k = ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd), memory_config=_qkv_mc)
            v = ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp), memory_config=_qkv_mc)
            _qkv_head_dims = (Nk, Dk, Nv, Dv)
        else:
            q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd), memory_config=_qkv_mc), (1, T, Nk, Dk))
            k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd), memory_config=_qkv_mc), (1, T, Nk, Dk))
            v = ttnn.reshape(
                ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp), memory_config=_qkv_mc), (1, T, Nv, Dv)
            )
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
        # const_tiles only applies to the fused op; the seq adapter has no such param.
        _extra = {"const_tiles": self._fused_const_tiles} if _use_fused else {}
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
        # The recurrent state outlives this call (assigned to self.rec_state, or handed to the caller to
        # stitch), so it must not sit in L1: it is exactly Nv*Dk*Dv*4 = 1MB (16KB/bank) and it is live
        # across the NEXT chunk's kernels. Cheap no-op when the op already returned DRAM.
        if final_state.memory_config().buffer_type != ttnn.BufferType.DRAM:
            _fs_l1 = final_state
            final_state = ttnn.to_memory_config(final_state, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(_fs_l1)
        captured = None
        if return_state:
            # Per-user prefill: return this user's state for assemble_batched_state to stitch
            # into the batched buffers. No self.* writeback; tensors are not deallocated here.
            # NOTE: these come back DRAM-resident already (verified: out/rec/conv are all
            # BufferType.DRAM at every u), so retaining them across users costs no L1. Spilling them
            # explicitly was tried as a fix for the native-conv1d clash and changed nothing.
            captured = (final_state, conv_new_state)
        else:
            # ---- Carry recurrent + conv state for the NEXT chunk (chunk-outer prefill). ----
            # In place (ttnn.copy) when _stable_state so the addresses the prefill/decode traces
            # baked in stay valid across execute_trace replays and across sequences.
            if carry:
                ttnn.copy(final_state, self.rec_state)
                ttnn.deallocate(final_state)
                ttnn.copy(conv_new_state, self.conv_carry)  # [1, K-1, D] last-K-1 conv inputs
            else:
                self.rec_state = final_state
            # ---- Finalize the decode conv window (last chunk / short prompt). ----
            # conv_states[1..K-1] = the last K-1 real conv inputs; [0] is the (shifted-out) zero.
            # Harmless to refresh every chunk — the last chunk's values are the ones decode reads.
            if capture_state:
                if self.conv_states is None:
                    self.reset_state()
                if self._zero_conv0 is not None:
                    ttnn.copy(self._zero_conv0, self.conv_states[0])
                else:
                    # Allocated on device: a host zeros() + upload buys nothing for a buffer whose
                    # every element is 0, and this runs per prefill chunk.
                    zero = ttnn.zeros(
                        ttnn.Shape([1, B, D]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh
                    )
                    ttnn.copy(zero, self.conv_states[0])
                    ttnn.deallocate(zero)
                for j in range(self.K - 1):
                    src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                    ttnn.copy(src, self.conv_states[j + 1])
            ttnn.deallocate(conv_new_state)
        # Gated RMSNorm + SiLU(z); norm/flatten in L1, gated output in DRAM for out-proj.
        # The norm/flatten tensors are [1,Nv,T,Dv] — they scale with the prefill chunk length, so at
        # long T they are far too big for L1 (fp32 at T=2048, Nv_tp=16, Dv=128 is 16MB EACH, and the
        # norm + concat_heads outputs are live together). Blackhole's larger L1 absorbed this at the
        # chunk sizes it was tuned for; on WH it OOMs the allocator mid-prefill. Keep the L1 fast path
        # for short chunks (unchanged behaviour, incl. all of Blackhole) and spill to DRAM past that.
        _L1 = ttnn.L1_MEMORY_CONFIG
        # NOTE: _elem/_norm_mc are deliberately computed from the PRE-cast dtype (see the bf16 cast
        # below). In bf16 the [1,Nv,T,Dv] norm tensor is exactly 8MB at T=2048, which lands precisely
        # on this threshold and would flip the choice to L1; keeping the fp32-based decision leaves the
        # L1/DRAM split byte-identical to before.
        _elem = 4 if o.dtype == ttnn.float32 else 2
        _norm_mc = _L1 if (tpc.is_blackhole() or Nv * T * Dv * _elem <= (8 << 20)) else ttnn.DRAM_MEMORY_CONFIG
        # Wormhole: run the OUTPUT path in bf16 rather than the chunk kernel's fp32. Everything
        # downstream inherits the dtype — rms_norm -> nlp_concat_heads -> z-gate -> o_proj -> the
        # reduce-scatter — so this halves the bytes through the layer's largest data-movement op (the
        # fp32 RS is 1,804us of 21,668 in the single-layer profile).
        #
        # Only `o` (the per-token output) is cast. `final_state` stays fp32, so the recurrence carried
        # across chunks and into decode is untouched and nothing compounds.
        #
        # Blackhole stays fp32: there o_proj's row-parallel RS sums 4 per-device partials (TP=4), where
        # bf16 measured PCC ~0.69 even at ISL 2048 (test_oproj_dtype_isl). An N300 is TP=2 and sums 2
        # partials, so that result does not carry over — measured separately here.
        # QWEN35_GDN_OUT_FP32=1 forces the old fp32 path back on Wormhole.
        if not tpc.is_blackhole() and o.dtype == ttnn.float32 and os.environ.get("QWEN35_GDN_OUT_FP32") != "1":
            _o_fp32 = o
            # L1 destination: the cast reads 16MB of fp32 and writes 8MB of bf16, and writing that to
            # L1 instead of DRAM measured 127 -> 93us in the real layer (reproduced twice; the isolated
            # sweep in tests/perf/test_gdn_tail_sweep.py says 124.5 -> 92.0us). rms_norm consumes it
            # immediately, so it is short-lived.
            o = ttnn.typecast(o, ttnn.bfloat16, memory_config=_L1)
            ttnn.deallocate(_o_fp32)
        if self._gdn_fuse_out:
            # Fuse adapter relayout with per-head rms_norm + head-flatten.
            # TILE-native head->token relayout (transpose + fold), dropping the
            # TILE->ROW_MAJOR->TILE round-trip. o is head-major (1,Nv,T,Dv).
            n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_norm_mc)
            ttnn.deallocate(o)
            n = ttnn.reshape(n, (1, Nv, T, Dv))
            # Fused head->token relayout: [1,Nv,T,Dv] -> [1,1,T,Nv*Dv].
            # Free the rms_norm output explicitly instead of just rebinding `n` over it. Rebinding
            # leaves the old buffer alive until Python GC happens to collect the shadowed tensor, and at
            # short chunks _norm_mc is L1 (the [1,Nv,T,Dv] threshold below), so the leak is 1MB of L1
            # (16KB/bank) surviving for a nondeterministic number of subsequent ops. That was enough to
            # fragment L1 and tip the native conv1d past its ~2% L1 headroom on some calls but not
            # others — the "clash with L1 buffers" that made _gdn_conv1d unshippable on WH.
            _n_pre = n
            # L1 rather than _norm_mc (DRAM at T=2048): 89 -> 67us in the real layer, reproduced.
            # Note this is the head-flatten OUTPUT only. The rms_norm above deliberately keeps
            # _norm_mc: L1 measured SLOWER for it (100.8 -> 106.6us), so the conservative fp32-based
            # threshold that lands it in DRAM at T=2048 is already the right call.
            n = ttnn.experimental.nlp_concat_heads(n, memory_config=_L1)
            ttnn.deallocate(_n_pre)
            out_f = ttnn.reshape(n, (1, T, self.value_dim_tp))
        else:
            out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_norm_mc)
            ttnn.deallocate(o)
            out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp), memory_config=_norm_mc)
            ttnn.deallocate(out_n)
        # gated stays DRAM. L1 measured 262.3 -> 217.7us in ISOLATION and still fails the full layer:
        # "statically allocated circular buffers clash with L1 buffers ... L1 buffer allocated at 949120".
        # gated is 8MB and feeds the out-proj matmul, which runs one K pass and so spends nearly all of
        # L1 on CBs -- the same collision that rules out an L1 in-proj output. Tested individually: the
        # typecast and head-concat above pass the full layer, this one does not.
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
        _cps, _wpl = tpc.prefill_ccl_tuning()
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            chunks_per_sync=_cps,
            num_workers_per_link=_wpl,
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

        Under _stable_state (decode-trace path) the result is copied into the fixed-address
        buffers; otherwise (demo/standalone) it is assigned.
        """
        assert len(rec_list) == self.B and len(conv_new_list) == self.B, "need one state per batch row"
        D = self.qkv_dim_tp
        rec_batched = ttnn.concat(rec_list, dim=0)  # [B, Nv, Dk, Dv]
        conv_states = [
            ttnn.zeros(ttnn.Shape([1, self.B, D]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
        ]
        for m in range(1, self.K):  # conv_states[m] row u = conv_new_list[u][:, m-1]
            rows = [
                ttnn.reshape(ttnn.slice(conv_new_list[u], (0, m - 1, 0), (1, m, D)), (1, 1, D)) for u in range(self.B)
            ]
            cs = ttnn.concat(rows, dim=1)  # [1, B, D]
            for r in rows:
                ttnn.deallocate(r)
            conv_states.append(cs)

        if self._stable_state and self.rec_state is not None:
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
        else:
            self.rec_state = rec_batched
            self.conv_states = conv_states
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

    def _write_recurrent_state_prefix(self, new_rec, B):
        """Write active rows [0:B] without reading or copying idle rows."""
        grid_size = self.mesh.compute_with_storage_grid_size()
        assert (
            grid_size.x >= 8 and grid_size.y >= 6
        ), f"GDN prefix state write needs an 8x6 core rectangle, got {grid_size.x}x{grid_size.y}"
        nhw = B * self.Nv * self.Dk
        assert (
            nhw % ttnn.TILE_SIZE == 0
        ), f"GDN prefix state rows B={B}, Nv={self.Nv}, Dk={self.Dk} -> {nhw} is not tile-aligned"
        n_tiles = nhw // ttnn.TILE_SIZE

        # Prefer the tuned 8x6=48-core rectangle, which every TP=4 shape hits (Nv=12 -> nhw=B*1536
        # -> 48*B tiles). At TP=8 Nv halves to 6, so B=1 gives only 24 tiles and cannot fill 48
        # cores with tile-aligned shards — fall back to the largest core count that divides evenly.
        if n_tiles % 48 == 0:
            num_cores = 48
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))})
        else:
            num_cores = max(c for c in range(1, min(48, grid_size.x * grid_size.y) + 1) if n_tiles % c == 0)
            grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

        shard_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid,
                (nhw // num_cores, self.Dv),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
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

    def _write_index(self, buf, src, idx, dim):
        """Replace slice `idx` of `buf` along `dim` with `src` (extent 1 along `dim`), preserving
        the other slices, via an in-place copy into `buf`. Consumes `src` (and the temporary
        slices). `src` must already match `buf`'s dtype."""
        n = buf.shape[dim]
        if n == 1:
            ttnn.copy(src, buf)
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
            ttnn.deallocate(p)

    def write_slot(self, slot, rec, convs):
        """Write one user's B=1 prefill state into decode `slot`, preserving every other (live)
        row. The per-slot analogue of assemble_batched_state for vLLM continuous batching.

        rec:   [1, Nv, Dk, Dv] the user's recurrent state.
        convs: list of K [1, 1, qkv_dim_tp] the user's conv taps (conv_states[m] column). Unlike
               assemble_batched_state (which zeroes tap 0), every tap is written straight from the
               user's B=1 prefill state, so decode continues from exactly the produced shift register.
        Consumes rec and convs. Requires the batched buffers (allocate_kv_caches(batch_size=B))."""
        assert self.rec_state is not None and self.conv_states is not None, "batched GDN state not allocated"
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        rec_src = rec if rec.dtype == self.rec_state.dtype else ttnn.typecast(rec, self.rec_state.dtype)
        if rec_src is not rec:
            ttnn.deallocate(rec)
        self._write_index(self.rec_state, rec_src, slot, dim=0)
        for m in range(self.K):
            c = convs[m]
            c_src = c if c.dtype == self.conv_states[m].dtype else ttnn.typecast(c, self.conv_states[m].dtype)
            if c_src is not c:
                ttnn.deallocate(c)
            self._write_index(self.conv_states[m], c_src, slot, dim=1)

    def remap_slots(self, remap):
        """Reindex the batched decode state after a vLLM batch condense: slot i takes the state
        previously at slot remap[i] (identity entries are no-ops). Mirrors
        seed_manager.apply_slot_remap for GDN's per-slot recurrent+conv state, which the plugin's
        slot_remap does not itself move. In-place copy into the fixed buffers (preserves the decode
        trace's baked addresses)."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self._gather_indices(self.rec_state, idx, dim=0)
        for m in range(self.K):
            self._gather_indices(self.conv_states[m], idx, dim=1)

    def _gather_indices(self, buf, idx, dim):
        """Rebuild `buf` so slice i along `dim` becomes old slice idx[i], then copy back in place.
        `new` is fully materialized before the copy, so gathering from `buf` into itself is safe."""
        rows = [self._slice_along(buf, dim, idx[i], idx[i] + 1) for i in range(len(idx))]
        new = ttnn.concat(rows, dim=dim)
        ttnn.copy(new, buf)
        ttnn.deallocate(new)
        for r in rows:
            ttnn.deallocate(r)

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
                    sequence start, so the first chunk reads zeros (== from scratch). Requires
                    _stable_state (the batched decode buffers).

        KERNEL CAP: gated_delta_attn_seq maps one BH = B*Nv_tp row per core and is L1-bound, so BH
        must stay <= ~32 (at TP=4, Nv_tp=8 => B <= 4). Larger B trips an L1 clash (B=8) or the
        kernel's `BH <= compute_grid` assert (B=32); B>4 would need grouped launches (groups <=4).
        The model currently prefills per-user instead (see prefill_paged_peruser).
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        self._spill_rec_state_to_dram()
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
            self._batched_conv_carry = ttnn.zeros(
                ttnn.Shape([B, self.K - 1, D]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh
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
        _extra = {"const_tiles": self._fused_const_tiles} if _use_fused else {}
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
        if self._stable_state and self.rec_state is not None:
            rec_src = (
                final_state
                if final_state.dtype == self.rec_state.dtype
                else ttnn.typecast(final_state, self.rec_state.dtype)
            )
            ttnn.copy(rec_src, self.rec_state)
            if rec_src is not final_state:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(final_state)
        else:
            self.rec_state = final_state  # [B, Nv, Dk, Dv]
        # conv_states[0] = shifted-out zero; conv_states[m] row u = conv_new_state[u, m-1].
        zero0 = ttnn.zeros(ttnn.Shape([1, B, D]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh)
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
        if self._stable_state and self.conv_states is not None:
            for m in range(self.K):
                ttnn.copy(new_conv[m], self.conv_states[m])
                ttnn.deallocate(new_conv[m])
        else:
            self.conv_states = new_conv

        # ---- output (gated RMSNorm + SiLU(z) gate + row-parallel out proj + all-reduce) ----
        # Wormhole: halve the bytes through the norm/reshape/gate tail, same as forward_prefill
        # (see the comment there) -- this path lacked the cast even though it hits the same tensors.
        if not tpc.is_blackhole() and o.dtype == ttnn.float32 and os.environ.get("QWEN35_GDN_OUT_FP32") != "1":
            _o_fp32 = o
            o = ttnn.typecast(o, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(_o_fp32)
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

    def forward_decode(self, x):
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        Bmax = self.B
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode conv→recurrence→norm/gate chain L1-resident
        if self.conv_states is None:
            self.reset_state()
        self._promote_rec_state_to_l1()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))

        # Active decode width, taken from the input. Normally == Bmax. BUCKETED decode: a request
        # feeds B<Bmax tokens and the whole step runs on state rows [0:B]; idle rows [B:Bmax] are
        # preserved. Conv taps are per-channel (broadcast over batch), so the conv weighted-sum
        # works at any width. The B==Bmax path is byte-identical to before.
        B = x.shape[-2]

        qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)

        # Conv1d shift-register + weighted sum + SiLU
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
        ttnn.deallocate(qkv)
        # FIR over the K taps. The per-tap chain (multiply + K-1 mac) costs K*2-1 = 7 DEVICE ops at
        # K=4, because ttnn.mac does NOT fuse into a single kernel -- it dispatches a multiply and an
        # add, which the Tracy profile shows as 7 BinaryNg ops for these 4 calls. Stacking the states
        # into [K,B,C] and the taps into [K,1,C] turns the whole FIR into ONE broadcast multiply plus
        # ONE reduction: 3 device ops (concat + multiply + sum) instead of 7.
        # MEASURED (WH, K=4 B=32 C=4096, full shift+FIR+silu step): 43.0us / 8 ops vs the per-tap
        # chain's 47.1us / 12 ops, with PCC 0.999997 vs 0.999995 (marginally BETTER, since the
        # reduction accumulates once instead of rounding after every mac).
        # The 4 in-place shift copies above are deliberately NOT folded in: they keep conv_states at
        # fixed addresses for decode-trace replay, and every stacked-shift alternative measured worse
        # (slice+concat 13.1us vs 11.2us; in-place slice_write 238us -- see git history).
        # Blackhole keeps the per-tap chain: these numbers are WH-measured and BH's grid/L1 differ.
        if tpc.is_blackhole():
            conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
            for j in range(1, self.K):
                conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        else:
            _stk = ttnn.concat(st, dim=0, memory_config=_L1)  # [K, B, qkv_dim_tp]
            _prod = ttnn.multiply(_stk, tw["conv_taps_stack"], memory_config=_L1)
            ttnn.deallocate(_stk)
            conv = ttnn.sum(_prod, dim=0, keepdim=True, memory_config=_L1)  # [1, B, qkv_dim_tp]
            ttnn.deallocate(_prod)
        conv = ttnn.silu(conv, memory_config=_L1)

        kd = self.key_dim_tp
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        rf = Nv // Nk
        if self._decode_tile_opt:
            # WH 9B: v has no GQA expand — slice straight into the recurrent kernel's 4D shape, and
            # expand Q/K in TILE (concat on tile-aligned Dk, not repeat_interleave's untilize-forcing
            # path — see _gqa_expand_heads). Recurrence L2-norms + scales internally.
            v = ttnn.reshape(
                ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, 1, Nv, Dv), memory_config=_L1
            )
            ttnn.deallocate(conv)
            q = _gqa_expand_heads(q, rf, B, Nv, Dk, _L1)
            k = _gqa_expand_heads(k, rf, B, Nv, Dk, _L1)
            # Left at (1, B, Nv) -- their natural post-activation shape -- rather than reshaped to
            # (B, 1, Nv): the kernel reshapes beta/g to [B, H] itself (B/H come from q's shape, not
            # beta/g's rank), so (1,B,Nv)->(B,H) is exactly as cheap and one reshape cheaper here.
            # The batch-split branch below reshapes to (B,1,Nv) where it needs B leading to slice.
            beta = ttnn.sigmoid(b, memory_config=_L1)
            g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        else:
            # Blackhole / 27B: unchanged pre-rework path.
            v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
            ttnn.deallocate(conv)
            # GQA expand Q/K Nk→Nv; recurrence L2-norms + scales internally
            q = ttnn.repeat_interleave(q, rf, dim=1)
            k = ttnn.repeat_interleave(k, rf, dim=1)
            # Decode: hand q/k/v to the recurrent kernel in L1. The kernel typecasts + does a LOCAL
            # l2-norm (no cross-device gather), so placement is output-neutral here (unlike SDPA-q,
            # which hard-requires DRAM, and unlike the residual→DistributedNorm all-gather).
            q = ttnn.reshape(q, (B, 1, Nv, Dk), memory_config=_L1)
            k = ttnn.reshape(k, (B, 1, Nv, Dk), memory_config=_L1)
            v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)
            beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
            g = ttnn.reshape(
                ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1), (B, 1, Nv)
            )
        ttnn.deallocate(b)
        ttnn.deallocate(a)

        # fp32 decode step on Blackhole (QWEN35_GDN_DECODE_BF16=1 reverts to bf16 there too).
        # Wormhole always runs bf16 -- matches reset_state's state dtype above, same tradeoff.
        _hp = tpc.is_blackhole() and os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"
        # Only the first B users' state participates when the batch is under the allocated max.
        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
        _bstep = self._decode_batch_split(B)
        # tile_opt only exists on the Wormhole-local kernel; the Blackhole dispatch target is the
        # shared upstream function, which does not take it. _decode_tile_opt is False on BH anyway,
        # so gate the kwarg itself rather than passing tile_opt=False into a signature without it.
        _rec_kw = {"tile_opt": True} if self._decode_tile_opt else {}
        if _bstep >= B:
            o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
                q,
                k,
                v,
                beta,
                g,
                scale=self.scale,
                initial_state=init_state,
                device=self.mesh,
                high_precision=_hp,
                **_rec_kw,
            )
        else:
            # Batch-split decode: the recurrence is per-user independent (each user owns its own
            # [Nv,Dk,Dv] state; nothing crosses the batch dim), so running the batch in slices is
            # mathematically EXACT — not an approximation. Needed because the kernel makes the state
            # L1-resident and holds a second same-sized tensor alongside it; see _decode_batch_split.
            # beta/g need B as a leading (non-tiled) dim here to slice per-batch-chunk without
            # hitting tile-alignment costs -- reshape to (B,1,Nv) only in this (rare, large-B) path.
            # Off the tile-opt path they already arrive as (B,1,Nv), so reuse them as-is.
            if self._decode_tile_opt:
                beta_bn1 = ttnn.reshape(beta, (B, 1, Nv), memory_config=_L1)
                g_bn1 = ttnn.reshape(g, (B, 1, Nv), memory_config=_L1)
            else:
                beta_bn1, g_bn1 = beta, g
            o_parts, rec_parts = [], []
            for s in range(0, B, _bstep):
                e = min(s + _bstep, B)
                q_s = ttnn.slice(q, (s, 0, 0, 0), (e, 1, Nv, self.Dk))
                k_s = ttnn.slice(k, (s, 0, 0, 0), (e, 1, Nv, self.Dk))
                v_s = ttnn.slice(v, (s, 0, 0, 0), (e, 1, Nv, Dv))
                beta_s = ttnn.slice(beta_bn1, (s, 0, 0), (e, 1, Nv))
                g_s = ttnn.slice(g_bn1, (s, 0, 0), (e, 1, Nv))
                rec_s = ttnn.slice(init_state, (s, 0, 0, 0), (e, Nv, self.Dk, Dv))
                o_s, rec_new_s = recurrent_gated_delta_rule_decode_ttnn(
                    q_s,
                    k_s,
                    v_s,
                    beta_s,
                    g_s,
                    scale=self.scale,
                    initial_state=rec_s,
                    device=self.mesh,
                    high_precision=_hp,
                    **_rec_kw,
                )
                for t in (q_s, k_s, v_s, beta_s, g_s, rec_s):
                    ttnn.deallocate(t)
                # The kernel hands back an L1-resident state slice. Spill it to DRAM before the next
                # slice runs: holding every slice's state in L1 at once would defeat the whole point
                # of splitting (and collides with the next iteration's circular buffers).
                rec_dram = ttnn.to_memory_config(rec_new_s, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(rec_new_s)
                o_parts.append(o_s)
                rec_parts.append(rec_dram)
            o = ttnn.concat(o_parts, dim=0)
            new_rec = ttnn.concat(rec_parts, dim=0)
            for t in o_parts + rec_parts:
                ttnn.deallocate(t)
            if self._decode_tile_opt:
                ttnn.deallocate(beta_bn1)
                ttnn.deallocate(g_bn1)
        if init_state is not self.rec_state:
            ttnn.deallocate(init_state)
        if self._stable_state:
            # In-place update preserves rec_state address for decode trace replay
            if B == Bmax:
                ttnn.copy(new_rec, self.rec_state)
                ttnn.deallocate(new_rec)
            else:
                self._write_recurrent_state_prefix(new_rec, B)
        else:
            # WH 9B eager path: keep the L1-resident state the kernel just produced. forward_prefill
            # / forward_prefill_batched call _spill_rec_state_to_dram so the next conv1d does not
            # inherit this L1 allocation (the clash documented on _gdn_conv1d). The always-spill
            # below paid an L1->DRAM copy every decode step for a buffer the very next decode step
            # (_promote_rec_state_to_l1) immediately copied straight back.
            #
            # Blackhole / 27B keep the original unconditional spill: the kernel hands the state back
            # L1-RESIDENT (the batch-split branch above already relies on that and spills each slice
            # for exactly this reason); the un-split branch used to just assign it, leaving
            # Nv*Dk*Dv*4 = 1MB of L1 (16KB/bank) parked in self.rec_state for the whole lifetime of
            # the next prefill call. That is the block that fragmented L1 and tipped the native
            # conv1d over its ~2% L1 headroom — see _gdn_conv1d.
            if not self._decode_tile_opt and new_rec.memory_config().buffer_type != ttnn.BufferType.DRAM:
                _nr_l1 = new_rec
                new_rec = ttnn.to_memory_config(new_rec, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(_nr_l1)
            self.rec_state = new_rec

        # Under _decode_tile_opt the WH kernel returns [B,H,1,V] (matmul-native, see
        # recurrent_decode_wh.py) -- rms_norm reduces the last dim (V==Dv) regardless of where the
        # singleton sits, so the [B,1,H,V]-crossing reshape is dead weight. Every other config gets
        # [B,1,H,V] as before, where the reshape to [B,Nv,Dv] is a cheap view (tiled pair stays H,V).
        #
        # TRIED AND REVERTED (2026-08-19): the [...,1,V] tiling of [B,H,1,V] pads each (B,H) row to a
        # full 32-row tile, and an ISOLATED probe (test_gdn_outnorm_probe.py, single-device, B=32
        # H=16 V=128) showed reshaping to (B,Nv,Dv) before the norm netting -21% (44.4us -> 35.1us,
        # reshape included) by escaping that padding. Reproduced the profiled 86us (2-device merged
        # report) being far above a weighted norm's ~12us floor. Applied here, then checked against a
        # real single-layer decode capture (test_profile_single_layer_gdn_decode.py, batch32,
        # TT_METAL_DEVICE_PROFILER Tracy) -- and REGRESSED: the norm did drop 46us -> 9us as predicted,
        # but the added reshape cost 40us in this TP=2 context (not the ~27us the single-device probe
        # implied), for a net +3us on this pair and +12us on the whole captured window (2,925 ->
        # 2,937us, 111 -> 112 device ops). Same class of trap as the ROPE Q+K merge and the q_norm/
        # k_norm flattening elsewhere in this codebase: an isolated op's reshape cost does not
        # transfer to the real TP layer. Left as upstream had it; do not re-apply without re-checking
        # a full-layer capture, not just the isolated probe.
        if self._decode_tile_opt:
            out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)  # gated norm (no +1)
            ttnn.deallocate(o)
        else:
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
