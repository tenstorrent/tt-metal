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
from models.demos.blackhole.qwen36.tt.gdn.recurrent_decode_wh import (
    recurrent_gated_delta_rule_decode_dispatch as recurrent_gated_delta_rule_decode_ttnn,  # Wormhole skips q's fp32 promotion (q never feeds the state write); see recurrent_decode_wh.py.
)

# Spec verify's multi-token recurrence; the only source of per-token state.
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    fused_recurrent_gated_delta_rule_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce

# Splice-carry patch conv is off: it does not beat the concat it replaces.
_SPLICE_CARRY = False


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


def _gqa_expand_heads(t, repeats, batch, n_heads, head_dim, memory_config):
    """GQA expand along tile-aligned Dk so the result stays TILE and lands [B, 1, Nv, Dk]."""
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
        # Trailing zero rows after b so the fused width's tile count factors; never sliced.
        _pad_rows = getattr(args, "gdn_qkvzab_pad_tiles", 0) * 32
        # Zero rows so b starts tile-aligned.
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
            # Pad and gap both qualify the cache key; as_tensor reloads a cache file as-is.
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
        # Same ab_gap as the fused path so the a/b split stays tile-aligned.
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
    if getattr(args, "num_devices", 1) > 1 and tpc.is_blackhole():
        # COLUMN-parallel copy of the out-proj for prefill.
        # Decode keeps the row-sharded tw["out"] (matmul + all-reduce).
        tw["out_colpar"] = tpc.shard_w(
            sd[P + "out_proj.weight"],
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=c("out.colpar"),
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
    # Same taps stacked to [K, 1, qkv_dim] for the one-shot FIR. Kept alongside the per-tap list.
    tw["conv_taps_stack"] = tpc.shard_small(torch.stack([t.reshape(1, -1) for t in taps], dim=0), mesh, c("tap_stack"))
    # Depthwise conv1d weight [qkv_dim, 1, K], host-held mesh-sharded (dim=0) for prepare_conv_weights /
    # _conv1d_prefill. When gdn_conv_channel_chunks > 1 it is a list of per-device channel-chunk weights
    # (see TPGatedDeltaNet.__init__ for why); chunks=1 keeps the single tensor.
    W1d = torch.stack(taps, dim=-1).reshape(args.gdn_qkv_dim, 1, args.gdn_conv_kernel_size).contiguous()
    n_cc = getattr(args, "gdn_conv_channel_chunks", 1)

    def _shard_conv_w(w):
        return ttnn.from_torch(
            w.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )

    if n_cc > 1:
        C_dev = args.gdn_qkv_dim // tp
        assert C_dev % n_cc == 0, f"GDN per-device channels {C_dev} not divisible by gdn_conv_channel_chunks {n_cc}"
        cw = C_dev // n_cc
        Wd = W1d.reshape(tp, C_dev, 1, args.gdn_conv_kernel_size)  # per-device channel block
        tw["conv_w1d"] = [
            _shard_conv_w(Wd[:, i * cw : (i + 1) * cw].reshape(tp * cw, 1, args.gdn_conv_kernel_size))
            for i in range(n_cc)
        ]
    else:
        tw["conv_w1d"] = _shard_conv_w(W1d)
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
        # Zero columns so b starts on a tile boundary. 0 except Wormhole 9B.
        self._ab_gap = getattr(args, "gdn_ab_gap", 0)
        # Wormhole 9B only.
        self._decode_tile_opt = tpc.wh_9b_n300(args)
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
        self._fuse_agmm = self._fuse_ab and tpc.is_blackhole()
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        # BH only: the default MMRS grid needs rows 8-9, which Wormhole's 8-row grid lacks.
        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1 and tpc.is_blackhole()
        # PREFILL out-proj as column-parallel AG+matmul (takes precedence over the MMRS arm when the
        # col-sharded weight was loaded).
        # BH only. Column-parallel gated stays in L1; on Wormhole that clashes with the out-proj CBs.
        self._out_colpar_prefill = "out_colpar" in tw and tpc.is_blackhole()
        # Pre-build chunk masks once (trace-safe; avoids from_torch inside captured trace)
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        # Prefill fused-op constant tiles, owned by this layer (avoids process-lifetime C++ cache vs device lifetime).
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

        self._fused_const_tiles = build_fused_const_tiles(mesh, _FUSED_CHUNK_SIZE)
        self.conv_states = None
        self.rec_state = None
        # Spec-decode verify captures per-token state so commit is a slot select, not a re-run.
        self._capture_slots = False
        self._verify_slots = None
        self._verify_states = None  # per-token rec states from the last verify (token-major)
        self._verify_states_buf = None  # same tensor; kept so traced replays can re-arm the handle
        # Persistent slot buffers: verify copies into fixed addresses so a trace does not allocate.
        self._slot_bufs = None
        # Fully-batched verify: no per-token loop. The False branch is the per-token reference.
        self.use_fullbatch_verify = True
        # Batched-conv verify window. None means per-token slots.
        self._verify_win_buf = None
        # Durable [1, K, qkv_dim_tp] shift register. The traced verify reads its carry from a fixed slice.
        self._conv_win_buf = None
        # After a fullbatch verify the window is ahead of the taps; outside writers leave the window behind.
        self._conv_taps_stale = False
        self._conv_win_stale = False
        self._win_captured = False  # did THIS verify populate the window? (buffer is always allocated)
        self._conv_taps_T = None  # conv taps expanded to T rows for the batched-conv path
        # In-place state updates for decode/prefill traces (set by model allocate_kv_caches)
        self._stable_state = False
        # Spec decode only: one fused recurrent op, full batch only.
        self.use_fused_recurrent_decode = False
        self.conv_carry = None  # cross-chunk prefill conv carry [1, K-1, qkv_dim_tp]
        # Native ttnn.conv1d depthwise prefill; L1_FULL slice keeps it trace-safe.
        # Only used when valid_len is None (masked buckets keep the MAC FIR).
        # QWEN35_GDN_CONV1D=0 falls back to the MAC FIR. Do not leave recurrent state L1-resident into the next prefill.
        _conv1d_env = os.environ.get("QWEN35_GDN_CONV1D")
        self._gdn_conv1d = True if _conv1d_env is None else (_conv1d_env == "1")
        # Split the depthwise conv over channel chunks so each native L1_FULL conv fits L1: the
        # per-channel-independent depthwise CB is channel-dominated (not reducible by act-block or
        # DRAM width/height slicing), and the 35B-A3B GDN qkv_dim_tp overflows a single call on BH.
        # 27B runs a single chunk (unchanged); see model_config.gdn_conv_channel_chunks.
        self._conv_chunks = getattr(args, "gdn_conv_channel_chunks", 1)
        # BH: list over channel chunks. WH: dict keyed by (input width, padding).
        self._conv1d_wprep = None
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill

    # Spare L1 left for the recurrent state after the decode kernel's own allocations.
    _DECODE_STATE_L1_BUDGET = 31 * (1 << 20)
    # Bytes per element by rec_state dtype, so the split threshold follows the allocated dtype.
    _STATE_BYTES_PER_ELEM = {ttnn.bfloat8_b: 1, ttnn.bfloat16: 2, ttnn.float32: 4}

    def _decode_batch_split(self, B):
        """Largest batch slice whose recurrent state fits the decode kernel's spare L1."""
        if tpc.is_blackhole():
            return B
        elem_bytes = self._STATE_BYTES_PER_ELEM.get(self.rec_state.dtype, 4)
        per_user = self.Nv * self.Dk * self.Dv * elem_bytes  # one user's state, actual dtype
        # Budget covers the pre-decay state and its decayed copy at once.
        max_b = max(1, self._DECODE_STATE_L1_BUDGET // (2 * max(1, per_user)))
        if max_b >= B:
            return B
        # Even split into power-of-two slices.
        step = 1
        while step * 2 <= max_b:
            step *= 2
        return step

    def _spill_rec_state_to_dram(self):
        """Move rec_state to DRAM before prefill. Do not call under _stable_state (traces bake the address)."""
        if self.rec_state is None or self._stable_state:
            return
        if self.rec_state.memory_config().buffer_type == ttnn.BufferType.DRAM:
            return
        spilled = ttnn.to_memory_config(self.rec_state, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(self.rec_state)
        self.rec_state = spilled

    def _promote_rec_state_to_l1(self):
        """Eager Wormhole-9B decode only: hoist rec_state into L1. Not under _stable_state or a split batch."""
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
        # Device zeros: the buffers are pure zeros, so skip a host upload.
        def z(shape, dtype=ttnn.bfloat16):
            return ttnn.zeros(ttnn.Shape(list(shape)), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh)

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        # fp32 recurrent state on Blackhole unless QWEN35_GDN_STATE_BF16=1; Wormhole stays bf16.
        if not tpc.wh_9b_n300(self.args) and os.environ.get("QWEN35_GDN_STATE_BF16") != "1":
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
        # rec_state/conv_states got fresh addresses here, so any verify slot buffers cloned from the
        # old ones are stale — drop them (re-allocated lazily on the next captured verify).
        if self._slot_bufs is not None:
            for rec, convs in self._slot_bufs:
                ttnn.deallocate(rec)
                for c in convs:
                    ttnn.deallocate(c)
            self._slot_bufs = None
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
        # Taps are now the truth (zeros); the window mirror still holds the previous sequence's.
        self._conv_taps_stale = False
        self._conv_win_stale = True

    def _col_proj(self, x, weight, decode_progcfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG, prefill_out_dtype=None):
        """Column-parallel qkvz projection; DRAM-sharded decode matmul when enabled.
        prefill_out_dtype pins the prefill result when in0 is bf8."""
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
            _kpass1 or self.args.prefill_progcfg,
            self.args.dim,
            decode_out_memory_config=out_memory_config,
            prefill_compute_cfg=tpc.COMPUTE_HIFI2_NO_FP32_ACC if _kpass1 is not None else None,
            prefill_out_dtype=prefill_out_dtype,
        )

    def _normalize_valid_len(self, valid_len, T):
        """valid_len >= T means no padding in this chunk, so return None. Wormhole only."""
        if (
            not tpc.is_blackhole()
            and valid_len is not None
            and not isinstance(valid_len, (list, tuple))
            and valid_len >= T
        ):
            return None
        return valid_len

    def prefill_uses_native_conv1d(self, T, valid_len=None):
        """True when this chunk takes native conv1d. Callers use it to keep the gather dtype off the FIR path."""
        return self._gdn_conv1d and self._normalize_valid_len(valid_len, T) is None and self._conv1d_native_fits_l1(T)

    def _conv1d_native_fits_l1(self, T):
        """Whether native conv1d CBs fit L1: one output tile per core, and qkv_dim_tp * K at the TP=4 width."""
        if tpc.is_blackhole():
            return True
        grid = self.mesh.compute_with_storage_grid_size()
        _fits_grid = -(-T // tpc.TILE_SIZE) <= grid.x * grid.y
        _fits_channel_width = self.qkv_dim_tp * self.K <= 4096 * 4
        return _fits_grid and _fits_channel_width

    def _carry_tail_tile_stable(self, src, carry_len, C, full_T):
        """Last K-1 valid rows via a tile-aligned window, so slice programs are keyed on the bucket."""
        K = self.K
        TS = tpc.TILE_SIZE
        W = 2 * TS
        start = carry_len - (K - 1)
        base = min(max((start // TS) * TS, 0), full_T - W)
        off = start - base  # 0 <= off <= W-(K-1)
        _dram = ttnn.DRAM_MEMORY_CONFIG
        win = ttnn.slice(src, (0, base, 0), (1, base + W, C))
        win_t = ttnn.to_layout(win, ttnn.TILE_LAYOUT, memory_config=_dram)
        ttnn.deallocate(win)
        idx = ttnn.arange(0, W, 1, dtype=ttnn.float32, device=self.mesh)
        idx = ttnn.reshape(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), (1, 1, W))
        picks = []
        for i in range(K - 1):
            sel = ttnn.eq(idx, float(off + i))
            picks.append(ttnn.typecast(sel, win_t.dtype))
            ttnn.deallocate(sel)
        ttnn.deallocate(idx)
        sel_all = picks[0] if K - 1 == 1 else ttnn.concat(picks, dim=1)  # [1, K-1, W]
        if K - 1 > 1:
            for p in picks:
                ttnn.deallocate(p)
        out = ttnn.matmul(sel_all, win_t)  # [1, K-1, C]
        ttnn.deallocate(sel_all)
        ttnn.deallocate(win_t)
        return out

    def _shift_register_tail(self, src, T, conv_state, C, full_T=None):
        """Last K-1 rows of [conv_state ; src]. T < K-1 must not slice src alone (negative start)."""
        K = self.K
        if T >= K - 1:
            if full_T is not None and full_T > T and full_T >= 2 * tpc.TILE_SIZE and not tpc.is_blackhole():
                return self._carry_tail_tile_stable(src, T, C, full_T)
            return ttnn.slice(src, (0, T - (K - 1), 0), (1, T, C))
        _dram = ttnn.DRAM_MEMORY_CONFIG
        if conv_state is None:
            # Nothing carried and fewer than K-1 rows: the register is zero-padded on the left.
            return ttnn.slice(src, (0, 0, 0), (1, T, C))
        cs = (
            conv_state
            if conv_state.layout == src.layout
            else ttnn.to_layout(conv_state, src.layout, memory_config=_dram)
        )
        win = ttnn.concat([cs, src], dim=1, memory_config=_dram)
        if cs is not conv_state:
            ttnn.deallocate(cs)
        n = win.shape[1]
        tail = ttnn.slice(win, (0, max(0, n - (K - 1)), 0), (1, n, C))
        ttnn.deallocate(win)
        return tail

    def _conv1d_raw(self, x, clen, cpad, ppad):
        """One depthwise conv1d. Weights are cached per (input width, padding); those are different programs."""
        dev, K, C = self.mesh, self.K, self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        conv_cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        if self._conv1d_wprep is None:
            self._conv1d_wprep = {}
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
            # L1_FULL slice: the DRAM-slice path does host reads that begin_trace_capture rejects.
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            return_output_dim=False,
            return_weights_and_bias=False,
        )

    def _conv1d_window(self, xin, T):
        """Conv1d over an already-built [carry ; tokens] window. Shares the concat form's weight-prep cache."""
        C = self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        Lin = (self.K - 1) + T
        xin = ttnn.to_layout(xin, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        xin = ttnn.reshape(xin, (1, Lin, 1, C))
        out = self._conv1d_raw(xin, Lin, 0, (0, 0))
        ttnn.deallocate(xin)
        out = ttnn.sharded_to_interleaved(out, _dram)
        out = ttnn.reshape(out, (1, T, C))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
        # SiLU stays separate: folding it into conv_config.activation drops accuracy.
        return ttnn.silu(out, memory_config=_dram)

    def _conv1d_verify(self, win, T):
        """_conv1d_prefill for the fullbatch verify: window pre-built, new_state dead."""
        return self._conv1d_window(win, T)

    def _conv1d_prefill(self, qkv, T, conv_state, _force_splice=False, carry_len=None):
        """Depthwise causal conv1d + SiLU via ttnn.conv1d. Returns (out [1,T,C], new_state [1,K-1,C]) DRAM TILE.

        carry_len: carry comes from the first carry_len rows, not all T (masked buckets).

        Prepends K-1 carry rows with padding=0 so one program serves every chunk (native pad only zeros,
        so it can't inject cross-chunk carry into a shared trace).
        """
        dev, K, C = self.mesh, self.K, self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        Lin = (K - 1) + T
        # From-scratch chunks use conv1d's causal pad instead of a carry concat.
        _splice = (
            (_SPLICE_CARRY or _force_splice) and conv_state is not None and not tpc.is_blackhole() and T > tpc.TILE_SIZE
        )
        _native_pad = (conv_state is None or _splice) and not tpc.is_blackhole()
        _conv_len = T if _native_pad else Lin
        _conv_pad = [K - 1, 0] if _native_pad else 0
        _prep_pad = (0, 0, K - 1, 0) if _native_pad else (0, 0)
        _xfix = None  # splice input; stays None on Blackhole and on the from-scratch/concat paths
        if tpc.is_blackhole():
            # new_state: last K-1 real input tokens (for the next chunk's carry), TILE/DRAM.
            new_state = self._shift_register_tail(qkv, T if carry_len is None else carry_len, conv_state, C)
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
            # conv1d wants ROW_MAJOR; a TILE concat of the off-tile carry retile anyway.
            _rm = ttnn.ROW_MAJOR_LAYOUT
            # qkv is usually already ROW_MAJOR; keep to_layout for callers that still pass TILE.
            qkv_rm = qkv if qkv.layout == _rm else ttnn.to_layout(qkv, _rm, memory_config=_dram)
            new_state = self._shift_register_tail(
                qkv_rm, T if carry_len is None else carry_len, conv_state, C, full_T=T
            )
            new_state = ttnn.to_memory_config(ttnn.to_layout(new_state, ttnn.TILE_LAYOUT), _dram)
            # Build the splice input before deallocate(xin), which can free these tensors.
            if _splice:
                _cs_rm = ttnn.to_layout(conv_state, _rm, memory_config=_dram)
                _head = ttnn.slice(qkv_rm, (0, 0, 0), (1, tpc.TILE_SIZE, C), memory_config=_dram)
                _xfix = ttnn.concat([_cs_rm, _head], dim=1, memory_config=_dram)
                ttnn.deallocate(_head)
                if _cs_rm is not conv_state:
                    ttnn.deallocate(_cs_rm)
            if _native_pad:
                # No concat: conv1d applies the K-1 causal zero pad itself.
                xin = qkv_rm
            elif conv_state is None:
                pad = ttnn.zeros([1, K - 1, C], device=dev, dtype=ttnn.bfloat16, layout=_rm, memory_config=_dram)
                xin = ttnn.concat([pad, qkv_rm], dim=1, memory_config=_dram)
                ttnn.deallocate(pad)
            else:
                cs_rm = ttnn.to_layout(conv_state, _rm, memory_config=_dram)
                xin = ttnn.concat([cs_rm, qkv_rm], dim=1, memory_config=_dram)
                if cs_rm is not conv_state:
                    ttnn.deallocate(cs_rm)
            # When _native_pad, xin is qkv_rm; freeing it would free the conv input.
            if qkv_rm is not qkv and xin is not qkv_rm:
                ttnn.deallocate(qkv_rm)
        # Latch before reshape: reshape aliases xin's buffer, so identity checks stop working.
        _xin_aliases_qkv = xin is qkv
        xin = ttnn.reshape(xin, (1, _conv_len, 1, C))
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # Needs l1_small_size on the device (prefill/demo set 24576); matches the validated A/B config.
        conv_cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        if tpc.is_blackhole():
            # Channel-chunked conv: the depthwise CB is channel-dominated and can overflow one L1_FULL call.
            w1d_chunks = self.tw["conv_w1d"] if isinstance(self.tw["conv_w1d"], list) else [self.tw["conv_w1d"]]
            n_cc = len(w1d_chunks)
            assert C % n_cc == 0, f"GDN conv channels {C} not divisible by n_cc {n_cc}"
            cw = C // n_cc
            if self._conv1d_wprep is None:
                self._conv1d_wprep = [
                    ttnn.prepare_conv_weights(
                        weight_tensor=w,
                        input_memory_config=_dram,
                        input_layout=ttnn.ROW_MAJOR_LAYOUT,
                        weights_format="OIHW",
                        in_channels=cw,
                        out_channels=cw,
                        batch_size=1,
                        input_height=1,
                        input_width=Lin,
                        kernel_size=(1, K),
                        stride=(1, 1),
                        padding=(0, 0),
                        dilation=(1, 1),
                        has_bias=False,
                        groups=cw,
                        device=dev,
                        input_dtype=ttnn.bfloat16,
                        conv_config=conv_cfg,
                        compute_config=cc,
                    )
                    for w in w1d_chunks
                ]
            conv_outs = []
            for i, wprep in enumerate(self._conv1d_wprep):
                xin_i = xin if n_cc == 1 else ttnn.slice(xin, (0, 0, 0, i * cw), (1, Lin, 1, (i + 1) * cw))
                out_i = ttnn.conv1d(
                    input_tensor=xin_i,
                    weight_tensor=wprep,
                    device=dev,
                    in_channels=cw,
                    out_channels=cw,
                    batch_size=1,
                    input_length=Lin,
                    kernel_size=K,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=cw,
                    dtype=ttnn.bfloat16,
                    conv_config=conv_cfg,
                    compute_config=cc,
                    # L1_FULL: the DRAM-slice path does host reads that begin_trace_capture rejects.
                    slice_config=ttnn.Conv2dL1FullSliceConfig,
                    return_output_dim=False,
                    return_weights_and_bias=False,
                )
                if n_cc > 1:
                    ttnn.deallocate(xin_i)
                conv_outs.append(ttnn.reshape(ttnn.sharded_to_interleaved(out_i, _dram), (1, T, cw)))
            ttnn.deallocate(xin)
            out = conv_outs[0] if n_cc == 1 else ttnn.concat(conv_outs, dim=-1, memory_config=_dram)
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
            return ttnn.silu(out, memory_config=_dram), new_state

        # One conv over all channels. Cache weights per (width, padding); those are different programs.
        if self._conv1d_wprep is None:
            self._conv1d_wprep = {}

        # Splice path is implemented and left off: the extra conv costs more than the concat it saves.
        out = self._conv1d_raw(xin, _conv_len, _conv_pad, _prep_pad)
        # xin may alias the caller's qkv; freeing it would free that buffer.
        if not _xin_aliases_qkv:
            ttnn.deallocate(xin)
        # SiLU stays separate: folding it into the conv activation drops accuracy.
        if out.is_sharded():
            _pre_silu = out
            out = ttnn.silu(out, memory_config=out.memory_config())
            ttnn.deallocate(_pre_silu)
            out = ttnn.sharded_to_interleaved(out, _dram)
            if _xfix is not None:
                # Patch the carry-dependent rows on ROW_MAJOR interleaved out, before the TILE relayout.
                _fix = self._conv1d_raw(
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
                # 27B on Wormhole: bf8 matmul output so the reduce-scatter carries less. Not a later typecast.
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
        keep the intermediates resident. qkv_row_major returns qkv already ROW_MAJOR; only a zero offset can fuse the untilize.
        """
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
                # Pin the result to bf16, not in0's dtype: unfused prefill in0 can be bf8.
                _qkvzab_dt = ttnn.bfloat16 if x.dtype != ttnn.bfloat16 else None
                qkvzab = self._col_proj(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvzab_progcfg,
                    out_memory_config=_proj_mc,
                    prefill_out_dtype=_qkvzab_dt,
                )
            if qkv_row_major:
                qkv = ttnn.untilize_with_unpadding(qkvzab, (0, S - 1, qz - 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz), memory_config=out_mc)
            # z (output gate) lives across the chunk kernel; L1 z (8MB@S=2048)
            # clashes with the scan kernel CBs -> keep DRAM in chunk-prefill; decode (small S) keeps out_mc.
            _z_mc = ttnn.DRAM_MEMORY_CONFIG if (self._fuse_agmm and S > tpc.TILE_SIZE) else out_mc
            z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az), memory_config=_z_mc)
            # A slice start must be tile-aligned; ab_gap puts b on that boundary.
            if self._ab_gap:
                b_start = Nv + self._ab_gap
                a = ttnn.slice(qkvzab, (0, 0, az), (1, S, az + Nv), memory_config=out_mc)
                b = ttnn.slice(qkvzab, (0, 0, az + b_start), (1, S, az + b_start + Nv), memory_config=out_mc)
                ttnn.deallocate(qkvzab)
            else:
                # No gap: b is not tile-aligned, so slice an enclosing aligned block and split a/b from that.
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

        # Prefill qkvzab tensors grow with chunk length.
        _big_prefill = not tpc.is_blackhole()
        _proj_mc = ttnn.DRAM_MEMORY_CONFIG if _big_prefill else ttnn.L1_MEMORY_CONFIG
        # Native conv1d only where its CBs fit L1; otherwise the MAC FIR.
        _use_native_conv1d = self.prefill_uses_native_conv1d(T, valid_len)
        _qkv_rm = False
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=_proj_mc, qkv_row_major=_qkv_rm)

        # FIR conv1d; conv_state = previous chunk's last K-1 inputs (None/zero from scratch)
        _cstate = self.conv_carry if carry else None
        # Masked native conv only for a scalar valid_len >= K-1, so the tail lies inside qkv.
        _vl_native = self._normalize_valid_len(valid_len, T)
        _masked_native = (
            not _use_native_conv1d
            and self._gdn_conv1d
            and _vl_native is not None
            and not isinstance(_vl_native, (list, tuple))
            and int(_vl_native) >= self.K - 1
            and self._conv1d_native_fits_l1(T)
        )
        if _use_native_conv1d or _masked_native:
            conv, conv_new_state = self._conv1d_prefill(
                qkv, T, _cstate, carry_len=(int(_vl_native) if _masked_native else None)
            )
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

        # q/k/v may sit in L1; beta/g stay in DRAM. The L1 placement is for dim <= 4096 only.
        kd = self.key_dim_tp
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
        # Recurrent state outlives this call, so it must not stay in L1.
        if final_state.memory_config().buffer_type != ttnn.BufferType.DRAM:
            _fs_l1 = final_state
            final_state = ttnn.to_memory_config(final_state, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(_fs_l1)
        captured = None
        if return_state:
            # Per-user prefill: return this user's state for assemble_batched_state to stitch
            # into the batched buffers. No self.* writeback; tensors are not deallocated here.
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
                # Prefill wrote the taps, so the window mirror is behind until the next eager verify.
                self._conv_taps_stale = False
                self._conv_win_stale = True
            ttnn.deallocate(conv_new_state)
        # Gated RMSNorm + SiLU(z); norm/flatten in L1, gated output in DRAM for out-proj.
        _L1 = ttnn.L1_MEMORY_CONFIG
        # _elem/_norm_mc use the pre-cast dtype.
        _elem = 4 if o.dtype == ttnn.float32 else 2
        _norm_mc = _L1 if (tpc.is_blackhole() or Nv * T * Dv * _elem <= (8 << 20)) else ttnn.DRAM_MEMORY_CONFIG
        # Wormhole output path is bf16 unless QWEN35_GDN_OUT_FP32=1.
        if not tpc.is_blackhole() and o.dtype == ttnn.float32 and os.environ.get("QWEN35_GDN_OUT_FP32") != "1":
            _o_fp32 = o
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
            # Free the rms_norm output; rebinding would not deallocate it.
            _n_pre = n
            n = ttnn.experimental.nlp_concat_heads(n, memory_config=_L1)
            ttnn.deallocate(_n_pre)
            out_f = ttnn.reshape(n, (1, T, self.value_dim_tp))
        else:
            out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_norm_mc)
            ttnn.deallocate(o)
            out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp), memory_config=_norm_mc)
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
        # gated stays DRAM: L1 clashes with the out-proj circular buffers.
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
        self._conv_taps_stale = False  # taps written from outside: the window mirror is now behind
        self._conv_win_stale = True
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
        self.sync_conv_taps()  # read-modify-write of the taps: they must be current first
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
        self._conv_win_stale = True

    def remap_slots(self, remap):
        """Reindex the batched decode state after a vLLM batch condense: slot i takes the state
        previously at slot remap[i] (identity entries are no-ops). Mirrors
        seed_manager.apply_slot_remap for GDN's per-slot recurrent+conv state, which the plugin's
        slot_remap does not itself move. In-place copy into the fixed buffers (preserves the decode
        trace's baked addresses)."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self.sync_conv_taps()  # read-modify-write of the taps: they must be current first
        self._gather_indices(self.rec_state, idx, dim=0)
        for m in range(self.K):
            self._gather_indices(self.conv_states[m], idx, dim=1)
        self._conv_win_stale = True

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
        self._conv_taps_stale = False  # taps written from outside: the window mirror is now behind
        self._conv_win_stale = True

        # ---- output (gated RMSNorm + SiLU(z) gate + row-parallel out proj + all-reduce) ----
        # Wormhole: cast the output tail to bf16, same as forward_prefill, unless QWEN35_GDN_OUT_FP32=1.
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

        # Taps must be current: a fullbatch verify advances only the window mirror.
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
            # v has no GQA expand; expand Q/K on tile-aligned Dk.
            v = ttnn.reshape(
                ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, 1, Nv, Dv), memory_config=_L1
            )
            ttnn.deallocate(conv)
            q = _gqa_expand_heads(q, rf, B, Nv, Dk, _L1)
            k = _gqa_expand_heads(k, rf, B, Nv, Dk, _L1)
            # Leave beta/g at (1, B, Nv); the kernel reshapes them to [B, H].
            beta = ttnn.sigmoid(b, memory_config=_L1)
            g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        else:
            v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
            ttnn.deallocate(conv)
            q = ttnn.repeat_interleave(q, rf, dim=1)
            k = ttnn.repeat_interleave(k, rf, dim=1)
            # q/k/v stay in L1; this kernel does not gather across devices.
            q = ttnn.reshape(q, (B, 1, Nv, Dk), memory_config=_L1)
            k = ttnn.reshape(k, (B, 1, Nv, Dk), memory_config=_L1)
            v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)
            beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
            g = ttnn.reshape(
                ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1), (B, 1, Nv)
            )
        ttnn.deallocate(b)
        ttnn.deallocate(a)

        # fp32 decode on Blackhole unless QWEN35_GDN_DECODE_BF16=1; Wormhole stays bf16.
        _hp = tpc.is_blackhole() and os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"
        # Only the first B users' state participates when the batch is under the allocated max.
        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
        _bstep = self._decode_batch_split(B)
        # Do not pass tile_opt into the Blackhole upstream kernel; it has no such argument.
        _rec_kw = {"tile_opt": True} if self._decode_tile_opt else {}
        # model_args is accepted by the dispatch and ignored; it does not select the Wormhole fork.
        _rec_kw["model_args"] = self.args
        if self.use_fused_recurrent_decode:
            # Spec decode and verify must share this fused op.
            assert B == Bmax, "fused recurrent decode path supports full-batch only (spec decode, B=1)"
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
        elif _bstep >= B:
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
                # Spill each L1 state slice to DRAM before the next slice, or the slices collide in L1.
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
            # Eager Wormhole-9B keeps the L1 state; prefill spills it before the next conv1d.
            if not self._decode_tile_opt and new_rec.memory_config().buffer_type != ttnn.BufferType.DRAM:
                _nr_l1 = new_rec
                new_rec = ttnn.to_memory_config(new_rec, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(_nr_l1)
            self.rec_state = new_rec

        # tile_opt returns [B,H,1,V]; rms_norm still reduces the last dim.
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
        # decode_ccl_tuning: without it this all-reduce uses different defaults than decode.
        _dt = tpc.decode_ccl_tuning(self.args)
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **({"chunks_per_sync": _dt[0], "num_workers_per_link": _dt[1]} if _dt else {}),
        )
        return out

    def forward_verify_recurrent(self, x, valid_len, pre_gathered=False):
        """Spec-decode verify over the first valid_len rows, using the decode kernel. pre_gathered skips the all-gather."""
        assert valid_len <= tpc.TILE_SIZE, f"verify bucket {valid_len} exceeds one tile"
        return self._forward_verify_recurrent_batched(x, valid_len, pre_gathered=pre_gathered)

    def _forward_verify_recurrent_batched(self, x, valid_len, pre_gathered=False):
        """Same math as the per-token decode loop, with one decode matmul over the valid rows. Do not use the prefill AGMM."""
        tw, B, Nk, Nv, Dk, Dv = self.tw, self.B, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        if self.conv_states is None:
            self.reset_state()
        # Decode-config verify input is L1 width-sharded; interleave before a row slice. We own that copy.
        _x_owned = False
        if pre_gathered and x.is_sharded():
            x = ttnn.to_memory_config(x, mc)
            _x_owned = True
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        bucket = x.shape[-2]  # x is K-sharded [1, bucket, dim/tp] (full-dim when pre_gathered)
        T = valid_len
        kd = self.key_dim_tp

        # Gather valid_len rows, then one decode qkvzab matmul. S <= one tile uses matmul_1d_decode.
        x_valid = x if T == bucket else ttnn.slice(x, (0, 0, 0), (1, T, x.shape[-1]))
        if pre_gathered:
            # pre_gathered x is already full-dim. Gathering again would quadruple the feature dim.
            xg = x_valid
        else:
            # x is 3D [1, T, dim/tp] here (reshaped above), so gather the LAST (feature) dim = -1, not 3.
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
        qkv_all, z_all, a_all, b_all = self._project_qkvzab(xg, T, out_mc=mc)
        if xg is not x:
            ttnn.deallocate(xg)
        if _x_owned:
            ttnn.deallocate(x)

        # 2) Sequential conv + recurrence per token — identical building blocks to forward_decode, so
        #    self.conv_states / self.rec_state advance exactly as decode does (bit-exact slot capture).
        capture = getattr(self, "_capture_slots", False)
        if capture:
            self._ensure_verify_slot_bufs(T)
            self._verify_slots = self._slot_bufs
        else:
            self._verify_slots = None
        # Fullbatch verify removes the per-token loop; the conv stays one native conv1d over T.
        if self.use_fullbatch_verify:
            return self._verify_fullbatch(qkv_all, z_all, a_all, b_all, T, bucket, capture)
        # Per-token path: the taps below ARE the shift register (same contract as forward_decode).
        self.sync_conv_taps()
        self._conv_win_stale = True
        rf = Nv // Nk
        out_f_rows = []
        q_seq, k_seq, v_seq, beta_seq, g_seq = [], [], [], [], []
        for t in range(T):
            qkv_t = ttnn.reshape(ttnn.slice(qkv_all, (0, t, 0), (1, t + 1, self.qkv_dim_tp)), (1, B, self.qkv_dim_tp))
            st = self.conv_states
            for j in range(self.K - 1):
                ttnn.copy(st[j + 1], st[j])
            ttnn.copy(qkv_t, st[self.K - 1])
            ttnn.deallocate(qkv_t)
            conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
            for j in range(1, self.K):
                conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
            conv = ttnn.silu(conv, memory_config=_L1)

            q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
            k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
            v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
            ttnn.deallocate(conv)
            q = ttnn.reshape(ttnn.repeat_interleave(q, rf, dim=1), (B, 1, Nv, Dk), memory_config=_L1)
            k = ttnn.reshape(ttnn.repeat_interleave(k, rf, dim=1), (B, 1, Nv, Dk), memory_config=_L1)
            v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)

            a_t = ttnn.reshape(ttnn.slice(a_all, (0, t, 0), (1, t + 1, Nv)), (1, B, Nv))
            b_t = ttnn.reshape(ttnn.slice(b_all, (0, t, 0), (1, t + 1, Nv)), (1, B, Nv))
            beta = ttnn.reshape(ttnn.sigmoid(b_t, memory_config=_L1), (B, 1, Nv))
            ttnn.deallocate(b_t)
            g = ttnn.reshape(
                ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_t, tw["dt_bias"]), memory_config=_L1), (B, 1, Nv)
            )
            ttnn.deallocate(a_t)

            q_seq.append(q)
            k_seq.append(k)
            v_seq.append(v)
            beta_seq.append(beta)
            g_seq.append(g)
            if capture:
                _, conv_bufs = self._slot_bufs[t]
                for j, c in enumerate(self.conv_states):
                    ttnn.copy(c, conv_bufs[j])

        # One recurrence over T. The wrapper applies L2-norm, scale, and exp(g).
        def _stack(seq, d):
            if T == 1:
                return seq[0]
            cat = ttnn.concat(seq, dim=1, memory_config=mc)
            for x in seq:
                ttnn.deallocate(x)
            return cat

        q_all = _stack(q_seq, Dk)
        k_all = _stack(k_seq, Dk)
        v_all = _stack(v_seq, Dv)
        beta_all = _stack(beta_seq, Nv)
        g_all = _stack(g_seq, Nv)
        o_all, states = fused_recurrent_gated_delta_rule_ttnn(
            q_all,
            k_all,
            v_all,
            beta_all,
            g_all,
            scale=self.scale,
            initial_state=self.rec_state,
            device=self.mesh,
            output_per_token_state=capture,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        ttnn.deallocate(q_all)
        ttnn.deallocate(k_all)
        ttnn.deallocate(v_all)
        ttnn.deallocate(beta_all)
        ttnn.deallocate(g_all)
        # Keep the token-major per-token states; commit slices the accepted slot. Do not copy all T.
        if capture:
            self._verify_states = self._verify_states_buf = states  # [B,T,Nv,Dk,Dv]
            if self._stable_state:
                last = ttnn.reshape(ttnn.slice(states, (0, T - 1, 0, 0, 0), (B, T, Nv, Dk, Dv)), (B, Nv, Dk, Dv))
                ttnn.copy(last, self.rec_state)
                ttnn.deallocate(last)
            else:
                self.rec_state = ttnn.reshape(
                    ttnn.slice(states, (0, T - 1, 0, 0, 0), (B, T, Nv, Dk, Dv)), (B, Nv, Dk, Dv)
                )
        else:
            if self._stable_state:
                ttnn.copy(states, self.rec_state)
                ttnn.deallocate(states)
            else:
                self.rec_state = states
        for t in range(T):
            o_t = ttnn.reshape(ttnn.slice(o_all, (0, t, 0, 0), (B, t + 1, Nv, Dv)), (B, Nv, Dv))
            out_n = ttnn.rms_norm(o_t, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o_t)
            out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
            ttnn.deallocate(out_n)
            out_f_rows.append(ttnn.to_layout(out_f, rm))
            ttnn.deallocate(out_f)
        ttnn.deallocate(o_all)

        ttnn.deallocate(qkv_all)
        ttnn.deallocate(a_all)
        ttnn.deallocate(b_all)

        if T == 1:
            out_f_b = ttnn.to_layout(out_f_rows[0], ttnn.TILE_LAYOUT)
            for r in out_f_rows:
                ttnn.deallocate(r)
        else:
            cat = ttnn.concat(out_f_rows, dim=1, memory_config=mc)  # [1, T, value_dim_tp] ROW_MAJOR
            out_f_b = ttnn.to_layout(cat, ttnn.TILE_LAYOUT)
            ttnn.deallocate(cat)
            for r in out_f_rows:
                ttnn.deallocate(r)
        gated = _silu_mul(out_f_b, z_all, mc)
        ttnn.deallocate(out_f_b)
        ttnn.deallocate(z_all)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        o_red = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=mc,
        )
        if T < bucket:
            o_rm = ttnn.to_layout(o_red, rm)
            ttnn.deallocate(o_red)
            # ttnn.zeros is a host write that TT_FATALs inside a captured trace.
            pad = self._verify_pad_buf(bucket - T, o_rm.shape[-1], o_rm.dtype, rm, mc)
            o_full = ttnn.concat([o_rm, pad], dim=2, memory_config=mc)
            ttnn.deallocate(o_rm)
            o_red = ttnn.to_memory_config(ttnn.to_layout(o_full, ttnn.TILE_LAYOUT), mc)
            ttnn.deallocate(o_full)
        return o_red

    def _verify_fullbatch(self, qkv_all, z_all, a_all, b_all, T, bucket, capture):
        """Fully-batched verify with no per-token loop. Inputs are already projected [1,T,*]."""
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        kd, C, rf = self.key_dim_tp, self.qkv_dim_tp, Nv // Nk
        self._ensure_conv_win()  # no-op after the first (eager) call; never allocates inside a trace

        # Carry is a fixed-offset slice of the persistent shift register, so the trace can replay it.
        carry = ttnn.slice(self._conv_win_buf, (0, 1, 0), (1, self.K, C))
        # One window shared by the conv and the state stash.
        E = ttnn.concat([carry, qkv_all], dim=1, memory_config=mc)  # [1, K-1+T, C]
        ttnn.deallocate(carry)
        conv_all = self._conv1d_verify(E, T)  # [1,T,C], SiLU applied

        # q/k/v: feature-dim slices, then one repeat_interleave each.
        q_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        k_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
        v_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 2 * kd), (1, T, C)), (1, T, Nv, Dv))
        ttnn.deallocate(conv_all)
        if rf != 1:
            q_all = ttnn.repeat_interleave(q_all, rf, dim=2)
            k_all = ttnn.repeat_interleave(k_all, rf, dim=2)

        beta_all = ttnn.sigmoid(b_all, memory_config=_L1)
        g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_all, tw["dt_bias"]), memory_config=_L1)

        o_all, states = fused_recurrent_gated_delta_rule_ttnn(
            q_all,
            k_all,
            v_all,
            beta_all,
            g_all,
            scale=self.scale,
            initial_state=self.rec_state,
            device=self.mesh,
            output_per_token_state=capture,
            high_precision=(os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"),
        )
        for t in (q_all, k_all, v_all, beta_all, g_all):
            ttnn.deallocate(t)

        # Conv slots are rows [t, t+K) of window E.
        if capture:
            ttnn.copy(E, self._verify_win_buf)
            self._win_captured = True
            self._verify_states = self._verify_states_buf = states
            last = ttnn.reshape(ttnn.slice(states, (0, T - 1, 0, 0, 0), (self.B, T, Nv, Dk, Dv)), (self.B, Nv, Dk, Dv))
            if self._stable_state:
                ttnn.copy(last, self.rec_state)
                ttnn.deallocate(last)
            else:
                self.rec_state = last
        elif self._stable_state:
            ttnn.copy(states, self.rec_state)
            ttnn.deallocate(states)
        else:
            self.rec_state = states
        # Do not refill the K taps here. At T == 1 the tail slice aliases E, so do not deallocate it.
        tail = ttnn.slice(E, (0, T - 1, 0), (1, T - 1 + self.K, C))
        ttnn.copy(tail, self._conv_win_buf)
        if T > 1:
            # T == 1: the slice is a full-span alias of E. Deallocating it would double-free.
            ttnn.deallocate(tail)
        ttnn.deallocate(E)
        self._conv_taps_stale, self._conv_win_stale = True, False

        out_n = ttnn.rms_norm(ttnn.reshape(o_all, (T, Nv, Dv)), weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
        ttnn.deallocate(o_all)
        out_f_b = ttnn.reshape(out_n, (1, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f_b, z_all, mc)
        ttnn.deallocate(out_f_b)
        ttnn.deallocate(z_all)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        # Same collective config as decode's out-projection all-reduce.
        _dt = tpc.decode_ccl_tuning(self.args)
        o_red = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=mc,
            **({"chunks_per_sync": _dt[0], "num_workers_per_link": _dt[1]} if _dt else {}),
        )
        ttnn.deallocate(qkv_all)
        ttnn.deallocate(a_all)
        ttnn.deallocate(b_all)
        if T < bucket:
            o_rm = ttnn.to_layout(o_red, rm)
            ttnn.deallocate(o_red)
            pad = self._verify_pad_buf(bucket - T, o_rm.shape[-1], o_rm.dtype, rm, mc)
            o_full = ttnn.concat([o_rm, pad], dim=2, memory_config=mc)
            ttnn.deallocate(o_rm)
            o_red = ttnn.to_memory_config(ttnn.to_layout(o_full, ttnn.TILE_LAYOUT), mc)
            ttnn.deallocate(o_full)
        return o_red

    def _verify_pad_buf(self, rows, width, dtype, layout, mc):
        """Persistent zero pad so the verify output concat does not call ttnn.zeros inside a trace."""
        cache = getattr(self, "_verify_pad_cache", None)
        if cache is None:
            cache = self._verify_pad_cache = {}
        key = (rows, width, dtype, layout)
        buf = cache.get(key)
        if buf is None:
            buf = ttnn.zeros([1, 1, rows, width], device=self.mesh, dtype=dtype, layout=layout, memory_config=mc)
            cache[key] = buf
        return buf

    def _ensure_verify_slot_bufs(self, n):
        """Persistent per-token slot buffers so a captured trace does not allocate."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        # Allocate the conv window before trace warmup so capture and replay take the same copy.
        if self.use_fullbatch_verify:
            _wrows = self.K - 1 + n
            if self._verify_win_buf is None or self._verify_win_buf.shape[-2] != _wrows:
                if self._verify_win_buf is not None:
                    ttnn.deallocate(self._verify_win_buf)
                self._verify_win_buf = ttnn.zeros(
                    [1, _wrows, self.qkv_dim_tp],
                    device=self.mesh,
                    dtype=self.conv_states[0].dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mc,
                )
            self._ensure_conv_win()
        if self._slot_bufs is not None and len(self._slot_bufs) >= n:
            return
        if self._slot_bufs is not None:
            for rec, convs in self._slot_bufs:
                ttnn.deallocate(rec)
                for c in convs:
                    ttnn.deallocate(c)
        self._slot_bufs = [
            (ttnn.clone(self.rec_state, memory_config=mc), [ttnn.clone(c, memory_config=mc) for c in self.conv_states])
            for _ in range(n)
        ]

    def _ensure_conv_win(self):
        """Allocate the persistent shift register on the first eager call, never inside a captured trace."""
        if self._conv_win_buf is None:
            self._conv_win_buf = ttnn.zeros(
                [1, self.K, self.qkv_dim_tp],
                device=self.mesh,
                dtype=self.conv_states[0].dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._conv_win_stale = True
        if self._conv_win_stale:
            self.sync_conv_win()

    def sync_conv_taps(self):
        """Rebuild conv_states from the window. No-op unless a fullbatch verify left the taps behind."""
        if not self._conv_taps_stale:
            return
        self._conv_taps_stale = False
        if self._conv_win_buf is None or self.conv_states is None:
            return
        C = self.qkv_dim_tp
        for j in range(self.K):
            row = ttnn.slice(self._conv_win_buf, (0, j, 0), (1, j + 1, C))
            ttnn.copy(row, self.conv_states[j])
            ttnn.deallocate(row)

    def sync_conv_win(self):
        """Copy conv_states into the persistent window the traced verify reads its carry from."""
        if self._conv_win_buf is None or self.conv_states is None:
            return
        if self._conv_taps_stale:
            # The window is ahead of the taps; copying taps over it would undo the verify.
            self.sync_conv_taps()
            self._conv_win_stale = False
            return
        c = ttnn.concat(list(self.conv_states), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(c, self._conv_win_buf)
        ttnn.deallocate(c)
        self._conv_win_stale = False

    def commit_verify_slot(self, idx):
        """Copy verify slot idx into the persistent state. Does not free the slot buffers."""
        assert self._verify_states is not None, "commit_verify_slot called without a captured verify"
        # A verify (traced or eager) just advanced the window; the taps are behind either way, so
        # arm the rebuild BEFORE the full-acceptance early-out below.
        if self._win_captured:
            self._conv_taps_stale, self._conv_win_stale = True, False
        if idx == self._verify_states.shape[1] - 1:
            # Full acceptance: the verify already left durable state at the last token.
            self._verify_states = None
            return
        st = ttnn.reshape(
            ttnn.slice(self._verify_states, (0, idx, 0, 0, 0), (self.B, idx + 1, self.Nv, self.Dk, self.Dv)),
            (self.B, self.Nv, self.Dk, self.Dv),
        )
        ttnn.copy(st, self.rec_state)
        ttnn.deallocate(st)
        convs = self._slot_bufs[idx][1] if self._slot_bufs is not None else None
        if self._win_captured:
            # Shift register at token idx is rows [idx, idx+K) of the stashed window.
            w = ttnn.slice(self._verify_win_buf, (0, idx, 0), (1, idx + self.K, self.qkv_dim_tp))
            ttnn.copy(w, self._conv_win_buf)
            ttnn.deallocate(w)
        else:
            for j, c in enumerate(convs):
                ttnn.copy(c, self.conv_states[j])
            self._conv_taps_stale, self._conv_win_stale = False, True
        self._verify_states = None

    # Traced commit: every tensor the commit ops touch is persistent, so the body can be captured.

    def commit_verify_slot_ops(self, idx):
        """Device half of a traced commit. Reads _verify_states_buf, whose address the trace bakes in."""
        assert self._verify_states_buf is not None, "traced commit needs a captured verify"
        assert self._win_captured, "traced commit is the batched-conv (window) verify path only"
        st = ttnn.reshape(
            ttnn.slice(self._verify_states_buf, (0, idx, 0, 0, 0), (self.B, idx + 1, self.Nv, self.Dk, self.Dv)),
            (self.B, self.Nv, self.Dk, self.Dv),
        )
        ttnn.copy(st, self.rec_state)
        ttnn.deallocate(st)
        w = ttnn.slice(self._verify_win_buf, (0, idx, 0), (1, idx + self.K, self.qkv_dim_tp))
        ttnn.copy(w, self._conv_win_buf)
        ttnn.deallocate(w)

    def commit_verify_slot_host(self, idx):
        """Host half of a traced commit. Runs every iteration; execute_trace does not re-run Python."""
        assert self._verify_states is not None, "commit_verify_slot called without a captured verify"
        if self._win_captured:
            self._conv_taps_stale, self._conv_win_stale = True, False
        self._verify_states = None

    def _traced_commit_blockers(self):
        """Which traced-commit preconditions this layer fails, as a list of names. Empty == ready."""
        return [
            name
            for name, ok in (
                ("use_fullbatch_verify", self.use_fullbatch_verify),
                ("_win_captured", self._win_captured),
                ("_stable_state", self._stable_state),
                ("_verify_states_buf", self._verify_states_buf is not None),
                ("_verify_win_buf", self._verify_win_buf is not None),
                ("_conv_win_buf", self._conv_win_buf is not None),
                ("rec_state", self.rec_state is not None),
            )
            if not ok
        ]

    def traced_commit_ready(self):
        """Commit can be traced only when the verify window and every touched buffer are persistent."""
        return not self._traced_commit_blockers()

    def traced_commit_why(self):
        """Human-readable reason traced_commit_ready() is False (for the capture-time log)."""
        return ", ".join(self._traced_commit_blockers()) or "ready"
