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

# Wormhole takes a local fork of the decode step: q skips the fp32 promotion (it never feeds the
# state write), q_row/k_row use transpose instead of a reshape across tiled dims, and the GQA
# expansion runs there so repeat_interleave hits its TILE-native kernel. Blackhole dispatches to
# the shared upstream function unchanged. See recurrent_decode_wh.py.
from models.demos.blackhole.qwen36.tt.gdn.recurrent_decode_wh import (
    recurrent_gated_delta_rule_decode_dispatch as recurrent_gated_delta_rule_decode_ttnn,
)
from models.demos.blackhole.qwen36.tt.gdn.recurrent_decode_wh import wh_decode_fork_applies
from models.demos.blackhole.qwen36.tt.wh_compat import apply as _apply_wh_compat
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_reduce

_apply_wh_compat()  # Wormhole GDN L1 adjustments (see tt/wh_compat.py)


def _softplus_add(a, bias):
    """g-gate: softplus(a + bias) fused into one op (softplus as a post-activation on the add)."""
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def _silu_mul(x, z, memory_config, dtype=None):
    """out-gate: x * silu(z). NOT fused into one op: fusing silu via input_tensor_b_activations
    overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small inputs).
    dtype: optional output dtype (bf16 for the column-parallel prefill out-proj; default = x's).

    RE-TESTED and the warning above holds. An op-level probe against torch at [1,B,1024] shows the
    fused form BIT-EQUIVALENT to the separate one -- bf16 and fp32, at |z| up to ~120, no NaN
    anywhere -- so it looks safe in isolation. It is not. With real weights it takes
    test_gdn_tp_prefill (prefill-vs-decode) from 0.99996 to 0.000137, fused-chunk-vs-seq from
    0.99871 to -0.0012, and test_gdn_tp_peruser_state[B32] to PCC = nan. The real layer reaches
    magnitudes a synthetic probe does not. Do not re-try this without running test_gdn_tp.py.
    """
    s = ttnn.silu(z, memory_config=memory_config)
    if dtype is None:
        return ttnn.multiply(x, s, memory_config=memory_config)
    return ttnn.multiply(x, s, memory_config=memory_config, dtype=dtype)


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
        # a and b are padded to a full TILE of rows each (they are only nv_per=8 wide) so that both
        # land on a 32-column boundary of the fused output. The pad rows are zeros and are never
        # sliced back out -- _project_qkvzab takes exactly nv_per columns from each tile start.
        # Without this, b starts mid-tile and its slice costs an untilize+tilize every layer.
        _T = tpc.TILE_SIZE

        def _pad_to_tile(w):
            if w.shape[0] == _T:
                return w
            return torch.cat([w, torch.zeros(_T - w.shape[0], w.shape[1], dtype=w.dtype)], dim=0)

        assert nv_per <= _T, f"gdn_nv_tp ({nv_per}) > TILE ({_T}); a/b tile packing assumes one tile"
        fused = torch.cat(
            [
                torch.cat(
                    [
                        qkv_re[d * qkv_per : (d + 1) * qkv_per],
                        z_w[d * z_per : (d + 1) * z_per],
                        _pad_to_tile(a_w[d * nv_per : (d + 1) * nv_per]),
                        _pad_to_tile(b_w[d * nv_per : (d + 1) * nv_per]),
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
            cache_path=c("qkvzab_abtile" + (".il" if _proj1d else ".dramshard")),
            # STAYS bfloat8_b. This matmul is the single biggest op in the decode layer (~37us,
            # weight-bandwidth-bound: 6.4 MB of weight per token at ~173 GB/s), and bfloat4_b halves
            # that -- MEASURED -2.5% end to end (0.3336 -> 0.3254 ms/step). REJECTED on accuracy:
            # worst per-user decode PCC fell 0.99981 -> 0.93529 (B=8) and 0.99963 -> 0.93565 (B=32).
            # This projection feeds the recurrent state, so the error compounds over every decode
            # step of all 30 GDN layers. The routed experts use bf4 safely; this weight does not.
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
        # Separate A+B projection (column-parallel fallback)
        ab = torch.cat(
            [
                torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        tw["ab"] = tpc.shard_w(
            ab, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab"), dtype=ttnn.bfloat8_b
        )
    # Row-parallel out projection: DRAM-width-sharded (like the in-proj) — decode tput win.
    _out_sharded = getattr(args, "gdn_out_weight_memcfg", None) is not None
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.gdn_out_weight_memcfg if _out_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out.dramshard" if _out_sharded else "out"),
        # STAYS bfloat8_b. bfloat4_b halves this 2.23 MB weight read and MEASURED -0.8%
        # (0.3036 -> 0.3012 ms/step), far milder on accuracy than the in-projection's bf4 (this
        # output feeds the residual stream, not the recurrent state, so it does not compound the
        # same way) -- but still PCC 0.99994 -> 0.99501 at B=1 and 0.99981 -> 0.99064 at B=8.
        # 0.8% is not worth ~0.009 PCC per layer across 30 GDN layers of residual.
        dtype=ttnn.bfloat8_b,
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
            dtype=ttnn.bfloat8_b,
        )
    # Per-head params
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    # Constant per-channel weights that fold the q/k L2 scaling INTO their rms_norm instead of
    # paying a separate broadcast BinaryNg for each. l2_norm(x) is rms_norm(x) * K**-0.5, and q
    # additionally wants * scale; rms_norm already applies an optional per-channel weight, so both
    # multiplies ride along for free. q's folds scale too (scale defaults to K**-0.5 -> 1/K).
    _dk = args.gdn_dk
    tw["qn_w"] = tpc.replicate(torch.full((_dk,), (_dk**-0.5) * (_dk**-0.5), dtype=torch.float32), mesh, c("qn_w"))
    tw["kn_w"] = tpc.replicate(torch.full((_dk,), _dk**-0.5, dtype=torch.float32), mesh, c("kn_w"))
    # Conv taps (4), sharded per Q/K/V head grouping
    taps = tpc.prepare_conv_taps(conv1d_w, key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    # Stacked [K,1,C] taps for the Wormhole decode FIR (one broadcast multiply + one reduction
    # instead of the per-tap multiply/mac chain). See forward_decode. PR #54572.
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
        # GDN in/out projections. Wormhole drops to LoFi/no-fp32-acc: the weights are bfloat8_b,
        # so HiFi2's extra math passes buy precision the operands cannot hold (see
        # tpc.COMPUTE_LOFI_NO_FP32_ACC for the per-matmul sweep numbers). Blackhole unchanged.
        # GDN in/out projections. STAYS HiFi2 + fp32 dest-acc on BOTH arches.
        #
        # A full program-config sweep of all four GDN matmuls (tests/perf/test_sweep_gdn_matmuls.py,
        # n150x4 / 35B-A3B) found fidelity to be the ONLY lever -- grid and in0_block_w did nothing.
        # Both reduced-fidelity rungs were measured end to end and REJECTED:
        #
        #   config              sweep (isolated)      worst decode PCC   END-TO-END GDN time
        #   HiFi2 + fp32acc     baseline              0.99961            183.68 ms   <- kept
        #   HiFi2 no-fp32acc    in_proj_prefill -11%  0.99812            182.45 ms  (-0.7%)
        #   LoFi  no-fp32acc    decode -16% / -23%    0.99548            (not measured)
        #
        # The isolated -11% did NOT transfer: matmul went 37.47 -> 37.33 ms, 0.7% overall, because
        # this profile is decode-dominated and decode takes the 1D progcfg path. Paying 4x (HiFi2
        # no-fp32acc) or 10x (LoFi) the error for 0.7% is a bad trade anywhere, and especially here:
        # these projections feed the recurrent state, which compounds over every decode step and all
        # 30 GDN layers, and full-model batched decode is already marginal (test_model_tp).
        self.cfg = tpc.COMPUTE_HIFI2
        # Must match load_gdn_weights_tp gates
        self._dram_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
        self._fuse_ab = self._dram_sharded
        # Fuse prefill norm-allgather + qkvzab in-proj into all_gather_minimal_matmul_async.
        # Requires the folded qkvzab weight; norm's post-AG is disabled in layer.py (GDN, prefill).
        self._fuse_agmm = self._fuse_ab
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        # mmrs_prefill_supported() is the ARCH GUARD: the fused matmul+reduce-scatter hangs on
        # Wormhole (the 1-link LINEAR hop wants 18 RS cores and no split of the 8x8 grid
        # completes), so this arm is Blackhole-only. It returns True on BH, leaving the Blackhole
        # path exactly as main had it.
        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1 and tpc.mmrs_prefill_supported()
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
        # In-place state updates for decode/prefill traces (set by model allocate_kv_caches)
        self._stable_state = False
        self.conv_carry = None  # cross-chunk prefill conv carry [1, K-1, qkv_dim_tp]
        # Native ttnn.conv1d depthwise prefill; L1_FULL slice keeps it trace-safe.
        # Only used when valid_len is None (masked buckets keep the MAC FIR).
        self._gdn_conv1d = True
        # Split the depthwise conv over channel chunks so each native L1_FULL conv fits L1: the
        # per-channel-independent depthwise CB is channel-dominated (not reducible by act-block or
        # DRAM width/height slicing), and the 35B-A3B GDN qkv_dim_tp overflows a single call on BH.
        # 27B runs a single chunk (unchanged); see model_config.gdn_conv_channel_chunks.
        self._conv_chunks = getattr(args, "gdn_conv_channel_chunks", 1)
        self._conv1d_wprep = None  # prepared depthwise weight (populated on first prefill call)
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill

    def reset_state(self):
        def z(shape, mc=None):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                memory_config=mc,
            )

        # The conv shift register is L1-RESIDENT: it is the only state the decode hot path reads
        # every step (the FIR's per-tap multiply + K-1 mac, plus the K-1 shift copies), and it is
        # small -- K x [1,B,C] bf16 = 16 KB at B=1, 512 KB at B=32, per layer. Leaving it in DRAM
        # made all four of those ops read across the interface. MEASURED -1.1% (0.3016 -> 0.2982
        # ms/step, n150x4 35B-A3B B=1 traced).
        #
        # ONLY these. The other buffers below (conv_carry, the _zero_* sources, rec_state) are
        # touched once per SEQUENCE by reset_state_inplace / the prefill carry, never per token, so
        # L1 residency buys them nothing and _zero_rec alone is [B,Nv,Dk,Dv] = 8 MB at B=32.
        self.conv_states = [z((1, self.B, self.qkv_dim_tp), ttnn.L1_MEMORY_CONFIG) for _ in range(self.K)]
        # fp32 recurrent state by default (QWEN35_GDN_STATE_BF16=1 reverts).
        #
        # UNVALIDATED CANDIDATE: allocating this in L1 instead of DRAM removes both halves of the
        # per-layer state round trip (the kernel's DRAM->L1 hoist and the ttnn.copy write-back,
        # 512 KB each at B=1) and MEASURED -2.5% on the single-layer decode benchmark (0.3211 ->
        # 0.3132 ms/step). NOT taken: that benchmark holds one layer, while the real model keeps 30
        # GDN states resident at once (~15 MB of L1 at B=1, more at batch), and L1 pressure here is
        # already a known failure mode -- see _spill_rec_state_to_dram and the B=32 "clash with L1
        # buffers" note in recurrent_decode_wh. Needs a full-model memory check before it lands.
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

    def _spill_rec_state_to_dram(self):
        """Move rec_state out of L1 before prefill (ported from PR #54572).

        The prefill conv1d + chunk kernel run with very little L1 headroom on Wormhole; a resident
        [B,Nv,Dk,Dv] recurrent state is the documented "clash with L1 buffers" trigger. No-op when
        already DRAM or unset. Must NOT run under _stable_state: that path bakes rec_state's
        address into the prefill/decode traces."""
        if self.rec_state is None or self._stable_state:
            return
        if self.rec_state.memory_config().buffer_type == ttnn.BufferType.DRAM:
            return
        spilled = ttnn.to_memory_config(self.rec_state, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(self.rec_state)
        self.rec_state = spilled

    def _col_proj(self, x, weight, decode_progcfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """Column-parallel qkvz projection; DRAM-sharded decode matmul when enabled.
        out_memory_config: decode result placement (default DRAM; L1 keeps it resident)."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.cfg, memory_config=out_memory_config)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
            decode_out_memory_config=out_memory_config,
        )

    def _conv1d_prefill(self, qkv, T, conv_state):
        """Depthwise causal conv1d + SiLU via ttnn.conv1d. Returns (out [1,T,C], new_state [1,K-1,C]) DRAM TILE.

        Prepends K-1 carry rows with padding=0 so one program serves every chunk (native pad only zeros,
        so it can't inject cross-chunk carry into a shared trace).
        """
        dev, K, C = self.mesh, self.K, self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        _RM = ttnn.ROW_MAJOR_LAYOUT
        Lin = (K - 1) + T
        # ONE tile->row-major conversion of the [1,T,C] activation, up front, and every step that
        # follows stays in ROW_MAJOR. ttnn.conv1d requires ROW_MAJOR input anyway, so the tensor has
        # to cross the layout boundary at some point; doing it first means it crosses ONCE.
        #
        # Previously it crossed four times, because both the carry slice and the carry concat act on
        # dim 1 -- a TILE-tiled dim -- while the conv then wanted ROW_MAJOR regardless:
        #     slice(qkv, T-(K-1)..T)  -> untilize + tilize   (mid-tile row range)
        #     concat([carry, qkv], 1) -> untilize + tilize   (tiled-dim concat)
        #     to_layout(xin, ROW_MAJOR) -> untilize again
        # In ROW_MAJOR both are stick operations: a row slice selects whole sticks and a dim-1
        # concat appends them, neither needing a relayout.
        qkv_rm = ttnn.to_layout(qkv, _RM, memory_config=_dram)

        # last K-1 real input rows = the next chunk's carry
        new_state_rm = ttnn.slice(qkv_rm, (0, T - (K - 1), 0), (1, T, C))

        if conv_state is None:
            pad = ttnn.zeros([1, K - 1, C], device=dev, dtype=ttnn.bfloat16, layout=_RM, memory_config=_dram)
            xin = ttnn.concat([pad, qkv_rm], dim=1, memory_config=_dram)
            ttnn.deallocate(pad)
        else:
            # callers hold the carry in TILE (it lands in TILE persistent buffers); it is only
            # [1,K-1,C] = one tile-row, so converting it is cheap next to the chunk itself.
            cs_rm = conv_state if conv_state.layout == _RM else ttnn.to_layout(conv_state, _RM, memory_config=_dram)
            xin = ttnn.concat([cs_rm, qkv_rm], dim=1, memory_config=_dram)
            if cs_rm is not conv_state:
                ttnn.deallocate(cs_rm)
        ttnn.deallocate(qkv_rm)
        xin = ttnn.reshape(xin, (1, Lin, 1, C))
        # carry returned in TILE for the callers; one tile-row, not the full chunk.
        new_state = ttnn.to_memory_config(ttnn.to_layout(new_state_rm, ttnn.TILE_LAYOUT), _dram)
        ttnn.deallocate(new_state_rm)
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # Needs l1_small_size on the device (prefill/demo set 24576); matches the validated A/B config.
        conv_cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        # Depthwise conv over channel chunks (per-channel-independent → concatenation is exact);
        # n_cc=1 (27B) is the original single call. Weights prepped once per chunk (warmup) so trace
        # replay stays device-only.
        # n_cc comes from args.gdn_conv_channel_chunks (=4 on a WH 8x8 grid). Each chunk costs its
        # own slice + Halo + InterleavedToSharded + conv + ShardedToInterleaved, so fewer chunks
        # looks like free speed. DO NOT LOWER IT. Swept on n150x4 (T=128, traced, single layer):
        #     n_cc    1       2       4 (default)   8
        #     ms   0.8884  0.8746     0.8965     0.9478
        # n_cc=2 is -2.4% and passes the ENTIRE test_gdn_tp suite (all PCC clean, no L1 error) --
        # and it BREAKS THE REAL MODEL. demo/text_demo.py traced_128 on the full 40 layers dies with
        #   "Statically allocated circular buffers in program 120 clash with L1 buffers
        #    on core range [0-0 - 7-7]"
        # which is precisely what this heuristic exists to prevent (ModelArgs._init_tp_config).
        # The default passes the same demo at ttft=0.66s / 15.95 tok/s. A single-layer benchmark
        # cannot see this: one layer has the whole L1 to itself. Validate any change to the conv's
        # L1 footprint with the demo, not with test_gdn_tp.
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
                # L1_FULL slice: keep the conv in L1 instead of DRAM-width-slicing. The DRAM-slice path does
                # host reads that begin_trace_capture rejects (see uniad); L1_FULL is trace-safe (as UNet).
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
        # SiLU stays separate (folding via conv_config.activation drops PCC to ~0.84 on this depthwise).
        return ttnn.silu(out, memory_config=_dram), new_state

    def _row_proj(self, x, weight):
        """Row-parallel out projection: DRAM-sharded decode/prefill matmul (K=gdn_value_dim_tp),
        matching the in-proj. Falls back to plain interleaved on single device (no sharded memcfg)."""
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            # L1 out, not DRAM: the reduce-scatter that consumes this reads it straight back, and
            # at decode the partial is one tile row (8 KB), so keeping it resident beats a DRAM
            # round trip. MEASURED -0.5% (0.3030 -> 0.3017 ms/step, n150x4 B=1 traced, reproduced
            # over two pairs), PCC unchanged. Placement only -- no numerics change.
            return tpc.matmul_1d_decode(
                x, weight, self.args.gdn_out_decode_1d_progcfg, self.cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG
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
                    self.cfg,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                qkvzab = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvzab_progcfg, out_memory_config=_proj_mc)
            qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz), memory_config=out_mc)
            # z (output gate) lives across the chunk kernel (gated = out_f * silu(z)); L1 z (6MB@S=2048)
            # clashes with the scan kernel CBs -> keep DRAM in chunk-prefill; decode (small S) keeps out_mc.
            _z_mc = ttnn.DRAM_MEMORY_CONFIG if (self._fuse_agmm and S > tpc.TILE_SIZE) else out_mc
            z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az), memory_config=_z_mc)
            # a and b each own a whole tile of columns (load_gdn_weights_tp pads them), so both
            # slices START tile-aligned and are plain width truncations -- no untilize/tilize, and
            # no enclosing `ab` block to carve up first. Under the old bare-nv packing b began at
            # az+Nv, mid-tile, which forced untilize -> slice -> tilize on every layer.
            _T = tpc.TILE_SIZE
            a = ttnn.slice(qkvzab, (0, 0, az), (1, S, az + Nv), memory_config=out_mc)
            b = ttnn.slice(qkvzab, (0, 0, az + _T), (1, S, az + _T + Nv), memory_config=out_mc)
            ttnn.deallocate(qkvzab)
            return qkv, z, a, b
        qkvz = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvz_progcfg)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, S, qz))
        z = ttnn.slice(qkvz, (0, 0, qz), (1, S, az))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, self.tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv))
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
        self._spill_rec_state_to_dram()
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

        # Cross-chunk carry (chunk-outer prefill): when _stable_state, the recurrent + conv
        # state continue from the persistent buffers (zeroed at sequence start by
        # reset_state_inplace, so a from-scratch single pass reads zeros == None). The demo
        # path (_stable_state False) is unchanged: no carry, reassign state.
        # Per-user prefill (return_state) is always from scratch: must not carry the shared
        # batched buffer (other users' state) as its initial recurrent/conv state.
        carry = self._stable_state and not return_state
        if carry and self.conv_carry is None:
            self.reset_state()

        # Prefill qkvzab in L1: keeps proj + q/k/v/z/a/b resident for conv+gate prep.
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=ttnn.L1_MEMORY_CONFIG)

        # FIR conv1d; conv_state = previous chunk's last K-1 inputs (None/zero from scratch)
        _cstate = self.conv_carry if carry else None
        if self._gdn_conv1d and valid_len is None:
            # Native depthwise ttnn.conv1d (masked buckets keep the MAC FIR: valid_len new_state differs)
            conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)
        else:
            conv, conv_new_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                self.K,
                self.mesh,
                # Conv in L1 (output freed before chunk kernel; new_state lands in DRAM internally)
                memory_config=ttnn.L1_MEMORY_CONFIG,
                conv_state=_cstate,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                valid_len=valid_len,
            )
        ttnn.deallocate(qkv)

        # q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes it.
        kd = self.key_dim_tp
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
            ttnn.deallocate(conv_new_state)
        # Gated RMSNorm + SiLU(z); norm/flatten in L1, gated output in DRAM for out-proj
        _L1 = ttnn.L1_MEMORY_CONFIG
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

        Under _stable_state (decode-trace path) the result is copied into the fixed-address
        buffers; otherwise (demo/standalone) it is assigned.
        """
        assert len(rec_list) == self.B and len(conv_new_list) == self.B, "need one state per batch row"
        D = self.qkv_dim_tp
        rec_batched = ttnn.concat(rec_list, dim=0)  # [B, Nv, Dk, Dv]
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
        self._spill_rec_state_to_dram()
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
        if self._stable_state and self.conv_states is not None:
            for m in range(self.K):
                ttnn.copy(new_conv[m], self.conv_states[m])
                ttnn.deallocate(new_conv[m])
        else:
            self.conv_states = new_conv

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

    def forward_decode(self, x):
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
            # pad_and_free, NOT pad + deallocate: this pad can alias its input (tpc.pad_and_free).
            qkv = tpc.pad_and_free(qkv, [(0, 0), (0, Bmax - B), (0, 0)], value=0.0, memory_config=_L1)
        # The FIR runs BEFORE the shift, reading the newest token straight out of `qkv` rather than
        # from a slot it was copied into first. That removes one of the four shift copies -- at
        # B=1 each is a 4 KB [1,B,C] transfer costing ~5.6us of pure launch overhead, so the four
        # of them are 7.1% of the decode layer (probe: deleting all four takes 0.3154 -> 0.2931).
        #
        # Equivalent because the taps are ordered oldest->newest: the old code shifted so that
        # st[0..K-1] held (t-K+1 .. t) and summed st[j]*taps[j]; reading st[1..K-1] plus `qkv`
        # before the shift indexes exactly those same K values. st[0] becomes dead (prefill still
        # zeroes it, harmlessly).
        conv = ttnn.multiply(st[1], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, self.K - 1):
            conv = ttnn.mac(st[j + 1], tw["conv_taps"][j], conv)
        conv = ttnn.mac(qkv, tw["conv_taps"][self.K - 1], conv)
        # now roll the window forward for the next step: K-1 copies, not K. The remaining K-1 are
        # structural under tracing -- the traced op sequence bakes in buffer addresses, so the
        # window cannot be advanced by rotating an index instead of moving data. Collapsing them
        # into one [K,1,B,C] buffer (slice + slice_write + a stacked broadcast-multiply/reduce FIR,
        # using the conv_taps_stack weight that is already loaded) would get most of the remaining
        # ~5%, but conv_states has 57 references across tp.py and model.py including the per-user
        # state assembly, so it is its own piece of work.
        for j in range(1, self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        ttnn.deallocate(qkv)
        # FIR over the K taps: per-tap multiply + (K-1) mac.
        # PR #54572's stacked variant (concat[K,B,C] -> broadcast multiply -> sum) was MEASURED
        # THERE at B=32 and is a REGRESSION here at B=1: ttnn.concat on four [1,B,C] tensors emits
        # a FillPad to tile-align, and on this config that cost +10.4ms FillPad +3.4ms concat
        # +2.1ms reduce against the 7.3ms of mac it removed -- net +8.7ms over a 128-step decode
        # (n150x4, 35B-A3B, C=2048). At B=32 the pad amortizes over 32 rows; at B=1 it does not.
        # KEEP ttnn.mac here. Note 6 in recurrent_decode_wh._write_state_wh measures ttnn.mac as
        # ~2x SLOWER than multiply+add -- but that is on the rank-1 BROADCAST shape
        # ([B,H,K,1] x [B,H,1,V]). These taps are a plain same-shape elementwise product, where mac
        # is the better op: replacing these three with multiply+add pairs MEASURED +3.8%
        # (0.3746 -> 0.3888 ms/step). The two cases genuinely differ; do not generalise either way.
        conv = ttnn.silu(conv, memory_config=_L1)

        kd = self.key_dim_tp
        # GQA expansion (Nk→Nv) is NOT done here: it is handed to the recurrent kernel as
        # gqa_repeat and applied there, after the L2-norm, on the [B,H,1,K] row layout where
        # ttnn.repeat_interleave hits its TILE-native kernel instead of a ROW_MAJOR round trip.
        # See recurrent_decode_wh.recurrent_gated_delta_rule_decode_wh's q_row/k_row block.
        # Decode: hand q/k/v to the recurrent kernel in L1. The kernel typecasts + does a LOCAL
        # l2-norm (no cross-device gather), so placement is output-neutral here (unlike SDPA-q,
        # which hard-requires DRAM, and unlike the residual→DistributedNorm all-gather).
        # fp32 decode step by default (QWEN35_GDN_DECODE_BF16=1 reverts)
        _hp = os.environ.get("QWEN35_GDN_DECODE_BF16") != "1"
        # q and k are ADJACENT in the conv output and the same width, so when the recurrence will
        # take the Wormhole fork, hand them over as ONE tensor and let it run the norm, transpose
        # and GQA expansion once instead of twice. Gated on the SAME predicate the dispatch uses --
        # the upstream fallback takes q/k separately and applies its own q scale, so it must get
        # the unfused pair and the uncompensated epsilon below.
        _fuse_qk = wh_decode_fork_applies(B, Nk, Dk, Dv, gqa_repeat=Nv // Nk, high_precision=_hp)
        q = k = qk = None
        if _fuse_qk:
            qk = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, 2 * kd)), (B, 1, 2 * Nk, Dk), memory_config=_L1)
        else:
            q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, 1, Nk, Dk), memory_config=_L1)
            k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, 1, Nk, Dk), memory_config=_L1)
        # v stays [B,1,Nv,Dv] and the recurrence transposes it. Building it ROWED as [B,Nv,1,Dv]
        # instead deletes that transpose and MEASURED -1.2% at B=1 (0.3032 -> 0.2997 ms/step), but
        # REJECTED: the rowed shape pads to 8x the tiles ([32,8,1,128] = 2 MB vs 256 KB at B=32),
        # and that tips the already-marginal B=32 decode into "Statically allocated circular
        # buffers clash with L1 buffers". Faster at B=1, broken at B=32 -- not a trade worth 1.2%.
        # When the fork applies, reshape v STRAIGHT to the rowed [B,Nv,1,Dv] the recurrence's
        # matmuls want, folding its transpose into a reshape that had to happen anyway. This is the
        # "rowed v" rejected just above -- sound here only because the same predicate that gates the
        # q/k fusion also excludes the B=32 case whose L1 it broke.
        _v_shape = (B, Nv, 1, Dv) if _fuse_qk else (B, 1, Nv, Dv)
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), _v_shape, memory_config=_L1)
        ttnn.deallocate(conv)
        rf = Nv // Nk

        beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
        ttnn.deallocate(b)
        # DO NOT add dtype=ttnn.float32 here to save the recurrence's typecast of g. It reads free
        # (the bf16 x bf16 product is already accumulated at fp32 in dest, so writing it out as
        # fp32 is one extra tile-store) and it IS one op fewer, but MEASURED (n150x4, 35B-A3B) it
        # breaks test_gdn_tp_peruser_state[B32]: users 29 and 31 drop to PCC 0.78 / 0.92 against
        # their B=1 references while the other 30 stay at 1.00000. Bisected against exactly this
        # line -- reverting it alone restores the test. The bf16 g and the kernel-side typecast stay.
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a)
        g = ttnn.reshape(g, (B, 1, Nv))

        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
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
            gqa_repeat=rf,
            q_norm_weight=tw["qn_w"],
            k_norm_weight=tw["kn_w"],
            qk_fused=qk,
            v_rowed=_fuse_qk,
        )
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
            self.rec_state = new_rec

        # o arrives [B,Nv,1,Dv] (the recurrence's native read-query layout). rms_norm reduces the
        # last dim, which is Dv either way, so it runs directly on that -- no [B,Nv,Dv] detour.
        # Gated norm (no +1). EPSILON IS NOT A CONSTANT HERE. The fused-q/k path norms q with k's
        # fold weight, which is q's times Dk**0.5, so o arrives Dk**0.5 larger. rms_norm is only
        # scale-free in the eps->0 limit and this o is SMALL (absmax ~0.019, see the dispatch
        # docstring), so eps genuinely sets the answer -- dropping q's scale and leaving eps alone
        # MEASURED test_gdn_tp[B8] PCC 0.826. The identity rms_norm(s*x, s^2*eps) == rms_norm(x, eps)
        # restores it exactly at zero op cost: scale eps by s^2 = Dk.
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6 * (Dk if _fuse_qk else 1), memory_config=_L1)
        out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f, z, _L1)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        # This reduce-scatter is ~32us/layer (10% of the decode layer) for an 8 KB payload, i.e. it
        # is sync/latency bound, not bandwidth bound. Two routes out were tried and neither works:
        #
        #   * Fusing it into the out-proj via tpc.matmul_reduce_scatter_decode. That op HANGS on
        #     Wormhole (see tpc.mmrs_prefill_supported: the 1-link LINEAR hop wants 18 RS cores and
        #     no split of the 8x8 grid completes), and its own docstring records that the fusion
        #     LOSES at decode M=1 even on Blackhole, where the 2D matmul collapses to ~8 cores.
        #     That is why matmul_reduce_scatter_decode has no caller -- deliberately, not by
        #     oversight.
        #   * Tuning the CCL. Swept chunks_per_sync x num_workers_per_link over
        #     {1,2,4,10,20} x {1,2,4} (the only two knobs tt_all_reduce exposes -- it has no
        #     num_buffers_per_channel parameter): best 0.3196 vs 0.3211 ms/step at the defaults,
        #     0.5% and inside run-to-run spread. Left at the defaults rather than adding a knob.
        # ~32us/layer and the single biggest non-matmul op, but NOT reducible from here:
        #
        #  * Fusing it into the out-proj (tpc.matmul_reduce_scatter_decode) is a dead end twice over:
        #    matmul_reduce_scatter_async does not complete at all on a (1,4) Wormhole mesh (see
        #    tpc.mmrs_prefill_supported), and even on Blackhole the fusion LOSES at decode M=1,
        #    where the 2D matmul collapses to ~8 cores. That helper is deliberately uncalled.
        #  * The CCL knobs tt_all_reduce exposes do nothing here. Swept chunks_per_sync in
        #    {1,2,4,10,20} x num_workers_per_link in {1,2,4}: every result landed in 0.3197-0.3255
        #    ms/step against 0.3209 at the defaults, i.e. inside run-to-run noise. The payload is
        #    ~8 KB at B=1, so this op is sync/latency-bound, not bandwidth- or worker-bound.
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
