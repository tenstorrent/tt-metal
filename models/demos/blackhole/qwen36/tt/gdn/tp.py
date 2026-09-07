# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel Gated DeltaNet for Qwen3.5.

Recurrence is per value-head (no cross-device comms inside); all-reduce after row-parallel out.
Reuses `recurrent_gated_delta_rule_decode_ttnn`; weights interleaved. GDN norm uses raw weight
(no +1) + SiLU(z) gate — distinct from QK/layer norms.
"""
import math
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_reduce


def _softplus_add(a, bias):
    """g-gate: softplus(a + bias) fused into one op (softplus as a post-activation on the add)."""
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def _silu_mul(x, z, memory_config):
    """out-gate: x * silu(z). NOT fused into one op: fusing silu via input_tensor_b_activations
    overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small inputs)."""
    return ttnn.multiply(x, ttnn.silu(z, memory_config=memory_config), memory_config=memory_config)


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's linear_attn.* weights across the mesh."""
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp

    if cache_dir is not None:
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
        # QWEN36_GDN_PROJ_CHUNKS: the chunked-AGMM in-projection needs every chunk width to be a
        # multiple of TILE_SIZE and the widths to sum to the weight's logical N, so the per-device
        # width must itself be tile-aligned (4120 -> 4128 at TP=4). The 8 appended columns are zeros,
        # which is byte-for-byte what TILE layout already pads the 4120-wide weight with, so the
        # on-device weight bytes and every matmul result are unchanged; the extra output columns sit
        # after b and are never read (a/b are taken at [0:Nv] / [Nv:2*Nv]). Distinct cache key because
        # the cached tensor records the logical width.
        _pad_n = 0
        if tpc.proj_chunks_mode():
            _per_dev = fused.shape[0] // tp
            _pad_n = (-_per_dev) % tpc.TILE_SIZE
        if _pad_n:
            _per_dev = fused.shape[0] // tp
            fused = torch.cat(
                [
                    torch.cat(
                        [fused[d * _per_dev : (d + 1) * _per_dev], fused.new_zeros(_pad_n, fused.shape[1])], dim=0
                    )
                    for d in range(tp)
                ],
                dim=0,
            )
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            # The DRAM-sharded memcfg rounds N up to TILE_SIZE*DRAM_CORES, so the 8 pad columns do
            # not change its shard shape (tp_common.create_dram_sharded_mem_config:97-109).
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.gdn_qkvzab_weight_memcfg,
            cache_path=c("qkvzab" + (".il" if _proj1d else ".dramshard") + (".padN" if _pad_n else "")),
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
        dtype=ttnn.bfloat8_b,
    )
    # AGMM out-proj mode (QWEN36_GDN_OUT_MODE=agmm): column-sharded copy of out_proj so the out-proj
    # runs as all-gather(gated) + column-parallel matmul (fp32 accumulation inside the matmul; no
    # cross-device partial sums) instead of row-parallel matmul + reduce-scatter.
    if os.environ.get("QWEN36_GDN_OUT_MODE", "mmrs_fp32") in ("agmm", "ag_mm"):
        tw["out_col"] = tpc.shard_w(
            sd[P + "out_proj.weight"],
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=c("out_col"),
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
        self._fuse_agmm = self._fuse_ab
        # QWEN36_GDN_PROJ_CHUNKS (default 0 = off): let the in-proj AGMM write qkv | z | ab as three
        # tensors instead of one wide tensor + three ttnn.slice ops. Read once here so it matches the
        # (env-gated) tile padding load_gdn_weights_tp applied to tw["qkvz"]; the padded-width check
        # in _project_qkvzab is the real guard if the two ever disagree.
        self._proj_chunks = tpc.proj_chunks_mode()
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        # QWEN36_GDN_OUT_MODE:
        #   "mmrs_fp32" (default) fused fp32 matmul + reduce-scatter (matmul_reduce_scatter_async), ~1.05 ms/layer @2048.
        #   "agmm"  all-gather the bf16 gated activation + column-parallel fused matmul (all_gather_minimal_matmul_async)
        #           on the column-sharded out_proj (tw["out_col"]): ~0.42 ms/layer, fp32 accumulation inside the matmul,
        #           no cross-device partial sums. Validated at TP=4 (P150x4) only.
        #   "ag_mm" same math, un-fused (all_gather_async + 2D matmul); used automatically for TP != 4.
        # Both AG modes need tw["out_col"], which load_gdn_weights_tp creates only when the env var is set at load time
        # (costs a second ~8 MB/device copy of out_proj per layer because decode/batched paths still use tw["out"]).
        _mode = os.environ.get("QWEN36_GDN_OUT_MODE", "mmrs_fp32")
        if _mode not in ("mmrs_fp32", "agmm", "ag_mm"):
            logger.warning(f"QWEN36_GDN_OUT_MODE={_mode!r} unknown; using mmrs_fp32")
            _mode = "mmrs_fp32"
        if _mode != "mmrs_fp32" and "out_col" not in tw:
            logger.warning(
                "QWEN36_GDN_OUT_MODE set after the weights were loaded without tw['out_col']; using mmrs_fp32"
            )
            _mode = "mmrs_fp32"
        if _mode == "agmm" and args.num_devices != 4:
            logger.warning("QWEN36_GDN_OUT_MODE=agmm is validated at TP=4 only; using ag_mm")
            _mode = "ag_mm"
        self._gdn_out_mode = _mode
        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1 and _mode == "mmrs_fp32"
        # QWEN36_GDN_CONV=kda: fused depthwise causal conv1d + SiLU + q/k/v split
        # (ttnn.experimental.kda.qkv_causal_conv1d_silu, 4 taps) instead of the conv2d/halo relayout chain
        # (~-0.3 ms/layer @2048). Full (unmasked) chunks only; the masked eager path keeps the FIR.
        self._gdn_conv_kda = os.environ.get("QWEN36_GDN_CONV", "conv2d") == "kda"
        if self._gdn_conv_kda and self.K != 4:
            logger.warning(f"QWEN36_GDN_CONV=kda needs a 4-tap conv (got K={self.K}); using conv2d")
            self._gdn_conv_kda = False
        # channel_chunk_size must divide the per-device qkv width and be tile-aligned: 2560 -> 512 (TP=4), 1280 -> 256 (TP=8).
        self._kda_chunk = math.gcd(self.qkv_dim_tp, int(os.environ.get("QWEN36_GDN_KDA_CHUNK", "512")))
        assert self._kda_chunk % tpc.TILE_SIZE == 0, f"KDA channel chunk {self._kda_chunk} must be tile aligned"
        # QWEN36_KDA_TILE_IN=1: feed the fused conv op the TILE-layout activation directly. The op then reads
        # whole tiles and shifts rows on the math engine (matmul against constant 0/1 matrices) instead of
        # requiring a ROW_MAJOR gather, which removes the ~85 us full-chunk untilize per layer. The carry still
        # comes back as TILE [1,K-1,C], so every other piece of conv state bookkeeping is unchanged.
        self._kda_tile_in = self._gdn_conv_kda and os.environ.get("QWEN36_KDA_TILE_IN") == "1"
        # QWEN36_GDN_DECODE_KDA=1: decode recurrence through the fused KDA chunk ops on a 32-row chunk (token in
        # row 0, beta/g zero on the padding rows -> t_inv is the identity and the chunk math reduces exactly to
        # the single-step delta rule). Replaces the ~35-op ttnn recurrent step. Constants built lazily.
        self._decode_kda = os.environ.get("QWEN36_GDN_DECODE_KDA") == "1"
        self._kda_dec_const = None
        # QWEN36_KDA_RM_L1=1: ROW_MAJOR fallback tuning. The untilize itself cannot use more than 64 cores
        # for a [1,2048,2560] chunk -- UntilizeCodegen splits work over tile-rows only
        # (untilize_codegen_program_factory.cpp:562 choose_2d_ncol returns 1 whenever
        # total_tile_rows >= valid_cores/2, then build_main_split hands each of the 64 tile-rows to one core).
        # sub_core_grids is not an escape hatch either: it forces the native path, whose sub-core-grid factory
        # TT_FATALs on anything taller than one tile row (untilize_device_operation.cpp:118-127).
        # The one lever left is where the row-major copy lands: L1 makes the untilize writer and the conv
        # reader L1-local instead of DRAM. It may also shrink get_max_l1_space enough to drop the codegen CB
        # tier to Native (untilize_codegen_cb_plan.cpp:189-206), which reaches a genuinely 2D block split
        # (UntilizeMultiCoreBlockProgramFactory) -- so check the op's CORE COUNT, not just its duration, when
        # measuring this. Costs ~10.5 MB of interleaved L1 at T=2048 (~96 KB/core), so it is off by default.
        # Ignored when _kda_tile_in is on (there is no full-chunk untilize left to tune).
        self._kda_rm_l1 = self._gdn_conv_kda and os.environ.get("QWEN36_KDA_RM_L1") == "1"
        # Zeros [1, K-1, qkv_dim_tp] for a from-scratch prefill (layout follows the input mode); see reset_state.
        self._kda_zero_hist = None
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
        self._conv1d_wprep = None  # prepared depthwise weight (populated on first prefill call)
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill

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
        if self._gdn_conv_kda and self._kda_zero_hist is None:
            # Zero history for a from-scratch KDA prefill (eager path); allocated here, never under trace.
            # Layout must match the activation the op is given: TILE under QWEN36_KDA_TILE_IN, else ROW_MAJOR.
            self._kda_zero_hist = ttnn.from_torch(
                torch.zeros(1, self.K - 1, self.qkv_dim_tp, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT if self._kda_tile_in else ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
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
        Lin = (K - 1) + T
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
        xin = ttnn.reshape(xin, (1, Lin, 1, C))
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # Needs l1_small_size on the device (prefill/demo set 24576); matches the validated A/B config.
        conv_cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        # Prepare conv weight once (warmup); avoids host reprocess + keeps traced replay device-only.
        if self._conv1d_wprep is None:
            self._conv1d_wprep = ttnn.prepare_conv_weights(
                weight_tensor=self.tw["conv_w1d"],
                input_memory_config=_dram,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=C,
                out_channels=C,
                batch_size=1,
                input_height=1,
                input_width=Lin,
                kernel_size=(1, K),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=False,
                groups=C,
                device=dev,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
            )
        out = ttnn.conv1d(
            input_tensor=xin,
            weight_tensor=self._conv1d_wprep,
            device=dev,
            in_channels=C,
            out_channels=C,
            batch_size=1,
            input_length=Lin,
            kernel_size=K,
            stride=1,
            padding=0,
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
        ttnn.deallocate(xin)
        out = ttnn.sharded_to_interleaved(out, _dram)
        out = ttnn.reshape(out, (1, T, C))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
        # SiLU stays separate (folding via conv_config.activation drops PCC to ~0.84 on this depthwise).
        return ttnn.silu(out, memory_config=_dram), new_state

    def _conv1d_prefill_kda(self, qkv, T, conv_state):
        """Fused depthwise causal conv1d + SiLU + q/k/v split (ttnn.experimental.kda.qkv_causal_conv1d_silu).

        qkv: TILE [1,T,C] (L1 or DRAM). conv_state: TILE [1,K-1,C] carry or None (zeros).
        Returns ((q,k,v) TILE DRAM flat [1,T,kd]/[1,T,kd]/[1,T,vd], new_state TILE [1,K-1,C] DRAM).
        One untilize of qkv + one tiny RM slice/tilize for the carry replace the conv2d/halo chain
        (untilize x3, concat, tilize x2, i2s, halo, conv2d, s2i, silu, 3 slices).

        QWEN36_KDA_TILE_IN=1 drops the full-chunk untilize entirely: the op's TILE kernels read whole
        tiles and shift rows on the math engine, so only the last row-tile is untilized (for the carry).
        Outputs are bit-identical -- the in-kernel shift is a matmul by exact 0/1 matrices, so the tap
        products and their bf16 partial accumulation are unchanged.
        """
        C, K = self.qkv_dim_tp, self.K
        _dram = ttnn.DRAM_MEMORY_CONFIG
        _tile = tpc.TILE_SIZE
        _scratch = []  # tensors this call allocated and must free before returning
        if self._kda_tile_in:
            # The op takes the TILE activation as-is: no full-chunk untilize, and history stays TILE.
            x_in = qkv  # caller-owned; deallocated by forward_prefill, not here
            # Carry for the next chunk still has to be TILE [1,K-1,C]. Only the last tile-row can contain it,
            # so untilize that one row-tile (1/Mt of the chunk) instead of the whole activation.
            tail = ttnn.slice(qkv, (0, T - _tile, 0), (1, T, C), memory_config=_dram)  # tile-aligned: tile copy
            tail_rm = ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
            ttnn.deallocate(tail)
            new_state = ttnn.slice(tail_rm, (0, _tile - (K - 1), 0), (1, _tile, C), memory_config=_dram)
            new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT, memory_config=_dram)
            ttnn.deallocate(tail_rm)
        else:
            _rm_mc = ttnn.L1_MEMORY_CONFIG if self._kda_rm_l1 else _dram
            x_in = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=_rm_mc)
            _scratch.append(x_in)
            # Carry for the next chunk: last K-1 real rows (cheap RM slice), tilized to match conv_carry.
            new_state = ttnn.slice(x_in, (0, T - (K - 1), 0), (1, T, C), memory_config=_dram)
            new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT, memory_config=_dram)
        if conv_state is None:
            if self._kda_zero_hist is None:  # normally allocated in reset_state; eager fallback
                self._kda_zero_hist = ttnn.from_torch(
                    torch.zeros(1, K - 1, C, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT if self._kda_tile_in else ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh,
                    memory_config=_dram,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
            hist = self._kda_zero_hist
        elif self._kda_tile_in:
            hist = conv_state  # already TILE [1,K-1,C]; the op reads its three carry rows in place
        else:
            hist = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
            _scratch.append(hist)
        taps = self.tw["conv_taps"]
        q, k, v = ttnn.experimental.kda.qkv_causal_conv1d_silu(
            x_in,
            hist,
            taps[0],
            taps[1],
            taps[2],
            taps[3],
            self.key_dim_tp,
            self.key_dim_tp,
            self.value_dim_tp,
            program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=self._kda_chunk),
            memory_config=_dram,
            # Op default accumulates the 4 taps in bf16 DEST; match the conv2d path (HiFi4, fp32 acc).
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )
        for t in _scratch:
            ttnn.deallocate(t)
        return (q, k, v), new_state

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
            # QWEN36_GDN_PROJ_CHUNKS: ask the AGMM for three output tensors (qkv | z | ab) instead of
            # one [1,S,4128] tensor plus three ttnn.slice ops (~30 us/layer/chunk). Needs the
            # tile-padded weight (load_gdn_weights_tp), so gate on the actual weight width, not the
            # env alone: a stale unpadded cache must fall back to slicing rather than TT_FATAL.
            _ab_w = self.tw["qkvz"].shape[-1] - az  # 32 with the padded weight, 24 without
            if self._proj_chunks and self._fuse_agmm and S > tpc.TILE_SIZE and _ab_w % tpc.TILE_SIZE == 0:
                # ONE memory config covers every chunk (compute_output_specs builds all chunk specs
                # from attributes.output_mem_config), so all three land where the widest/longest-lived
                # consumer needs them: z spans the chunk kernel, so DRAM (mode 2 forces L1 for A/B).
                _chunk_mc = tpc.proj_chunks_memcfg(self._proj_chunks)
                qkv, z, ab = tpc.all_gather_matmul_prefill(
                    x,
                    self.tw["qkvz"],
                    self.tt_ccl,
                    self.cfg,
                    self.args.ccl_topology(),
                    out_memory_config=_chunk_mc,
                    chunk_sizes=[qz, az - qz, _ab_w],
                )
                # [1,1,S,W] -> [1,S,W] is a metadata-only view (last dim unchanged).
                qkv = ttnn.reshape(qkv, (1, S, qz))
                z = ttnn.reshape(z, (1, S, az - qz))
                ab = ttnn.reshape(ab, (1, S, _ab_w))
                # a/b stay slices: the consumers reshape them to [B,T,Nv], so the 12-wide logical
                # width is load-bearing. Both are tiny (S x 12) and read from a tile-aligned block.
                a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
                b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv), memory_config=out_mc)
                ttnn.deallocate(ab)
                return qkv, z, a, b
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
            # a,b end mid-tile; slicing straight from qkvzab untilizes the full 4120-wide tensor.
            # Grab the enclosing tile-aligned block once (no untilize), then split a/b from it (test_gdn_slice_opt).
            _ab_end = min(az + -(-2 * Nv // tpc.TILE_SIZE) * tpc.TILE_SIZE, qkvzab.shape[-1])  # 2*Nv up to a tile
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
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv))
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def forward_prefill(
        self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False, gdn_masks=None
    ):
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

        # gdn_masks: (mask_f32, mask_q, conv_sel) PERSISTENT DEVICE tensors owned by the model's per-bucket
        # prefill trace (Qwen36Model._alloc_masked_bucket_bufs). They carry exactly the values the int-valid_len
        # path builds with ttnn.from_torch (a host write that TT_FATALs inside a captured trace). When given, the
        # masked branches are taken UNCONDITIONALLY (an all-ones mask is bit-identical to no mask), so the captured
        # op sequence depends only on the bucket. None (default) => every branch below is exactly as before.
        _mask_f32 = _mask_q = _conv_sel = None
        if gdn_masks is not None:
            _mask_f32, _mask_q, _conv_sel = gdn_masks
            # Trace preconditions: every remaining host-write fallback below must be unreachable.
            assert not return_state, "gdn_masks is the single-sequence traced path (no per-user collect)"
            assert carry and self.conv_carry is not None, "gdn_masks needs the persistent conv carry (_stable_state)"
            assert self._zero_conv0 is not None, "gdn_masks needs the persistent zero conv row (reset_state first)"
            assert tw["conv_taps"] is not None, "gdn_masks needs pre-built conv taps"

        # Prefill qkvzab in L1: keeps proj + q/k/v/z/a/b resident for conv+gate prep.
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=ttnn.L1_MEMORY_CONFIG)

        # FIR conv1d; conv_state = previous chunk's last K-1 inputs (None/zero from scratch)
        _cstate = self.conv_carry if carry else None
        _kda_qkv = None
        if self._gdn_conv_kda and valid_len is None and gdn_masks is None and self._gdn_flat_qkv:
            # Fused depthwise conv1d + SiLU + q/k/v split in ONE program (no conv2d/halo relayout chain).
            _kda_qkv, conv_new_state = self._conv1d_prefill_kda(qkv, T, _cstate)
            conv = None
        elif self._gdn_conv1d and valid_len is None and gdn_masks is None:
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
                conv_sel=_conv_sel,
            )
        ttnn.deallocate(qkv)

        # q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes it.
        kd = self.key_dim_tp
        if _kda_qkv is not None:
            # Fused conv op already emitted flat TILE q/k/v (DRAM).
            q, k, v = _kda_qkv
            _qkv_head_dims = (Nk, Dk, Nv, Dv)
        elif self._gdn_flat_qkv:
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
        if conv is not None:
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
        if gdn_masks is not None:
            # Device masks replace the int valid_len: same five multiplies, no host write.
            assert _use_fused, "gdn_masks requires the fused chunk adapter"
            _extra["valid_mask"] = (_mask_f32, _mask_q)
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
            valid_len=None if gdn_masks is not None else valid_len,
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
        if self._gdn_out_mode in ("agmm", "ag_mm"):
            # bf16 gated activation for the all-gather + column-parallel out-proj, produced directly by the
            # gate multiply (output dtype bf16). NOTE: a separate ttnn.typecast(fp32->bf16) here gave wrong
            # model output under the traced multi-layer path (exact standalone and in a single layer), so the
            # cast is folded into the multiply instead.
            gated = ttnn.multiply(
                out_f,
                ttnn.silu(z, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            gated = _silu_mul(out_f, z, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        # AGMM out-proj: all-gather the K-sharded gated activation and multiply by the column-sharded
        # out_proj -> output already hidden-sharded [1,1,T,dim/tp]; fp32 accumulation, no bf16 partial sums.
        if self._gdn_out_mode in ("agmm", "ag_mm"):
            # `gated` is bf16 here (see above): the all-gather+matmul kernels assume bf16 in0 (an fp32 in0
            # produced garbage) and bf16 halves the gather bytes.
            x_out = ttnn.reshape(gated, (1, 1, T, gated.shape[-1]))
            if self._gdn_out_mode == "agmm":
                out = tpc.all_gather_matmul_prefill(
                    x_out, tw["out_col"], self.tt_ccl, self.cfg, self.args.ccl_topology(), role="out"
                )
            else:
                out = tpc.all_gather_then_matmul_prefill(
                    x_out, tw["out_col"], self.tt_ccl, self.cfg, self.args.ccl_topology()
                )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        # Prefill: fused out-proj matmul + reduce-scatter (matmul_reduce_scatter_async), flag-gated.
        if self._fuse_out_mmrs_prefill:
            x_out = ttnn.reshape(gated, (1, 1, T, gated.shape[-1]))
            # fp32 output is load-bearing: o_proj is row-parallel, so the RS SUMS 4 per-device partials
            # across devices — bf16 there tanks PCC to ~0.69 even at ISL 2048 (test_oproj_dtype_isl). Keep fp32.
            out = tpc.matmul_reduce_scatter_prefill(
                x_out,
                tw["out"],
                self.tt_ccl,
                self.cfg,
                self.args.ccl_topology(),
                self.args.num_devices,
                ttnn.float32,
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

    def _kda_decode_constants(self):
        """0/1 expansion matrices + a fixed sigmoid gate (row 0 open, padding rows closed) for the KDA decode step."""
        if self._kda_dec_const is not None:
            return self._kda_dec_const
        Nk, Nv, Dk, Dv = self.Nk, self.Nv, self.Dk, self.Dv
        rf = Nv // Nk
        T = tpc.TILE_SIZE
        e_q = torch.zeros(Nk * Dk, Nv * Dk)  # GQA expand: value head j reads q/k head j // rf
        for j in range(Nv):
            src = j // rf
            e_q[src * Dk : (src + 1) * Dk, j * Dk : (j + 1) * Dk] = torch.eye(Dk)
        e_g = torch.zeros(Nv, Nv * Dk)  # per-head log decay -> per-key (KDA prep takes [1,T,H*K])
        for j in range(Nv):
            e_g[j, j * Dk : (j + 1) * Dk] = 1.0
        gate = torch.full((1, T, Nv * Dv), -30.0)
        gate[0, 0, :] = 30.0  # sigmoid -> 1 on the token row, 0 on the padding rows

        def dev(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        # sigmoid_gated_rms_norm wants the norm weight as a 1-D [V] tensor (tw["norm_w"] is stored 4-D).
        w_host = ttnn.to_torch(ttnn.get_device_tensors(self.tw["norm_w"])[0]).float().reshape(-1)[:Dv]
        self._kda_dec_const = {"e_q": dev(e_q), "e_g": dev(e_g), "gate": dev(gate), "norm_w": dev(w_host)}
        return self._kda_dec_const

    def _decode_kda_step(self, conv, z, a, b):
        """One decode step of the gated delta rule via the fused KDA chunk ops (see _decode_kda).
        conv: TILE [1,1,qkv_dim_tp] bf16 (post conv+silu, B=1). z/a/b: [1,1,value_dim]/[1,1,Nv]/[1,1,Nv].
        Returns gated = rmsnorm(o) * norm_w * silu(z) as TILE [1,1,value_dim] (L1), rec_state updated in place."""
        tw, Nv, Dk, Dv = self.tw, self.Nv, self.Dk, self.Dv
        T = tpc.TILE_SIZE
        C, kd, vd = self.qkv_dim_tp, self.key_dim_tp, self.value_dim_tp
        _L1 = ttnn.L1_MEMORY_CONFIG
        _dram = ttnn.DRAM_MEMORY_CONFIG
        const = self._kda_decode_constants()
        exact = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # 32-row chunk with the token in row 0; the pad rows are exact zeros.
        conv32 = ttnn.pad(conv, [(0, 0), (0, T - 1), (0, 0)], 0.0, memory_config=_L1)
        ttnn.deallocate(conv)
        q = ttnn.slice(conv32, (0, 0, 0), (1, T, kd), memory_config=_L1)
        k = ttnn.slice(conv32, (0, 0, kd), (1, T, 2 * kd), memory_config=_L1)
        v = ttnn.slice(conv32, (0, 0, 2 * kd), (1, T, C), memory_config=_L1)
        ttnn.deallocate(conv32)
        q_e = ttnn.matmul(q, const["e_q"], compute_kernel_config=exact, memory_config=_L1)
        k_e = ttnn.matmul(k, const["e_q"], compute_kernel_config=exact, memory_config=_L1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        # beta [Nv,1,32,1] fp32 and per-key g [1,32,Nv*Dk] bf16; both zero on rows 1..31.
        beta_row = ttnn.pad(ttnn.sigmoid(b, memory_config=_L1), [(0, 0), (0, T - 1), (0, 0)], 0.0, memory_config=_L1)
        ttnn.deallocate(b)
        beta_t = ttnn.transpose(beta_row, -2, -1, memory_config=_L1)  # [1, Nv, 32]
        ttnn.deallocate(beta_row)
        beta_col = ttnn.reshape(beta_t, (Nv, 1, T, 1), memory_config=_L1)
        ttnn.deallocate(beta_t)
        beta_col = ttnn.typecast(beta_col, ttnn.float32, memory_config=_L1)
        g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a)
        if g.dtype != ttnn.bfloat16:
            g = ttnn.typecast(g, ttnn.bfloat16, memory_config=_L1)
        g_row = ttnn.pad(g, [(0, 0), (0, T - 1), (0, 0)], 0.0, memory_config=_L1)
        ttnn.deallocate(g)
        g_flat = ttnn.matmul(g_row, const["e_g"], compute_kernel_config=exact, memory_config=_L1)
        ttnn.deallocate(g_row)
        prep = ttnn.experimental.kda.prepare_chunk_recurrence(q_e, k_e, v, g_flat, beta_col, Nv, memory_config=_dram)
        for t in (q_e, k_e, v, g_flat, beta_col):
            ttnn.deallocate(t)
        state3 = ttnn.reshape(self.rec_state, (Nv, Dk, Dv))
        y, s_new = ttnn.experimental.kda.recurrent_chunk_scan(*prep, state3, memory_config=_dram)
        for t in prep:
            ttnn.deallocate(t)
        ttnn.copy(ttnn.reshape(s_new, (1, Nv, Dk, Dv)), self.rec_state)
        ttnn.deallocate(s_new)
        y3 = ttnn.reshape(y, (Nv, T, Dv))
        out = ttnn.experimental.kda.sigmoid_gated_rms_norm(
            y3, const["gate"], const["norm_w"], Nv, epsilon=1e-6, memory_config=_L1, output_dtype=ttnn.bfloat16
        )  # [1, 32, Nv*Dv]; padding rows gated to zero
        ttnn.deallocate(y)
        out_row = ttnn.slice(out, (0, 0, 0), (1, 1, vd), memory_config=_L1)
        ttnn.deallocate(out)
        gated = _silu_mul(out_row, z, _L1)
        ttnn.deallocate(out_row)
        ttnn.deallocate(z)
        return gated

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
            qkv_p = ttnn.pad(qkv, [(0, 0), (0, Bmax - B), (0, 0)], value=0.0, memory_config=_L1)
            ttnn.deallocate(qkv)
            qkv = qkv_p
        for j in range(self.K - 1):
            ttnn.copy(st[j + 1], st[j])
        ttnn.copy(qkv, st[self.K - 1])
        ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)

        kd = self.key_dim_tp
        if self._decode_kda and B == 1:
            gated = self._decode_kda_step(conv, z, a, b)
            partial = self._row_proj(gated, tw["out"])
            ttnn.deallocate(gated)
            partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
            return tt_all_reduce(
                partial,
                self.mesh,
                self.tt_ccl,
                cluster_axis=0,
                dim=3,
                topology=self.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
        k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
        v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
        ttnn.deallocate(conv)

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
        if self._stable_state:
            # In-place update preserves rec_state address for decode trace replay
            if B == Bmax:
                ttnn.copy(new_rec, self.rec_state)
                ttnn.deallocate(new_rec)
            else:
                self._write_recurrent_state_prefix(new_rec, B)
        else:
            self.rec_state = new_rec

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
