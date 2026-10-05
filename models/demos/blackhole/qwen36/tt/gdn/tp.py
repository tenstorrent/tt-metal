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
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_reduce


def gdn_fused_decode_enabled():
    """QWEN36_GDN_FUSED_DECODE (default "1"): single-user (max batch 1) GDN decode through the fused kernels
    (qkv_causal_conv1d_silu + chunk_gated_delta_rule + sigmoid_gated_rms_norm). "0" keeps the original
    shift-register / recurrent_gated_delta_rule_decode_ttnn path byte-for-byte."""
    return os.environ.get("QWEN36_GDN_FUSED_DECODE", "1") != "0"


_fused_decode_logged = False


def _log_fused_decode_once(msg):
    """One info line per process (the layer is built 48x)."""
    global _fused_decode_logged
    if not _fused_decode_logged:
        _fused_decode_logged = True
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


def kda_conv_prefill(qkv, T, history, taps, widths, actual_start, xin_memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Depthwise causal conv (K=4) + SiLU + q/k/v split in ONE program (ttnn.experimental.kda.qkv_causal_conv1d_silu).

    qkv:          [1, T, C] bf16 TILE, the projection's conv columns (q | k | v).
    history:      [1, 3, C] bf16, the three rows preceding this chunk (zeros from scratch). TILE or ROW_MAJOR;
                  the op reads ROW_MAJOR, so a TILE carry is converted here (three rows).
    taps:         four [1, 1, C] bf16 TILE tensors in kernel-position order, tap j multiplying row t-3+j
                  (tw["conv_taps"], exactly the op's tap0..tap3 contract).
    widths:       (q_width, k_width, v_width), tile-aligned, summing to C.
    actual_start: uint32 [1] device tensor holding 0, allocated once before any trace capture.
    Returns q [1,T,Q], k [1,T,K], v [1,T,V] bf16 TILE DRAM, and new_state [1, 3, C] bf16 TILE DRAM: the chunk's
    last three INPUT rows, i.e. the next chunk's history (the op does not emit it).
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
    # Next chunk's history: the last three input rows, taken from the row-major copy (page reads, no
    # untilize), then re-tiled because the layer-wide carry (conv_carry, decode seeding, per-user assembly)
    # is TILE.
    new_state = ttnn.slice(xin, (0, T - n_hist, 0), (1, T, C), memory_config=_dram)
    ttnn.deallocate(xin)
    new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT, memory_config=_dram)
    return q, k, v, new_state


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
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
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
        # In-place state updates for decode/prefill traces (set by model allocate_kv_caches)
        self._stable_state = False
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
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill
        # Fused single-user decode (QWEN36_GDN_FUSED_DECODE, default on). Decided here, once, so the state format
        # is fixed for the model's lifetime: only a max_batch_size == 1 model takes it (forward_decode re-checks
        # self.B == 1 and the active width == 1); every other configuration runs the original decode path.
        #   _conv_hist_rm: the fused decode's conv history, [1, K-1, qkv_dim_tp] ROW_MAJOR bf16, the three raw
        #                  conv inputs preceding the next token, oldest first (the layout/order the kda conv op
        #                  reads). Allocated once (reset_state / first write) and only ever updated IN PLACE
        #                  (ttnn.copy), so the decode and prefill traces keep a fixed address. Decode maintains
        #                  it INSTEAD of conv_states; every code path that writes conv_states for decode
        #                  (prefill capture_state, assemble_batched_state, write_slot, forward_prefill_batched)
        #                  also refreshes it. It is deliberately NOT part of the model's per-binding state swaps
        #                  (model.py rebinds rec_state/conv_states/conv_carry on prefill scratch): with
        #                  max_batch_size == 1 there is one user, so the latest prefill IS the decode state.
        self._conv_hist_rm = None
        self._fused_decode = self._fused_decode_supported(args, tw)

    def _fused_decode_supported(self, args, tw):
        """Whether this layer takes the fused single-user decode path (QWEN36_GDN_FUSED_DECODE). Logs one info line
        per process stating the active path."""
        reasons = []
        if not gdn_fused_decode_enabled():
            reasons.append("QWEN36_GDN_FUSED_DECODE=0")
        else:
            if args.max_batch_size != 1:
                reasons.append(f"max_batch_size={args.max_batch_size} (fused decode is single-user)")
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
        _log_fused_decode_once(
            "[GDN] decode path: FUSED for max_batch_size == 1 (qkv_causal_conv1d_silu + chunk_gated_delta_rule + "
            "sigmoid_gated_rms_norm; QWEN36_GDN_FUSED_DECODE=0 reverts). conv_states is not maintained during decode; "
            "out-proj input is bf16."
        )
        return True

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
        if self._fused_decode:
            # Fused-decode conv history: allocated once, zeroed IN PLACE on every later reset (it may already be
            # baked into a decode/prefill trace, and reset_state also runs for the B=1 prefill scratch).
            _had_hist = self._conv_hist_rm is not None
            self._ensure_conv_hist()
            if _had_hist:
                ttnn.copy(self._kda_zero_history, self._conv_hist_rm)
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
        if self._fused_decode and self._conv_hist_rm is not None:
            ttnn.copy(self._kda_zero_history, self._conv_hist_rm)  # preallocated RM zeros (no allocation here)

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

    def _kda_conv_prefill(self, qkv, T, conv_state):
        """kda_conv_prefill on this layer's taps and widths. conv_state: the previous chunk's carry
        [1, K-1, C] TILE, or None from scratch. Returns (q, k, v, new_state), see kda_conv_prefill."""
        self._ensure_kda_consts()
        history = conv_state if conv_state is not None else self._kda_zero_history
        kd, vd = self.key_dim_tp, self.value_dim_tp
        return kda_conv_prefill(qkv, T, history, self.tw["conv_taps"], (kd, kd, vd), self._kda_actual_start)

    # ------------------------------------------------------------------ #
    # Fused single-user decode: conv history (_conv_hist_rm) + its handoff from prefill.
    # ------------------------------------------------------------------ #
    def _ensure_conv_hist(self):
        """The persistent [1, K-1, qkv_dim_tp] ROW_MAJOR bf16 conv history, allocated (zeros) on first use. Host
        write: reset_state calls it before any trace capture; the lazy callers run eagerly."""
        self._ensure_kda_consts()
        if self._conv_hist_rm is None:
            self._conv_hist_rm = ttnn.from_torch(
                torch.zeros(1, self.K - 1, self.qkv_dim_tp, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._conv_hist_rm

    def _hist_hook_active(self):
        """Writers of the decode conv state refresh the fused history only on a single-user binding."""
        return self._fused_decode and self.B == 1

    def _write_conv_hist(self, last_inputs):
        """Handoff prefill -> fused decode: copy the last K-1 conv inputs [1, K-1, C] TILE (oldest first, the
        same tensor that seeds conv_states[1..K-1]) into the RM history, in place. Device-only (trace-safe once
        the history exists)."""
        hist = self._ensure_conv_hist()
        rm = ttnn.to_layout(last_inputs, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(rm, hist)
        ttnn.deallocate(rm)

    def _sync_conv_hist_from_states(self):
        """Rebuild the RM history from conv_states[1..K-1] (each [1, 1, C] TILE; B == 1), for the writers that
        produce conv_states but not a stacked [1, K-1, C] tensor (write_slot, batched prefill)."""
        hist = self._ensure_conv_hist()
        rows = [
            ttnn.to_layout(self.conv_states[m], ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for m in range(1, self.K)
        ]
        stacked = ttnn.concat(rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, K-1, C]
        for r in rows:
            ttnn.deallocate(r)
        ttnn.copy(stacked, hist)
        ttnn.deallocate(stacked)

    def snapshot_fused_decode_state(self):
        """Host copy of the fused-decode conv history (None when the fused path is off). conv_states is NOT the
        decode conv state when the fused path is on, so a caller that snapshots/restores GDN state around a
        throwaway decode run (demo trace capture) must also snapshot/restore this: pair with
        restore_fused_decode_state. Mirrors the demos' rec_state/conv_states host snapshot."""
        if not self._fused_decode or self._conv_hist_rm is None:
            return None
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

        # Causal conv + SiLU; conv_state = previous chunk's last K-1 inputs (None/zero from scratch).
        # q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes it.
        _cstate = self.conv_carry if carry else None
        kd = self.key_dim_tp
        if self._gdn_kda_conv and valid_len is None:
            # KDA op: conv + SiLU + q/k/v split in one program; its outputs are already the three
            # token-major tensors the flat-qkv path wants (masked buckets keep the MAC FIR below).
            q, k, v, conv_new_state = self._kda_conv_prefill(qkv, T, _cstate)
            ttnn.deallocate(qkv)
            if self._gdn_flat_qkv:
                _qkv_head_dims = (Nk, Dk, Nv, Dv)
            else:
                q = ttnn.reshape(q, (1, T, Nk, Dk))
                k = ttnn.reshape(k, (1, T, Nk, Dk))
                v = ttnn.reshape(v, (1, T, Nv, Dv))
                _qkv_head_dims = None
        else:
            # The MAC FIR: masked buckets (their one-hot new_state selection) and QWEN_GDN_CONV=fir.
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
        # const_tiles / program_config only apply to the fused op; the seq adapter has neither param.
        _extra = (
            {"const_tiles": self._fused_const_tiles, "program_config": self.gdn_program_config} if _use_fused else {}
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
                if self._hist_hook_active():
                    # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
                    self._write_conv_hist(conv_new_state)
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
        if self._hist_hook_active() and len(conv_new_list) == 1:
            # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
            self._write_conv_hist(conv_new_list[0])
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
        if self._hist_hook_active():
            # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
            self._sync_conv_hist_from_states()

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
        _extra = (
            {"const_tiles": self._fused_const_tiles, "program_config": self.gdn_program_config} if _use_fused else {}
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
        if B == 1 and self._hist_hook_active():
            # Fused single-user decode reads its conv history from the RM buffer, not conv_states.
            self._sync_conv_hist_from_states()

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
        if self._fused_decode and Bmax == 1 and B == 1:
            # QWEN36_GDN_FUSED_DECODE (default on), single-user: fused conv / chunk-delta-rule / gated-norm kernels.
            return self._forward_decode_fused(x, decode_ar)

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
        if self._stable_state:
            ttnn.copy(new_rec, self.rec_state)
            ttnn.deallocate(new_rec)
        else:
            self.rec_state = new_rec

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
