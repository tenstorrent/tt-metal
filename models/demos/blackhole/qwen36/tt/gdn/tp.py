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
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        self._fuse_out_mmrs_prefill = not self._out_sharded and args.num_devices > 1
        # Pre-build chunk masks once (trace-safe; avoids from_torch inside captured trace)
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        # Prefill fused-op constant tiles, owned by this layer (avoids process-lifetime C++ cache vs device lifetime).
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

        self._fused_const_tiles = build_fused_const_tiles(mesh, _FUSED_CHUNK_SIZE)
        self.conv_states = None
        self.rec_state = None
        # Spec-decode hybrid verify slot capture (set by SpeculativeDecoder): per-token recurrent-state
        # snapshots buffered during a captured verify so commit is a slot-select, not a re-run.
        self._capture_slots = False
        self._verify_slots = None
        self._verify_states = None  # per-token rec states from the last verify (token-major)
        self._verify_states_buf = None  # same tensor; kept so traced replays can re-arm the handle
        # Pre-allocated persistent slot buffers (rec_state + conv_states shaped). Verify copies state
        # INTO these fixed addresses (trace-safe + no per-call alloc) instead of fresh ttnn.clone.
        self._slot_bufs = None
        # Fully-batched hybrid verify (_verify_fullbatch): no per-token loop at all. This is the
        # production spec-decode verify — 2.3 ms/candidate vs ~15 ms for the per-token path — and is
        # validated lossless + deterministic (test_spec_lossless.py, test_spec_determinism.py). The
        # per-token loop below survives as this flag's False branch: an A/B reference for numerics
        # work, flipped from python (no env), never from the demo.
        self.use_fullbatch_verify = True
        # Batched-conv verify only: the [1, K-1+T, qkv_dim_tp] conv window stashed with ONE copy, from
        # which commit_verify_slot slices the accepted slot's shift-register. None => per-token slots.
        self._verify_win_buf = None
        # Batched-conv verify only: the DURABLE shift register as one [1, K, qkv_dim_tp] tensor,
        # mirroring conv_states[0..K-1]. The traced verify reads its carry from a constant-offset
        # slice of this, and commit_verify_slot writes it with ONE slice+copy instead of touching K
        # separate conv_states taps — see commit_verify_slot's COST NOTE.
        self._conv_win_buf = None
        # Which of the two mirrors is authoritative. The traced fullbatch verify advances ONLY
        # _conv_win_buf (refilling the K conv_states taps inside the trace cost ~10 ms/iteration over
        # 48 layers and nothing in the spec loop reads them), so after a verify the taps are BEHIND:
        # _conv_taps_stale. Conversely everything that writes the taps from outside (prefill
        # capture_state, reset, slot edits, decode's own shift register) leaves the window behind:
        # _conv_win_stale. Exactly one can be set at a time — each setter clears the other. The
        # rebuilds are sync_conv_taps() (window -> taps, at every tap CONSUMER) and sync_conv_win()
        # (taps -> window, lazily in _ensure_conv_win, which only ever runs eagerly).
        self._conv_taps_stale = False
        self._conv_win_stale = False
        self._win_captured = False  # did THIS verify populate the window? (buffer is always allocated)
        self._conv_taps_T = None  # conv taps expanded to T rows for the batched-conv path
        # In-place state updates for decode/prefill traces (set by model allocate_kv_caches)
        self._stable_state = False
        # Spec decode only (set by SpeculativeDecoder): run the ONE fused recurrent device op in
        # forward_decode instead of the composite, so decode and hybrid verify share GDN math.
        # Full-batch (B == self.B) only — it has no bucketed B<Bmax state slice/writeback.
        self.use_fused_recurrent_decode = False
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
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        self._zero_conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_rec = z((self.B, self.Nv, self.Dk, self.Dv))
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
        # new_state: last K-1 real input tokens (for the next chunk's carry), TILE/DRAM.
        # A chunk SHORTER than the register (T < K-1) has no K-1 rows of its own -- the spec-decode
        # seed/commit calls this with T=1. Take the tail of [conv_state ; qkv] instead, which IS the
        # register after T shifts; slicing qkv alone would ask for a negative start and TT_FATAL.
        if T >= K - 1:
            new_state = ttnn.slice(qkv, (0, T - (K - 1), 0), (1, T, C))
        else:
            _w = ttnn.concat([conv_state, qkv], dim=1, memory_config=_dram) if conv_state is not None else qkv
            _n = _w.shape[1]
            new_state = ttnn.slice(_w, (0, max(0, _n - (K - 1)), 0), (1, _n, C))
            if _w is not qkv:
                ttnn.deallocate(_w)
        new_state = ttnn.to_memory_config(ttnn.to_layout(new_state, ttnn.TILE_LAYOUT), _dram)
        if conv_state is None:
            pad = ttnn.zeros(
                [1, K - 1, C], device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_dram
            )
            xin = ttnn.concat([pad, qkv], dim=1, memory_config=_dram)
            ttnn.deallocate(pad)
        else:
            xin = ttnn.concat([conv_state, qkv], dim=1, memory_config=_dram)
        return self._conv1d_window(xin, T), new_state

    def _conv1d_verify(self, win, T):
        """``_conv1d_prefill`` for the fullbatch verify: the [carry ; tokens] window is already built
        (it is the same tensor the state stash needs) and the new_state return is dead there.

        Byte-identical conv output — same op, same weights, same input — for two ops/layer less: the
        duplicate window concat (~0.11 ms/layer) and the discarded new_state's tile-unaligned row
        slice + relayout. Both were pure waste in the verify, ~7 ms/iteration over 48 GDN layers."""
        return self._conv1d_window(win, T)

    def _conv1d_window(self, xin, T):
        """The ttnn.conv1d call itself, over an already-concatenated [1, K-1+T, C] TILE window."""
        dev, K, C = self.mesh, self.K, self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        Lin = (K - 1) + T
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
        return ttnn.silu(out, memory_config=_dram)

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
                # Prefill wrote the taps directly, so they are the truth and the [1,K,C] window
                # mirror the fullbatch verify reads its carry from is now behind (re-seeded lazily
                # by _ensure_conv_win, on the next EAGER verify — the spec loop's seed).
                self._conv_taps_stale = False
                self._conv_win_stale = True
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
        self._conv_taps_stale = False  # taps written from outside: the window mirror is now behind
        self._conv_win_stale = True

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

        # Conv1d shift-register + weighted sum + SiLU.
        # The K taps below ARE the shift register, so they must be current: a preceding fullbatch
        # verify advanced only the [1,K,C] window mirror. No-op (zero device ops, so trace-safe)
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

    def forward_verify_recurrent(self, x, valid_len, pre_gathered=False):
        """Hybrid spec-decode verify for GDN: advance the recurrent state token-by-token over the
        first ``valid_len`` rows of the bucket. This is BIT-EXACT to ``valid_len`` sequential
        ``forward_decode`` steps (identical conv shift-register + recurrent kernel + in-place state
        updates), so verify uses the SAME kernel as decode instead of the lossy chunk kernel.

        Rows past ``valid_len`` are zero and are never read downstream: full attention is causal
        (real queries < valid_len never attend to padded keys) and verify only row-selects rows
        < valid_len. The rest of the layer stack (attn/MLP/norm/lm_head) still runs batched over the
        bucket, so only the GDN recurrence is sequential — that is the whole point of the hybrid.

        x : [1, 1, bucket, dim] prefill-normed input (same shape forward_prefill receives).
        Returns [1, 1, bucket, dim] full-dim (matches forward_prefill's output for the layer add).

        pre_gathered : the caller already handed us a FULL-dim activation (decode-config verify runs
        the layer norms in Mode.DECODE, which gathers pre-norm), so skip the internal all-gather.
        """
        assert valid_len <= tpc.TILE_SIZE, f"verify bucket {valid_len} exceeds one tile"
        return self._forward_verify_recurrent_batched(x, valid_len, pre_gathered=pre_gathered)

    def _forward_verify_recurrent_batched(self, x, valid_len, pre_gathered=False):
        """Batched hybrid verify — BIT-IDENTICAL to the per-token forward_decode loop, but with
        valid_len x fewer matmul/all-reduce launches. Key fact: the decode matmul (matmul_1d_decode)
        is row-independent and processes a full 32-row M-tile regardless of how many rows are real, so
        packing all `valid_len` (<= TILE) tokens into ONE decode matmul gives per-row-identical results
        while collapsing valid_len separate launches into one. Structure (cf. the reference's
        fused_sigmoid_gating_delta_rule_update: project once, loop the recurrence, output once):

          1. Gather the valid_len rows to full dim, then ONE decode qkvzab matmul (same kernel + weights
             forward_decode uses per token -> per-row bit-identical q/k/v/z/a/b).
          2. Per-token loop over valid_len: conv shift-register + fp32 recurrence step (the ONLY
             sequential part; carries rec_state/conv_states, so slot capture is unchanged).
          3. ONE gated-norm + out-proj (decode kernel) + all-reduce over the valid_len rows.

        Because every matmul is the decode kernel and row-independent, this is numerically identical to
        the per-token loop (verified: same accept rate + same trajectory), NOT an approximation. The
        AGMM prefill projection was avoided precisely because it rounds differently and drifts the state.
        """
        tw, B, Nk, Nv, Dk, Dv = self.tw, self.B, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        if self.conv_states is None:
            self.reset_state()
        # Decode-config verify hands us the DECODE attn-norm output, which is L1 WIDTH-SHARDED.
        # We slice valid rows out of it below, so interleave first (row-slicing a width-shard is not
        # supported). We then own that copy and must free it.
        _x_owned = False
        if pre_gathered and x.is_sharded():
            x = ttnn.to_memory_config(x, mc)
            _x_owned = True
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        bucket = x.shape[-2]  # x is K-sharded [1, bucket, dim/tp] (full-dim when pre_gathered)
        T = valid_len
        kd = self.key_dim_tp

        # 1) Gather only the valid_len rows to full dim (like the per-token loop's gather, but over
        #    valid_len rows not the whole bucket), then ONE decode qkvzab matmul. S=valid_len <= TILE
        #    routes _project_qkvzab through matmul_1d_decode — the exact per-token decode projection.
        x_valid = x if T == bucket else ttnn.slice(x, (0, 0, 0), (1, T, x.shape[-1]))
        if pre_gathered:
            # Decode-config verify: the layer already ran its norm in Mode.DECODE, which gathers
            # PRE-norm, so x is already full-dim [1, bucket, dim]. Gathering again would quadruple
            # the feature dim. Only free xg below if we own it (i.e. it is the slice, not the caller's x).
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
        # The recurrence is ALWAYS the fused device op now: the T sequential dispatches (~13 ops
        # each) collapse into one. The conv shift-register (a cheap FIR) stays sequential — it
        # produces per-token q/k/v that we collect, then run the whole recurrence in a single call
        # that also emits the state AFTER every token (output_per_token_state) for slot acceptance.
        # Measured: verify marginal 18.0 -> 15.2 ms/candidate (the recurrence is only ~15% of it).
        # self.use_fullbatch_verify (default True): eliminate the per-token loop ENTIRELY (marginal
        # -> 2.3 ms). The ~18 ms/candidate is ~50 device ops per token per GDN layer at a few us each
        # — launch-bound with no hot spot — so the only fix is to stop launching them. Three pieces
        # make that possible without any sliding-window slicing:
        #   conv    -> the NATIVE depthwise ttnn.conv1d (_conv1d_prefill), the same op prefill uses,
        #              which takes all T rows at once and is trace-safe (weights prepared at warmup);
        #   q/k/v   -> sliced out of the batched conv output on the FEATURE dim (contiguous columns,
        #              not tiled rows) and repeat_interleaved once for all T;
        #   beta/g  -> a_all/b_all are already [1,T,Nv], so the gating is one sigmoid / softplus pass;
        #   recur.  -> the C++ fused_recurrent_gated_delta_rule kernel consumes [B,T,Nv,D] for all T
        #              tokens in ONE dispatch and emits per-token state for slot acceptance.
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

            # Defer the recurrence: collect this token's inputs. conv_states slot capture stays
            # per-token here (the shift-register is inherently sequential); rec_state slots come
            # from the fused call's per-token output below.
            q_seq.append(q)
            k_seq.append(k)
            v_seq.append(v)
            beta_seq.append(beta)
            g_seq.append(g)
            if capture:
                _, conv_bufs = self._slot_bufs[t]
                for j, c in enumerate(self.conv_states):
                    ttnn.copy(c, conv_bufs[j])

        # ONE recurrence over all T tokens. q/k/v -> [B,T,Nv,D]; beta/g -> [B,T,Nv]. The wrapper
        # applies the L2-norm + query scale + exp(g) internally (same contract as the per-token op).
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
        # Advance durable rec_state to the last token's state; keep the per-token states for slot
        # acceptance. The kernel writes them token-major, so slot t is a contiguous row block of
        # `states` and NOTHING has to be copied here: hold the tensor and let commit_verify_slot
        # slice the one accepted slot. The old code ran T x (slice + reshape + copy + dealloc) per
        # layer — ~770 device ops per verify across 48 GDN layers, for state that is thrown away
        # for every slot except the accepted one.
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
        # Per-token gated-norm to build out_f_rows (identical to the sequential tail).
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

        # 3) Output tail: one gated SiLU + one out-proj + one all-reduce over the valid rows.
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
        # z_all is already [1, T, value_dim_tp] (projected over exactly the valid rows).
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
        # Pad the valid rows out to the full bucket (rows >= valid_len are never read downstream).
        if T < bucket:
            o_rm = ttnn.to_layout(o_red, rm)
            ttnn.deallocate(o_red)
            # Trace-safe pad: ttnn.zeros is a host write that TT_FATALs inside a captured trace, so use
            # a PERSISTENT zero buffer (allocated once, fixed address) when verify is being traced. The
            # values are identical to ttnn.zeros; only the allocation site differs.
            pad = self._verify_pad_buf(bucket - T, o_rm.shape[-1], o_rm.dtype, rm, mc)
            o_full = ttnn.concat([o_rm, pad], dim=2, memory_config=mc)
            ttnn.deallocate(o_rm)
            o_red = ttnn.to_memory_config(ttnn.to_layout(o_full, ttnn.TILE_LAYOUT), mc)
            ttnn.deallocate(o_full)
        return o_red

    def _verify_fullbatch(self, qkv_all, z_all, a_all, b_all, T, bucket, capture):
        """Fully-batched hybrid verify: NO per-token loop. See the use_fullbatch_verify note above.

        Inputs are the already-projected [1,T,*] tensors. Returns the same padded [1,1,bucket,dim]
        the per-token path returns, and advances conv_states / rec_state identically.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        kd, C, rf = self.key_dim_tp, self.qkv_dim_tp, Nv // Nk
        self._ensure_conv_win()  # no-op after the first (eager) call; never allocates inside a trace

        # 1) Causal conv over all T tokens. The carry is the shift register's previous K-1 inputs,
        #    i.e. conv_states[1:] (conv_states[K-1] is the most recent input). Constant-offset slice
        #    of the persistent shift register — trace-safe (fixed address, fixed offsets) and what
        #    lets commit write ONE buffer.
        carry = ttnn.slice(self._conv_win_buf, (0, 1, 0), (1, self.K, C))
        # ONE window E = [carry(K-1) ; tokens(T)], shared by the conv and the state stash in step 5.
        # _conv1d_prefill built this very concat a second time as its own input, and a concat costs
        # ~0.11 ms/layer — ~5 ms/iteration over the 48 GDN layers for nothing.
        E = ttnn.concat([carry, qkv_all], dim=1, memory_config=mc)  # [1, K-1+T, C]
        ttnn.deallocate(carry)
        conv_all = self._conv1d_verify(E, T)  # [1,T,C], SiLU applied

        # 2) q/k/v for all T: FEATURE-dim slices (contiguous columns) + one repeat_interleave each.
        q_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
        k_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
        v_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 2 * kd), (1, T, C)), (1, T, Nv, Dv))
        ttnn.deallocate(conv_all)
        if rf != 1:
            q_all = ttnn.repeat_interleave(q_all, rf, dim=2)
            k_all = ttnn.repeat_interleave(k_all, rf, dim=2)

        # 3) Gating for all T at once (a_all/b_all are already [1,T,Nv]).
        beta_all = ttnn.sigmoid(b_all, memory_config=_L1)
        g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_all, tw["dt_bias"]), memory_config=_L1)

        # 4) ONE recurrence dispatch over all T tokens, emitting per-token state when capturing.
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

        # 5) State bookkeeping. rec_state slots come from the kernel's per-token output; conv slots
        #    are rows [t, t+K) of the window E built in step 1, stashed with ONE copy.
        if capture:
            ttnn.copy(E, self._verify_win_buf)
            self._win_captured = True
            # Token-major kernel output: keep it and let commit_verify_slot slice the accepted slot.
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
        # Durable shift register = the window's last K rows (what T sequential shifts would leave),
        # written to the persistent [1,K,C] buffer the next replay's carry reads. ONE slice + copy.
        #
        # The K conv_states taps are deliberately NOT refilled here. Nothing in the verify loop reads
        # them, and doing it inside the trace cost K x (slice + copy) per layer x 48 layers ~= 10 ms
        # of the iteration. They are rebuilt from this window on demand instead (sync_conv_taps), at
        # every consumer: the snapshot/restore round-trip, forward_decode's shift register, and the
        # end of a spec generate. `_conv_taps_stale` below is what arms that — and note it is a HOST
        # side effect, so it only fires on the eager passes; the per-REPLAY marking lives in
        # model.verify_traced (python does not re-run during execute_trace).
        tail = ttnn.slice(E, (0, T - 1, 0), (1, T - 1 + self.K, C))
        ttnn.copy(tail, self._conv_win_buf)
        if T > 1:
            # At T == 1 (the one-token eager seed) E is exactly K rows, so that slice is FULL-SPAN
            # and ttnn.slice hands back an ALIAS of E — deallocating it would double-free with E.
            ttnn.deallocate(tail)
        ttnn.deallocate(E)
        self._conv_taps_stale, self._conv_win_stale = True, False

        # 6) Batched output tail: rms_norm normalises over the last dim, so one call covers all T.
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
        o_red = tt_all_reduce(
            partial, self.mesh, self.tt_ccl, cluster_axis=0, dim=3, topology=self.args.ccl_topology(), memory_config=mc
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

    def _ensure_verify_slot_bufs(self, n):
        """Lazily allocate n persistent per-token slot buffers (rec_state + conv_states shaped). Verify
        copies state INTO these fixed addresses so a captured trace replays without allocating (clones
        would mint new addresses each call). Reallocated only if fewer than n slots exist."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        # Conv window buffer, ONLY for the fully-batched conv path: [1, K-1+n, qkv_dim_tp], matching
        # the concat that builds E. Allocated HERE (before the trace warmup) so the warmup and
        # captured passes take the identical ttnn.copy branch — a lazy allocate-then-copy makes
        # capture hit an uncompiled program. Gated so the per-token A/B path allocates nothing.
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
        """Allocate the persistent [1, K, qkv_dim_tp] shift register once, seeded from conv_states.

        Allocate-and-seed happens on the FIRST eager fullbatch verify (the spec loop's seed forward),
        never lazily inside a captured trace: by capture time the buffer exists and the trace body
        only ever slices/copies it, which the warmup pass has already compiled.

        Re-seeds an EXISTING buffer whose taps moved underneath it (`_conv_win_stale`: a new prompt's
        prefill, a reset, a plain decode step). That re-seed likewise only ever happens on an eager
        call — by capture time the flag is clear (the pre-capture _restore_gdn_verify syncs), so the
        trace body records no copy and a replay can never clobber the window with stale taps."""
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
        """Rebuild conv_states[0..K-1] from the persistent window — the inverse of sync_conv_win.

        The traced fullbatch verify advances ONLY _conv_win_buf (see _verify_fullbatch), so the K tap
        buffers go stale for as long as nothing reads them. Every tap CONSUMER calls this first;
        it is a no-op — zero device ops, so calling it from a traced body is safe — unless a
        fullbatch verify actually ran since the taps were last written.

        K x (slice + copy) per layer, eager, and only on the handful of iterations that read taps
        (snapshot, an eager decode step, the end of a generate) instead of every verify replay."""
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
        """Mirror conv_states[0..K-1] into the persistent [1,K,qkv_dim_tp] shift-register buffer.

        conv_states stay the source of truth outside the spec loop (prefill fills them, snapshot /
        restore round-trips them, the decode path reads them); this buffer is the copy the traced
        fullbatch verify reads its carry from, so it has to be re-seeded whenever conv_states are
        set from outside — at slot-buffer setup and after every _restore_gdn_verify."""
        if self._conv_win_buf is None or self.conv_states is None:
            return
        if self._conv_taps_stale:
            # The WINDOW is the truth here (a fullbatch verify advanced it and the taps were left
            # behind), so copying the taps over it would undo the verify. Bring the taps forward
            # instead; both mirrors then agree and there is nothing left to copy.
            self.sync_conv_taps()
            self._conv_win_stale = False
            return
        c = ttnn.concat(list(self.conv_states), dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.copy(c, self._conv_win_buf)
        ttnn.deallocate(c)
        self._conv_win_stale = False

    def commit_verify_slot(self, idx):
        """Set the recurrent state to the buffered verify slot `idx` (state after consuming the
        accepted-prefix's last token). Copies in place (preserves buffer addresses), so no commit
        forward runs. Slot buffers are persistent (reused next verify), so they are NOT freed here.

        The recurrent state comes straight out of the kernel's token-major per-token output, so only
        the ONE accepted slot is ever touched — the verify no longer copies all T states aside.

        COST NOTE. This is the one EAGER step of the spec iteration that scales with layer count: 48
        GDN layers x a handful of device ops, at this stack's ~35 us eager op cost. The ops move a
        few KB each — it is dispatch count, not data — so the only thing that matters here is how
        many ttnn calls the loop makes. On the fully-batched verify it used to make ten per layer
        (rec slice+copy, then K conv taps x slice+copy+deallocate) and cost ~17 ms, ~3x its per-token
        self, whose trace had already materialised the per-token conv slots. Two changes took it to
        ~5 ms: the full-acceptance early-out below, and committing the conv shift register as ONE
        [1,K,qkv_dim_tp] buffer (_conv_win_buf) that the traced verify reads its carry from, instead
        of K separate conv_states taps."""
        assert self._verify_states is not None, "commit_verify_slot called without a captured verify"
        # A verify (traced or eager) just advanced the window; the taps are behind either way, so
        # arm the rebuild BEFORE the full-acceptance early-out below.
        if self._win_captured:
            self._conv_taps_stale, self._conv_win_stale = True, False
        if idx == self._verify_states.shape[1] - 1:
            # Full acceptance: the verify already LEFT the durable state at the last token (rec_state
            # = states[T-1], _conv_win_buf = window rows [T-1, T-1+K)), so every copy below would write
            # what is already there. 48 layers x ~6 device ops saved on those iterations.
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
            # Batched-conv path: conv slots were not materialised per token. The shift-register as of
            # token idx is rows [idx, idx+K) of the stashed window (see _forward_verify_recurrent_batched).
            #
            # ONE slice + ONE copy into the persistent shift register the next replay's carry reads.
            # This used to write the K conv_states taps individually (K x slice+copy+deallocate per
            # layer, x48 layers) and that dispatch count WAS the fb=1 commit regression. conv_states
            # themselves are left at the end-of-window state the trace wrote; nothing in the spec
            # loop reads them (see sync_conv_win).
            w = ttnn.slice(self._verify_win_buf, (0, idx, 0), (1, idx + self.K, self.qkv_dim_tp))
            ttnn.copy(w, self._conv_win_buf)
            ttnn.deallocate(w)
        else:
            for j, c in enumerate(convs):
                ttnn.copy(c, self.conv_states[j])
            self._conv_taps_stale, self._conv_win_stale = False, True
        self._verify_states = None
