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


# --------------------------------------------------------------------------- #
# Batched (multi-user) spec-decode host helpers. Pure torch: no ttnn, no device.
# --------------------------------------------------------------------------- #
# Row layout for a batched verify: B users x T = K+1 candidate rows, packed USER-MAJOR into one
# 32-row decode tile (row r = u*T + j). After a verify the host knows mi[u], the index of the last
# ACCEPTED row of user u, and commits by SELECTING rather than by re-running anything: the next
# replay reads user u's GDN state from the ring block mi[u] wrote, and rebuilds user u's conv
# shift-register from the window rows mi[u] wrote. These two helpers build the tiny selector
# tensors that carry mi into the trace.


def spec_state_blk_idx(mi, n_users, nv):
    """Ring block index per (user, value-head) for the fused recurrent op's deferred-select mode.

    The ring holds T*n_users*nv state blocks laid out (token slot, user, head)-major, so the state
    user u produced after accepting through row mi[u] lives at block (mi[u]*n_users + u)*nv + h.
    Entry (u*nv + h) of the returned tensor is exactly that block.

    mi: per-user accepted-prefix last index (len n_users, each in [0, T)).
    Returns torch.int32 [n_users*nv].
    """
    assert len(mi) == n_users, f"need one mi per user: got {len(mi)} for {n_users} users"
    idx = torch.empty(n_users * nv, dtype=torch.int32)
    head = torch.arange(nv, dtype=torch.int32)
    for u in range(n_users):
        idx[u * nv : (u + 1) * nv] = (int(mi[u]) * n_users + u) * nv + head
    return idx


def spec_conv_sel(mi, n_users, T, kc):
    """One-hot row selector that rebuilds every user's conv window for the next verify.

    The verify concatenates last iteration's window with this iteration's new qkv rows:
        concat = cat([E_prev [n_users, kc-1+T, C], qkv_new [n_users, T, C]], dim=1)
    and left-multiplies it by this selector, so for user u:
        out row r        (r < kc-1) <- concat row mi[u] + 1 + r      (E_prev, post-commit window)
        out row kc-1 + j            <- concat row (kc-1+T) + j       (this iteration's qkv row j)
    i.e. the kc-1 conv taps that survive the commit, followed by the T new inputs.

    Returns torch.bfloat16 [n_users, kc-1+T, kc-1+2T] (one-hot, so the matmul is exact).
    """
    assert len(mi) == n_users, f"need one mi per user: got {len(mi)} for {n_users} users"
    rows, cols = kc - 1 + T, kc - 1 + 2 * T
    sel = torch.zeros(n_users, rows, cols, dtype=torch.bfloat16)
    for u in range(n_users):
        m = int(mi[u])
        assert 0 <= m < T, f"mi[{u}]={m} out of range [0,{T})"
        for r in range(kc - 1):
            sel[u, r, m + 1 + r] = 1.0
        for j in range(T):
            sel[u, kc - 1 + j, (kc - 1 + T) + j] = 1.0
    return sel


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
        # ---- Batched (multi-user) spec-decode verify buffers; see prepare_spec_verify. ----
        # (n_users, T) the spec buffers below are sized for; None => prepare_spec_verify not run.
        self._spec_shape = None
        # Per-token recurrent-state RING, fp32 TILE [T*B*Nv, Dk, Dv]. The fused recurrent op reads
        # each (user, head)'s initial state from the block `state_blk_idx` points at and writes that
        # token's state back IN PLACE, so "commit" is just the host choosing next iteration's index.
        self._spec_ring = None
        # E_prev: the conv window each verify consumes and rewrites, bf16 TILE [B, K-1+T, C].
        self._verify_win_buf = None
        # Persistent zero rows [B, T-1, C] used once by seed_spec_state to fill _verify_win_buf's
        # tail (never read: the first replay runs with mi = 0).
        self._spec_win_pad = None
        # The DURABLE shift register as one [B, K, qkv_dim_tp] tensor, mirroring conv_states[0..K-1]
        # with the user axis folded in (tap j of user u == _conv_win_buf[u, j, :]).
        self._conv_win_buf = None
        # One-hot matmul config: HiFi4 + fp32 accumulate so a 0/1 selector matmul is EXACT in bf16.
        self._cfg_onehot = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # Which of the two mirrors is authoritative. The traced batched verify advances ONLY the
        # window buffers (refilling the K conv_states taps inside the trace cost ~10 ms/iteration
        # over 48 layers and nothing in the spec loop reads them), so after materialize_spec_state
        # the taps are BEHIND: _conv_taps_stale. Conversely everything that writes the taps from
        # outside (prefill capture_state, reset, slot edits, decode's own shift register) leaves the
        # window behind: _conv_win_stale. Exactly one can be set at a time — each setter clears the
        # other. The rebuilds are sync_conv_taps() (window -> taps, at every tap CONSUMER) and
        # sync_conv_win() (taps -> window, lazily in _ensure_conv_win, which only ever runs eagerly).
        self._conv_taps_stale = False
        self._conv_win_stale = False
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
        self._conv1d_wprep = {}  # prepared depthwise weights, keyed by (batch_size, input_width)
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill

    def _release_spec_bufs(self):
        """Free the batched spec-verify buffers and forget their shape (prepare_spec_verify reallocs)."""
        for name in ("_spec_ring", "_verify_win_buf", "_spec_win_pad"):
            buf = getattr(self, name, None)
            if buf is not None:
                ttnn.deallocate(buf)
            setattr(self, name, None)
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
        # Chunk-outer batched-prefill conv left-context (allocated lazily by forward_prefill_batched).
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
        self._batched_conv_carry = None
        # rec_state/conv_states got fresh addresses here, so every spec-verify buffer derived from
        # the old ones is stale — drop them (re-allocated by the next prepare_spec_verify).
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
        (the selector matmul produced it) and the new_state return is dead there.

        Byte-identical conv output — same op, same weights, same input — for two ops/layer less: the
        duplicate window concat (~0.11 ms/layer) and the discarded new_state's tile-unaligned row
        slice + relayout. Both were pure waste in the verify, ~7 ms/iteration over 48 GDN layers."""
        return self._conv1d_window(win, T)

    def _conv1d_window(self, xin, T):
        """The ttnn.conv1d call itself, over an already-concatenated [Bc, K-1+T, C] TILE window.

        Bc is the window's leading dim: 1 for prefill and the single-user verify, n_users for the
        batched spec verify (each user an independent conv batch row — the depthwise conv has no
        cross-row term, so batching is exact). Weights are prepared once per (Bc, input_width);
        the prep is a HOST call, so every (Bc, L) a trace will replay must appear in the warmup."""
        dev, K, C = self.mesh, self.K, self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        Bc = xin.shape[0] if len(xin.shape) == 3 else 1
        Lin = (K - 1) + T
        xin = ttnn.to_layout(xin, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        xin = ttnn.reshape(xin, (Bc, Lin, 1, C))
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # Needs l1_small_size on the device (prefill/demo set 24576); matches the validated A/B config.
        conv_cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        # Prepare conv weight once per shape (warmup); avoids host reprocess + keeps replay device-only.
        wprep = self._conv1d_wprep.get((Bc, Lin))
        if wprep is None:
            wprep = ttnn.prepare_conv_weights(
                weight_tensor=self.tw["conv_w1d"],
                input_memory_config=_dram,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=C,
                out_channels=C,
                batch_size=Bc,
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
            self._conv1d_wprep[(Bc, Lin)] = wprep
        out = ttnn.conv1d(
            input_tensor=xin,
            weight_tensor=wprep,
            device=dev,
            in_channels=C,
            out_channels=C,
            batch_size=Bc,
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
        out = ttnn.reshape(out, (Bc, T, C))
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

    def _row_shard_memcfg(self, nhw, width):
        """L1 height-shard config for a rank-4 TILE tensor flattened to (nhw, width) rows.

        ``ttnn.experimental.slice_write`` only writes STRAIGHT into an interleaved TILE output when
        its input is sharded; with an interleaved input it round-trips the output through ROW_MAJOR,
        which mints a new buffer and breaks any address a trace baked in. So every slice_write into
        a persistent buffer shards its source with this.
        """
        grid_size = self.mesh.compute_with_storage_grid_size()
        assert (
            grid_size.x >= 8 and grid_size.y >= 6
        ), f"GDN state write needs an 8x6 core rectangle, got {grid_size.x}x{grid_size.y}"
        assert nhw % ttnn.TILE_SIZE == 0, f"GDN state write rows {nhw} is not tile-aligned"
        n_tiles = nhw // ttnn.TILE_SIZE

        # Prefer the tuned 8x6=48-core rectangle, but only when the tile count actually divides by
        # 48 — it depends on the per-device head count, not just on B. At Nv_tp=12, nhw=B*1536 gives
        # 48*B tiles and the fast path always fires; at Nv_tp=8 (TP=4 on this model) nhw=B*1024 gives
        # 32*B tiles, so it fires only for B a multiple of 3, and at Nv_tp=6 (TP=8) B=1 gives 24
        # tiles. Otherwise fall back to the largest core count that divides the tiles evenly.
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

    # ------------------------------------------------------------------ #
    # Batched (multi-user) speculative verify
    # ------------------------------------------------------------------ #
    # One verify advances B users x T = K+1 candidate rows in a single pass over the 32-row decode
    # tile (rows are USER-MAJOR: row u*T + j). Nothing here commits: the recurrent op writes EVERY
    # token's state into a persistent ring and the conv window keeps every candidate's shift
    # register, so after the host reads the accepted prefix it "commits" by handing the NEXT replay
    # two tiny selector tensors (state_blk_idx, conv_sel) instead of running a commit phase.
    #
    # Everything the body touches is persistent and allocated by prepare_spec_verify, because the
    # body runs under a captured trace: a first-time allocation or a host write inside it would
    # either TT_FATAL at capture or bake a throwaway address into the replay.

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

        Call it EAGERLY, before the verify warmup pass, so the warmup compiles every program the
        captured body replays.
        """
        assert n_users == self.B, f"spec verify runs the whole decode batch: n_users={n_users}, B={self.B}"
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

    def seed_spec_state(self):
        """Load the live decode state into the spec buffers so the first replay can run with mi = 0.

        rec_state -> ring token-slot 0 (blocks [0, B*Nv), which is where a mi = 0 state_blk_idx
        points), and the conv shift register -> the first K rows of E_prev (which is what a mi = 0
        conv_sel reads: concat rows 1 .. K-1). The window's remaining T-1 rows are zeroed padding
        that no mi = 0 selector can reach.
        """
        assert self._spec_ring is not None, "seed_spec_state before prepare_spec_verify"
        B, Nv, Dk, Dv = self.B, self.Nv, self.Dk, self.Dv
        T = self._spec_shape[1]
        # --- recurrent state -> ring blocks [0, B*Nv) ---
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
        # --- conv shift register -> E_prev rows [0, K) ---
        self.sync_conv_win()  # taps are the truth outside the spec loop; mirror them into the window
        full = (
            self._conv_win_buf
            if self._spec_win_pad is None
            else ttnn.concat([self._conv_win_buf, self._spec_win_pad], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        ttnn.copy(full, self._verify_win_buf)  # full-shape copy: _verify_win_buf keeps its address
        if full is not self._conv_win_buf:
            ttnn.deallocate(full)
        # Both mirrors now hold the same (live) shift register.
        self._conv_taps_stale, self._conv_win_stale = False, False

    def materialize_spec_state(self, mi):
        """Pull the accepted state out of the spec buffers back into the durable decode state.

        The spec loop never commits on device; this is what ends it (or hands a plain decode step
        the right state). For user u: rec_state row u <- ring block (mi[u]*B + u)*Nv, and
        _conv_win_buf row u <- E_prev rows [mi[u], mi[u]+K) — exactly the shift register T
        sequential decode steps would have left after accepting through row mi[u].

        Leaves the K conv_states taps BEHIND the window (``_conv_taps_stale``); the next tap
        consumer rebuilds them via sync_conv_taps.
        """
        assert self._spec_ring is not None, "materialize_spec_state before prepare_spec_verify"
        B, Nv, Dk, Dv, K, C = self.B, self.Nv, self.Dk, self.Dv, self.K, self.qkv_dim_tp
        T = self._spec_shape[1]
        assert len(mi) == B, f"need one mi per user: got {len(mi)} for {B} users"
        assert all(0 <= int(m) < T for m in mi), f"mi {list(mi)} out of range [0,{T})"
        # --- recurrent state: one slice per user, ONE concat, ONE in-place copy ---
        rows = []
        for u in range(B):
            blk = (int(mi[u]) * B + u) * Nv
            rows.append(ttnn.reshape(ttnn.slice(self._spec_ring, (blk, 0, 0), (blk + Nv, Dk, Dv)), (1, Nv, Dk, Dv)))
        new_rec = ttnn.concat(rows, dim=0) if B > 1 else rows[0]
        src = new_rec if new_rec.dtype == self.rec_state.dtype else ttnn.typecast(new_rec, self.rec_state.dtype)
        ttnn.copy(src, self.rec_state)
        if src is not new_rec:
            ttnn.deallocate(src)
        if B > 1:
            for r in rows:
                ttnn.deallocate(r)
        ttnn.deallocate(new_rec)
        # --- conv window: user u's K rows starting at its own mi[u] ---
        wins = [ttnn.slice(self._verify_win_buf, (u, int(mi[u]), 0), (u + 1, int(mi[u]) + K, C)) for u in range(B)]
        new_win = ttnn.concat(wins, dim=0) if B > 1 else wins[0]
        ttnn.copy(new_win, self._conv_win_buf)
        if B > 1:
            for w in wins:
                ttnn.deallocate(w)
        ttnn.deallocate(new_win)
        self._conv_taps_stale, self._conv_win_stale = True, False

    def forward_verify_recurrent(self, x, valid_len, pre_gathered=False, n_users=1, state_blk_idx=None, conv_sel=None):
        """Batched hybrid spec-decode verify for GDN: advance B users x T candidate rows at once.

        x : [1, 1, bucket, dim] prefill-normed input; the first ``valid_len`` rows are real and
        USER-MAJOR (row u*T + j == user u, candidate j). Returns [1, 1, bucket, dim/tp], zero-padded
        past ``valid_len`` (nothing downstream reads those rows: attention is causal and verify only
        row-selects real rows).

        valid_len  : n_users * T real rows; must fit one 32-row decode tile.
        pre_gathered: the caller already handed us a FULL-dim activation (decode-config verify runs
                     the layer norms in Mode.DECODE, which gathers pre-norm), so skip the all-gather.
        n_users    : number of real users packed into the tile; must equal self.B.
        state_blk_idx: device uint32 ROW_MAJOR [B*Nv], from spec_state_blk_idx(mi, B, Nv). Selects
                     each (user, head)'s initial state block inside the persistent ring.
        conv_sel   : device bf16 TILE [B, K-1+T, K-1+2T] one-hot, from spec_conv_sel(mi, B, T, K).
                     Rebuilds each user's conv window from the accepted prefix + the new rows.

        Side effects: the ring is updated IN PLACE (every candidate's state, for every user) and
        _verify_win_buf becomes the new E_prev. Neither rec_state nor conv_states moves — the host
        commits later, by index (materialize_spec_state), not by copying state here.
        """
        assert valid_len <= tpc.TILE_SIZE, f"verify bucket {valid_len} exceeds one tile"
        assert n_users == self.B, f"spec verify runs the whole decode batch: n_users={n_users}, B={self.B}"
        assert valid_len % n_users == 0, f"valid_len {valid_len} is not {n_users} whole user row-groups"
        T = valid_len // n_users
        assert n_users * T <= tpc.TILE_SIZE
        if state_blk_idx is None or conv_sel is None:
            raise ValueError(
                "forward_verify_recurrent needs state_blk_idx and conv_sel (the deferred-select verify "
                "is the only verify path; call prepare_spec_verify + seed_spec_state first)"
            )
        assert self._spec_shape == (n_users, T), (
            f"spec buffers are sized for {self._spec_shape}, verify asked for {(n_users, T)}; "
            "call prepare_spec_verify(n_users, T) before capture"
        )
        return self._forward_verify_recurrent_batched(x, valid_len, T, n_users, state_blk_idx, conv_sel, pre_gathered)

    def _forward_verify_recurrent_batched(self, x, valid_len, T, n_users, state_blk_idx, conv_sel, pre_gathered):
        """Gather + ONE decode projection over all valid rows, then the batched conv + recurrence.

        Key fact this rests on: the decode matmul (matmul_1d_decode) is row-independent and
        processes a full 32-row M-tile however many rows are real, so packing every user's T rows
        into ONE projection is per-row identical to projecting them one at a time — while collapsing
        valid_len separate launches into one. The AGMM prefill projection is deliberately avoided:
        it rounds differently and would drift the state away from plain decode.
        """
        qkv_all, z_all, a_all, b_all, bucket = self._verify_project(x, valid_len, pre_gathered)
        return self._verify_fullbatch(qkv_all, z_all, a_all, b_all, T, n_users, bucket, state_blk_idx, conv_sel)

    def _verify_project(self, x, valid_len, pre_gathered):
        """Gather the real rows to full dim and run ONE decode qkvzab projection over them.

        The front half of every spec-decode GDN pass, shared by the batched verify and the T=1 seed
        so the two can never drift apart: same slice, same all-gather (or none), same
        ``_project_qkvzab`` at S = valid_len <= TILE_SIZE, which routes through matmul_1d_decode —
        the exact projection a plain decode step runs. Returns (qkv, z, a, b, bucket).
        """
        mc = ttnn.DRAM_MEMORY_CONFIG
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
        R = valid_len  # real rows, user-major

        x_valid = x if R == bucket else ttnn.slice(x, (0, 0, 0), (1, R, x.shape[-1]))
        if pre_gathered:
            # Already full-dim [1, bucket, dim]: gathering again would quadruple the feature dim.
            xg = x_valid
        else:
            # x is 3D [1, R, dim/tp] here (reshaped above), so gather the LAST (feature) dim.
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
        qkv_all, z_all, a_all, b_all = self._project_qkvzab(xg, R, out_mc=mc)
        if xg is not x:
            ttnn.deallocate(xg)
        if _x_owned:
            ttnn.deallocate(x)
        return qkv_all, z_all, a_all, b_all, bucket

    def forward_seed_recurrent(self, x, valid_len, pre_gathered=False, n_users=1):
        """The spec loop's SEED: advance GDN by ONE row per user, with the VERIFY's arithmetic.

        Same shape as a plain decode step (B users x T = 1 token) but built out of the verify's
        pieces, not the decode path's: the native depthwise ``ttnn.conv1d`` over a [B, K, C] window
        instead of the K-tap shift-register MAC. Those two are NOT bit-equal, and the state the seed
        leaves behind is what every later verify replay resumes from, so seeding with the decode
        formulation plants a low-bit mismatch between the seed state and the formulation the whole
        loop then uses — enough to fork the greedy trajectory at a near tie.

        NON-RING, unlike ``forward_verify_recurrent``: the spec ring and E_prev do not exist yet
        (``prepare_spec_verify`` runs later, inside ``capture_verify_trace``). This advances the
        DURABLE state — ``rec_state`` in place and ``_conv_win_buf`` (plus the K taps) to the new
        window — which is exactly what ``seed_spec_state`` copies into the ring and E_prev after the
        capture.

        x : [1, 1, B, dim] (full-dim when ``pre_gathered``). Returns [1, 1, B, dim/tp].
        """
        assert n_users == self.B, f"the seed runs the whole decode batch: n_users={n_users}, B={self.B}"
        assert valid_len == n_users, f"the seed is one row per user: valid_len={valid_len}, n_users={n_users}"
        assert n_users <= tpc.TILE_SIZE, f"{n_users} users exceeds the {tpc.TILE_SIZE}-row decode tile"
        qkv_all, z_all, a_all, b_all, bucket = self._verify_project(x, valid_len, pre_gathered)
        return self._seed_fullbatch(qkv_all, z_all, a_all, b_all, n_users, bucket)

    def _seed_fullbatch(self, qkv_all, z_all, a_all, b_all, n_users, bucket):
        """``_verify_fullbatch`` at T = 1 against the DURABLE state instead of the ring.

        Op for op the pre-ring verify body: window slice -> concat -> conv1d -> q/k/v slices ->
        sigmoid/softplus gating -> ONE fused recurrent dispatch -> state writeback -> gated norm,
        SiLU gate, out-proj, all-reduce. At B = 1 every shape here is the shape that body had at
        T = 1, so the arithmetic is the same arithmetic.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        kd, C, rf, K = self.key_dim_tp, self.qkv_dim_tp, Nv // Nk, self.K
        B = R = n_users

        # 1) Conv window. The carry is the shift register's previous K-1 inputs, i.e. taps [1, K) of
        #    the [B, K, C] mirror; E = [carry(K-1) ; this row(1)] is BOTH the conv input and the new
        #    shift register, so it is built once. _ensure_conv_win() makes the mirror current from
        #    the live taps first (prefill/reset/a plain decode step all move the taps under it).
        self._ensure_conv_win()
        qkv_new = self._rows_to_users(qkv_all, B, 1, C)  # [B, 1, C]
        carry = ttnn.slice(self._conv_win_buf, (0, 1, 0), (B, K, C))  # [B, K-1, C]
        E = ttnn.concat([carry, qkv_new], dim=1, memory_config=mc)  # [B, K, C]
        ttnn.deallocate(carry)
        ttnn.deallocate(qkv_new)
        conv_all = self._conv1d_verify(E, 1)  # [B, 1, C], SiLU applied

        # 2) q/k/v: FEATURE-dim slices (contiguous columns) + one repeat_interleave each.
        q_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 0), (B, 1, kd)), (B, 1, Nk, Dk))
        k_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, kd), (B, 1, 2 * kd)), (B, 1, Nk, Dk))
        v_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 2 * kd), (B, 1, C)), (B, 1, Nv, Dv))
        ttnn.deallocate(conv_all)
        if rf != 1:
            q_all = ttnn.repeat_interleave(q_all, rf, dim=2)
            k_all = ttnn.repeat_interleave(k_all, rf, dim=2)

        # 3) Gating for every user at once.
        a_new = self._rows_to_users(a_all, B, 1, Nv)
        b_new = self._rows_to_users(b_all, B, 1, Nv)
        beta_all = ttnn.sigmoid(b_new, memory_config=_L1)
        ttnn.deallocate(b_new)
        g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_new, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a_new)

        # 4) ONE recurrence dispatch, PLAIN mode: initial state is the durable rec_state and the op
        #    returns the final state only (there is no per-token ring yet, and at T = 1 the two are
        #    the same state anyway). Identical call — and identical [B,1,Nv,*] shapes — to the fused
        #    recurrent decode step, so it adds no program the decode path does not already have.
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

        # 5) State writeback, to the DURABLE buffers. rec_state in place (its address is baked into
        #    the decode traces); the shift register is E itself, which already has exactly K rows.
        if self._stable_state:
            ttnn.copy(states, self.rec_state)
            ttnn.deallocate(states)
        else:
            self.rec_state = states
        ttnn.copy(E, self._conv_win_buf)
        ttnn.deallocate(E)
        # Push the window straight back out to the K taps rather than leaving them stale. Two
        # reasons, both about WHEN programs compile: sync_conv_win() branches on _conv_taps_stale,
        # and the throwaway seed_spec_state() inside capture_verify_trace exists only to compile the
        # branch the REAL post-capture seed_spec_state() will take — so both calls must see the same
        # flags. And a stale-tap window would make the next plain decode step (sync_conv_taps) pay
        # its K slice+copy per layer with the verify trace parked. Costs K copies per layer, once.
        self._conv_taps_stale, self._conv_win_stale = True, False
        self.sync_conv_taps()  # clears _conv_taps_stale; both mirrors now hold the seeded window

        # 6) Output tail over the B rows.
        out_n = ttnn.rms_norm(ttnn.reshape(o_all, (R, Nv, Dv)), weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
        ttnn.deallocate(o_all)
        out_f_b = ttnn.reshape(out_n, (1, R, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f_b, z_all, mc)
        ttnn.deallocate(out_f_b)
        ttnn.deallocate(z_all)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
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

    def _rows_to_users(self, t, B, T, width):
        """[1, B*T, width] -> [B, T, width]. Rows are user-major, so this is a pure regroup.

        B == 1 is the identity (SAME buffer returned — do not free both handles). For B > 1 the
        reshape allocates, because each user's T rows have to start a fresh tile row; the source is
        freed here. The tiled reshape builds a host page-map on its program-cache MISS, so a traced
        body must have been warmed up at this exact shape.
        """
        if B == 1:
            return t
        out = ttnn.reshape(t, (B, T, width))
        ttnn.deallocate(t)
        return out

    def _verify_fullbatch(self, qkv_all, z_all, a_all, b_all, T, n_users, bucket, state_blk_idx, conv_sel):
        """The device body of a batched verify. Inputs are the already-projected [1, B*T, *] rows.

        No per-token loop and no commit phase:
          conv   -> one one-hot matmul rebuilds every user's window, then ONE native depthwise
                    ttnn.conv1d over the B window rows;
          recur. -> ONE fused_recurrent_gated_delta_rule dispatch in ring mode, which reads each
                    (user, head)'s initial state from the block state_blk_idx names and writes all
                    B*T per-token states back into the same ring;
          output -> one gated norm + out-proj + all-reduce over the B*T rows.
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1, mc, rm = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.ROW_MAJOR_LAYOUT
        kd, C, rf = self.key_dim_tp, self.qkv_dim_tp, Nv // Nk
        B, R = n_users, n_users * T

        # 1) Conv window. cat = [E_prev(K-1+T) ; new qkv(T)] per user; conv_sel is one-hot, so the
        #    matmul is an exact row gather (bf16 in, HiFi4 + fp32 accumulate) that both drops the
        #    rows the last commit rejected and appends this iteration's inputs. E_new is stored as
        #    the next E_prev AND is the conv input, so nothing is built twice.
        qkv_new = self._rows_to_users(qkv_all, B, T, C)
        cat = ttnn.concat([self._verify_win_buf, qkv_new], dim=1, memory_config=mc)  # [B, K-1+2T, C]
        ttnn.deallocate(qkv_new)
        E_new = ttnn.matmul(conv_sel, cat, compute_kernel_config=self._cfg_onehot, memory_config=mc)
        ttnn.deallocate(cat)
        # In place, so the trace's baked address survives. NOTE the staleness flags are deliberately
        # NOT touched here: for the whole spec loop the durable conv truth is _verify_win_buf plus
        # the host's mi (which rows of it each user accepted). BOTH mirrors — the K conv_states taps
        # AND _conv_win_buf — are behind it, and marking one stale would claim the OTHER is current,
        # which neither is. So no in-loop consumer may read the taps or _conv_win_buf; the loop's
        # own carry is this buffer and conv_sel. materialize_spec_state restores the two-mirror
        # invariant when the loop ends: it rebuilds _conv_win_buf from the accepted rows and sets
        # _conv_taps_stale=True so the next tap consumer (forward_decode, a slot edit) resyncs.
        ttnn.copy(E_new, self._verify_win_buf)
        conv_all = self._conv1d_verify(E_new, T)  # [B, T, C], SiLU applied
        ttnn.deallocate(E_new)

        # 2) q/k/v for all B*T: FEATURE-dim slices (contiguous columns) + one repeat_interleave each.
        q_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 0), (B, T, kd)), (B, T, Nk, Dk))
        k_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, kd), (B, T, 2 * kd)), (B, T, Nk, Dk))
        v_all = ttnn.reshape(ttnn.slice(conv_all, (0, 0, 2 * kd), (B, T, C)), (B, T, Nv, Dv))
        ttnn.deallocate(conv_all)
        if rf != 1:
            q_all = ttnn.repeat_interleave(q_all, rf, dim=2)
            k_all = ttnn.repeat_interleave(k_all, rf, dim=2)

        # 3) Gating for every row at once.
        a_new = self._rows_to_users(a_all, B, T, Nv)
        b_new = self._rows_to_users(b_all, B, T, Nv)
        beta_all = ttnn.sigmoid(b_new, memory_config=_L1)
        ttnn.deallocate(b_new)
        g_all = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a_new, tw["dt_bias"]), memory_config=_L1)
        ttnn.deallocate(a_new)

        # 4) ONE recurrence dispatch, ring mode: initial state per (user, head) comes from the ring
        #    block state_blk_idx names, and every token's state is written back in place. The
        #    returned state IS self._spec_ring, so there is nothing to copy or free.
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

        # 5) Output tail over the B*T rows (the inverse of the step-1 regroup: rms_norm normalises
        #    over the last dim, so one call covers every row).
        out_n = ttnn.rms_norm(ttnn.reshape(o_all, (R, Nv, Dv)), weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
        ttnn.deallocate(o_all)
        out_f_b = ttnn.reshape(out_n, (1, R, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = _silu_mul(out_f_b, z_all, mc)
        ttnn.deallocate(out_f_b)
        ttnn.deallocate(z_all)
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, R, partial.shape[-1]))
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

        Row u, tap j of this buffer is user u's conv_states[j] column — the same K taps decode
        shifts, transposed into one tensor so a commit is one slice and one copy.

        Allocate-and-seed happens on an EAGER call (prepare_spec_verify, the spec loop's seed),
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

        A batched verify advances ONLY _conv_win_buf / _verify_win_buf (see _verify_fullbatch), so
        the K tap buffers go stale for as long as nothing reads them. Every tap CONSUMER calls this
        first; it is a no-op — zero device ops, so calling it from a traced body is safe — unless a
        verify actually ran since the taps were last written.

        Tap j is _conv_win_buf[:, j, :] ([B,1,C]) reshaped to conv_states[j]'s [1,B,C]."""
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
            if B > 1:  # B == 1 makes the reshape an identity (src IS row)
                ttnn.deallocate(row)

    def sync_conv_win(self):
        """Mirror conv_states[0..K-1] into the persistent [B, K, qkv_dim_tp] shift-register buffer.

        conv_states stay the source of truth outside the spec loop (prefill fills them, the decode
        path shifts them, slot edits rewrite them); this buffer is the copy the batched verify seeds
        E_prev from, so it has to be re-seeded whenever conv_states are set from outside."""
        if self._conv_win_buf is None or self.conv_states is None:
            return
        if self._conv_taps_stale:
            # The WINDOW is the truth here (a verify advanced it and the taps were left behind), so
            # copying the taps over it would undo the verify. Bring the taps forward instead; both
            # mirrors then agree and there is nothing left to copy.
            self.sync_conv_taps()
            self._conv_win_stale = False
            return
        B, C = self.B, self.qkv_dim_tp
        rows = [ttnn.reshape(c, (B, 1, C)) for c in self.conv_states]  # [1,B,C] -> [B,1,C]
        w = ttnn.concat(rows, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [B, K, C]
        ttnn.copy(w, self._conv_win_buf)
        ttnn.deallocate(w)
        if B > 1:  # B == 1 makes the reshape an identity (rows[j] IS conv_states[j])
            for r in rows:
                ttnn.deallocate(r)
        self._conv_win_stale = False
