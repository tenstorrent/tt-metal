# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B GatedDeltaNet (linear-attention) block — v2 port.

This is the v2 (llama3_70b_galaxy-derived tree) port of the v1 reference at
``models/demos/qwen3_6_galaxy/tt/qwen36_deltanet.py``. The math is identical;
the adaptations relative to v1 are:

  (a) Persistent recurrent_state / conv_state device buffers allocated once in
      ``__init__`` and written in place during ``forward_decode``. This is the
      pattern used by ``paged_fused_update_cache`` for the llama KV cache and
      is the prerequisite for trace replay (a captured trace expects fixed
      buffer addresses, so a per-call new allocation breaks replay).
  (b) View-aliasing fix from the v1 commit ``e5f496ba69a`` — the
      ``_causal_conv1d_fir_mesh`` helper marks the pad as persistent in the
      ``conv_state``-supplied branch so ``pad.deallocate(True)`` after the
      concat does NOT free the caller's conv_state.
  (c) Uses v2's ``llama_ccl.TT_CCL`` plumbing — the constructor accepts a
      ``tt_ccl`` argument so the broader decoder can dispatch uniformly.
      The current reduction path (all_gather + fast_reduce_nc) does not yet
      route through ``tt_ccl``; that swap is deferred to the optimization
      phase where the trace-safe CCL primitive is selected. The argument is
      stored as ``self.tt_ccl`` so future swaps don't require an API change.

Mesh sharding strategy (identical to v1)
-----------------------------------------
Mesh is 8 rows × 4 cols (32 chips).
- 48 V-heads sharded across 8 rows → 6 V-heads/row  (n_v_per_row = 6)
- 16 K-heads sharded across 8 rows → 2 K-heads/row  (n_k_per_row = 2)
- Cols are replicated (all 4 cols on a row do identical work)
- Hidden dim H=5120 is replicated (matches residual-stream convention)
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy_v2.tt.ttnn_delta_rule_ops_fp32 import recurrent_gated_delta_rule_ttnn_fp32
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_ttnn,  # kept for back-compat; not used in fp32 path
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import chunk_gated_delta_rule_ttnn


# ----------------------------------------------------------------------------
# Mesh-compatible depthwise causal conv1d + SiLU via FIR decomposition.
# Ported verbatim from v1 (commit e5f496ba69a) — including the view-alias fix
# for the conv_state-supplied branch.
# ----------------------------------------------------------------------------
def _causal_conv1d_fir_mesh(
    x,
    w_per_tap,
    kernel_size,
    mesh_device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    conv_state=None,
    conv_state_zero_pad=None,
):
    """FIR depthwise conv1d + SiLU. See v1 docstring for details.

    Returns (output [B, T, D] TILE_LAYOUT, new_conv_state [B, K-1, D] ROW_MAJOR).
    """
    B, T, D = x.shape[0], x.shape[1], x.shape[2]

    # Convert x to ROW_MAJOR to allow non-tile-aligned slicing.
    x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)

    pad_is_persistent = False
    if conv_state is not None:
        # View-alias fix: ttnn.slice short-circuits to a VIEW when the slice
        # extents match the source shape exactly. Since conv_state arrives
        # already at [B, K-1, D] every decode step, pad ALIASES conv_state's
        # storage. Mark pad as persistent so the deallocate below is skipped.
        if conv_state.layout == ttnn.TILE_LAYOUT:
            cs_rm = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=memory_config)
        else:
            cs_rm = conv_state
        pad = ttnn.slice(cs_rm, [0, 0, 0], [B, kernel_size - 1, D], memory_config=memory_config)
        pad_is_persistent = True
    elif conv_state_zero_pad is not None:
        zp_shape = list(conv_state_zero_pad.shape)
        if zp_shape == [B, kernel_size - 1, D]:
            pad = conv_state_zero_pad
            pad_is_persistent = True
        else:
            pad = ttnn.slice(
                conv_state_zero_pad,
                [0, 0, 0],
                [B, kernel_size - 1, D],
                memory_config=memory_config,
            )
    else:
        pad_torch = torch.zeros(B, kernel_size - 1, D, dtype=torch.bfloat16)
        pad = ttnn.from_torch(
            pad_torch,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    x_padded = ttnn.concat([pad, x_rm], dim=1, memory_config=memory_config)
    if not pad_is_persistent:
        pad.deallocate(True)
    # If we built a ROW_MAJOR copy of a TILE_LAYOUT conv_state above, free
    # the copy now that concat has captured its data — but never the caller's
    # original conv_state.
    if conv_state is not None and conv_state.layout == ttnn.TILE_LAYOUT:
        cs_rm.deallocate(True)

    # Compute new conv state: last K-1 tokens of x.
    if T >= kernel_size - 1:
        new_conv_state_rm = ttnn.slice(x_rm, [0, T - (kernel_size - 1), 0], [B, T, D], memory_config=memory_config)
    else:
        new_conv_state_rm = ttnn.slice(x_padded, [0, T, 0], [B, T + kernel_size - 1, D], memory_config=memory_config)
    new_conv_state = new_conv_state_rm  # [B, K-1, D] ROW_MAJOR
    x_rm.deallocate(True)

    out = None
    for k in range(kernel_size):
        start = k
        end = k + T
        x_slice = ttnn.slice(x_padded, [0, start, 0], [B, end, D], memory_config=memory_config)
        x_slice_tl = ttnn.to_layout(x_slice, ttnn.TILE_LAYOUT, memory_config=memory_config)
        x_slice.deallocate(True)
        term = ttnn.multiply(x_slice_tl, w_per_tap[k], memory_config=memory_config)
        x_slice_tl.deallocate(True)
        if out is None:
            out = term
        else:
            prev = out
            out = ttnn.add(prev, term, memory_config=memory_config)
            prev.deallocate(True)
            term.deallocate(True)

    x_padded.deallocate(True)
    return ttnn.silu(out, memory_config=memory_config), new_conv_state


class TtQwen36DeltaAttention(LightweightModule):
    """Linear-attention (GatedDeltaNet) block for Qwen3.6-27B on BH GLX 8×4.

    Constructor signature matches ``TtLlamaAttention`` so the v2 decoder can
    dispatch ``lin`` vs ``full`` layers uniformly. Extra args are accepted via
    ``**kwargs`` and ignored.

    Persistent buffer model (v2 adaptation): both the recurrent state and the
    conv state are allocated as fixed device tensors at init and updated in
    place during ``forward_decode``. The generator threads
    ``self.dn_state_buffer`` / ``self.conv_state_buffer`` across decode steps;
    ``clear_state()`` zeros them at the start of each fresh sequence.
    """

    def __init__(
        self,
        mesh_device,
        args,
        layer_num,
        weights_dict,
        tt_ccl,
        dtype=ttnn.bfloat16,
        **kwargs,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.layer_num = layer_num
        self.tt_ccl = tt_ccl
        # Keep DeltaNet weights at the requested model dtype (bf8 by default).
        # V2-7b: forcing self.dtype=bfloat16 unexpectedly dropped layer-0 PCC
        # from 0.999 → 0.84 — DeltaNet relies on bf8-quantised recurrent inputs
        # to keep its `exp(-A_log * softplus(...))` term within range.  The
        # full-attention path (TtLlamaAttention) gets a separate bf16 weight
        # override because its quantisation noise compounds differently.
        self.dtype = dtype

        # --- Mesh topology ---
        self.cluster_shape = list(mesh_device.shape)  # [8, 4]
        self.mesh_rows = self.cluster_shape[0]
        self.mesh_cols = self.cluster_shape[1]

        # --- Model dimensions ---
        self.hidden_size = args.dim  # 5120
        self.n_k_heads = args.linear_num_key_heads  # 16
        self.n_v_heads = args.linear_num_value_heads  # 48
        self.head_dim = args.linear_head_dim  # 128
        self.conv_kernel = args.linear_conv_kernel  # 4
        self.eps = args.norm_eps
        self.max_batch_size = args.max_batch_size

        # --- Per-row head counts ---
        assert (
            self.n_v_heads % self.mesh_rows == 0
        ), f"n_v_heads={self.n_v_heads} must be divisible by mesh_rows={self.mesh_rows}"
        assert (
            self.n_k_heads % self.mesh_rows == 0
        ), f"n_k_heads={self.n_k_heads} must be divisible by mesh_rows={self.mesh_rows}"
        self.n_k_per_row = self.n_k_heads // self.mesh_rows  # 2
        self.n_v_per_row = self.n_v_heads // self.mesh_rows  # 6
        self.q_per_row = self.n_k_per_row * self.head_dim  # 256
        self.v_per_row = self.n_v_per_row * self.head_dim  # 768
        self.conv_per_row = self.q_per_row + self.q_per_row + self.v_per_row  # 1280

        # --- Compute kernel: HiFi4 + fp32 dest accumulation ---
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # --- Weights ---
        self._build_weights(weights_dict)

        # Host-side per-row conv weight blocks (for diagnostic tests).
        self._conv_w_host = self._build_conv_host_blocks(weights_dict)

        # --- Persistent buffers ---
        # Persistent zero pad for the conv-first-step branch (no per-call
        # ttnn.from_torch host write → trace-capture-safe).
        self._conv_zero_pad = self._build_conv_zero_pad()

        # Persistent recurrent state buffer: [B, H_per_row, K, V] sharded per
        # row by the per-row H dimension. The kernel reads/writes [B, H, K, V]
        # with H=n_v_per_row=6.
        self.dn_state_buffer = self._build_dn_state_buffer()

        # Persistent conv state buffer: [B, K-1, D_per_row] ROW_MAJOR, sharded
        # across rows by D (since each row holds its own conv channels).
        self.conv_state_buffer = self._build_conv_state_buffer()

    # ------------------------------------------------------------------
    # Weight construction helpers (ported from v1 _build_weights)
    # ------------------------------------------------------------------

    def _to_device(self, t, mapper, dtype=None, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            t,
            device=self.mesh_device,
            dtype=dtype or self.dtype,
            layout=layout,
            memory_config=memory_config,
            mesh_mapper=mapper,
        )

    def _resolve_weight(self, weights_dict, *candidate_keys):
        """Look up the first matching weight in weights_dict.

        load_checkpoints.py may emit the HF-native keys (``linear_attn.*``) or
        the v1-style stripped keys (e.g. ``in_proj_qkv.weight``). We accept
        either, in the order given.
        """
        for k in candidate_keys:
            if k in weights_dict:
                return weights_dict[k]
        raise KeyError(
            f"DeltaNet weight not found. Tried: {candidate_keys}. "
            f"Available keys: {sorted(weights_dict.keys())[:20]}..."
        )

    def _build_conv_host_blocks(self, sd):
        conv_w_src = self._resolve_weight(sd, "linear_attn.conv1d.weight", "conv1d.weight")
        conv_w = conv_w_src.squeeze(1)  # [10240, 4]
        conv_Q = conv_w[: self.n_k_heads * self.head_dim]
        conv_K = conv_w[self.n_k_heads * self.head_dim : 2 * self.n_k_heads * self.head_dim]
        conv_V = conv_w[2 * self.n_k_heads * self.head_dim :]
        blocks = []
        for i in range(self.mesh_rows):
            qc = conv_Q[i * self.q_per_row : (i + 1) * self.q_per_row]
            kc = conv_K[i * self.q_per_row : (i + 1) * self.q_per_row]
            vc = conv_V[i * self.v_per_row : (i + 1) * self.v_per_row]
            blocks.append(torch.cat([qc, kc, vc], dim=0))
        return blocks

    def _build_weights(self, sd):
        H = self.hidden_size
        mesh_rows = self.mesh_rows
        hd_k = self.head_dim
        n_k = self.n_k_heads
        n_v = self.n_v_heads
        q_per_row = self.q_per_row
        v_per_row = self.v_per_row

        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        row_shard_out = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(1, None), mesh_shape=self.cluster_shape)

        # -- QKV / Z / A / B projections --
        # HF Qwen3.6 uses a fused in_proj_qkvz weight in some snapshots; v1
        # consumed separately-split tensors via load_checkpoints. Support both.
        if any(k in sd for k in ("linear_attn.in_proj_qkv.weight", "in_proj_qkv.weight")):
            qkv_w = self._resolve_weight(sd, "linear_attn.in_proj_qkv.weight", "in_proj_qkv.weight")
            z_w = self._resolve_weight(sd, "linear_attn.in_proj_z.weight", "in_proj_z.weight")
            a_w = self._resolve_weight(sd, "linear_attn.in_proj_a.weight", "in_proj_a.weight")
            b_w = self._resolve_weight(sd, "linear_attn.in_proj_b.weight", "in_proj_b.weight")
        else:
            # Fused HF case: linear_attn.in_proj_qkvz.weight is
            # [n_k*hd + n_k*hd + n_v*hd + n_v*hd, H] = [16384, 5120] and
            # linear_attn.in_proj_ba.weight is [2*n_v, H] = [96, 5120].
            qkvz_w = self._resolve_weight(sd, "linear_attn.in_proj_qkvz.weight", "in_proj_qkvz.weight")
            ba_w = self._resolve_weight(sd, "linear_attn.in_proj_ba.weight", "in_proj_ba.weight")
            qkv_dim = 2 * n_k * hd_k + n_v * hd_k  # Q + K + V
            qkv_w = qkvz_w[:qkv_dim]
            z_w = qkvz_w[qkv_dim:]
            b_w = ba_w[:n_v]
            a_w = ba_w[n_v:]

        Q_w = qkv_w[: n_k * hd_k]
        K_w = qkv_w[n_k * hd_k : 2 * n_k * hd_k]
        V_w = qkv_w[2 * n_k * hd_k :]
        Q_w_T = Q_w.T.contiguous()
        K_w_T = K_w.T.contiguous()
        V_w_T = V_w.T.contiguous()
        Z_w_T = z_w.T.contiguous()

        self.w_q = self._to_device(Q_w_T, row_shard_out)  # per-row: [5120, 256]
        self.w_k = self._to_device(K_w_T, row_shard_out)  # per-row: [5120, 256]
        self.w_v = self._to_device(V_w_T, row_shard_out)  # per-row: [5120, 768]
        self.w_z = self._to_device(Z_w_T, row_shard_out)  # per-row: [5120, 768]

        a_w_T = a_w.T.contiguous()
        b_w_T = b_w.T.contiguous()
        self.w_a = self._to_device(a_w_T, row_shard_out)  # per-row: [5120, 6]
        self.w_b = self._to_device(b_w_T, row_shard_out)  # per-row: [5120, 6]

        # V2-11 (lever E): fused projection weights.
        # ShardTensor2dMesh(dims=(1, None)) distributes contiguous column
        # chunks across the 8 mesh-rows. Naively `cat([Q_w_T, K_w_T], dim=-1)`
        # would give rows 0..3 a Q-only chunk and rows 4..7 a K-only chunk,
        # which is wrong. We need to INTERLEAVE per-row so that each row's
        # contiguous shard is `[Q_row_i | K_row_i | ...]`.
        # This produces the same per-row matmul output as separate Q, K
        # matmuls — the fused matmul's output slice [0:256] is the row's
        # Q, [256:512] is the row's K, and so on.
        mesh_rows = self.mesh_rows
        # Q+K (per-row 256+256=512, tile-multiple)
        QK_rows = []
        for i in range(mesh_rows):
            q_row = Q_w_T[:, i * 256 : (i + 1) * 256]
            k_row = K_w_T[:, i * 256 : (i + 1) * 256]
            QK_rows.append(torch.cat([q_row, k_row], dim=-1))  # [5120, 512]
        QK_w_T_interleaved = torch.cat(QK_rows, dim=-1)  # [5120, 4096]
        self.w_qk = self._to_device(QK_w_T_interleaved, row_shard_out)

        # V+Z (per-row 768+768=1536, tile-multiple)
        VZ_rows = []
        for i in range(mesh_rows):
            v_row = V_w_T[:, i * 768 : (i + 1) * 768]
            z_row = Z_w_T[:, i * 768 : (i + 1) * 768]
            VZ_rows.append(torch.cat([v_row, z_row], dim=-1))  # [5120, 1536]
        VZ_w_T_interleaved = torch.cat(VZ_rows, dim=-1)  # [5120, 12288]
        self.w_vz = self._to_device(VZ_w_T_interleaved, row_shard_out)

        # V2-12 Lever 2: Q+K+V+Z fused single matmul.
        # Per-row 256+256+768+768=2048 (tile-multiple). Combines all 4 into one
        # matmul launch — saves 1 matmul / DeltaNet layer (V2-11 had QK+VZ as
        # 2 launches; V2-12 has QKVZ as 1). Same per-row interleave pattern
        # to preserve ``ShardTensor2dMesh(dims=(1, None))`` contiguous contract.
        QKVZ_rows = []
        for i in range(mesh_rows):
            q_row = Q_w_T[:, i * 256 : (i + 1) * 256]
            k_row = K_w_T[:, i * 256 : (i + 1) * 256]
            v_row = V_w_T[:, i * 768 : (i + 1) * 768]
            z_row = Z_w_T[:, i * 768 : (i + 1) * 768]
            QKVZ_rows.append(torch.cat([q_row, k_row, v_row, z_row], dim=-1))  # [5120, 2048]
        QKVZ_w_T_interleaved = torch.cat(QKVZ_rows, dim=-1)  # [5120, 16384]
        self.w_qkvz = self._to_device(QKVZ_w_T_interleaved, row_shard_out)

        # B+A (per-row 6+6=12, NOT tile-multiple but matmul pads internally)
        BA_rows = []
        for i in range(mesh_rows):
            b_row = b_w_T[:, i * self.n_v_per_row : (i + 1) * self.n_v_per_row]
            a_row = a_w_T[:, i * self.n_v_per_row : (i + 1) * self.n_v_per_row]
            BA_rows.append(torch.cat([b_row, a_row], dim=-1))  # [5120, 12]
        BA_w_T_interleaved = torch.cat(BA_rows, dim=-1)  # [5120, 96]
        self.w_ba = self._to_device(BA_w_T_interleaved, row_shard_out)

        # -- Conv1d weight: pre-interleave by row (Bug1 fix from v1) --
        conv_w_src = self._resolve_weight(sd, "linear_attn.conv1d.weight", "conv1d.weight")
        conv_w = conv_w_src.squeeze(1)
        conv_Q_w = conv_w[: n_k * hd_k]
        conv_K_w = conv_w[n_k * hd_k : 2 * n_k * hd_k]
        conv_V_w = conv_w[2 * n_k * hd_k :]

        chunks = []
        for i in range(mesh_rows):
            qc = conv_Q_w[i * q_per_row : (i + 1) * q_per_row]
            kc = conv_K_w[i * q_per_row : (i + 1) * q_per_row]
            vc = conv_V_w[i * v_per_row : (i + 1) * v_per_row]
            chunks.append(torch.cat([qc, kc, vc], dim=0))
        conv_w_interleaved = torch.cat(chunks, dim=0)  # [10240, 4]

        self.conv_weight_taps = []
        for tap in range(self.conv_kernel):
            tap_vec = conv_w_interleaved[:, tap]
            tap_3d = tap_vec.reshape(1, 1, mesh_rows * self.conv_per_row)
            row_shard_3d_chan = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(2, None), mesh_shape=self.cluster_shape)
            tap_tt = ttnn.from_torch(
                tap_3d,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=row_shard_3d_chan,
            )
            self.conv_weight_taps.append(tap_tt)

        # -- A_log, dt_bias: 3-D [1, 1, n_v_per_row] (Bug2 fix from v1) --
        A_log = self._resolve_weight(sd, "linear_attn.A_log", "A_log")
        dt_bias = self._resolve_weight(sd, "linear_attn.dt_bias", "dt_bias")

        A_log_3d = A_log.reshape(1, 1, n_v)
        dt_bias_3d = dt_bias.reshape(1, 1, n_v)
        row_shard_3d = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(2, None), mesh_shape=self.cluster_shape)
        self.A_log = self._to_device(A_log_3d, row_shard_3d, layout=ttnn.TILE_LAYOUT)
        self.dt_bias = self._to_device(dt_bias_3d, row_shard_3d, layout=ttnn.TILE_LAYOUT)
        # V2-11 (lever C) attempted to precompute -exp(A_log) once at init
        # to elide the per-step `exp` + `neg` (saves 2 ops × 48 DeltaNet
        # layers / step). Coherency broke when the precomputed tensor was
        # quantized at init time vs the runtime exp(bf8-stored A_log) → neg
        # chain — the per-step path picks up a fresh L1 intermediate that
        # the static precompute cannot replicate. Left out.

        # -- Norm weight (replicated, standard RMSNorm — Bug5 fix from v1) --
        # NB: ROW_MAJOR_LAYOUT requires bfloat16 (bfloat8_b requires TILE). The
        # norm weight is tiny (single head_dim vector); dtype has no perf impact.
        norm_w = self._resolve_weight(sd, "linear_attn.norm.weight", "norm.weight")
        norm_w_4d = norm_w.reshape(1, 1, self.head_dim // 32, 32)
        self.norm_weight = self._to_device(norm_w_4d, replicate, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # -- Output projection: [H, n_v*hd] → shard input dim across rows --
        out_proj_w = self._resolve_weight(sd, "linear_attn.out_proj.weight", "out_proj.weight")
        out_proj_w_T = out_proj_w.T.contiguous()
        row_shard_out0 = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, None), mesh_shape=self.cluster_shape)
        self.w_out = self._to_device(out_proj_w_T, row_shard_out0)  # per-row [768, 5120]

    # ------------------------------------------------------------------
    # Persistent buffer constructors
    # ------------------------------------------------------------------

    def _build_conv_zero_pad(self):
        """Replicated zero buffer used by _causal_conv1d_fir_mesh when no
        conv_state has been written yet. Sized for max_batch_size."""
        pad_torch = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_row,
            dtype=torch.bfloat16,
        )
        return ttnn.from_torch(
            pad_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _build_dn_state_buffer(self):
        """Persistent recurrent state buffer.

        Shape: ``[B, n_v_per_row, head_dim, head_dim]`` per device.
        The kernels expect ``[B, H, K, V]`` where H is the per-row head count
        (= 6) and K = V = head_dim (= 128). The buffer is replicated across
        the cluster_shape — each row holds the state for its own 6 V-heads;
        the 4 cols on a row are independent replicas (they all do the same
        work, see module docstring).

        V2-decode-debug: dtype is ``float32`` so the recurrent state does NOT
        round-trip through bf16 between decode steps.  Compounding across 48
        DeltaNet layers had previously pinched 64L decode logits PCC to 0.30;
        keeping the state at fp32 (and running the recurrent kernel at fp32
        internally — see ``ttnn_delta_rule_ops_fp32``) restores the precision
        floor.  Prefill (``chunk_gated_delta_rule_ttnn``) ALREADY returns the
        state at fp32, so this change is consistent with the prefill seed.

        V2-12 Lever 1: live in L1 (interleaved) instead of DRAM. The recurrent
        kernel ``to_memory_config(initial_state, L1)`` round-trip becomes a
        ~no-op (L1→L1 instead of DRAM→L1). Per-step savings: ~24 µs × 48 layers
        ≈ 1.15 ms / decode step. Each buffer is 6 × 128 × 128 × 4B = 384 KB
        fp32 tiled; 48 layers × 384 KB = 18 MB per chip — comfortably fits in
        L1 interleaved (distributed across ~130 cores ≈ ~140 KB / core).
        """
        state_torch = torch.zeros(
            self.max_batch_size,
            self.n_v_per_row,
            self.head_dim,
            self.head_dim,
            dtype=torch.float32,
        )
        return ttnn.from_torch(
            state_torch,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _build_conv_state_buffer(self):
        """Persistent conv-state buffer.

        Shape: ``[B, K-1, conv_per_row]`` ROW_MAJOR per device — same layout
        as ``_conv_zero_pad``. Each row gets its own conv_per_row=1280
        channels (sharded by the conv-weight loader); the buffer itself is
        replicated and only read/written through the per-row conv path.
        """
        buf_torch = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_row,
            dtype=torch.bfloat16,
        )
        return ttnn.from_torch(
            buf_torch,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def clear_state(self):
        """Zero both persistent buffers (fresh-sequence entry point)."""
        # dn_state_buffer is fp32 (see _build_dn_state_buffer docstring); use a
        # matching fp32 source for the in-place copy.
        zero_state = torch.zeros(
            self.max_batch_size, self.n_v_per_row, self.head_dim, self.head_dim, dtype=torch.float32
        )
        zero_conv = torch.zeros(self.max_batch_size, self.conv_kernel - 1, self.conv_per_row, dtype=torch.bfloat16)
        new_state = ttnn.from_torch(
            zero_state,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        new_conv = ttnn.from_torch(
            zero_conv,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy(new_state, self.dn_state_buffer)
        ttnn.copy(new_conv, self.conv_state_buffer)
        new_state.deallocate(True)
        new_conv.deallocate(True)

    # ------------------------------------------------------------------
    # Forward-stage helpers (ported verbatim from v1; same math)
    # ------------------------------------------------------------------

    def _project_inputs(self, x):
        """V2-12 Lever 2: collapse Q+K+V+Z into a single fused matmul + B+A.

        Q+K+V+Z → w_qkvz (per-row output dim 256+256+768+768=2048, tile-multiple)
        B+A     → w_ba (output dim 12, padded to 32 internally)

        Total 2 matmul launches (down from V2-11's 3 launches: QK + VZ + BA).
        Saves 1 matmul launch / DeltaNet layer × 48 layers ≈ ~1 ms / decode step.
        Same per-row interleave pattern as V2-11 to preserve the
        ``ShardTensor2dMesh(dims=(1, None))`` contiguous-shard contract — naive
        cat would steer Q-only chunks to rows 0..3 and silently break coherency.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        ck = self.compute_kernel
        # Q+K+V+Z fused
        qkvz = ttnn.linear(x, self.w_qkvz, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        out_rank = len(qkvz.shape)
        q_per_row = self.q_per_row  # 256
        v_per_row = self.v_per_row  # 768
        # Slice Q | K | V | Z along the last dim.
        # Per-row layout (after the ShardTensor2dMesh split): [Q_256 | K_256 | V_768 | Z_768]
        # repeated across the 8 rows. ttnn.slice over the global dim picks the
        # same per-row offset on each row.
        if out_rank == 3:
            B_, T_, _ = list(qkvz.shape)
            q = ttnn.slice(qkvz, [0, 0, 0], [B_, T_, q_per_row], memory_config=mem)
            k = ttnn.slice(qkvz, [0, 0, q_per_row], [B_, T_, 2 * q_per_row], memory_config=mem)
            v = ttnn.slice(qkvz, [0, 0, 2 * q_per_row], [B_, T_, 2 * q_per_row + v_per_row], memory_config=mem)
            z = ttnn.slice(
                qkvz,
                [0, 0, 2 * q_per_row + v_per_row],
                [B_, T_, 2 * q_per_row + 2 * v_per_row],
                memory_config=mem,
            )
        elif out_rank == 4:
            B_, D1_, T_, _ = list(qkvz.shape)
            q = ttnn.slice(qkvz, [0, 0, 0, 0], [B_, D1_, T_, q_per_row], memory_config=mem)
            k = ttnn.slice(qkvz, [0, 0, 0, q_per_row], [B_, D1_, T_, 2 * q_per_row], memory_config=mem)
            v = ttnn.slice(qkvz, [0, 0, 0, 2 * q_per_row], [B_, D1_, T_, 2 * q_per_row + v_per_row], memory_config=mem)
            z = ttnn.slice(
                qkvz,
                [0, 0, 0, 2 * q_per_row + v_per_row],
                [B_, D1_, T_, 2 * q_per_row + 2 * v_per_row],
                memory_config=mem,
            )
        else:
            raise RuntimeError(f"Unexpected qkvz rank {out_rank}: shape={qkvz.shape}")
        qkvz.deallocate(True)

        # B+A fused (note: matches in_proj_ba layout which is b|a, not a|b)
        ba = ttnn.linear(x, self.w_ba, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        n_v_per_row = self.n_v_per_row
        if out_rank == 3:
            B_, T_, _ = list(ba.shape)
            b = ttnn.slice(ba, [0, 0, 0], [B_, T_, n_v_per_row], memory_config=mem)
            a = ttnn.slice(ba, [0, 0, n_v_per_row], [B_, T_, 2 * n_v_per_row], memory_config=mem)
        else:
            B_, D1_, T_, _ = list(ba.shape)
            b = ttnn.slice(ba, [0, 0, 0, 0], [B_, D1_, T_, n_v_per_row], memory_config=mem)
            a = ttnn.slice(ba, [0, 0, 0, n_v_per_row], [B_, D1_, T_, 2 * n_v_per_row], memory_config=mem)
        ba.deallocate(True)

        return q, k, v, z, a, b

    def _apply_conv_and_split(self, q, k, v, B, T, conv_state=None):
        mem = ttnn.DRAM_MEMORY_CONFIG
        mixed = ttnn.concat([q, k, v], dim=-1, memory_config=mem)
        mixed_conv, new_conv_state = _causal_conv1d_fir_mesh(
            mixed,
            self.conv_weight_taps,
            self.conv_kernel,
            self.mesh_device,
            memory_config=mem,
            conv_state=conv_state,
            conv_state_zero_pad=self._conv_zero_pad,
        )
        mixed.deallocate(True)

        q_conv = ttnn.slice(mixed_conv, [0, 0, 0], [B, T, self.q_per_row], memory_config=mem)
        k_conv = ttnn.slice(mixed_conv, [0, 0, self.q_per_row], [B, T, 2 * self.q_per_row], memory_config=mem)
        v_conv = ttnn.slice(mixed_conv, [0, 0, 2 * self.q_per_row], [B, T, self.conv_per_row], memory_config=mem)
        mixed_conv.deallocate(True)
        return q_conv, k_conv, v_conv, new_conv_state

    def _compute_beta_g(self, b, a, B, T):
        """V2-11 (lever C): attempted unary-chain fusion, NOT LANDED.

        Original (6 ops):
          beta = sigmoid(b); a_biased = add(a, dt_bias); sp = softplus(a_biased)
          A_exp = exp(A_log); neg_A_exp = neg(A_exp); g = multiply(neg_A_exp, sp)

        Attempts:
          (i)  Precompute neg_exp(A_log) once at init, multiply directly.
          (ii) Fuse add+softplus via activations=[SOFTPLUS].
          (iii) Fuse exp+neg into multiply via input_tensor_a_activations.
        All three variants broke coherency — the generated text became
        gibberish (~80 alpha chars of mojibake). The fused-activation path
        evaluates softplus / exp / neg at a slightly different precision
        than the standalone unary launches (likely L1-vs-DRAM intermediate
        dtype or in-tile activation chain rounding), and the 48 DeltaNet
        layers compound that drift past the coherency tolerance. Reverted
        to the verified per-op pattern; the device-time saving from this
        lever alone (~0.4 ms / step) was not worth the coherency cost.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        beta = ttnn.sigmoid(b, memory_config=mem)
        a_biased = ttnn.add(a, self.dt_bias, memory_config=mem)
        sp = ttnn.softplus(a_biased, memory_config=mem)
        A_exp = ttnn.exp(self.A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
        g = ttnn.multiply(ttnn.neg(A_exp, memory_config=mem), sp, memory_config=mem)
        return beta, g

    def _gqa_expand_q_k(self, q, k, B, T):
        ratio = self.n_v_per_row // self.n_k_per_row
        mem = ttnn.DRAM_MEMORY_CONFIG
        q_e = ttnn.repeat_interleave(q, ratio, dim=2, memory_config=mem)
        k_e = ttnn.repeat_interleave(k, ratio, dim=2, memory_config=mem)
        return q_e, k_e

    def _apply_norm_gated(self, core_out, z, B, T):
        """V2-11 (lever D): attempted silu(z) into multiply fusion, NOT LANDED.

        The fused path
          out = multiply(out, z, input_tensor_b_activations=[SILU])
        ran at the same speed (~77.71 ms / step vs 77.77 baseline) but
        the compile-pass token shifted from 248068 (<think>) → 232, and
        subsequent decode tokens became gibberish (~96 alpha chars of
        mojibake across 32 generated tokens). The pre-multiply SILU
        activation evaluates at slightly different precision than the
        standalone unary launch (likely the fused activation pipeline
        sees an L1-vs-DRAM intermediate dtype it would not otherwise
        hit), and the 48 DeltaNet layer compounding pushes the output
        past tolerance. Reverted to the verified two-op pattern.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.rms_norm(
            core_out,
            weight=self.norm_weight,
            epsilon=self.eps,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel,
        )
        z_silu = ttnn.silu(z, memory_config=mem)
        out = ttnn.multiply(out, z_silu, memory_config=mem)
        z_silu.deallocate(True)
        out = ttnn.reshape(out, [B, T, self.v_per_row])
        return out

    def _output_proj_and_reduce(self, out_flat, B, T):
        mem = ttnn.DRAM_MEMORY_CONFIG
        # qwen3.6 residual-stream dtype lock (olmo session-11 lesson):
        # DeltaNet runs on 48 of 64 layers; its output projection writes
        # directly into the post-attention residual add in TtTransformerBlock.
        # Even though out_proj weight stays bfloat8_b (forcing weights to bf16
        # dropped layer-0 PCC 0.999 → 0.84 — see _build_weights comment),
        # the OUTPUT activation must stay bfloat16 so the residual stream
        # does not get quantized at every full-layer boundary.
        partial = ttnn.linear(
            out_flat,
            self.w_out,
            dtype=ttnn.bfloat16,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel,
        )
        # V2-11 (lever B): collapse `all_gather + fast_reduce_nc` (2 ops) into
        # a single `ttnn.all_reduce` (1 op). 48 DeltaNet layers × 1 saved op
        # per decode step = 48 fewer device ops + lower per-CCL launch
        # latency (single barrier semaphore vs two). The math is identical
        # (Sum reduction across cluster_axis=0). Topology defaults to the
        # Linear fabric configured for BH GLX 8x4.
        reduced = ttnn.all_reduce(
            partial,
            cluster_axis=0,
            num_links=1,
            memory_config=mem,
        )
        partial.deallocate(True)
        return reduced

    # ------------------------------------------------------------------
    # Public API — matches TtLlamaAttention surface
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        kv_cache=None,
        batch_size=1,
    ):
        if mode == "prefill":
            return self.forward_prefill(x, current_pos, rot_mats, kv_cache=kv_cache, page_table=page_table)
        else:
            return self.forward_decode(x, current_pos, rot_mats, kv_cache=kv_cache, page_table=page_table)

    def forward_prefill(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        kv_cache=None,
        page_table=None,
        **kwargs,
    ):
        """Prefill (T>1) using the chunked delta-rule kernel.

        ``rot_mats``, ``kv_cache``, ``page_table`` are ignored — DeltaNet has
        no RoPE and no KV cache; state lives in self.dn_state_buffer.
        """
        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape

        # 1. Projections
        q, k, v, z, a, b = self._project_inputs(x)

        # 2. Conv1d + split (conv_state from previous prefill chunk or fresh)
        q_conv, k_conv, v_conv, new_conv_state = self._apply_conv_and_split(q, k, v, B, T, conv_state=None)
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)

        # 3. Reshape to per-head layout
        q_h = ttnn.reshape(q_conv, [B, T, self.n_k_per_row, self.head_dim])
        k_h = ttnn.reshape(k_conv, [B, T, self.n_k_per_row, self.head_dim])
        v_h = ttnn.reshape(v_conv, [B, T, self.n_v_per_row, self.head_dim])
        z_h = ttnn.reshape(z, [B, T, self.n_v_per_row, self.head_dim])
        q_conv.deallocate(True)
        k_conv.deallocate(True)
        v_conv.deallocate(True)

        # 4. beta and g
        beta, g = self._compute_beta_g(b, a, B, T)
        b.deallocate(True)
        a.deallocate(True)

        # 5. GQA expand q, k
        q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h, B, T)
        q_h.deallocate(True)
        k_h.deallocate(True)

        # 6. Chunked delta rule kernel
        core_out, new_state = chunk_gated_delta_rule_ttnn(
            q=q_exp,
            k=k_exp,
            v=v_h,
            beta=beta,
            g=g,
            chunk_size=32,
            initial_state=None,
            device=self.mesh_device,
        )
        q_exp.deallocate(True)
        k_exp.deallocate(True)
        v_h.deallocate(True)
        beta.deallocate(True)
        g.deallocate(True)

        # 7. GroupRMSNormGated
        out = self._apply_norm_gated(core_out, z_h, B, T)
        core_out.deallocate(True)
        z_h.deallocate(True)

        # 8. Output projection + all-reduce across rows
        output = self._output_proj_and_reduce(out, B, T)
        out.deallocate(True)

        # 9. Write the final state into the persistent buffers (in place) so
        # the subsequent decode steps see them. ttnn.copy preserves the buffer
        # address — required for trace replay.
        ttnn.copy(new_state, self.dn_state_buffer)
        ttnn.copy(new_conv_state, self.conv_state_buffer)
        new_state.deallocate(True)
        new_conv_state.deallocate(True)

        return output

    def forward_decode(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        kv_cache=None,
        page_table=None,
        **kwargs,
    ):
        """Decode (T=1) using the recurrent delta-rule kernel.

        Reads recurrent_state / conv_state from the persistent buffers and
        writes the new state back into the same buffers in place — trace-safe.
        """
        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
        assert T == 1, f"Decode expects T=1, got T={T}"

        # 1. Projections
        q, k, v, z, a, b = self._project_inputs(x)

        # 2. Conv1d + split — read persistent conv_state buffer
        q_conv, k_conv, v_conv, new_conv_state = self._apply_conv_and_split(
            q, k, v, B, T, conv_state=self.conv_state_buffer
        )
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)

        # 3. Reshape to per-head layout
        q_h = ttnn.reshape(q_conv, [B, T, self.n_k_per_row, self.head_dim])
        k_h = ttnn.reshape(k_conv, [B, T, self.n_k_per_row, self.head_dim])
        v_h = ttnn.reshape(v_conv, [B, T, self.n_v_per_row, self.head_dim])
        z_h = ttnn.reshape(z, [B, T, self.n_v_per_row, self.head_dim])
        q_conv.deallocate(True)
        k_conv.deallocate(True)
        v_conv.deallocate(True)

        # 4. beta and g
        beta, g = self._compute_beta_g(b, a, B, T)
        b.deallocate(True)
        a.deallocate(True)

        # 5. GQA expand q, k
        q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h, B, T)
        q_h.deallocate(True)
        k_h.deallocate(True)

        # 6. Recurrent delta rule — read persistent recurrent state buffer.
        # V2-decode-debug: use the fp32-state fork so the recurrent state stays
        # at fp32 across decode steps (eliminates the per-layer bf16 round-trip
        # that was compounding to 64L PCC 0.30 — see ttnn_delta_rule_ops_fp32).
        core_out, new_state = recurrent_gated_delta_rule_ttnn_fp32(
            q=q_exp,
            k=k_exp,
            v=v_h,
            beta=beta,
            g=g,
            initial_state=self.dn_state_buffer,
            device=self.mesh_device,
        )
        q_exp.deallocate(True)
        k_exp.deallocate(True)
        v_h.deallocate(True)
        beta.deallocate(True)
        g.deallocate(True)

        # 7. GroupRMSNormGated
        out = self._apply_norm_gated(core_out, z_h, B, T)
        core_out.deallocate(True)
        z_h.deallocate(True)

        # 8. Output projection + all-reduce across rows
        output = self._output_proj_and_reduce(out, B, T)
        out.deallocate(True)

        # 9. In-place write of new state into persistent buffers (trace-safe).
        ttnn.copy(new_state, self.dn_state_buffer)
        ttnn.copy(new_conv_state, self.conv_state_buffer)
        new_state.deallocate(True)
        new_conv_state.deallocate(True)

        return output

    # ------------------------------------------------------------------
    # Diagnostic accessor (test_deltanet_sharding_correctness)
    # ------------------------------------------------------------------
    def get_conv_weight_row(self, row_i):
        """Return host-side conv weight block for row_i [1280, 4]."""
        return self._conv_w_host[row_i]
