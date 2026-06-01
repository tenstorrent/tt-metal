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

import os
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy_v2.tt.gdn_chunk_ops_seq import chunk_gated_delta_rule_seq

# Prefill chunk kernel: use the qwen35-27b vendored kernel (clip-before-exp +
# decay-offset normalization + HiFi2/fp32) instead of the shared experimental
# one, which produces Inf on one Galaxy mesh row at T=4096 (bf16 rounding over
# the 4096-token decay cumsum violates the g<=0 invariant -> exp overflow).
# See the module docstring of qwen35_chunk_delta_rule_ops for details.
from models.demos.qwen3_6_galaxy_v2.tt.qwen35_chunk_delta_rule_ops import (
    chunk_gated_delta_rule_ttnn,
    create_chunk_masks,
    l2_norm_ttnn,
)
from models.demos.qwen3_6_galaxy_v2.tt.ttnn_delta_rule_ops_fp32 import (
    _fp32_compute_cfg_hifi4,
    recurrent_gated_delta_rule_ttnn_fp32,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_ttnn,  # kept for back-compat; not used in fp32 path
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
    l2_norm_ttnn,
)


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
        # V4: follow x_rm dtype so fp32 activation mode doesn't trip
        # ttnn.concat([pad, x_rm])'s same-dtype assertion.
        x_dtype = x_rm.dtype
        pad_torch_dtype = torch.float32 if x_dtype == ttnn.float32 else torch.bfloat16
        pad_torch = torch.zeros(B, kernel_size - 1, D, dtype=pad_torch_dtype)
        pad = ttnn.from_torch(
            pad_torch,
            device=mesh_device,
            dtype=x_dtype,
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
        use_tt_lang_beta_g: bool = False,
        use_tt_lang_recurrent: bool = False,
        use_tt_lang_recurrent_v2: bool = False,
        use_tt_lang_recurrent_v3: bool = False,
        **kwargs,
    ):
        super().__init__()
        # V2-16: optional fused tt-lang beta/g kernel for decode (T=1).
        # Env override is preferred (matches QWEN36_DELTA_LAR / QWEN36_FULLATTN_LAR
        # pattern) so the test plumbing doesn't need to thread the bool through
        # the TtTransformer / decoder layer hierarchy.
        env_flag = os.environ.get("QWEN36_TT_LANG_BETA_G", "").strip()
        self.use_tt_lang_beta_g = use_tt_lang_beta_g or (env_flag == "1")
        # V2-18: optional partial tt-lang recurrent kernel (per-head state
        # update only, readout matmul remains external).
        env_flag_rec = os.environ.get("QWEN36_TT_LANG_RECURRENT", "").strip()
        self.use_tt_lang_recurrent = use_tt_lang_recurrent or (env_flag_rec == "1")
        # V2-17b: full multi-head + multi-core + fused-readout V3 kernel
        # (recurrent_delta_rule_v3). One ttnn.generic_op launch handles all 6
        # heads on 24 cores. Mutually exclusive with V2-18 path.
        env_flag_rec_v2 = os.environ.get("QWEN36_TT_LANG_RECURRENT_V2", "").strip()
        self.use_tt_lang_recurrent_v2 = use_tt_lang_recurrent_v2 or (env_flag_rec_v2 == "1")
        if self.use_tt_lang_recurrent_v2:
            self.use_tt_lang_recurrent = False
        # V2-17d: full multi-head batched + readout-fused + in-place state
        # kernel (recurrent_delta_rule_v3). ONE ttnn.generic_op launch handles
        # all 6 heads on 24 cores. Mutually exclusive with V2-17/V2-17c paths.
        env_flag_rec_v3 = os.environ.get("QWEN36_TT_LANG_RECURRENT_V3", "").strip()
        self.use_tt_lang_recurrent_v3 = use_tt_lang_recurrent_v3 or (env_flag_rec_v3 == "1")
        if self.use_tt_lang_recurrent_v3:
            self.use_tt_lang_recurrent = False
            self.use_tt_lang_recurrent_v2 = False

        # GDN decode create-heads fusion: produce per-head q/k/v/beta/g already
        # in the recurrent core's [B, H, T, D] layout so the pure-ttnn fp32
        # recurrent core can SKIP its 5 input transposes. PCC-safe at T=1
        # (reshape-then-transpose == direct reshape for a T=1 contiguous tensor).
        # Only wired into the pure-ttnn else-branch (recurrent_gated_delta_rule_
        # ttnn_fp32); the tt-lang kernel paths keep the old [B,T,H,D] layout.
        env_flag_fused_heads = os.environ.get("QWEN36_DN_FUSED_HEADS", "1").strip()
        self.use_dn_fused_heads = env_flag_fused_heads == "1"

        self.mesh_device = mesh_device
        self.args = args
        self.model_config = args.get_model_config()
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

        # Prefill chunk-rule config. Match qwen35-27b/P150 (gdn_chunk_size=64):
        # chunk_size=64 halves the number of sequential chunk iterations vs 32
        # at a given ISL (4k -> 64 steps instead of 128). Build the triu/tril/
        # eye masks ONCE here and reuse them every layer/call (the P150 path
        # passes cached_masks; rebuilding them per call host-uploads 5 tensors
        # on every one of the 48 linear layers, every prefill).
        # NOTE: chunk_size=64 (P150's value) measured SLOWER here (warm prefill
        # 4k 27.3s -> 31s) — the larger intra-chunk matmuls outweigh the halved
        # iteration count on the Galaxy shapes. Keep 32 (faster + coherence-
        # validated); cached_masks below is still a free win.
        self.prefill_chunk_size = int(os.environ.get("QWEN36_GDN_CHUNK_SIZE", "32"))
        self._chunk_masks = create_chunk_masks(self.prefill_chunk_size, mesh_device)

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
        # V2-CONFIG-C: optionally downgrade to HiFi2 to match llama70b's
        # production-tuned config (its w_out / wqkv all use HiFi2). Same env
        # var as FA so a single toggle flips both attention block types.
        _attn_hifi2 = os.environ.get("QWEN36_ATTN_HIFI2", "0") == "1"
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2 if _attn_hifi2 else ttnn.MathFidelity.HiFi4,
            math_approx_mode=_attn_hifi2,
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

        # V2-16: tt-lang fused beta/g kernel state (persistent buffers + program
        # descriptor). Only constructed when the flag is on — keeps the safe
        # 6-op path zero-cost when disabled.
        self._beta_g_kernel_state = None
        if self.use_tt_lang_beta_g:
            self._beta_g_kernel_state = self._build_beta_g_kernel_state()

        # V2-18: partial tt-lang recurrent kernel state. The kernel handles
        # state_new[h] = state[h] * decay[h] + outer(k_col, v_row) * beta[h]
        # for one head per launch; readout (q @ state_new) is external. Only
        # constructed when the flag is on so the safe path stays free.
        self._recurrent_kernel_state = None
        if self.use_tt_lang_recurrent:
            self._recurrent_kernel_state = self._build_recurrent_kernel_state()

        # V2-17b: multi-head V3 kernel state (24-core, fused readout).
        self._recurrent_v2_kernel_state = None
        if self.use_tt_lang_recurrent_v2:
            self._recurrent_v2_kernel_state = self._build_recurrent_v2_kernel_state()

        # V2-17d: V3 multi-head batched kernel state. Persistent per-head
        # broadcast tiles (ones_per_head [1, H=6, 32, 32]) live in DRAM so
        # they don't compete for L1 with the dn_state_buffer.
        self._recurrent_v3_kernel_state = None
        if self.use_tt_lang_recurrent_v3:
            self._recurrent_v3_kernel_state = self._build_recurrent_v3_kernel_state()

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
        # V2-DN-TP: 2D-TP DeltaNet — rows split output dim (heads), cols split
        # input dim (hidden).  Input contract becomes col-sharded H/4 = 1280
        # per chip; col-axis all_reduce after the matmul completes the inner-
        # product sum.  Math verified bit-equivalent to 1D-TP in
        # test_2d_tp_matmul_math.py (V2-DN-TP-0).
        row_shard_out = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(1, 0), mesh_shape=self.cluster_shape)

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

        # T1: width-sharded GroupRMSNormGated gamma.  The sharded
        # ``ttnn.rms_norm`` gives each core a contiguous slice of the gamma
        # matching its shard width, so for the per-head (head_dim-wide) grouped
        # norm with one head per core we need gamma tiled across all heads:
        # a [1,1,1,v_per_row] vector = the head_dim weight repeated n_v_per_row
        # times.  Replicated across the mesh (same as norm_weight).
        norm_w_tiled = norm_w.reshape(-1).repeat(self.n_v_per_row).reshape(1, 1, 1, self.v_per_row)
        self.norm_weight_sharded = self._to_device(
            norm_w_tiled, replicate, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        # -- Output projection: [H, n_v*hd] -- V2-DN-TP 2D-TP layout --
        # rows split dim 0 (input n_v*hd=6144 → 768 per row, head subset)
        # cols split dim 1 (output H=5120 → 1280 per col, col-sharded output)
        # Per chip shape: [768, 1280].  After row-axis all_reduce the output
        # is naturally col-sharded H/4 — matches the V2-TP residual contract.
        out_proj_w = self._resolve_weight(sd, "linear_attn.out_proj.weight", "out_proj.weight")
        out_proj_w_T = out_proj_w.T.contiguous()
        row_shard_out0 = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=self.cluster_shape)
        self.w_out = self._to_device(out_proj_w_T, row_shard_out0)  # per-chip [768, 1280]

        # ------------------------------------------------------------------
        # Fused-prefill kernel: DEFAULT ON (disable with QWEN36_GDN_FUSED_PREFILL=0).
        # Port of the P150 gdn_prefill_fused path: one ttnn.generic_op launch
        # for the whole-sequence recurrence (vs the op-heavy chunk reference).
        # Galaxy 8-row head sharding => half of P150's per-chip head counts.
        # Made default because in EAGER execution it is the wall-clock win: it
        # collapses ~856 ops/layer -> ~12, cutting per-op dispatch overhead
        # (~814 -> ~502 ms/layer @4k, ~2.2x @128k). Tradeoff: block PCC 0.9854
        # (< the 0.99 bar; e2e demos coherent) and higher DEVICE KERNEL DURATION
        # (one 111 ms 6-core GenericOp vs chunk's 21 ms across many cores) -- so
        # revisit this default if/when prefill is traced (chunk wins traced).
        # ------------------------------------------------------------------
        # Seq parallel-scan prefill (C++ gated_delta_attn_seq kernel, ported from
        # the P150 qwen35 path). Default ON — takes precedence over the fused and
        # pure-TTNN chunk paths. Disable with QWEN36_GDN_SEQ_PREFILL=0.
        self._use_seq_prefill = os.environ.get("QWEN36_GDN_SEQ_PREFILL", "1") != "0"
        self._seq_prefill_chunk_size = int(os.environ.get("QWEN36_SEQ_CHUNK", "128"))
        # Pre-build chunk masks at init (NOT lazily mid-forward — see _build_seq_masks: lazy
        # mid-prefill build corrupted DRAM under galaxy TP=32).
        self._seq_masks = None
        if self._use_seq_prefill:
            self._build_seq_masks()

        self._use_fused_prefill = os.environ.get("QWEN36_GDN_FUSED_PREFILL", "1") != "0"
        if self._use_fused_prefill:
            import math as _math

            from models.demos.qwen3_6_galaxy_v2.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused as _gpf

            self._gdn_prefill_fused = _gpf
            self._fused_Nv_TP = self.n_v_per_row  # 6
            self._fused_Nk_TP = self.n_k_per_row  # 2
            self._fused_repeat = self.n_v_per_row // self.n_k_per_row  # 3
            self._fused_key_dim_tp = self.n_k_per_row * self.head_dim  # 256
            self._fused_qkv_dim_tp = 2 * self.q_per_row + self.v_per_row  # 256+256+768 = 1280
            # -exp(A_log), built in bf16, same [1,1,6] row sharding as A_log.
            self.neg_exp_A = ttnn.neg(ttnn.exp(ttnn.typecast(self.A_log, ttnn.bfloat16)))

            def _scalar_tile(val):
                return ttnn.from_torch(
                    torch.full((1, 1, 1), float(val), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )

            self._fused_scale_tt = _scalar_tile(self.head_dim**-0.5)
            self._fused_rms_scale_tt = _scalar_tile(_math.sqrt(self.head_dim))
            self._fused_rms_eps_tt = _scalar_tile(self.head_dim * self.eps)
            # Post-kernel ttnn.rms_norm weight: [1, 1, Dv=head_dim] bf16 (replicated).
            self._fused_norm_w = self._to_device(
                norm_w.reshape(1, 1, self.head_dim), replicate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

    # ------------------------------------------------------------------
    # Persistent buffer constructors
    # ------------------------------------------------------------------

    def _build_conv_zero_pad(self):
        """Replicated zero buffer used by _causal_conv1d_fir_mesh when no
        conv_state has been written yet. Sized for max_batch_size.

        V4: dtype follows self.dtype so fp32-weights mode produces an fp32 pad
        that matches the fp32 x_rm at the concat site.
        """
        is_fp32 = self.dtype == ttnn.float32
        pad_torch = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_row,
            dtype=torch.float32 if is_fp32 else torch.bfloat16,
        )
        return ttnn.from_torch(
            pad_torch,
            device=self.mesh_device,
            dtype=ttnn.float32 if is_fp32 else ttnn.bfloat16,
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
        # Seq parallel-scan prefill (gated_delta_attn_seq) needs ~157 KB of L1 on
        # the head cores for its C×C circular buffers, which clashes with the 48×
        # 384 KB L1-interleaved dn_state buffers. When the seq path is active, hold
        # the state in DRAM instead (prefill computes state fresh; decode already
        # does to_memory_config(initial_state, L1), so this only costs the L1→L1
        # round-trip optimization, ~1.15 ms/decode step).
        _seq = os.environ.get("QWEN36_GDN_SEQ_PREFILL", "1") != "0"
        _state_mem = ttnn.DRAM_MEMORY_CONFIG if _seq else ttnn.L1_MEMORY_CONFIG
        return ttnn.from_torch(
            state_torch,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_state_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _build_conv_state_buffer(self):
        """Persistent conv-state buffer.

        Shape: ``[B, K-1, conv_per_row]`` ROW_MAJOR per device — same layout
        as ``_conv_zero_pad``. Each row gets its own conv_per_row=1280
        channels (sharded by the conv-weight loader); the buffer itself is
        replicated and only read/written through the per-row conv path.

        V4: dtype follows self.dtype so fp32-weights mode keeps the buffer
        fp32 — required because `ttnn.copy(new_conv_state, self.conv_state_buffer)`
        cannot do dtype conversion on ROW_MAJOR tensors.
        """
        is_fp32 = self.dtype == ttnn.float32
        buf_torch = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_row,
            dtype=torch.float32 if is_fp32 else torch.bfloat16,
        )
        return ttnn.from_torch(
            buf_torch,
            device=self.mesh_device,
            dtype=ttnn.float32 if is_fp32 else ttnn.bfloat16,
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
        # V4: match conv_state_buffer dtype (which follows self.dtype)
        is_fp32 = self.dtype == ttnn.float32
        zero_conv = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_row,
            dtype=torch.float32 if is_fp32 else torch.bfloat16,
        )
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
        # V2-DN-TP: x is now COL-SHARDED H/4 = 1280 per chip (was full-H 5120).
        # The matmul produces a partial output on the input dim; complete the
        # sum with an all_reduce on cluster_axis=1 (4-way col ring).
        # QWEN36_PREFILL_OPT: pass the tuned 2D-TP prefill program config on the
        # QKVZ matmul (T>1 only, so decode is untouched). _project_inputs is
        # shared by prefill + decode.
        _T_qkvz = x.shape[-2]
        if os.environ.get("QWEN36_PREFILL_OPT", "0") == "1" and _T_qkvz > 1:
            # Reshape long T into 2048-wide chunks (llama70b pattern) so the 2D
            # matmul's M (per_core_M=8) fits the 10-row grid; otherwise M=T
            # tiles / 8 exceeds the grid rows for T>2048.
            _xh = x.shape[-1]
            x_mm = ttnn.reshape(x, [1, _T_qkvz // 2048, 2048, _xh]) if _T_qkvz > 2048 else x
            qkvz_partial = ttnn.linear(
                x_mm,
                self.w_qkvz,
                dtype=self.dtype,
                memory_config=mem,
                compute_kernel_config=ck,
                program_config=self.model_config["QWEN36_DN_QKVZ_PREFILL_PROGCFG"](_T_qkvz),
            )
            if x_mm is not x:
                x_mm.deallocate(True)
            if len(qkvz_partial.shape) == 4:
                qkvz_partial = ttnn.reshape(qkvz_partial, [1, _T_qkvz, qkvz_partial.shape[-1]])
        else:
            qkvz_partial = ttnn.linear(x, self.w_qkvz, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        # QWEN36_ABLATE_CCL: skip col-reduce (timing ablation; garbage values).
        if os.environ.get("QWEN36_ABLATE_CCL", "0") == "1":
            qkvz = qkvz_partial
        else:
            qkvz = ttnn.all_reduce(qkvz_partial, cluster_axis=1, num_links=1, memory_config=mem)
            qkvz_partial.deallocate(True)
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
        # V2-DN-TP: col-axis all_reduce to complete the inner-product sum.
        ba_partial = ttnn.linear(x, self.w_ba, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        if os.environ.get("QWEN36_ABLATE_CCL", "0") == "1":
            ba = ba_partial  # skip col-reduce (timing ablation)
        else:
            ba = ttnn.all_reduce(ba_partial, cluster_axis=1, num_links=1, memory_config=mem)
            ba_partial.deallocate(True)
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

    # ------------------------------------------------------------------
    # V2-16: fused tt-lang beta/g kernel (decode-only, T=1)
    # ------------------------------------------------------------------
    # The kernel emitted by ``tt/kernels/beta_g_kernel.py`` fuses the 6-op
    # beta/g chain (sigmoid + add + softplus + exp + neg + multiply) into a
    # single ttnn.generic_op launch on a 1×1 tile core grid. At decode the
    # input tensors ``b``, ``a`` (shape [1,1,6]) and the weight tensors
    # ``dt_bias``, ``A_log`` (shape [1,1,6]) all tilize to a single 32×32 tile
    # per device with the valid data in row 0, so no host-side broadcast is
    # needed — the per-tile element-wise math is identical to the TTNN chain's
    # implicit-broadcast result. V2-15B validated PCC 1.0 / 0.9999 + 4.96×
    # standalone speedup at this exact shape.
    #
    # Persistent ``beta_out`` / ``g_out`` / ``ones`` buffers live on-device
    # across decode steps so trace replay sees fixed addresses; the program
    # descriptor is rebuilt per call (the buffer addresses do not move across
    # calls so the descriptor is trace-safe).

    _BETA_G_KERNELS_DIR = Path(__file__).resolve().parent / "kernels" / "beta_g"
    _BETA_G_NUM_TENSORS = 7
    _BETA_G_CB_PAGE_SIZE = 2048  # bf16, 32×32 tile
    _BETA_G_CB_TOTAL_SIZE = 4096  # double-buffered
    # Kernel author script emitted these tensor indices (see _runner_emitted.py).
    _BETA_G_KERNEL_TENSOR_INDICES = [
        [],  # compute
        [3, 1, 0, 2, 4],  # read: al, a, b, dt, ones
        [5, 6],  # write: beta, g
    ]

    def _build_beta_g_kernel_state(self):
        kdir = self._BETA_G_KERNELS_DIR
        compute_path = str(kdir / "beta_g_compute.cpp")
        read_path = str(kdir / "beta_g_read.cpp")
        write_path = str(kdir / "beta_g_write.cpp")
        for p in (compute_path, read_path, write_path):
            assert Path(p).is_file(), (
                f"missing emitted beta/g kernel: {p}. Run "
                f"models/demos/qwen3_6_galaxy_v2/tt/kernels/beta_g_kernel.py "
                f"in the 3.12 venv to regenerate."
            )

        # The emitted kernel was authored for bfloat16 CBs (tile-page-size
        # 2048 bytes). The DeltaNet projection ``ba`` (and therefore ``b`` and
        # ``a`` at the call site) is bf8_b, so we (1) maintain bf16 copies of
        # the constant weights ``dt_bias`` / ``A_log`` and (2) per-call cast
        # ``b`` / ``a`` to bf16 before the kernel launch. The dtype-conversion
        # cost is two ttnn.typecast ops on a single tile each — << the saving
        # of fusing the 6-op chain into one launch.
        # Host shapes match the existing A_log / dt_bias build path: full
        # n_v_heads=48 along dim 2 sharded across 8 mesh-rows (6 per row).
        # This is the only sharding pattern that ShardTensor2dMesh accepts —
        # the per-device tile shape ends up at `[1, 1, 6]` which tilizes to
        # one 32×32 tile per device.
        n_v_heads = self.n_v_heads  # 48
        row_shard_3d = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(2, None), mesh_shape=self.cluster_shape)

        # bf16 copies of A_log / dt_bias (originals stay at bf8_b for the
        # 6-op fallback path — preserves bit-exact PCC parity in fallback).
        A_log_bf16 = ttnn.typecast(self.A_log, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        dt_bias_bf16 = ttnn.typecast(self.dt_bias, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # ones: shape matches dt_bias / A_log (`[1, 1, 48]` on host →
        # `[1, 1, 6]` per device after row-sharding on dim 2).
        ones_t = torch.ones((1, 1, n_v_heads), dtype=torch.bfloat16)
        ones = ttnn.from_torch(
            ones_t,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_shard_3d,
        )

        # beta_out / g_out have the same per-device shape as b / a at the
        # decode call site: `[B=1, T=1, 6]` per device. Build the template
        # via the same sharding pattern (host `[1, 1, 48]` → per-device
        # `[1, 1, 6]`). At max_batch_size > 1 this would need a host-shape
        # adjustment, but the qwen3.6 v2 config pins max_batch_size=1.
        assert self.max_batch_size == 1, (
            "V2-16 tt-lang beta/g kernel state assumes max_batch_size=1; " f"got {self.max_batch_size}"
        )
        template_t = torch.zeros((1, 1, n_v_heads), dtype=torch.bfloat16)
        template = ttnn.from_torch(
            template_t,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_shard_3d,
        )
        beta_out = ttnn.allocate_tensor_on_device(template.spec, self.mesh_device)
        g_out = ttnn.allocate_tensor_on_device(template.spec, self.mesh_device)
        template.deallocate(True)

        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        return {
            "compute_path": compute_path,
            "read_path": read_path,
            "write_path": write_path,
            "ones": ones,
            "A_log_bf16": A_log_bf16,
            "dt_bias_bf16": dt_bias_bf16,
            "beta_out": beta_out,
            "g_out": g_out,
            "core_ranges": core_ranges,
        }

    def _compute_beta_g_tt_lang(self, b, a):
        """Dispatch the fused tt-lang beta/g kernel via ttnn.generic_op.

        Returns the persistent ``beta_out`` / ``g_out`` device buffers.
        Trace-safe: the program-descriptor closure binds to fixed buffer
        addresses (the 5 inputs and 2 outputs are all DRAM-interleaved
        tensors whose addresses do not move across decode steps).
        """
        st = self._beta_g_kernel_state
        # Cast b/a to bf16 to match the kernel's CB data_format. The fp32-state
        # delta-rule downstream consumes both at bf16 anyway (typecast at
        # ttnn_delta_rule_ops_fp32.py:289-303), so this conversion is not
        # added overhead — it just moves the cast earlier in the pipeline.
        b_bf16 = ttnn.typecast(b, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a_bf16 = ttnn.typecast(a, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tensors = [
            b_bf16,
            a_bf16,
            st["dt_bias_bf16"],
            st["A_log_bf16"],
            st["ones"],
            st["beta_out"],
            st["g_out"],
        ]

        # Build TensorAccessorArgs (compile-time args) for all 7 tensors.
        tensor_accessor_args = []
        for t in tensors:
            tensor_accessor_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        cb_descriptors = []
        for i in range(self._BETA_G_NUM_TENSORS):
            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=i,
                data_format=ttnn.bfloat16,
                page_size=self._BETA_G_CB_PAGE_SIZE,
            )
            cb_descriptors.append(
                ttnn.CBDescriptor(
                    total_size=self._BETA_G_CB_TOTAL_SIZE,
                    core_ranges=st["core_ranges"],
                    format_descriptors=[cb_format],
                )
            )

        cb_indices = list(range(self._BETA_G_NUM_TENSORS))
        kernel_descriptors = []
        paths = [
            (st["compute_path"], "compute"),
            (st["read_path"], "noc"),
            (st["write_path"], "noc"),
        ]
        noc_idx = 0
        for kernel_idx, (kernel_path, thread_type) in enumerate(paths):
            tensor_indices = self._BETA_G_KERNEL_TENSOR_INDICES[kernel_idx]
            common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]
            if thread_type == "compute":
                compile_time_args = cb_indices
                config = ttnn.ComputeConfigDescriptor()
            else:
                compile_time_args = cb_indices + tensor_accessor_args
                if noc_idx == 0:
                    config = ttnn.ReaderConfigDescriptor()
                else:
                    config = ttnn.WriterConfigDescriptor()
                noc_idx += 1
            kernel_descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=st["core_ranges"],
                    compile_time_args=compile_time_args,
                    common_runtime_args=common_runtime_args,
                    config=config,
                )
            )

        program = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors,
            cbs=cb_descriptors,
            semaphores=[],
        )
        ttnn.generic_op(list(tensors), program)
        # b_bf16 / a_bf16 are per-call temporaries; the kernel has already
        # consumed them. Deallocate to keep L1/DRAM pressure flat across
        # decode steps. (The decode-mode caller deallocates b/a immediately
        # after _compute_beta_g returns, but those are bf8_b originals — our
        # bf16 copies need their own cleanup.)
        b_bf16.deallocate(True)
        a_bf16.deallocate(True)
        return st["beta_out"], st["g_out"]

    def _compute_beta_g(self, b, a, B, T):
        """V2-11 (lever C): attempted unary-chain fusion, NOT LANDED.

        Original (6 ops):
          beta = sigmoid(b); a_biased = add(a, dt_bias); sp = softplus(a_biased)
          A_exp = exp(A_log); neg_A_exp = neg(A_exp); g = multiply(neg_A_exp, sp)

        V2-16: when ``use_tt_lang_beta_g=True`` (or env QWEN36_TT_LANG_BETA_G=1)
        and T==1 (decode), the 6-op chain is replaced by a single tt-lang
        ``ttnn.generic_op`` launch — see ``_compute_beta_g_tt_lang``. Prefill
        keeps the 6-op chain because the kernel was emitted for the 1×1-tile
        decode shape; multi-tile prefill would require re-authoring with a
        broadcast-aware compute body.

        Attempts (pre-V2-16):
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
        if self.use_tt_lang_beta_g and T == 1 and self._beta_g_kernel_state is not None:
            return self._compute_beta_g_tt_lang(b, a)
        mem = ttnn.DRAM_MEMORY_CONFIG
        beta = ttnn.sigmoid(b, memory_config=mem)
        a_biased = ttnn.add(a, self.dt_bias, memory_config=mem)
        sp = ttnn.softplus(a_biased, memory_config=mem)
        A_exp = ttnn.exp(self.A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
        g = ttnn.multiply(ttnn.neg(A_exp, memory_config=mem), sp, memory_config=mem)
        return beta, g

    # ------------------------------------------------------------------
    # V2-18: partial tt-lang recurrent kernel (per-head state update only)
    # ------------------------------------------------------------------
    # The kernel emitted by ``tt/kernels/recurrent_delta_rule_kernel.py``
    # fuses the per-head state update (state*decay + outer(k,v)*beta) into
    # a single ttnn.generic_op launch on a single-core grid=(1,1). It
    # handles ONE head per launch — so wiring it into the 6-V-head/row
    # decode path costs 6 launches per layer per step. The readout matmul
    # (q @ state_new) is NOT fused into the kernel (the V2-17 kernel emits
    # zeros for the readout output); we keep the existing batched matmul
    # for the readout pattern.
    #
    # NOTE: this is a perf-measurement integration — see PERF.md V2-18.
    # The hypothesis under test: does the kernel saving on the state update
    # (6.68x at single-tile in isolation) yield ANY real-loop savings when
    # converted to a per-head Python loop, vs the existing fp32 chain that
    # runs the state update batched across all 6 heads via a single
    # MatmulMultiCoreReuseProgramConfig launch.

    _REC_KERNELS_DIR = Path(__file__).resolve().parent / "kernels" / "recurrent_delta_rule"
    _REC_NUM_TENSORS = 8
    _REC_TILE = 32
    _REC_K_TILES = 4  # head_dim / 32 = 4
    _REC_V_TILES = 4  # head_dim / 32 = 4
    _REC_K_DIM = 4 * 32  # 128
    _REC_V_DIM = 4 * 32  # 128
    # Tensor signature (from V2-17 author script):
    #   inputs:  [state, q, k, v, decay, beta]
    #   outputs: [state_out, o]
    # Reader reads [beta=5, decay=4, k=2, state=0, v=3].
    # Writer writes [o=7, state_out=6].
    _REC_KERNEL_TENSOR_INDICES = [
        [],  # compute
        [5, 4, 2, 0, 3],  # noc reader
        [7, 6],  # noc writer
    ]
    # CB layout. V2-17 emitted with block_count=2 (and =4 for CB6) for the
    # standalone test, totaling ~82KB. At real-decode L1 pressure that
    # collides with the model's interleaved-tensor backing storage on
    # core (0, 0). Shrink each CB to a single page (block_count=1) — the
    # kernel uses ``wait_front(1)`` / ``pop_front(1)`` so single-buffering
    # is functionally safe (sequential, no async pipeline depth needed).
    # New CB total: 9 × 4096 = 36864 bytes (45% of original).
    _REC_CB_CONFIGS = [
        (1, 4096, 4096),  # CB 0 (state)
        (1, 4096, 4096),  # CB 1 (q, unused in PoC)
        (1, 4096, 4096),  # CB 2 (k)
        (1, 4096, 4096),  # CB 3 (v)
        (1, 4096, 4096),  # CB 4 (decay)
        (1, 4096, 4096),  # CB 5 (beta)
        (1, 4096, 4096),  # CB 6 (state_out)
        (1, 4096, 4096),  # CB 7 (o)
        (1, 4096, 4096),  # CB 8 (o_acc internal)
    ]

    def _build_recurrent_kernel_state(self):
        kdir = self._REC_KERNELS_DIR
        compute_path = str(kdir / "recurrent_compute.cpp")
        read_path = str(kdir / "recurrent_read.cpp")
        write_path = str(kdir / "recurrent_write.cpp")
        for p in (compute_path, read_path, write_path):
            assert Path(p).is_file(), (
                f"missing emitted recurrent kernel: {p}. Run "
                f"models/demos/qwen3_6_galaxy_v2/tt/kernels/recurrent_delta_rule_kernel.py "
                f"in the 3.12 venv to regenerate."
            )

        # Single-core kernel grid. CBs total ~37KB (single-page each).
        # Avoid core (0, 0) — on BH that physical core has runtime/dispatch
        # L1 reservations that collide with the recurrent kernel's CBs
        # (which need a larger contiguous L1 region than the beta/g kernel).
        # The model's sub_core_grids documents cols 1-6 × rows 0-9 as
        # Tensix workers; pick (2, 5) — interior of the worker grid.
        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5))])

        # Pre-allocate the persistent broadcast "ones" tile [1, 1, 32, 32] fp32
        # in DRAM (not L1). Keeping it out of L1 avoids contributing to L1
        # pressure on core (0, 0) where dispatch + kernel CBs share L1.
        ones_torch = torch.ones((1, 1, self._REC_TILE, self._REC_TILE), dtype=torch.float32)
        ones_tile = ttnn.from_torch(
            ones_torch,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        zeros_torch = torch.zeros((self._REC_TILE, self._REC_TILE), dtype=torch.float32)
        zeros_tile = ttnn.from_torch(
            zeros_torch,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Pre-allocate persistent zero pad templates for k_col [128, 32]
        # and v_row [32, 128] in DRAM — used as the canvas for per-head k/v tiles.
        k_col_zeros = torch.zeros((self._REC_K_DIM, self._REC_TILE), dtype=torch.float32)
        v_row_zeros = torch.zeros((self._REC_TILE, self._REC_V_DIM), dtype=torch.float32)
        k_col_template = ttnn.from_torch(
            k_col_zeros,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        v_row_template = ttnn.from_torch(
            v_row_zeros,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return {
            "compute_path": compute_path,
            "read_path": read_path,
            "write_path": write_path,
            "core_ranges": core_ranges,
            "ones_tile": ones_tile,
            "zeros_tile": zeros_tile,
            "k_col_template_spec": k_col_template.spec,
            "v_row_template_spec": v_row_template.spec,
            "state_per_head_spec": None,  # filled lazily on first call
            "o_per_head_spec": None,  # filled lazily on first call
            "scalar_tile_spec": ones_tile.spec,
        }

    def _launch_recurrent_kernel(self, state, q, k, v, decay, beta, state_out, o):
        """Dispatch the V2-17 partial recurrent kernel via ttnn.generic_op.

        All tensors are PER-HEAD fp32 TILE_LAYOUT:
            state, state_out: [128, 128]
            q, v:             [32, 128]   (only row 0 valid for v)
            k:                [128, 32]   (only col 0 valid)
            decay, beta:      [32, 32]    (scalar broadcast)
            o:                [32, 128]   (kernel emits zeros — readout external)
        """
        st = self._recurrent_kernel_state
        tensors = [state, q, k, v, decay, beta, state_out, o]
        assert len(tensors) == self._REC_NUM_TENSORS

        tensor_accessor_args = []
        for t in tensors:
            tensor_accessor_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        cb_descriptors = []
        for i, (block_count, page_size, total_size) in enumerate(self._REC_CB_CONFIGS):
            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=i,
                data_format=ttnn.float32,
                page_size=page_size,
            )
            cb_descriptors.append(
                ttnn.CBDescriptor(
                    total_size=total_size,
                    core_ranges=st["core_ranges"],
                    format_descriptors=[cb_format],
                )
            )

        cb_indices = list(range(len(self._REC_CB_CONFIGS)))
        kernel_descriptors = []
        paths = [
            (st["compute_path"], "compute"),
            (st["read_path"], "noc"),
            (st["write_path"], "noc"),
        ]
        noc_idx = 0
        for kernel_idx, (kernel_path, thread_type) in enumerate(paths):
            tensor_indices = self._REC_KERNEL_TENSOR_INDICES[kernel_idx]
            common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]
            if thread_type == "compute":
                compile_time_args = cb_indices
                cfg = ttnn.ComputeConfigDescriptor()
                cfg.fp32_dest_acc_en = True
                cfg.math_fidelity = ttnn.MathFidelity.HiFi4
                cfg.math_approx_mode = False
                config = cfg
            else:
                compile_time_args = cb_indices + tensor_accessor_args
                if noc_idx == 0:
                    config = ttnn.ReaderConfigDescriptor()
                else:
                    config = ttnn.WriterConfigDescriptor()
                noc_idx += 1
            kernel_descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=st["core_ranges"],
                    compile_time_args=compile_time_args,
                    common_runtime_args=common_runtime_args,
                    config=config,
                )
            )

        program = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors,
            cbs=cb_descriptors,
            semaphores=[],
        )
        ttnn.generic_op(list(tensors), program)

    def _recurrent_step_tt_lang(self, state, q, k, v, beta_v, decay):
        """V2-18: per-head loop replacing _fused_decay_and_write_fp32 + readout.

        Args (4-D, fp32, TILE_LAYOUT):
            state : [B, H, K, V]    = [1, 6, 128, 128]
            q     : [B, H, T=1, K]  = [1, 6, 1, 128]
            k     : [B, H, T=1, K]  = [1, 6, 1, 128]
            v     : [B, H, T=1, V]  = [1, 6, 1, 128]   (this is `delta`, not raw v)
            beta_v: [B, H, T=1]     = [1, 6, 1]
            decay : [B, H, T=1]     = [1, 6, 1]

        Returns:
            o   : [B, T=1, H, V]   fp32 (transposed back to [B, T, H, V])
            h_new: [B, H, K, V]    fp32 — the updated state (caller writes
                    back to the persistent buffer via ttnn.copy).
        """
        # Kernel runs on core (0, 0) with grid=(1, 1). Its CBs reserve the
        # top ~100KB of L1 on that core, so any L1-INTERLEAVED tensor whose
        # bank-residency includes core (0,0) can collide with the kernel
        # CB region. To avoid the clash, route the per-head kernel inputs/
        # outputs through DRAM_MEMORY_CONFIG. The non-kernel reductions
        # (concat, matmul readout) live in L1 for bandwidth.
        mem = ttnn.L1_MEMORY_CONFIG
        kmem = ttnn.DRAM_MEMORY_CONFIG  # for tensors passed directly to the kernel
        st = self._recurrent_kernel_state
        cfg = _fp32_compute_cfg_hifi4()
        B = state.shape[0]
        H = state.shape[1]
        K = state.shape[2]
        V = state.shape[3]
        assert B == 1, f"V2-18 kernel path assumes B=1 (max_batch_size); got {B}"
        assert H == self.n_v_per_row, f"H={H} != n_v_per_row={self.n_v_per_row}"
        assert K == self._REC_K_DIM and V == self._REC_V_DIM, f"V2-18 kernel was emitted at K=V=128; got K={K}, V={V}"

        # All inputs are already 4-D TILE_LAYOUT:
        #   state [B, H, K, V], q/k/v [B, H, 1, D], beta_v/decay [B, H, 1].
        # beta/decay just need a fictitious last dim added for broadcast.
        beta_4d = ttnn.reshape(beta_v, [B, H, 1, 1], memory_config=mem)
        decay_4d = ttnn.reshape(decay, [B, H, 1, 1], memory_config=mem)

        q_4d_tile = q
        k_4d_tile = k
        v_4d_tile = v

        per_head_outs = []
        per_head_states_new = []
        for h_idx in range(H):
            # Per-head state slice routed through DRAM (kernel input).
            s_h = ttnn.slice(
                state,
                [0, h_idx, 0, 0],
                [1, h_idx + 1, K, V],
                memory_config=kmem,
            )

            # Per-head q/k/v slices — DRAM-resident for the kernel.
            q_h = ttnn.slice(q_4d_tile, [0, h_idx, 0, 0], [1, h_idx + 1, self._REC_TILE, K], memory_config=kmem)
            v_h = ttnn.slice(v_4d_tile, [0, h_idx, 0, 0], [1, h_idx + 1, self._REC_TILE, V], memory_config=kmem)
            k_h_row = ttnn.slice(k_4d_tile, [0, h_idx, 0, 0], [1, h_idx + 1, self._REC_TILE, K], memory_config=kmem)
            # k_col [128, 32] via transpose. transpose(-2, -1) on 4D → [1, 1, 128, 32].
            k_h_col = ttnn.transpose(k_h_row, -2, -1, memory_config=kmem)
            k_h_row.deallocate(True)

            # Per-head decay / beta scalar broadcast tiles [1, 1, 32, 32] in DRAM.
            decay_h_4d = ttnn.slice(decay_4d, [0, h_idx, 0, 0], [1, h_idx + 1, 1, 1], memory_config=mem)
            decay_tile = ttnn.multiply(st["ones_tile"], decay_h_4d, memory_config=kmem)
            decay_h_4d.deallocate(True)

            beta_h_4d = ttnn.slice(beta_4d, [0, h_idx, 0, 0], [1, h_idx + 1, 1, 1], memory_config=mem)
            beta_tile = ttnn.multiply(st["ones_tile"], beta_h_4d, memory_config=kmem)
            beta_h_4d.deallocate(True)

            # Outputs: per-head state_out [1, 1, 128, 128], per-head o [1, 1, 32, V] — DRAM.
            state_new_h = ttnn.allocate_tensor_on_device(s_h.spec, self.mesh_device)
            o_h_dummy = ttnn.allocate_tensor_on_device(q_h.spec, self.mesh_device)

            # Launch kernel (state update only — o is zeros from the kernel).
            self._launch_recurrent_kernel(
                state=s_h,
                q=q_h,
                k=k_h_col,
                v=v_h,
                decay=decay_tile,
                beta=beta_tile,
                state_out=state_new_h,
                o=o_h_dummy,
            )

            # External readout: o_h = q_row @ state_new_h → [1, 1, 32, V].
            o_h = ttnn.matmul(
                q_h,
                state_new_h,
                memory_config=mem,
                compute_kernel_config=cfg,
            )

            # Per-head temporaries cleanup (kernel + matmul complete).
            q_h.deallocate(True)
            v_h.deallocate(True)
            k_h_col.deallocate(True)
            decay_tile.deallocate(True)
            beta_tile.deallocate(True)
            o_h_dummy.deallocate(True)
            s_h.deallocate(True)

            per_head_outs.append(o_h)
            per_head_states_new.append(state_new_h)

        # NOTE: do NOT deallocate q_4d_tile / k_4d_tile / v_4d_tile —
        # they alias the caller's q / k / v via the metadata-only reshape +
        # no-op to_layout(TILE) when the source is already TILE_LAYOUT.
        # Similarly beta_4d / decay_4d alias the caller's beta_v / decay.
        # Let the caller manage cleanup.

        # ----- concat per-head results -----
        # per_head_outs: list of [1, 1, 32, V]. Concat along dim 1 → [B, H, 32, V].
        # Slice row 0 → [B, H, 1, V] (matches existing fp32 path's o_t shape).
        # Then transpose 1↔2 → [B, T=1, H, V].
        o_concat = ttnn.concat(per_head_outs, dim=1, memory_config=mem)
        for o in per_head_outs:
            o.deallocate(True)
        # Slice row 0 of the tile-padded readout.
        o_first_row = ttnn.slice(o_concat, [0, 0, 0, 0], [B, H, 1, V], memory_config=mem)
        o_concat.deallocate(True)
        # Transpose to [B, T=1, H, V] to match downstream contract.
        o_out = ttnn.transpose(o_first_row, 1, 2, memory_config=mem)
        # NOTE: o_first_row may alias o_out; don't deallocate.

        # per_head_states_new: list of [1, 1, K, V]. Concat along dim 1.
        h_new = ttnn.concat(per_head_states_new, dim=1, memory_config=mem)
        for s in per_head_states_new:
            s.deallocate(True)

        return o_out, h_new

    def recurrent_gated_delta_rule_tt_lang_decode(
        self,
        q,
        k,
        v,
        beta,
        g,
        scale=None,
        initial_state=None,
    ):
        """V2-18: full decode-step recurrent gated delta rule using V2-17 kernel.

        Mirrors ``recurrent_gated_delta_rule_ttnn_fp32`` but at T=1 with the
        per-head kernel for the inner state update. Same I/O contract:
            q, k: [B, T=1, H, K]
            v:    [B, T=1, H, V]
            beta: [B, T=1, H]
            g:    [B, T=1, H]
            initial_state: [B, H, K, V] fp32 — read-only entry state
        Returns:
            o: [B, T=1, H, V] bfloat16
            h_new: [B, H, K, V] fp32
        """
        mem = ttnn.L1_MEMORY_CONFIG
        cfg = _fp32_compute_cfg_hifi4()

        # ----- preprocessing (same as recurrent_gated_delta_rule_ttnn_fp32) -----
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)

        B = q.shape[0]
        T = q.shape[1]
        H = q.shape[2]
        K = q.shape[3]
        V = v.shape[3]
        assert T == 1, f"V2-18 decode helper expects T=1, got T={T}"

        if scale is None:
            scale = K**-0.5

        q_scaled = ttnn.multiply(q, scale, memory_config=mem)
        q.deallocate(True)
        q = q_scaled

        # Transpose [B, T, H, D] → [B, H, T, D]
        q_t = ttnn.transpose(q, 1, 2, memory_config=mem)
        q.deallocate(True)
        k_t = ttnn.transpose(k, 1, 2, memory_config=mem)
        k.deallocate(True)
        v_t = ttnn.transpose(v, 1, 2, memory_config=mem)
        beta_t = ttnn.transpose(beta, 1, 2, memory_config=mem)
        g_t = ttnn.transpose(g, 1, 2, memory_config=mem)

        # fp32 promote
        def _to_fp32(x):
            if x.dtype == ttnn.float32:
                return x
            x_new = ttnn.typecast(x, ttnn.float32, memory_config=mem)
            if x_new is not x:
                x.deallocate(True)
            return x_new

        q_t = _to_fp32(q_t)
        k_t = _to_fp32(k_t)
        v_t = _to_fp32(v_t)
        beta_t = _to_fp32(beta_t)
        g_t = _to_fp32(g_t)
        g_exp = ttnn.exp(g_t, memory_config=mem)
        g_t.deallocate(True)

        # Initial state into L1 (copy out of DRAM-resident persistent buffer).
        if initial_state.dtype != ttnn.float32:
            h = ttnn.typecast(initial_state, ttnn.float32, memory_config=mem)
        else:
            h = ttnn.to_memory_config(initial_state, mem)

        # Keep everything at 4-D [B, H, T=1, D] / [B, H, T=1]. Reshape ops
        # on tile-layout tensors with leading size-1 dims being dropped /
        # restored have caused physical-layout mismatches with the kernel's
        # tile-accessor expectations; avoid them by passing 4-D through.
        q_step = q_t
        k_step = k_t
        v_step = v_t
        beta_step = beta_t
        decay_step = g_exp

        # Compute v_read = k_row @ state  (batched, existing path).
        # Reshape h to TILE_LAYOUT first (it already is from the persistent
        # buffer / typecast, but be explicit).
        if h.layout != ttnn.TILE_LAYOUT:
            h_tile = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=mem)
            h.deallocate(True)
            h = h_tile
        read_query_prog_cfg = None
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(self.mesh_device, K, V)
        except Exception:
            pass

        # k_step is already [B, H, 1, K] in TILE_LAYOUT (no reshape needed).
        # v_read = k_row @ h → [B, H, 1, V]
        v_read_4d = ttnn.matmul(
            k_step,
            h,
            memory_config=mem,
            program_config=read_query_prog_cfg,
            compute_kernel_config=cfg,
        )
        delta_4d = ttnn.subtract(v_step, v_read_4d, memory_config=mem)
        v_read_4d.deallocate(True)

        # ----- per-head loop kernel state update + external readout -----
        # NOTE: do NOT deallocate `h` afterward — when the persistent
        # dn_state_buffer is already in L1, ttnn.to_memory_config(initial_state, L1)
        # is a NO-OP that returns the same tensor (aliasing the persistent
        # buffer). The existing fp32 path follows the same pattern.
        o_step, h_new = self._recurrent_step_tt_lang(
            state=h,
            q=q_step,
            k=k_step,
            v=delta_4d,
            beta_v=beta_step,
            decay=decay_step,
        )
        delta_4d.deallocate(True)

        # o_step is [B, T=1, H, V] from helper (already transposed back).
        # Downstream norm/out_proj expects bf16.
        o_bf16 = ttnn.typecast(o_step, ttnn.bfloat16, memory_config=mem)
        if o_bf16 is not o_step:
            o_step.deallocate(True)

        return o_bf16, h_new

    # ------------------------------------------------------------------
    # V2-17b: multi-head batched + multi-core + fused-readout V3 kernel
    # ------------------------------------------------------------------
    _REC_V3_KERNELS_DIR = Path(__file__).resolve().parent / "kernels" / "recurrent_delta_rule_v3"
    _REC_V3_NUM_TENSORS = 8
    _REC_V3_NUM_CBS = 10
    _REC_V3_GRID_COLS = 4
    _REC_V3_GRID_ROWS = 6
    _REC_V3_TILE = 32
    _REC_V3_K_DIM = 128
    _REC_V3_V_DIM = 128
    _REC_V3_KERNEL_TENSOR_INDICES = [
        [],  # compute
        [5, 4, 2, 1, 0, 3],  # noc reader: beta, decay, k, q, state, v
        [7, 6],  # noc writer: o, state_out
    ]
    _REC_V3_CB_CONFIGS = [
        (2, 4096, 8192),  # 0 state
        (2, 4096, 8192),  # 1 q
        (2, 4096, 8192),  # 2 k
        (2, 4096, 8192),  # 3 v
        (2, 4096, 8192),  # 4 decay
        (2, 4096, 8192),  # 5 beta
        (2, 4096, 8192),  # 6 state_out
        (2, 4096, 8192),  # 7 o
        (2, 4096, 8192),  # 8 state_readout (internal CB-fork)
        (2, 4096, 8192),  # 9 o_acc (internal accumulator)
    ]

    def _build_recurrent_v2_kernel_state(self):
        kdir = self._REC_V3_KERNELS_DIR
        compute_path = str(kdir / "recurrent_compute.cpp")
        read_path = str(kdir / "recurrent_read.cpp")
        write_path = str(kdir / "recurrent_write.cpp")
        for p in (compute_path, read_path, write_path):
            assert Path(p).is_file(), (
                f"missing emitted V3 kernel: {p}. Run "
                f"models/demos/qwen3_6_galaxy_v2/tt/kernels/recurrent_delta_rule_v3_kernel.py "
                f"in the 3.12 venv to regenerate."
            )

        # 4 x 6 = 24-core block at interior worker coords (cols 2-5, rows 0-5).
        core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(5, 5))])

        ones_torch = torch.ones((self._REC_V3_TILE, self._REC_V3_TILE), dtype=torch.float32)
        ones_tile = ttnn.from_torch(
            ones_torch,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        H = self._REC_V3_GRID_ROWS
        TILE = self._REC_V3_TILE
        V = self._REC_V3_V_DIM
        o_template = ttnn.from_torch(
            torch.zeros((H * TILE, V), dtype=torch.float32),
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return {
            "compute_path": compute_path,
            "read_path": read_path,
            "write_path": write_path,
            "core_ranges": core_ranges,
            "ones_tile": ones_tile,
            "o_spec": o_template.spec,
        }

    def _launch_recurrent_v2_kernel(self, state, q, k, v, decay, beta, state_out, o):
        """Dispatch the V3 multi-head kernel via ttnn.generic_op."""
        st = self._recurrent_v2_kernel_state
        tensors = [state, q, k, v, decay, beta, state_out, o]
        assert len(tensors) == self._REC_V3_NUM_TENSORS

        tensor_accessor_args = []
        for t in tensors:
            tensor_accessor_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        cb_descriptors = []
        for i, (_, page_size, total_size) in enumerate(self._REC_V3_CB_CONFIGS):
            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=i,
                data_format=ttnn.float32,
                page_size=page_size,
            )
            cb_descriptors.append(
                ttnn.CBDescriptor(
                    total_size=total_size,
                    core_ranges=st["core_ranges"],
                    format_descriptors=[cb_format],
                )
            )

        cb_indices = list(range(len(self._REC_V3_CB_CONFIGS)))
        kernel_descriptors = []
        paths = [
            (st["compute_path"], "compute"),
            (st["read_path"], "noc"),
            (st["write_path"], "noc"),
        ]
        noc_idx = 0
        for kernel_idx, (kernel_path, thread_type) in enumerate(paths):
            tensor_indices = self._REC_V3_KERNEL_TENSOR_INDICES[kernel_idx]
            common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]
            if thread_type == "compute":
                compile_time_args = cb_indices
                cfg = ttnn.ComputeConfigDescriptor()
                cfg.fp32_dest_acc_en = True
                cfg.math_fidelity = ttnn.MathFidelity.HiFi4
                cfg.math_approx_mode = False
                config = cfg
            else:
                compile_time_args = cb_indices + tensor_accessor_args
                if noc_idx == 0:
                    config = ttnn.ReaderConfigDescriptor()
                else:
                    config = ttnn.WriterConfigDescriptor()
                noc_idx += 1
            kernel_descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=st["core_ranges"],
                    compile_time_args=compile_time_args,
                    common_runtime_args=common_runtime_args,
                    config=config,
                )
            )

        program = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors,
            cbs=cb_descriptors,
            semaphores=[],
        )
        ttnn.generic_op(list(tensors), program)

    def recurrent_gated_delta_rule_tt_lang_decode_v2(
        self,
        q,
        k,
        v,
        beta,
        g,
        scale=None,
        initial_state=None,
    ):
        """V2-17b: full decode-step recurrent gated delta rule using the
        multi-head + multi-core + fused-readout V3 kernel.
        """
        mem = ttnn.L1_MEMORY_CONFIG
        kmem = ttnn.DRAM_MEMORY_CONFIG
        cfg = _fp32_compute_cfg_hifi4()
        st_v2 = self._recurrent_v2_kernel_state

        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)

        B = q.shape[0]
        T = q.shape[1]
        H = q.shape[2]
        K = q.shape[3]
        V = v.shape[3]
        TILE = self._REC_V3_TILE
        assert T == 1
        assert B == 1 and H == self._REC_V3_GRID_ROWS
        assert K == self._REC_V3_K_DIM and V == self._REC_V3_V_DIM

        if scale is None:
            scale = K**-0.5

        q_scaled = ttnn.multiply(q, scale, memory_config=mem)
        q.deallocate(True)
        q = q_scaled

        q_t = ttnn.transpose(q, 1, 2, memory_config=mem)
        q.deallocate(True)
        k_t = ttnn.transpose(k, 1, 2, memory_config=mem)
        k.deallocate(True)
        v_t = ttnn.transpose(v, 1, 2, memory_config=mem)
        beta_t = ttnn.transpose(beta, 1, 2, memory_config=mem)
        g_t = ttnn.transpose(g, 1, 2, memory_config=mem)

        def _to_fp32(x):
            if x.dtype == ttnn.float32:
                return x
            x_new = ttnn.typecast(x, ttnn.float32, memory_config=mem)
            if x_new is not x:
                x.deallocate(True)
            return x_new

        q_t = _to_fp32(q_t)
        k_t = _to_fp32(k_t)
        v_t = _to_fp32(v_t)
        beta_t = _to_fp32(beta_t)
        g_t = _to_fp32(g_t)
        g_exp = ttnn.exp(g_t, memory_config=mem)
        g_t.deallocate(True)

        if initial_state.dtype != ttnn.float32:
            h = ttnn.typecast(initial_state, ttnn.float32, memory_config=mem)
        else:
            h = ttnn.to_memory_config(initial_state, mem)
        if h.layout != ttnn.TILE_LAYOUT:
            h_tile = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=mem)
            h.deallocate(True)
            h = h_tile

        read_query_prog_cfg = None
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(self.mesh_device, K, V)
        except Exception:
            pass
        v_read_4d = ttnn.matmul(
            k_t,
            h,
            memory_config=mem,
            program_config=read_query_prog_cfg,
            compute_kernel_config=cfg,
        )
        delta_4d = ttnn.subtract(v_t, v_read_4d, memory_config=mem)
        v_read_4d.deallocate(True)

        # ----- Stage 2-D kernel inputs (DRAM, away from kernel CBs) -----
        h_dram = ttnn.to_memory_config(h, kmem)
        state_2d = ttnn.reshape(h_dram, [H * K, V], memory_config=kmem)

        q_t_padded = ttnn.pad(q_t, [(0, 0), (0, 0), (0, TILE - 1), (0, 0)], value=0.0)
        q_2d = ttnn.reshape(q_t_padded, [H * TILE, K], memory_config=kmem)

        delta_padded = ttnn.pad(delta_4d, [(0, 0), (0, 0), (0, TILE - 1), (0, 0)], value=0.0)
        v_2d = ttnn.reshape(delta_padded, [H * TILE, V], memory_config=kmem)

        k_col_4d = ttnn.transpose(k_t, -2, -1, memory_config=mem)
        k_col_padded = ttnn.pad(
            k_col_4d,
            [(0, 0), (0, 0), (0, 0), (0, TILE - 1)],
            value=0.0,
        )
        k_col_4d.deallocate(True)
        k_2d = ttnn.reshape(k_col_padded, [H * K, TILE], memory_config=kmem)

        ones_tile_4d = ttnn.reshape(
            st_v2["ones_tile"],
            [1, 1, TILE, TILE],
            memory_config=mem,
        )
        decay_4d_b = ttnn.reshape(g_exp, [B, H, 1, 1], memory_config=mem)
        beta_4d_b = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=mem)
        decay_full = ttnn.multiply(ones_tile_4d, decay_4d_b, memory_config=kmem)
        beta_full = ttnn.multiply(ones_tile_4d, beta_4d_b, memory_config=kmem)
        decay_4d_b.deallocate(True)
        beta_4d_b.deallocate(True)
        decay_2d = ttnn.reshape(decay_full, [H * TILE, TILE], memory_config=kmem)
        beta_2d = ttnn.reshape(beta_full, [H * TILE, TILE], memory_config=kmem)

        state_out_2d = ttnn.allocate_tensor_on_device(state_2d.spec, self.mesh_device)
        o_2d = ttnn.allocate_tensor_on_device(st_v2["o_spec"], self.mesh_device)

        self._launch_recurrent_v2_kernel(
            state=state_2d,
            q=q_2d,
            k=k_2d,
            v=v_2d,
            decay=decay_2d,
            beta=beta_2d,
            state_out=state_out_2d,
            o=o_2d,
        )

        decay_full.deallocate(True)
        beta_full.deallocate(True)
        k_col_padded.deallocate(True)
        delta_padded.deallocate(True)
        q_t_padded.deallocate(True)
        h_dram.deallocate(True)
        delta_4d.deallocate(True)

        h_new_4d = ttnn.reshape(state_out_2d, [B, H, K, V], memory_config=mem)
        o_4d_padded = ttnn.reshape(o_2d, [B, H, TILE, V], memory_config=mem)
        o_4d = ttnn.slice(o_4d_padded, [0, 0, 0, 0], [B, H, 1, V], memory_config=mem)
        o_4d_padded.deallocate(True)
        o_step = ttnn.transpose(o_4d, 1, 2, memory_config=mem)
        o_4d.deallocate(True)

        o_bf16 = ttnn.typecast(o_step, ttnn.bfloat16, memory_config=mem)
        if o_bf16 is not o_step:
            o_step.deallocate(True)

        return o_bf16, h_new_4d

    # ------------------------------------------------------------------
    # V2-17d: V3 multi-head batched + readout-fused + in-place kernel
    # ------------------------------------------------------------------
    # The kernel at tt/kernels/recurrent_delta_rule_v3/ runs grid=(4, 6) = 24
    # cores. Each (j_v, h) core handles one (V-tile-column, V-head) tile and
    # iterates over the 4 K-tiles internally. Per-head readout (q @ state_new)
    # is fused on the same core (V2-17c expression-recompute fork pattern).
    # The persistent state buffer is passed as BOTH state input AND state_out:
    # safe because read[h,i,j] happens before write[h,i,j] within the compute
    # thread, and tiles don't alias across cores.
    #
    # Wrapper-op cost (vs V2-17c per-head loop):
    #   - 1 ttnn.generic_op launch / layer  (vs 6)
    #   - 1 transpose of k     (vs 6)
    #   - 2 multiplies to build per-head broadcast tiles (vs 12)
    #   - 0 slice / concat over heads (vs ~24)

    _REC3_KERNELS_DIR = Path(__file__).resolve().parent / "kernels" / "recurrent_delta_rule_v3"
    _REC3_TILE = 32
    _REC3_HEAD_DIM = 128
    _REC3_K_TILES = 4
    _REC3_V_TILES = 4
    _REC3_V_HEADS = 6
    _REC3_NUM_TENSORS = 8
    _REC3_KERNEL_TENSOR_INDICES = [
        [],
        [5, 4, 2, 1, 0, 3],  # noc reader: beta, decay, k_col, q, state, v
        [7, 6],  # noc writer: o, state_out
    ]
    _REC3_NUM_CBS = 10
    _REC3_CB_PAGE = 4096
    _REC3_CB_TOTAL = 8192

    def _build_recurrent_v3_kernel_state(self):
        kdir = self._REC3_KERNELS_DIR
        compute_path = str(kdir / "recurrent_compute.cpp")
        read_path = str(kdir / "recurrent_read.cpp")
        write_path = str(kdir / "recurrent_write.cpp")
        for p in (compute_path, read_path, write_path):
            assert Path(p).is_file(), (
                f"missing emitted V3 recurrent kernel: {p}. Run "
                f"python_env_312/bin/python "
                f"models/demos/qwen3_6_galaxy_v2/tt/kernels/recurrent_delta_rule_v3_kernel.py "
                f"to regenerate."
            )

        # grid=(V_TILES=4, V_HEADS=6) — 24 cores. Use cores (2..5, 0..5):
        # the worker grid documented in `sub_core_grids` (cols 1-6 x rows 0-9)
        # has spare cores in cols 2-5; this keeps clear of dispatch
        # reservations on col 0.
        core_ranges = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(2, 0),
                    ttnn.CoreCoord(2 + self._REC3_V_TILES - 1, self._REC3_V_HEADS - 1),
                )
            ]
        )

        # Persistent per-head ones tile [1, H=6, 32, 32] fp32 in DRAM. Used
        # once per step to broadcast the scalar decay[h] / beta[h] across
        # each head's 32x32 tile (2 multiplies/layer instead of 12).
        ones_torch = torch.ones((1, self._REC3_V_HEADS, self._REC3_TILE, self._REC3_TILE), dtype=torch.float32)
        ones_per_head = ttnn.from_torch(
            ones_torch,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return {
            "compute_path": compute_path,
            "read_path": read_path,
            "write_path": write_path,
            "core_ranges": core_ranges,
            "ones_per_head": ones_per_head,
        }

    def _launch_recurrent_v3_kernel(self, state, q, k_col, v, decay, beta, o):
        """Dispatch the V3 multi-head kernel. ``state`` is BOTH input AND
        output (in-place writeback)."""
        st = self._recurrent_v3_kernel_state
        # state is passed as BOTH index 0 (input) and index 6 (output). The
        # kernel reads then writes each tile on the same core — safe.
        tensors = [state, q, k_col, v, decay, beta, state, o]
        assert len(tensors) == self._REC3_NUM_TENSORS

        tensor_accessor_args = []
        for t in tensors:
            tensor_accessor_args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

        cb_descriptors = []
        for i in range(self._REC3_NUM_CBS):
            cb_format = ttnn.CBFormatDescriptor(
                buffer_index=i,
                data_format=ttnn.float32,
                page_size=self._REC3_CB_PAGE,
            )
            cb_descriptors.append(
                ttnn.CBDescriptor(
                    total_size=self._REC3_CB_TOTAL,
                    core_ranges=st["core_ranges"],
                    format_descriptors=[cb_format],
                )
            )

        cb_indices = list(range(self._REC3_NUM_CBS))
        kernel_descriptors = []
        paths = [
            (st["compute_path"], "compute"),
            (st["read_path"], "noc"),
            (st["write_path"], "noc"),
        ]
        noc_idx = 0
        for kernel_idx, (kernel_path, thread_type) in enumerate(paths):
            tensor_indices = self._REC3_KERNEL_TENSOR_INDICES[kernel_idx]
            common_runtime_args = [tensors[idx].buffer_address() for idx in tensor_indices]
            if thread_type == "compute":
                compile_time_args = cb_indices
                cfg = ttnn.ComputeConfigDescriptor()
                cfg.fp32_dest_acc_en = True
                cfg.math_fidelity = ttnn.MathFidelity.HiFi4
                cfg.math_approx_mode = False
                config = cfg
            else:
                compile_time_args = cb_indices + tensor_accessor_args
                if noc_idx == 0:
                    config = ttnn.ReaderConfigDescriptor()
                else:
                    config = ttnn.WriterConfigDescriptor()
                noc_idx += 1
            kernel_descriptors.append(
                ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=st["core_ranges"],
                    compile_time_args=compile_time_args,
                    common_runtime_args=common_runtime_args,
                    config=config,
                )
            )

        program = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors,
            cbs=cb_descriptors,
            semaphores=[],
        )
        ttnn.generic_op(list(tensors), program)

    def recurrent_gated_delta_rule_tt_lang_v3_decode(
        self,
        q,
        k,
        v,
        beta,
        g,
        scale=None,
        initial_state=None,
    ):
        """V2-17d: full decode-step recurrent gated delta rule using V3.

        I/O contract identical to recurrent_gated_delta_rule_ttnn_fp32:
            q, k: [B=1, T=1, H=6, K=128]
            v:    [B=1, T=1, H=6, V=128]
            beta: [B=1, T=1, H=6]
            g:    [B=1, T=1, H=6]
            initial_state: [B, H, K, V] fp32 (the persistent dn_state_buffer)

        Returns:
            o:     [B=1, T=1, H=6, V=128] bfloat16
            h_new: alias of initial_state (state was written in place by the
                   kernel — h_new is identically the persistent buffer).
        """
        mem = ttnn.L1_MEMORY_CONFIG
        cfg = _fp32_compute_cfg_hifi4()
        st = self._recurrent_v3_kernel_state

        # ----- preprocessing (same as recurrent_gated_delta_rule_ttnn_fp32) -----
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)

        B = q.shape[0]
        T = q.shape[1]
        H = q.shape[2]
        K = q.shape[3]
        V = v.shape[3]
        assert B == 1, f"V3 decode helper assumes B=1; got {B}"
        assert T == 1, f"V3 decode helper expects T=1, got T={T}"
        assert H == self._REC3_V_HEADS, f"V3 H={H} != n_v_per_row={self._REC3_V_HEADS}"
        assert K == self._REC3_HEAD_DIM and V == self._REC3_HEAD_DIM, f"V3 emitted at K=V=128; got K={K}, V={V}"

        if scale is None:
            scale = K**-0.5

        q_scaled = ttnn.multiply(q, scale, memory_config=mem)
        q.deallocate(True)
        q = q_scaled

        # Transpose [B, T, H, D] → [B, H, T, D]; beta/g go [B, T, H] → [B, H, T].
        q_t = ttnn.transpose(q, 1, 2, memory_config=mem)
        q.deallocate(True)
        k_t = ttnn.transpose(k, 1, 2, memory_config=mem)
        k.deallocate(True)
        v_t = ttnn.transpose(v, 1, 2, memory_config=mem)
        beta_t = ttnn.transpose(beta, 1, 2, memory_config=mem)
        g_t = ttnn.transpose(g, 1, 2, memory_config=mem)

        # fp32 promote.
        def _to_fp32(x):
            if x.dtype == ttnn.float32:
                return x
            x_new = ttnn.typecast(x, ttnn.float32, memory_config=mem)
            if x_new is not x:
                x.deallocate(True)
            return x_new

        q_t = _to_fp32(q_t)
        k_t = _to_fp32(k_t)
        v_t = _to_fp32(v_t)
        beta_t = _to_fp32(beta_t)
        g_t = _to_fp32(g_t)
        g_exp = ttnn.exp(g_t, memory_config=mem)
        g_t.deallocate(True)

        # Initial state alias (persistent buffer). Cast to fp32 if needed.
        if initial_state.dtype != ttnn.float32:
            h = ttnn.typecast(initial_state, ttnn.float32, memory_config=mem)
        else:
            h = initial_state

        # ----- pre-update readout: v_read = k_row @ h, delta = v_t - v_read -----
        read_query_prog_cfg = None
        try:
            read_query_prog_cfg = _recurrent_read_query_program_config(self.mesh_device, K, V)
        except Exception:
            pass
        v_read_4d = ttnn.matmul(
            k_t,
            h,
            memory_config=mem,
            program_config=read_query_prog_cfg,
            compute_kernel_config=cfg,
        )
        delta_4d = ttnn.subtract(v_t, v_read_4d, memory_config=mem)
        v_read_4d.deallocate(True)
        v_t.deallocate(True)

        # ----- build V3 kernel inputs (DRAM-staged to avoid L1 collision with CBs) -----
        # The kernel was emitted at 2D shapes; the 4D inputs need T padded
        # 1→TILE so the logical reshape to 2D is volume-preserving.
        TILE = self._REC3_TILE
        kmem = ttnn.DRAM_MEMORY_CONFIG

        # 1. k_col: transpose [B, H, 1, K] → [B, H, K, 1], pad last dim 1→TILE.
        k_col_raw = ttnn.transpose(k_t, -2, -1, memory_config=mem)
        k_t.deallocate(True)
        k_col_padded = ttnn.pad(
            k_col_raw,
            [(0, 0), (0, 0), (0, 0), (0, TILE - 1)],
            value=0.0,
        )
        if k_col_padded is not k_col_raw:
            k_col_raw.deallocate(True)

        # 2 & 3. Per-head decay/beta broadcast tiles [1, H, 32, 32].
        decay_4d = ttnn.reshape(g_exp, [B, H, 1, 1], memory_config=mem)
        beta_4d = ttnn.reshape(beta_t, [B, H, 1, 1], memory_config=mem)
        decay_bcast = ttnn.multiply(st["ones_per_head"], decay_4d, memory_config=kmem)
        beta_bcast = ttnn.multiply(st["ones_per_head"], beta_4d, memory_config=kmem)
        g_exp.deallocate(True)
        beta_t.deallocate(True)

        # 4. v_in: pad T 1→TILE so reshape to [H*TILE, V] is volume-preserving.
        v_in_padded = ttnn.pad(
            delta_4d,
            [(0, 0), (0, 0), (0, TILE - 1), (0, 0)],
            value=0.0,
        )
        if v_in_padded is not delta_4d:
            delta_4d.deallocate(True)

        # 5. q_padded: pad T 1→TILE.
        q_padded = ttnn.pad(
            q_t,
            [(0, 0), (0, 0), (0, TILE - 1), (0, 0)],
            value=0.0,
        )
        if q_padded is not q_t:
            q_t.deallocate(True)

        # 6. state to DRAM (kernel input/output). The reads then writes
        #    happen inside the kernel via NoC DMA; bringing it to DRAM keeps
        #    the kernel CBs from colliding with the model's L1 state tiles
        #    on cores 2-5 / rows 0-5.
        h_dram = ttnn.to_memory_config(h, kmem)

        # 7. o output [H*TILE, V] in DRAM via v_in_padded's reshape spec.
        # Build by reshaping a zero-initialized tensor with the same spec.
        # The simplest path: use ttnn.allocate_tensor_on_device on the
        # reshape's resulting spec via the 2D form.
        v_2d_spec_tensor = ttnn.reshape(
            v_in_padded,
            [self._REC3_V_HEADS * TILE, self._REC3_HEAD_DIM],
            memory_config=kmem,
        )

        # Allocate o using v_2d_spec_tensor's spec (same shape).
        o_2d = ttnn.allocate_tensor_on_device(v_2d_spec_tensor.spec, self.mesh_device)

        # Metadata reshape to 2D logical layouts (tile counts identical).
        state_2d = ttnn.reshape(
            h_dram, [self._REC3_V_HEADS * self._REC3_HEAD_DIM, self._REC3_HEAD_DIM], memory_config=kmem
        )
        q_2d = ttnn.reshape(q_padded, [self._REC3_V_HEADS * TILE, self._REC3_HEAD_DIM], memory_config=kmem)
        k_col_2d = ttnn.reshape(
            k_col_padded,
            [self._REC3_V_HEADS * self._REC3_HEAD_DIM, TILE],
            memory_config=kmem,
        )
        v_2d = v_2d_spec_tensor  # already reshaped above
        decay_2d = ttnn.reshape(
            decay_bcast,
            [self._REC3_V_HEADS * TILE, TILE],
            memory_config=kmem,
        )
        beta_2d = ttnn.reshape(
            beta_bcast,
            [self._REC3_V_HEADS * TILE, TILE],
            memory_config=kmem,
        )

        self._launch_recurrent_v3_kernel(
            state=state_2d,
            q=q_2d,
            k_col=k_col_2d,
            v=v_2d,
            decay=decay_2d,
            beta=beta_2d,
            o=o_2d,
        )

        # state_2d aliases h_dram which is a SEPARATE DRAM staging copy of
        # the persistent L1 buffer. The kernel updated state_2d in place;
        # copy back into h (the persistent buffer).
        ttnn.copy(state_2d, h)
        h_dram.deallocate(True)
        q_padded.deallocate(True)
        k_col_padded.deallocate(True)
        v_in_padded.deallocate(True)
        decay_bcast.deallocate(True)
        beta_bcast.deallocate(True)
        # o_2d holds the kernel output.

        # Reshape o back to 4D [B, H, TILE, V], slice T=1, transpose to [B, T=1, H, V].
        o_4d_padded = ttnn.reshape(o_2d, [B, H, TILE, V], memory_config=mem)
        o_first_row = ttnn.slice(o_4d_padded, [0, 0, 0, 0], [B, H, 1, V], memory_config=mem)
        o_4d_padded.deallocate(True)
        o_2d.deallocate(True)
        o_out = ttnn.transpose(o_first_row, 1, 2, memory_config=mem)
        # o_first_row may alias o_out — don't deallocate.

        # Downstream norm/out_proj expects bf16.
        o_bf16 = ttnn.typecast(o_out, ttnn.bfloat16, memory_config=mem)
        if o_bf16 is not o_out:
            o_out.deallocate(True)

        return o_bf16, h

    def _gqa_expand_q_k(self, q, k, B, T, head_dim_axis=2):
        # head_dim_axis selects the head axis to expand:
        #   2 -> classic [B,T,H,D] layout (decode old / prefill).
        #   1 -> fused-heads [B,H,T,D] layout (decode create-heads fusion).
        # repeat_interleave on the head axis yields head order
        # [h0,h0,h0,h1,h1,h1] in BOTH cases (interleave repeats each head
        # contiguously regardless of which axis carries the heads).
        ratio = self.n_v_per_row // self.n_k_per_row
        mem = ttnn.DRAM_MEMORY_CONFIG
        q_e = ttnn.repeat_interleave(q, ratio, dim=head_dim_axis, memory_config=mem)
        k_e = ttnn.repeat_interleave(k, ratio, dim=head_dim_axis, memory_config=mem)
        return q_e, k_e

    def _build_dn_norm_sharded_cfg(self, B, T):
        """Lazily build the width-sharded program config + memcfg for the
        GroupRMSNormGated norm (T1).

        The DN norm is a *grouped* RMSNorm: ``norm_weight`` is head_dim (128)
        wide and the reduction is per-head over head_dim.  A
        ``LayerNormShardedMultiCoreProgramConfig`` reduces over the full
        per-core shard width, so by width-sharding the [B*T, v_per_row]
        activation so each core holds exactly ONE head (128 = 4 tiles), the
        sharded multi-core norm reduces over head_dim per core — bit-for-bit
        the same grouping as the single-core DRAM ``ttnn.rms_norm`` (which
        broadcasts the 128-wide weight over the [..,H,128] last dim).

        Layout: [1, 1, B*T(=32 → 1 tile), v_per_row(=768)] width-sharded over
        ``n_v_per_row`` (=6) cores, shard (32, 128).
        """
        if getattr(self, "_dn_norm_sharded_cfg", None) is not None:
            return self._dn_norm_sharded_cfg
        MT = B * T  # 32 for decode
        head_w = self.head_dim  # 128
        n_heads = self.n_v_per_row  # 6
        block_w = head_w // 32  # 4 tiles
        # Pick a core grid covering n_heads cores; one head per core.
        grid_x = min(n_heads, 8)
        grid_y = (n_heads + grid_x - 1) // grid_x
        core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))
        core_range_set = ttnn.CoreRangeSet({core_range})
        sharded_memcfg = ttnn.create_sharded_memory_config(
            shape=(1, 1, MT, head_w),
            core_grid=core_range_set,
            strategy=ttnn.ShardStrategy.WIDTH,
            use_height_and_width_as_shard_shape=True,
        )
        prgm_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=block_w,
            block_h=MT // 32,  # 1
            block_w=block_w,
            inplace=False,
        )
        self._dn_norm_sharded_cfg = (sharded_memcfg, prgm_cfg)
        return self._dn_norm_sharded_cfg

    def _apply_norm_gated(self, core_out, z, B, T):
        """GroupRMSNormGated: rms_norm(core_out) * silu(z).

        T1: when ``QWEN36_DN_NORM_SHARDED`` (default ON), run the norm
        width-sharded multi-core (one head per core) instead of the
        single-core DRAM ``ttnn.rms_norm``, keep silu+multiply on the same
        sharded L1 layout, then reshard back to DRAM at the block boundary so
        the existing out-proj linear contract ([B,T,v_per_row] DRAM) is
        unchanged.  Per-head reduction is identical to the DRAM path, so the
        precision footgun that killed the fused-silu attempt (token 248068 →
        232 + mojibake; see below) does not apply here.

        V2-11 (lever D) history: attempted silu(z) into multiply fusion via
        ``multiply(out, z, input_tensor_b_activations=[SILU])`` — ran at the
        same speed but shifted the compile-pass token 248068 → 232 and made
        decode gibberish (fused-activation precision drift compounding over 48
        layers).  Reverted to the verified two-op silu+multiply pattern, which
        is preserved here.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        if os.environ.get("QWEN36_DN_NORM_SHARDED", "1") == "1" and (B * T) == 32:
            sharded_memcfg, prgm_cfg = self._build_dn_norm_sharded_cfg(B, T)
            # core_out: [B,T,H,V] L1-interleaved from the recurrent core.
            # Flatten to [1,1,B*T,v_per_row] and reshard to width-sharded L1.
            core_2d = ttnn.reshape(core_out, [1, 1, B * T, self.v_per_row])
            core_sh = ttnn.to_memory_config(core_2d, sharded_memcfg)
            if core_sh is not core_2d:
                core_2d.deallocate(True)
            out = ttnn.rms_norm(
                core_sh,
                weight=self.norm_weight_sharded,
                epsilon=self.eps,
                memory_config=sharded_memcfg,
                program_config=prgm_cfg,
                compute_kernel_config=self.compute_kernel,
            )
            core_sh.deallocate(True)
            # silu(z) * out, all on the sharded L1 layout.
            z_2d = ttnn.reshape(z, [1, 1, B * T, self.v_per_row])
            z_sh = ttnn.to_memory_config(z_2d, sharded_memcfg)
            if z_sh is not z_2d:
                z_2d.deallocate(True)
            z_silu = ttnn.silu(z_sh, memory_config=sharded_memcfg)
            z_sh.deallocate(True)
            out = ttnn.multiply(out, z_silu, memory_config=sharded_memcfg)
            z_silu.deallocate(True)
            # Reshard back to DRAM at the block boundary (out-proj contract).
            out_dram = ttnn.to_memory_config(out, mem)
            if out_dram is not out:
                out.deallocate(True)
            out_dram = ttnn.reshape(out_dram, [B, T, self.v_per_row])
            return out_dram

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
        import os as _os

        mem = ttnn.DRAM_MEMORY_CONFIG
        # V2-14: optionally swap the post-linear ``ttnn.all_reduce`` to the
        # persistent-buffer ``line_all_reduce`` (3rd overload).  Gated by env
        # var so we can toggle for bisecting precision issues.
        #
        # V2-CCL (2026-05-16): mirror llama3_70b_galaxy `llama_mlp.py:219`
        # w2 epilogue → `line_all_reduce(cluster_axis=0,
        # num_links=GALAXY_NUM_LINKS, use_optimal_ccl_for_llama=True)`. The
        # task-named lever flag QWEN36_DELTA_OP_TUNED enables the
        # persistent-buffer LAR path; num_links is overridable via
        # QWEN36_CCL_NUM_LINKS (default 1, matching BH GLX FABRIC_1D).
        _use_pbuf = (
            _os.environ.get("QWEN36_DELTA_LAR", "0") == "1" or _os.environ.get("QWEN36_DELTA_OP_TUNED", "0") == "1"
        ) and getattr(self.tt_ccl, "qwen36_residual_buffers", [None, None])[0] is not None
        try:
            _do_num_links = int(
                _os.environ.get(
                    "QWEN36_CCL_NUM_LINKS_DELTA",
                    _os.environ.get("QWEN36_CCL_NUM_LINKS", "1"),
                )
            )
        except ValueError:
            _do_num_links = 1

        # V2-CONFIG-E: optionally use bf8 output for the DN out_proj matmul
        # (the residual write — analogous to llama70b WO at line 567 which
        # uses `dtype=ttnn.bfloat8_b`). Same env var as the FA WO flip in
        # `llama_attention.py`. The original olmo session-11 lesson said
        # bf16 was needed to avoid residual-stream quantization, but llama70b
        # ships with bf8 here — try matching it and measure.
        _dn_out_dtype = ttnn.bfloat8_b if _os.environ.get("QWEN36_ATTN_OUT_BF8", "0") == "1" else ttnn.bfloat16

        # QWEN36_PREFILL_OPT: tuned 2D-TP program config on the out-proj matmul
        # (T>1 only). Applied to the DRAM (non-pbuf) path; the pbuf path writes
        # width-sharded L1 and is left as-is to avoid program-config/sharded-
        # memcfg interaction.
        # Only for 1 < T <= 2048 (no long-T reshape needed; M fits the grid).
        # The non-pbuf path is unused when QWEN36_DELTA_OP_TUNED=1 (pbuf path),
        # so this stays conservative.
        _out_progcfg = (
            self.model_config["QWEN36_DN_OUT_PREFILL_PROGCFG"](T)
            if (_os.environ.get("QWEN36_PREFILL_OPT", "0") == "1" and 1 < T <= 2048)
            else None
        )
        if not _use_pbuf:
            partial = ttnn.linear(
                out_flat,
                self.w_out,
                dtype=_dn_out_dtype,
                memory_config=mem,
                compute_kernel_config=self.compute_kernel,
                program_config=_out_progcfg,
            )
            if _os.environ.get("QWEN36_ABLATE_CCL", "0") == "1":
                return partial  # skip row-reduce (timing ablation; garbage values)
            # V2-11 (lever B): collapse `all_gather + fast_reduce_nc` (2 ops)
            # into a single `ttnn.all_reduce` (1 op).
            reduced = ttnn.all_reduce(
                partial,
                cluster_axis=0,
                num_links=_do_num_links,
                memory_config=mem,
            )
            partial.deallocate(True)
            return reduced

        # V2-14 persistent-buffer path.  Linear writes directly into
        # width-sharded L1 (avoids inserted to_memory_config); the
        # all_reduce_async kernel-fused path consumes width-sharded input
        # and produces width-sharded output.
        sharded_memcfg = self.tt_ccl.qwen36_residual_output_memcfgs[0]
        _LIN_MODE = _os.environ.get("QWEN36_DELTA_LAR_LIN_MODE", "sharded")
        if _LIN_MODE == "sharded":
            partial = ttnn.linear(
                out_flat,
                self.w_out,
                dtype=_dn_out_dtype,
                memory_config=sharded_memcfg,
                compute_kernel_config=self.compute_kernel,
            )
        else:
            partial_dram = ttnn.linear(
                out_flat,
                self.w_out,
                dtype=_dn_out_dtype,
                memory_config=mem,
                compute_kernel_config=self.compute_kernel,
            )
            partial = ttnn.to_memory_config(partial_dram, sharded_memcfg)
            partial_dram.deallocate(True)
        reduced_sharded = self.tt_ccl.line_all_reduce(
            partial,
            cluster_axis=0,
            num_links=_do_num_links,
            memory_config=sharded_memcfg,
            use_optimal_ccl_for_llama=True,
            use_qwen36_residual_buffer=True,
        )
        partial.deallocate(True)
        if _os.environ.get("QWEN36_DELTA_LAR_SKIP_CVT", "0") == "1":
            return reduced_sharded
        reduced = ttnn.to_memory_config(reduced_sharded, mem)
        reduced_sharded.deallocate(True)
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

    def _build_seq_masks(self):
        """Build the chunk masks ONCE (call at __init__, not mid-forward). Building these
        from_torch/ones/triu tensors lazily during prefill collided with live activations in
        DRAM under galaxy TP=32 and corrupted the residual stream. Built at init they are
        stable for the whole forward."""
        if os.environ.get("QWEN36_NO_MASK_CACHE") or getattr(self, "_seq_masks", None) is not None:
            return
        import torch as _torch

        from models.demos.qwen3_6_galaxy_v2.tt.gdn_chunk_ops import _create_tril_ones, _create_triu_ones

        C = self._seq_prefill_chunk_size
        _dd = ttnn.DRAM_MEMORY_CONFIG

        def _eye_dram(n):
            return ttnn.from_torch(
                _torch.eye(n, dtype=_torch.float32).unsqueeze(0),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                memory_config=_dd,
            )

        self._seq_masks = {
            "triu_ones": ttnn.reshape(_create_triu_ones(C, self.mesh_device, ttnn.float32, _dd), [1, C, C]),
            "tril_mask": ttnn.reshape(_create_tril_ones(C, self.mesh_device, ttnn.float32, _dd), [1, C, C]),
            "eye": _eye_dram(C),
            "lower_causal": _create_tril_ones(C, self.mesh_device, ttnn.float32, _dd),
            "eye_32": _eye_dram(32),
        }

    def _chunk_gdr_seq(self, q_exp, k_exp, v_h, beta, g, B, T, initial_state=None):
        """Prefill DeltaNet via the C++ ``gated_delta_attn_seq`` parallel-scan
        kernel (ported from the P150 qwen35 path). Drop-in replacement for
        ``chunk_gated_delta_rule_ttnn``: same inputs/outputs.

        Inputs:  q_exp/k_exp [B,T,H,K], v_h [B,T,H,V], beta/g [B,T,H]
        Returns: core_out [B,T,H,V], new_state [B,H,K,V]

        The [B,T,H,X] -> [BH,T,X] conversion mirrors chunk_gated_delta_rule_ttnn
        exactly (transpose(1,2) + reshape). The seq kernel expects L2-normalized
        q/k (the pure-TTNN chunk path normalizes internally) and applies the
        K**-0.5 scale itself, so we normalize here and pass scale=None.
        """
        H = self.n_v_per_row
        K = self.head_dim
        V = self.head_dim
        BH = B * H

        _dram = ttnn.DRAM_MEMORY_CONFIG if T > 512 else ttnn.L1_MEMORY_CONFIG

        # Chunk masks are pre-built once at __init__ (see _build_seq_masks). Building them
        # lazily mid-forward corrupted DRAM under galaxy TP=32 (the from_torch/ones/triu
        # allocations mid-prefill collided with live activations → exploded the residual at
        # downstream layers). Fall back to a lazy build only if somehow not yet built.
        if not os.environ.get("QWEN36_NO_MASK_CACHE") and getattr(self, "_seq_masks", None) is None:
            self._build_seq_masks()

        # [B,T,H,X] -> [B,H,T,X] -> [BH,T,X] float32 (mirrors chunk_gated_delta_rule_ttnn).
        # Reshape to [BH,T,K] BEFORE l2-norm: the [B,T,H,hd] layout tile-pads the sub-tile H=6 dim
        # to 32 (~5x memory bloat), which OOMs L1 at large ISL. [BH,T,K] tiles cleanly on [T,K].
        q = ttnn.reshape(
            ttnn.typecast(ttnn.transpose(q_exp, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram),
            [BH, T, K],
        )
        k = ttnn.reshape(
            ttnn.typecast(ttnn.transpose(k_exp, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram),
            [BH, T, K],
        )
        q = l2_norm_ttnn(q, dim=-1)
        k = l2_norm_ttnn(k, dim=-1)
        v = ttnn.reshape(
            ttnn.typecast(ttnn.transpose(v_h, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram), [BH, T, V]
        )
        beta3 = ttnn.reshape(
            ttnn.typecast(ttnn.transpose(beta, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram),
            [BH, T, 1],
        )
        g3 = ttnn.reshape(
            ttnn.typecast(ttnn.transpose(g, 1, 2, memory_config=_dram), ttnn.float32, memory_config=_dram), [BH, T]
        )

        # Free the L1-resident inputs now (we hold DRAM copies). Otherwise these padded
        # [B,T,H,hd] tensors stay co-resident in L1 with the seq kernel's circular buffers on
        # the head cores and clash at large ISL. The caller skips its own dealloc for the seq path.
        q_exp.deallocate(True)
        k_exp.deallocate(True)
        v_h.deallocate(True)
        beta.deallocate(True)
        g.deallocate(True)

        # Carry the recurrent state across prefill chunks: the wrapper wants [BH,K,V];
        # our persistent dn_state_buffer / prior-chunk state is [B,H,K,V].
        init_state_bhkv = None
        if initial_state is not None:
            init_state_bhkv = ttnn.reshape(initial_state, [BH, K, V])

        out, final_state = chunk_gated_delta_rule_seq(
            q,
            k,
            v,
            beta3,
            g3,
            chunk_size=self._seq_prefill_chunk_size,
            scale=None,
            initial_state=init_state_bhkv,
            mesh_device=self.mesh_device,
            cached_masks=None if os.environ.get("QWEN36_NO_MASK_CACHE") else self._seq_masks,
        )

        # out: [BH, L, V] (L = padded to chunk multiple). Slice to T, back to [B,T,H,V].
        if out.shape[1] != T:
            out = ttnn.slice(out, (0, 0, 0), (BH, T, V))
        core_out = ttnn.transpose(ttnn.reshape(out, [B, H, T, V]), 1, 2)  # [B,T,H,V]
        new_state = ttnn.reshape(final_state, [B, H, K, V])
        return core_out, new_state

    def _forward_prefill_fused(self, q_conv, k_conv, v_conv, z, a, b, B, T):
        """Fused whole-sequence DeltaNet recurrence via the P150 gdn_prefill_fused
        kernel (one ttnn.generic_op launch), replacing the op-heavy chunk path.

        Inputs are post-conv (q/k/v) + raw gate inputs (a, b) + z gate; the kernel
        computes beta=sigmoid(b), g=neg_exp_A*softplus(a+dt_bias) and the
        recurrence internally. Post-kernel: per-head rms_norm + silu(z) gate +
        output projection (+ row all-reduce), matching ``_apply_norm_gated`` /
        ``_output_proj_and_reduce``.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        hd = self.head_dim
        Nv = self._fused_Nv_TP  # 6
        num_pairs = B * Nv  # B=1 -> 6

        # conv_out: concat(q|k|v) post-conv+silu -> [1, T, qkv_dim_tp=1280]
        conv_out = ttnn.concat([q_conv, k_conv, v_conv], dim=-1, memory_config=mem)
        if len(conv_out.shape) == 4:
            conv_out = ttnn.reshape(conv_out, [B, T, self._fused_qkv_dim_tp])

        # a, b -> [1, T, Nv]
        if len(a.shape) == 4:
            a = ttnn.reshape(a, [B, T, Nv])
            b = ttnn.reshape(b, [B, T, Nv])

        # recurrence state [num_pairs, Dk, Dv] bf16 (fresh: prefill from zero state)
        rec_states = ttnn.from_torch(
            torch.zeros(num_pairs, hd, hd, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # output [num_pairs * T, 1, Dv] bf16
        prefill_output = ttnn.from_torch(
            torch.zeros(num_pairs * T, 1, hd, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # QWEN36_ABLATE_RECURRENCE: skip the recurrence kernel (prefill_output
        # stays zeros) to measure how much of prefill wall-clock is the fused
        # DeltaNet recurrence itself. Timing-only; output is garbage.
        if os.environ.get("QWEN36_ABLATE_RECURRENCE", "0") != "1":
            self._gdn_prefill_fused(
                conv_out,
                a,
                b,
                self.neg_exp_A,
                self.dt_bias,
                self._fused_norm_w,
                self._fused_scale_tt,
                self._fused_rms_scale_tt,
                self._fused_rms_eps_tt,
                rec_states,
                prefill_output,
                num_pairs=num_pairs,
                num_tokens=T,
                num_cores=num_pairs,
                Nv_TP=Nv,
                Nk_TP=self._fused_Nk_TP,
                repeat_factor=self._fused_repeat,
                key_dim_tp=self._fused_key_dim_tp,
            )
        conv_out.deallocate(True)

        # per-head rms_norm, then reshape/permute to [1, T, v_per_row]
        out_n = ttnn.rms_norm(prefill_output, weight=self._fused_norm_w, epsilon=self.eps)
        prefill_output.deallocate(True)
        out_4d = ttnn.reshape(out_n, [1, num_pairs, T, hd])
        out_n.deallocate(True)
        out_4d = ttnn.permute(out_4d, (0, 2, 1, 3))  # [1, T, num_pairs, Dv]
        out_f = ttnn.reshape(out_4d, [B, T, self.v_per_row])
        out_4d.deallocate(True)

        # silu(z) gate
        z3 = ttnn.reshape(z, [B, T, self.v_per_row]) if len(z.shape) == 4 else z
        z_silu = ttnn.silu(z3, memory_config=mem)
        gated = ttnn.multiply(out_f, z_silu, memory_config=mem)
        out_f.deallocate(True)
        z_silu.deallocate(True)

        # write final recurrence state into the persistent fp32 buffer for decode
        rec_fp32 = ttnn.typecast(rec_states, ttnn.float32, memory_config=mem)
        rec_states.deallocate(True)
        rec_resh = ttnn.reshape(rec_fp32, [B, Nv, hd, hd])
        ttnn.copy(rec_resh, self.dn_state_buffer)
        rec_fp32.deallocate(True)

        output = self._output_proj_and_reduce(gated, B, T)
        gated.deallocate(True)
        return output

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

        # Chunked-prefill carry: when driven chunk-by-chunk (self._pf_chunk_idx set by the
        # model's prefill_chunked driver), seed conv + recurrent state from the persistent
        # buffers for chunks after the first. Chunk 0 (and single-pass, _pf_chunk_idx=None)
        # seeds fresh (None) — identical to the original single-pass behavior.
        _pf_ci = getattr(self, "_pf_chunk_idx", None)
        _pf_carry = _pf_ci is not None and _pf_ci > 0

        # 1. Projections
        q, k, v, z, a, b = self._project_inputs(x)

        # 2. Conv1d + split (conv_state carried from previous chunk, or fresh)
        _conv_in = self.conv_state_buffer if _pf_carry else None
        q_conv, k_conv, v_conv, new_conv_state = self._apply_conv_and_split(q, k, v, B, T, conv_state=_conv_in)
        q.deallocate(True)
        k.deallocate(True)
        v.deallocate(True)

        # --- Fast path: fused gdn_prefill kernel (one launch for the whole
        # sequence recurrence), mirroring the P150 gdn_prefill_fused flow. ---
        if getattr(self, "_use_fused_prefill", False) and not getattr(self, "_use_seq_prefill", False):
            output = self._forward_prefill_fused(q_conv, k_conv, v_conv, z, a, b, B, T)
            q_conv.deallocate(True)
            k_conv.deallocate(True)
            v_conv.deallocate(True)
            z.deallocate(True)
            a.deallocate(True)
            b.deallocate(True)
            ttnn.copy(new_conv_state, self.conv_state_buffer)
            new_conv_state.deallocate(True)
            return output

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

        if os.environ.get("QWEN36_MEMLOG"):
            try:
                ttnn.dump_device_memory_state(
                    self.mesh_device, prefix=f"[MEMLOG L{getattr(self,'layer_num','?')} T={T} pre-seq-kernel] "
                )
            except Exception as _e:
                print(f"[MEMLOG] dump failed: {str(_e)[:100]}", flush=True)

        # 6. Chunked delta rule kernel
        if getattr(self, "_use_seq_prefill", False):
            # Move z_h (gate, consumed in step 7) to DRAM so it doesn't co-reside in L1 with the
            # seq kernel's circular buffers on the head cores — the large-ISL clash. The seq kernel
            # reserves ~1.29 MB of the ~1.46 MB L1 on its 6 cores, so any big L1 tensor clashes.
            if z_h.memory_config().buffer_type == ttnn.BufferType.L1:
                _zd = ttnn.to_memory_config(z_h, ttnn.DRAM_MEMORY_CONFIG)
                z_h.deallocate(True)
                z_h = _zd
            if os.environ.get("QWEN36_DIAG_L1"):
                for _nm, _t in [
                    ("x", x),
                    ("q_conv?", None),
                    ("new_conv_state", new_conv_state),
                    ("q_exp", q_exp),
                    ("k_exp", k_exp),
                    ("v_h", v_h),
                    ("z_h", z_h),
                    ("beta", beta),
                    ("g", g),
                ]:
                    if _t is None:
                        continue
                    try:
                        _bt = _t.memory_config().buffer_type
                        _addr = _t.buffer_address()
                        print(f"[DIAG_L1] {_nm:16s} buf={_bt} addr={_addr} shape={list(_t.shape)}", flush=True)
                    except Exception as _e:
                        print(f"[DIAG_L1] {_nm:16s} (addr n/a: {str(_e)[:60]})", flush=True)
            # C++ gated_delta_attn_seq parallel-scan kernel (ported from P150).
            # _chunk_gdr_seq deallocates q_exp/k_exp/v_h/beta/g internally (after DRAM copies).
            _dn_in = self.dn_state_buffer if _pf_carry else None
            core_out, new_state = self._chunk_gdr_seq(q_exp, k_exp, v_h, beta, g, B, T, initial_state=_dn_in)
        else:
            core_out, new_state = chunk_gated_delta_rule_ttnn(
                q=q_exp,
                k=k_exp,
                v=v_h,
                beta=beta,
                g=g,
                chunk_size=self.prefill_chunk_size,
                initial_state=(self.dn_state_buffer if _pf_carry else None),
                device=self.mesh_device,
                cached_masks=self._chunk_masks,
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

        # The fused-heads layout is only wired into the pure-ttnn fp32 recurrent
        # core (the else-branch below). The tt-lang kernel paths still expect the
        # old [B,T,H,D] layout, so disable fusion when any of them is active.
        use_fused_heads = self.use_dn_fused_heads and not (
            (self.use_tt_lang_recurrent_v3 and self._recurrent_v3_kernel_state is not None)
            or (self.use_tt_lang_recurrent_v2 and self._recurrent_v2_kernel_state is not None)
            or (self.use_tt_lang_recurrent and self._recurrent_kernel_state is not None)
        )

        if use_fused_heads:
            # 3'. Reshape directly into the recurrent core's [B, H, T, D] layout.
            #     At T=1 this is bit-identical to the old reshape([B,T,H,D]) +
            #     transpose(1,2) (same contiguous memory, different target shape).
            #     z_h is left in [B,T,n_v,D] — the norm/output consume that layout.
            q_h = ttnn.reshape(q_conv, [B, self.n_k_per_row, T, self.head_dim])
            k_h = ttnn.reshape(k_conv, [B, self.n_k_per_row, T, self.head_dim])
            v_h = ttnn.reshape(v_conv, [B, self.n_v_per_row, T, self.head_dim])
            z_h = ttnn.reshape(z, [B, T, self.n_v_per_row, self.head_dim])
            q_conv.deallocate(True)
            k_conv.deallocate(True)
            v_conv.deallocate(True)

            # 4'. beta and g as [B, n_v, T] (= [1,6,1]) — pre-transposed.
            beta, g = self._compute_beta_g(b, a, B, T)
            b.deallocate(True)
            a.deallocate(True)
            beta = ttnn.reshape(beta, [B, self.n_v_per_row, T])
            g = ttnn.reshape(g, [B, self.n_v_per_row, T])

            # 5'. GQA expand q, k on the head dim (now dim 1).
            q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h, B, T, head_dim_axis=1)
            q_h.deallocate(True)
            k_h.deallocate(True)
        else:
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
        # V2-18: optional partial tt-lang recurrent kernel (per-head loop with
        # external readout matmul) — controlled by QWEN36_TT_LANG_RECURRENT=1.
        if self.use_tt_lang_recurrent_v3 and self._recurrent_v3_kernel_state is not None:
            # V2-17d: multi-head batched + readout-fused + in-place V3 kernel.
            # State is written in place to dn_state_buffer; new_state aliases
            # the persistent buffer (the post-forward ttnn.copy is a self-copy
            # which is short-circuited below).
            core_out, new_state = self.recurrent_gated_delta_rule_tt_lang_v3_decode(
                q=q_exp,
                k=k_exp,
                v=v_h,
                beta=beta,
                g=g,
                initial_state=self.dn_state_buffer,
            )
        elif self.use_tt_lang_recurrent_v2 and self._recurrent_v2_kernel_state is not None:
            # V2-17b: full multi-head + multi-core + fused-readout V3 kernel.
            core_out, new_state = self.recurrent_gated_delta_rule_tt_lang_decode_v2(
                q=q_exp,
                k=k_exp,
                v=v_h,
                beta=beta,
                g=g,
                initial_state=self.dn_state_buffer,
            )
        elif self.use_tt_lang_recurrent and self._recurrent_kernel_state is not None:
            core_out, new_state = self.recurrent_gated_delta_rule_tt_lang_decode(
                q=q_exp,
                k=k_exp,
                v=v_h,
                beta=beta,
                g=g,
                initial_state=self.dn_state_buffer,
            )
        else:
            core_out, new_state = recurrent_gated_delta_rule_ttnn_fp32(
                q=q_exp,
                k=k_exp,
                v=v_h,
                beta=beta,
                g=g,
                initial_state=self.dn_state_buffer,
                device=self.mesh_device,
                pre_transposed=use_fused_heads,
            )
        q_exp.deallocate(True)
        k_exp.deallocate(True)
        v_h.deallocate(True)
        # V2-16: when the fused tt-lang kernel is active, ``beta`` and ``g``
        # alias the persistent ``beta_out`` / ``g_out`` buffers — never
        # deallocate them. The next decode step re-uses the same addresses.
        if not (self.use_tt_lang_beta_g and self._beta_g_kernel_state is not None):
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
        # V2-17d: when V3 is active, new_state IS dn_state_buffer (kernel
        # wrote in place). Skip the self-copy to save one ttnn.copy per layer
        # per step.
        if not (self.use_tt_lang_recurrent_v3 and self._recurrent_v3_kernel_state is not None):
            ttnn.copy(new_state, self.dn_state_buffer)
            new_state.deallocate(True)
        ttnn.copy(new_conv_state, self.conv_state_buffer)
        new_conv_state.deallocate(True)

        return output

    # ------------------------------------------------------------------
    # Diagnostic accessor (test_deltanet_sharding_correctness)
    # ------------------------------------------------------------------
    def get_conv_weight_row(self, row_i):
        """Return host-side conv weight block for row_i [1280, 4]."""
        return self._conv_w_host[row_i]
