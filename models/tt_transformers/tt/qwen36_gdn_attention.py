# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B GatedDeltaNet (linear-attention) block — 1D tensor-parallel port.

This is the ``tt_transformers``-resident port of the 32-chip BH-Galaxy
``models/demos/qwen3_6_galaxy_v2/tt/qwen36_delta_attention.TtQwen36DeltaAttention``
class, retargeted for an 8-chip **1D tensor-parallel** mesh
(``MESH_DEVICE=p15x8`` → mesh shape ``(1, 8)``).

What is REUSED unchanged from the galaxy implementation
-------------------------------------------------------
The mesh-agnostic GDN cores all see the SAME per-chip shapes here as on the
galaxy mesh (the per-chip head counts are identical — see below), so they are
imported and reused verbatim:

  * ``recurrent_gated_delta_rule_ttnn_fp32`` (decode recurrence, fp32 state)
  * ``chunk_gated_delta_rule_ttnn`` / ``create_chunk_masks`` / ``l2_norm_ttnn``
    (chunked prefill)
  * ``_causal_conv1d_fir_mesh`` (FIR depthwise causal conv1d + SiLU)

Only the **weight sharding** and the **CCL** are redone for 1D TP.

Topology: galaxy 2D-TP vs this 1D-TP
------------------------------------
The galaxy class is a **2D-TP** layout on an ``(8, 4)`` mesh:
  * heads are split 8-way across mesh-ROWS (``mesh_rows=8``) via a per-row
    interleave + ``ShardTensor2dMesh(dims=(1, 0))``,
  * the hidden dim H is split 4-way across COLS (``mesh_cols=4``); the input
    projection therefore produces a PARTIAL inner-product sum that a
    ``ttnn.all_reduce(cluster_axis=1)`` (4-way col ring) completes,
  * the out-proj produces a partial sum over each row's 6 V-heads that a
    ``cluster_axis=0`` (8-way row ring) all_reduce completes.

For 8-chip 1D-TP we split **heads 8-way and DO NOT split H** (full H replicated
per chip). Per-chip head counts are IDENTICAL to galaxy
(``n_v_per_chip = 48 / 8 = 6``, ``n_k_per_chip = 16 / 8 = 2``,
``q_per_chip = 256``, ``v_per_chip = 768``), so the reused cores see the same
per-chip tensors.

Consequences for the two collectives:
  * **Input projection** ``x[full H] @ w_qkvz_chip[5120, 2048] -> [.., 2048]``:
    K is the full hidden dim ⇒ the per-chip output is the COMPLETE result, NOT
    a partial sum. So there is **NO input-side all_reduce** (the galaxy
    col-axis reduce is removed — 2 fewer CCLs / layer than galaxy). Same for
    ``w_ba``.
  * **Output projection** ``out_local[768] @ w_out_chip[768, 5120] -> [.., 5120]``:
    each chip owns its 6 V-heads ⇒ the per-chip output is a PARTIAL sum over
    the chip's heads. **One** ``ttnn.all_reduce(cluster_axis=1)`` over the
    single TP axis (axis 1 of the ``(1, 8)`` mesh) completes it.

Mesh mappers used (per weight)
------------------------------
  * ``w_qkvz`` [H=5120, 16384] → split the 16384 output dim 8-way (contiguous
    2048-wide chunks) with ``ShardTensorToMesh(dim=1)``. Built from the SAME
    per-chip interleave torch ``cat`` the galaxy class uses, so chunk ``i`` is
    ``[Q_i | K_i | V_i | Z_i]`` (2048 cols) — matching the per-chip
    ``_project_inputs`` slice offsets. dim-0 (H) is NOT split.
  * ``w_ba`` [H=5120, 96] → ``ShardTensorToMesh(dim=1)`` (12 cols/chip,
    ``[B_i | A_i]``).
  * ``w_out`` [n_v*hd=6144, H=5120] → ``ShardTensorToMesh(dim=0)`` (768
    rows/chip — the chip's 6 V-heads). Partial-sum output, reduced by the
    one out-proj all_reduce.
  * conv1d taps / ``A_log`` / ``dt_bias`` → per-head; shard the head dim 8-way
    with ``ShardTensorToMesh(dim=...)``.
  * ``norm.weight`` → ``ReplicateTensorToMesh`` (per-head-dim, head-local norm).

Persistent state buffers
-------------------------
Same per-chip layout as galaxy (``dn_state_buffer`` [B, 6, 128, 128] fp32 +
``conv_state_buffer`` [B, 3, 1280]), built with the 1D ``ReplicateTensorToMesh``
mapper — each chip already owns exactly its 6 local V-heads, so the buffers are
replicated and only ever touched through the chip-local conv / recurrent path.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy_v2.tt.qwen35_chunk_delta_rule_ops import (
    chunk_gated_delta_rule_ttnn,
    create_chunk_masks,
)
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_delta_attention import _causal_conv1d_fir_mesh
from models.demos.qwen3_6_galaxy_v2.tt.ttnn_delta_rule_ops_fp32 import recurrent_gated_delta_rule_ttnn_fp32

# --- Reused GDN cores (mesh-agnostic; per-chip shapes match galaxy) ----------
from models.tt_transformers.tt.ccl import tt_all_reduce


class TtQwen36GDNAttention(LightweightModule):
    """1D tensor-parallel GatedDeltaNet attention for Qwen3.6-27B (8-chip).

    Constructor signature matches the kwargs the ``tt_transformers`` decoder
    passes any attention class (``models/tt_transformers/tt/decoder.py``
    :53-66): ``mesh_device``, ``tt_ccl``, ``args``, ``state_dict``,
    ``weight_cache_path``, ``layer_num``, ``dtype``, ``transformation_mats``,
    ``configuration``, ``paged_attention_config``, ``use_paged_kv_cache``,
    ``prefetcher``. GDN ignores ``transformation_mats`` / ``rot_mats`` /
    ``page_table`` / ``kv_cache`` (no RoPE, no KV cache; state lives in
    ``dn_state_buffer`` / ``conv_state_buffer``).
    """

    def __init__(
        self,
        mesh_device,
        args,
        layer_num,
        dtype=ttnn.bfloat16,
        state_dict=None,
        weight_cache_path=None,
        transformation_mats=None,
        configuration=None,
        tt_ccl=None,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
        **kwargs,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args if args is not None else configuration
        self.model_config = self.args.get_model_config()
        self.layer_num = layer_num
        self.tt_ccl = tt_ccl
        # DeltaNet keeps its weights at the requested model dtype (bf8 by
        # default). Forcing bf16 dropped galaxy layer-0 PCC 0.999 -> 0.84 — the
        # recurrent exp(-A_log * softplus(...)) term relies on the bf8-quantised
        # inputs staying in range. Keep parity.
        self.dtype = dtype

        # --- Mesh topology: 1D-TP on a (1, 8) mesh. The single TP axis is the
        #     non-degenerate one. We do NOT assume axis index — derive it.
        self.cluster_shape = list(mesh_device.shape)  # [1, 8]
        assert 1 in self.cluster_shape, (
            f"TtQwen36GDNAttention is the 1D-TP port; expected a (1, N) or (N, 1) mesh, "
            f"got {self.cluster_shape}. Use the galaxy TtQwen36DeltaAttention for 2D meshes."
        )
        # tp_axis: the mesh axis that actually has the 8 chips (size > 1).
        self.tp_axis = 1 if self.cluster_shape[1] != 1 else 0
        self.tp_size = self.cluster_shape[self.tp_axis]

        # --- Model dimensions (read off ModelArgs; populated for qwen3.6) ----
        self.hidden_size = self.args.dim  # 5120
        self.n_k_heads = self.args.linear_num_key_heads  # 16
        self.n_v_heads = self.args.linear_num_value_heads  # 48
        self.head_dim = self.args.linear_head_dim  # 128
        self.conv_kernel = self.args.linear_conv_kernel  # 4
        self.eps = self.args.norm_eps
        self.max_batch_size = self.args.max_batch_size

        # --- Per-chip head counts (IDENTICAL to galaxy per-row counts) -------
        assert self.n_v_heads % self.tp_size == 0, f"n_v_heads={self.n_v_heads} % tp_size={self.tp_size}"
        assert self.n_k_heads % self.tp_size == 0, f"n_k_heads={self.n_k_heads} % tp_size={self.tp_size}"
        self.n_k_per_chip = self.n_k_heads // self.tp_size  # 2
        self.n_v_per_chip = self.n_v_heads // self.tp_size  # 6
        self.q_per_chip = self.n_k_per_chip * self.head_dim  # 256
        self.v_per_chip = self.n_v_per_chip * self.head_dim  # 768
        self.conv_per_chip = 2 * self.q_per_chip + self.v_per_chip  # 1280 = q+k+v
        # The reused galaxy cores read these "per_row" attribute names; mirror
        # them onto the same per-chip values so the imported helpers — and any
        # galaxy method we delegate to — find what they expect.
        self.mesh_rows = self.tp_size
        self.mesh_cols = 1
        self.n_k_per_row = self.n_k_per_chip
        self.n_v_per_row = self.n_v_per_chip
        self.q_per_row = self.q_per_chip
        self.v_per_row = self.v_per_chip
        self.conv_per_row = self.conv_per_chip

        # --- Prefill chunk masks (reuse galaxy chunk kernel) -----------------
        self.prefill_chunk_size = int(os.environ.get("QWEN36_GDN_CHUNK_SIZE", "32"))
        self._chunk_masks = create_chunk_masks(self.prefill_chunk_size, mesh_device)

        # --- Compute kernel: HiFi4 + fp32 dest accumulation ------------------
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # --- Number of TP links for the out-proj all_reduce ------------------
        try:
            self._out_num_links = int(os.environ.get("QWEN36_CCL_NUM_LINKS", "1"))
        except ValueError:
            self._out_num_links = 1

        # --- Weights (1D shard) ----------------------------------------------
        weights_dict = self._slice_layer_weights(state_dict, layer_num)
        self._build_weights(weights_dict)

        # --- Persistent buffers (1D replicated; chip already owns its heads) -
        self._conv_zero_pad = self._build_conv_zero_pad()
        self.dn_state_buffer = self._build_dn_state_buffer()
        self.conv_state_buffer = self._build_conv_state_buffer()

    # ------------------------------------------------------------------
    # Weight slicing + helpers
    # ------------------------------------------------------------------

    def _slice_layer_weights(self, state_dict, layer_num):
        """Pull this layer's GDN weights out of the full state_dict.

        Accepts either the meta-mapped key form ``layers.{n}.linear_attn.<rest>``
        (what ``load_checkpoints`` emits) or the raw HF form
        ``model.language_model.layers.{n}.linear_attn.<rest>`` (what the raw
        safetensors test loader uses). Returns a flat dict keyed
        ``linear_attn.<rest>`` — the form ``_resolve_weight`` expects.
        """
        if state_dict is None:
            return {}
        candidates = [
            f"layers.{layer_num}.linear_attn.",
            f"model.language_model.layers.{layer_num}.linear_attn.",
        ]
        out = {}
        for k, v in state_dict.items():
            for pfx in candidates:
                if k.startswith(pfx):
                    out["linear_attn." + k[len(pfx) :]] = v
                    break
        return out

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
        for k in candidate_keys:
            if k in weights_dict:
                return weights_dict[k]
        raise KeyError(
            f"DeltaNet weight not found. Tried: {candidate_keys}. "
            f"Available keys: {sorted(weights_dict.keys())[:20]}..."
        )

    def _build_weights(self, sd):
        H = self.hidden_size
        tp = self.tp_size
        hd_k = self.head_dim
        n_k = self.n_k_heads
        n_v = self.n_v_heads
        q_per_chip = self.q_per_chip
        v_per_chip = self.v_per_chip

        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        # 1D-TP output-dim shard: split the matmul OUTPUT dim (heads) 8-way; the
        # input dim (full H) is NOT split (replicated). ShardTensorToMesh(dim=1)
        # carves the output dim into ``tp`` contiguous chunks, one per chip.
        shard_out_dim1 = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)

        # ---- QKV / Z / B / A projection weights --------------------------------
        # HF Qwen3.6-27B ships block-wise separate keys (in_proj_qkv / in_proj_z /
        # in_proj_a / in_proj_b). Some checkpoints ship the fused in_proj_qkvz /
        # in_proj_ba. Support both, exactly as galaxy does.
        if any(k in sd for k in ("linear_attn.in_proj_qkv.weight", "in_proj_qkv.weight")):
            qkv_w = self._resolve_weight(sd, "linear_attn.in_proj_qkv.weight", "in_proj_qkv.weight")
            z_w = self._resolve_weight(sd, "linear_attn.in_proj_z.weight", "in_proj_z.weight")
            a_w = self._resolve_weight(sd, "linear_attn.in_proj_a.weight", "in_proj_a.weight")
            b_w = self._resolve_weight(sd, "linear_attn.in_proj_b.weight", "in_proj_b.weight")
        else:
            qkvz_w = self._resolve_weight(sd, "linear_attn.in_proj_qkvz.weight", "in_proj_qkvz.weight")
            ba_w = self._resolve_weight(sd, "linear_attn.in_proj_ba.weight", "in_proj_ba.weight")
            qkv_dim = 2 * n_k * hd_k + n_v * hd_k  # Q + K + V
            qkv_w = qkvz_w[:qkv_dim]
            z_w = qkvz_w[qkv_dim:]
            b_w = ba_w[:n_v]
            a_w = ba_w[n_v:]

        # in_proj_qkv is block-wise [Q(key_dim) | K(key_dim) | V(value_dim)].
        Q_w = qkv_w[: n_k * hd_k]
        K_w = qkv_w[n_k * hd_k : 2 * n_k * hd_k]
        V_w = qkv_w[2 * n_k * hd_k :]
        Q_w_T = Q_w.T.contiguous()  # [H, key_dim=2048]
        K_w_T = K_w.T.contiguous()  # [H, 2048]
        V_w_T = V_w.T.contiguous()  # [H, value_dim=6144]
        Z_w_T = z_w.T.contiguous()  # [H, 6144]

        # Build the QKVZ-fused weight with the SAME per-chip interleave the
        # galaxy class uses, so chip i's contiguous 2048-wide output chunk is
        # exactly [Q_i(256) | K_i(256) | V_i(768) | Z_i(768)]. The downstream
        # _project_inputs slices rely on this layout.
        #
        # NOTE: galaxy uses ShardTensor2dMesh(dims=(1,0)) to split this 16384
        # dim across mesh ROWS while ALSO splitting H across cols. For 1D we
        # split ONLY the 16384 output dim (dim=1) into ``tp`` contiguous 2048
        # chunks; H (dim 0) stays full → the matmul output is COMPLETE per chip
        # (no input-side all_reduce).
        QKVZ_chunks = []
        for i in range(tp):
            q_i = Q_w_T[:, i * q_per_chip : (i + 1) * q_per_chip]  # [H, 256]
            k_i = K_w_T[:, i * q_per_chip : (i + 1) * q_per_chip]  # [H, 256]
            v_i = V_w_T[:, i * v_per_chip : (i + 1) * v_per_chip]  # [H, 768]
            z_i = Z_w_T[:, i * v_per_chip : (i + 1) * v_per_chip]  # [H, 768]
            QKVZ_chunks.append(torch.cat([q_i, k_i, v_i, z_i], dim=-1))  # [H, 2048]
        QKVZ_w_T = torch.cat(QKVZ_chunks, dim=-1)  # [H, 16384]
        self.w_qkvz = self._to_device(QKVZ_w_T, shard_out_dim1)  # per-chip [H, 2048]

        # B + A fused (note in_proj_ba layout is b|a; here built b_i|a_i per chip).
        a_w_T = a_w.T.contiguous()  # [H, n_v=48]
        b_w_T = b_w.T.contiguous()  # [H, 48]
        BA_chunks = []
        for i in range(tp):
            b_i = b_w_T[:, i * self.n_v_per_chip : (i + 1) * self.n_v_per_chip]  # [H, 6]
            a_i = a_w_T[:, i * self.n_v_per_chip : (i + 1) * self.n_v_per_chip]  # [H, 6]
            BA_chunks.append(torch.cat([b_i, a_i], dim=-1))  # [H, 12]
        BA_w_T = torch.cat(BA_chunks, dim=-1)  # [H, 96]
        self.w_ba = self._to_device(BA_w_T, shard_out_dim1)  # per-chip [H, 12]

        # ---- conv1d weights: per-chip interleave [Q_i | K_i | V_i] on the
        #      channel dim, then shard that channel dim (dim 2) 8-way. --------
        conv_w_src = self._resolve_weight(sd, "linear_attn.conv1d.weight", "conv1d.weight")
        conv_w = conv_w_src.squeeze(1)  # [conv_dim=10240, K=4]
        conv_Q_w = conv_w[: n_k * hd_k]
        conv_K_w = conv_w[n_k * hd_k : 2 * n_k * hd_k]
        conv_V_w = conv_w[2 * n_k * hd_k :]
        chunks = []
        for i in range(tp):
            qc = conv_Q_w[i * q_per_chip : (i + 1) * q_per_chip]
            kc = conv_K_w[i * q_per_chip : (i + 1) * q_per_chip]
            vc = conv_V_w[i * v_per_chip : (i + 1) * v_per_chip]
            chunks.append(torch.cat([qc, kc, vc], dim=0))  # [conv_per_chip=1280, 4]
        conv_w_interleaved = torch.cat(chunks, dim=0)  # [10240, 4]

        shard_chan = ttnn.ShardTensorToMesh(self.mesh_device, dim=2)
        self.conv_weight_taps = []
        for tap in range(self.conv_kernel):
            tap_vec = conv_w_interleaved[:, tap]  # [10240]
            tap_3d = tap_vec.reshape(1, 1, tp * self.conv_per_chip)  # [1,1,10240]
            tap_tt = ttnn.from_torch(
                tap_3d,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard_chan,  # per-chip [1,1,1280]
            )
            self.conv_weight_taps.append(tap_tt)

        # ---- A_log / dt_bias: per-head [1, 1, n_v], shard head dim 8-way ----
        A_log = self._resolve_weight(sd, "linear_attn.A_log", "A_log")
        dt_bias = self._resolve_weight(sd, "linear_attn.dt_bias", "dt_bias")
        A_log_3d = A_log.reshape(1, 1, n_v)
        dt_bias_3d = dt_bias.reshape(1, 1, n_v)
        shard_head = ttnn.ShardTensorToMesh(self.mesh_device, dim=2)
        self.A_log = self._to_device(A_log_3d, shard_head, layout=ttnn.TILE_LAYOUT)  # per-chip [1,1,6]
        self.dt_bias = self._to_device(dt_bias_3d, shard_head, layout=ttnn.TILE_LAYOUT)

        # ---- norm weight (replicated; per-head-dim head-local RMSNorm) ------
        norm_w = self._resolve_weight(sd, "linear_attn.norm.weight", "norm.weight")
        norm_w_4d = norm_w.reshape(1, 1, self.head_dim // 32, 32)
        self.norm_weight = self._to_device(norm_w_4d, replicate, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # ---- output projection: [H, n_v*hd]; shard the INPUT (head) dim 8-way.
        # out_proj.weight HF shape is [H, value_dim=6144]; transpose to
        # [value_dim, H] then shard dim 0 (the 6144 = 48*128 head/input dim)
        # 8-way → each chip owns its 6 V-heads' 768-row slice. The per-chip
        # matmul out_local[768] @ w_out_chip[768, H] is a PARTIAL sum over the
        # chip's heads → completed by the one out-proj all_reduce.
        out_proj_w = self._resolve_weight(sd, "linear_attn.out_proj.weight", "out_proj.weight")
        out_proj_w_T = out_proj_w.T.contiguous()  # [value_dim=6144, H=5120]
        shard_in_dim0 = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        self.w_out = self._to_device(out_proj_w_T, shard_in_dim0)  # per-chip [768, 5120]

    # ------------------------------------------------------------------
    # Persistent buffer constructors (1D replicated)
    # ------------------------------------------------------------------

    def _build_conv_zero_pad(self):
        is_fp32 = self.dtype == ttnn.float32
        pad_torch = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_chip,
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
        """Persistent recurrent state [B, n_v_per_chip, head_dim, head_dim] fp32.

        fp32 (not bf16) so the recurrent state does not round-trip through bf16
        between decode steps (galaxy session lesson: bf16 state compounded to
        64L decode PCC 0.30). DRAM-resident; the recurrent core copies it into
        L1 each step.
        """
        state_torch = torch.zeros(
            self.max_batch_size, self.n_v_per_chip, self.head_dim, self.head_dim, dtype=torch.float32
        )
        return ttnn.from_torch(
            state_torch,
            device=self.mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _build_conv_state_buffer(self):
        is_fp32 = self.dtype == ttnn.float32
        buf_torch = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_chip,
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

    @staticmethod
    def _copy_state_into_buffer(new_state, buffer):
        """In-place writeback preserving the buffer device address (trace-safe).

        At ``max_batch_size`` rows == buffer rows this is a plain whole-buffer
        ``ttnn.copy``. (The galaxy per-user batched-prefill writeback path is
        not ported here — this 1D bring-up is single-user / batch-agnostic via
        dim-0; extend if batched per-user prefill is needed.)
        """
        ttnn.copy(new_state, buffer)

    def clear_state(self):
        """Zero both persistent buffers (fresh-sequence entry point)."""
        zero_state = torch.zeros(
            self.max_batch_size, self.n_v_per_chip, self.head_dim, self.head_dim, dtype=torch.float32
        )
        is_fp32 = self.dtype == ttnn.float32
        zero_conv = torch.zeros(
            self.max_batch_size,
            self.conv_kernel - 1,
            self.conv_per_chip,
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
            dtype=ttnn.float32 if is_fp32 else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy(new_state, self.dn_state_buffer)
        ttnn.copy(new_conv, self.conv_state_buffer)
        new_state.deallocate(True)
        new_conv.deallocate(True)

    # ------------------------------------------------------------------
    # Forward-stage helpers
    # ------------------------------------------------------------------

    def _project_inputs(self, x):
        """1D-TP input projection.

        ``x`` is the full-H residual (replicated across the TP chips). The
        per-chip QKVZ matmul therefore produces the COMPLETE per-chip output
        (K == full H) — no input-side all_reduce (unlike the galaxy 2D-TP
        col-axis reduce). Per-chip layout: [Q_256 | K_256 | V_768 | Z_768]
        repeated; the contiguous first 1280 cols ARE concat(q,k,v).

        Returns ``(mixed, z, a, b)`` where ``mixed`` is the contiguous Q|K|V
        slice (fed straight into the conv) and ``z`` the gate.
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        ck = self.compute_kernel

        qkvz = ttnn.linear(x, self.w_qkvz, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        out_rank = len(qkvz.shape)
        q_per = self.q_per_chip
        v_per = self.v_per_chip
        conv_per = self.conv_per_chip  # 1280 = q+k+v

        if out_rank == 3:
            B_, T_, _ = list(qkvz.shape)
            mixed = ttnn.slice(qkvz, [0, 0, 0], [B_, T_, conv_per], memory_config=mem)
            z = ttnn.slice(qkvz, [0, 0, conv_per], [B_, T_, conv_per + v_per], memory_config=mem)
        elif out_rank == 4:
            B_, D1_, T_, _ = list(qkvz.shape)
            mixed = ttnn.slice(qkvz, [0, 0, 0, 0], [B_, D1_, T_, conv_per], memory_config=mem)
            z = ttnn.slice(qkvz, [0, 0, 0, conv_per], [B_, D1_, T_, conv_per + v_per], memory_config=mem)
        else:
            raise RuntimeError(f"Unexpected qkvz rank {out_rank}: shape={qkvz.shape}")
        qkvz.deallocate(True)

        # B + A — again COMPLETE per chip (K == full H), no all_reduce.
        ba = ttnn.linear(x, self.w_ba, dtype=self.dtype, memory_config=mem, compute_kernel_config=ck)
        n_v_per = self.n_v_per_chip
        if out_rank == 3:
            B_, T_, _ = list(ba.shape)
            b = ttnn.slice(ba, [0, 0, 0], [B_, T_, n_v_per], memory_config=mem)
            a = ttnn.slice(ba, [0, 0, n_v_per], [B_, T_, 2 * n_v_per], memory_config=mem)
        else:
            B_, D1_, T_, _ = list(ba.shape)
            b = ttnn.slice(ba, [0, 0, 0, 0], [B_, D1_, T_, n_v_per], memory_config=mem)
            a = ttnn.slice(ba, [0, 0, 0, n_v_per], [B_, D1_, T_, 2 * n_v_per], memory_config=mem)
        ba.deallocate(True)
        return mixed, z, a, b

    def _apply_conv_and_split(self, mixed, B, T, conv_state=None):
        """Causal conv1d + SiLU on the contiguous Q|K|V block, then split.

        Reuses the galaxy ``_causal_conv1d_fir_mesh`` FIR helper verbatim
        (mesh-agnostic; it operates on the chip-local conv channels).
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        mixed_conv, new_conv_state = _causal_conv1d_fir_mesh(
            mixed,
            self.conv_weight_taps,
            self.conv_kernel,
            self.mesh_device,
            memory_config=mem,
            conv_state=conv_state,
            conv_state_zero_pad=self._conv_zero_pad,
        )
        q_conv = ttnn.slice(mixed_conv, [0, 0, 0], [B, T, self.q_per_chip], memory_config=mem)
        k_conv = ttnn.slice(mixed_conv, [0, 0, self.q_per_chip], [B, T, 2 * self.q_per_chip], memory_config=mem)
        v_conv = ttnn.slice(mixed_conv, [0, 0, 2 * self.q_per_chip], [B, T, self.conv_per_chip], memory_config=mem)
        mixed_conv.deallocate(True)
        return q_conv, k_conv, v_conv, new_conv_state

    def _compute_beta_g(self, b, a):
        """beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)."""
        mem = ttnn.DRAM_MEMORY_CONFIG
        beta = ttnn.sigmoid(b, memory_config=mem)
        a_biased = ttnn.add(a, self.dt_bias, memory_config=mem)
        sp = ttnn.softplus(a_biased, memory_config=mem)
        A_exp = ttnn.exp(self.A_log, memory_config=ttnn.L1_MEMORY_CONFIG)
        g = ttnn.multiply(ttnn.neg(A_exp, memory_config=mem), sp, memory_config=mem)
        return beta, g

    def _gqa_expand_q_k(self, q, k, head_dim_axis=2):
        ratio = self.n_v_per_chip // self.n_k_per_chip  # 3
        mem = ttnn.DRAM_MEMORY_CONFIG
        q_e = ttnn.repeat_interleave(q, ratio, dim=head_dim_axis, memory_config=mem)
        k_e = ttnn.repeat_interleave(k, ratio, dim=head_dim_axis, memory_config=mem)
        return q_e, k_e

    def _apply_norm_gated(self, core_out, z, B, T):
        """GroupRMSNormGated: rms_norm(core_out) * silu(z) (head-local)."""
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
        out = ttnn.reshape(out, [B, T, self.v_per_chip])
        return out

    def _output_proj_and_reduce(self, out_flat, B, T):
        """Out-proj + the SINGLE 1D-TP reduce-scatter.

        ``out_flat`` is [B, T, v_per_chip=768] (the chip's 6 V-heads). The
        matmul against ``w_out`` [768, H] produces a PARTIAL sum over the chip's
        heads (each chip's contribution to ALL H output columns). The framework
        residual stream is FRACTURED (H sharded H/tp per device), so we use the
        framework ``tt_all_reduce`` which, on a ``(1, tp)`` mesh, performs a
        reduce_scatter (dim=3): it SUMS the tp partials and SCATTERS H → each
        device gets its H/tp = 640-column slice. The fractured [.,.,T,H/tp]
        output then adds directly to the fractured residual (matches
        DefaultAttention's ``tt_all_reduce(cluster_axis=0, dim=3)``).
        """
        mem = ttnn.DRAM_MEMORY_CONFIG
        partial = ttnn.linear(
            out_flat,
            self.w_out,
            dtype=ttnn.bfloat16,
            memory_config=mem,
            compute_kernel_config=self.compute_kernel,
        )
        # tt_all_reduce reduce_scatters on dim=3, so the input must be 4D
        # [1, 1, B*T, H]. At decode (T=1) the partial is [1, 1, H] (3D) and
        # tt_all_reduce's auto-4D-reshape is skipped (dims 0,1 already 1) → it
        # would reduce_scatter a non-existent dim 3. Reshape explicitly.
        if len(partial.shape) == 3:
            _b, _t, _h = list(partial.shape)
            partial = ttnn.reshape(partial, [1, 1, _b * _t, _h])
        reduced = tt_all_reduce(
            partial,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=self._out_num_links,
            memory_config=mem,
            dtype=ttnn.bfloat16,
        )
        partial.deallocate(True)
        return reduced

    # ------------------------------------------------------------------
    # Public API — matches the tt_transformers attention surface
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
        kv_cache=None,
        **kwargs,
    ):
        # ``mode`` may be the string "prefill"/"decode" OR the tt_transformers
        # Mode enum (model.forward passes Mode.PREFILL/Mode.DECODE). Normalize.
        mode_val = mode.value if hasattr(mode, "value") else mode
        if mode_val == "prefill":
            return self.forward_prefill(x)
        return self.forward_decode(x)

    def forward_prefill(self, x):
        """Prefill (T>1) using the reused chunked delta-rule kernel."""
        orig_shape = list(x.shape)
        if len(orig_shape) == 4:
            B, _, T, H = orig_shape
            x = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape

        # 1. Projections (complete per chip; no input all_reduce)
        mixed, z, a, b = self._project_inputs(x)

        # 2. Conv1d + split (fresh conv state)
        q_conv, k_conv, v_conv, new_conv_state = self._apply_conv_and_split(mixed, B, T, conv_state=None)

        # 3. Per-head layout
        q_h = ttnn.reshape(q_conv, [B, T, self.n_k_per_chip, self.head_dim])
        k_h = ttnn.reshape(k_conv, [B, T, self.n_k_per_chip, self.head_dim])
        v_h = ttnn.reshape(v_conv, [B, T, self.n_v_per_chip, self.head_dim])
        z_h = ttnn.reshape(z, [B, T, self.n_v_per_chip, self.head_dim])
        q_conv.deallocate(True)
        k_conv.deallocate(True)
        v_conv.deallocate(True)

        # 4. beta and g
        beta, g = self._compute_beta_g(b, a)
        b.deallocate(True)
        a.deallocate(True)

        # 5. GQA expand q, k (n_k -> n_v)
        q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h)
        q_h.deallocate(True)
        k_h.deallocate(True)

        # 6. Chunked delta rule (reused galaxy kernel)
        core_out, new_state = chunk_gated_delta_rule_ttnn(
            q=q_exp,
            k=k_exp,
            v=v_h,
            beta=beta,
            g=g,
            chunk_size=self.prefill_chunk_size,
            initial_state=None,
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

        # 8. Output projection + 1D-TP all_reduce
        output = self._output_proj_and_reduce(out, B, T)
        out.deallocate(True)

        # 9. Write state into persistent buffers (trace-safe in-place copy)
        self._copy_state_into_buffer(new_state, self.dn_state_buffer)
        self._copy_state_into_buffer(new_conv_state, self.conv_state_buffer)
        new_state.deallocate(True)
        new_conv_state.deallocate(True)

        # Match the decoder/DefaultAttention prefill contract: [1, 1, B*T, H]
        # (the residual stream is 4D; a 3D [B,T,H] breaks the residual add).
        if len(output.shape) == 3:
            _Bo, _To, _Ho = list(output.shape)
            output = ttnn.reshape(output, [1, 1, _Bo * _To, _Ho])
        return output

    def forward_decode(self, x):
        """Decode (T=1) using the reused fp32-state recurrent delta-rule kernel.

        Reads recurrent_state / conv_state from the persistent buffers and
        writes the new state back in place (trace-safe).
        """
        orig_shape = list(x.shape)
        Bn = self.max_batch_size
        if len(orig_shape) == 4:
            _d0, _d1, R, H = orig_shape
            assert _d0 == 1 and _d1 == 1, f"Decode expects [1,1,R,H], got {orig_shape}"
        else:
            _d0, R, H = orig_shape
            assert _d0 == 1, f"Decode expects [1,R,H], got {orig_shape}"
        assert R >= Bn, f"row slot R={R} < max_batch_size={Bn}"

        # The decode row slot R is tile-padded (=32) even at batch-1. Move the R
        # rows into dim-0 and slice the Bn valid users → [Bn, 1, H] user-batch
        # (T stays 1). (The old Bn==1 fast path assumed R==1, which only held for
        # the galaxy R==max_batch_size decode; tt_transformers always pads to 32.)
        x = ttnn.reshape(x, [R, 1, H])
        if R != Bn:
            x = ttnn.slice(x, [0, 0, 0], [Bn, 1, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        B, T = Bn, 1

        # 1. Projections
        mixed, z, a, b = self._project_inputs(x)

        # 2. Conv1d + split — read persistent conv_state buffer
        q_conv, k_conv, v_conv, new_conv_state = self._apply_conv_and_split(
            mixed, B, T, conv_state=self.conv_state_buffer
        )

        # 3. Per-head layout
        q_h = ttnn.reshape(q_conv, [B, T, self.n_k_per_chip, self.head_dim])
        k_h = ttnn.reshape(k_conv, [B, T, self.n_k_per_chip, self.head_dim])
        v_h = ttnn.reshape(v_conv, [B, T, self.n_v_per_chip, self.head_dim])
        z_h = ttnn.reshape(z, [B, T, self.n_v_per_chip, self.head_dim])
        q_conv.deallocate(True)
        k_conv.deallocate(True)
        v_conv.deallocate(True)

        # 4. beta and g
        beta, g = self._compute_beta_g(b, a)
        b.deallocate(True)
        a.deallocate(True)

        # 5. GQA expand q, k
        q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h)
        q_h.deallocate(True)
        k_h.deallocate(True)

        # 6. Recurrent delta rule — fp32-state fork (reused galaxy core).
        #
        # Up-cast q/k/v to fp32 BEFORE the core. The core (pre_transposed=False)
        # transposes q/k/v on dims (1,2) and only THEN up-casts bf16→fp32. On
        # Blackhole, ttnn.transpose of a bf16 tensor whose head dim is sub-tile
        # (H=6, padded to a 32-tile) corrupts the data → the recurrent matmuls
        # produce ~1e36 and decode collapses to noise (PCC ≈ 0). Transposing in
        # fp32 is correct. This is perf-neutral: the core up-casts q/k/v to fp32
        # internally anyway (the default, bf16_in off), so we only move the cast
        # ahead of the transpose. (Isolated via per-tensor fp32 sweep: q/k/v fp32
        # → PCC 0.9996; beta/g fp32 alone → still 0.) The galaxy decode dodges
        # this by running pre_transposed=True (fused heads, no in-core transpose).
        _mem = ttnn.DRAM_MEMORY_CONFIG
        q_exp = ttnn.typecast(q_exp, ttnn.float32, memory_config=_mem)
        k_exp = ttnn.typecast(k_exp, ttnn.float32, memory_config=_mem)
        v_h = ttnn.typecast(v_h, ttnn.float32, memory_config=_mem)
        core_out, new_state = recurrent_gated_delta_rule_ttnn_fp32(
            q=q_exp,
            k=k_exp,
            v=v_h,
            beta=beta,
            g=g,
            initial_state=self.dn_state_buffer,
            device=self.mesh_device,
            pre_transposed=False,
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

        # 8. Output projection + 1D-TP all_reduce
        output = self._output_proj_and_reduce(out, B, T)
        out.deallocate(True)

        # 9. In-place state writeback (trace-safe)
        ttnn.copy(new_state, self.dn_state_buffer)
        new_state.deallocate(True)
        ttnn.copy(new_conv_state, self.conv_state_buffer)
        new_conv_state.deallocate(True)

        # Map the [B, 1, H_out] user-batch back to the tile-padded R-row decode
        # slot: users in rows 0..B-1, rows B..R-1 zero-padded (unused — only the
        # valid rows are sampled). Matches the [1, 1, R, H_out] residual stream.
        _Ho = list(output.shape)[-1]
        output = ttnn.reshape(output, [1, 1, B, _Ho])
        if R != B:
            output = ttnn.pad(output, [(0, 0), (0, 0), (0, R - B), (0, 0)], value=0.0)
        return output
