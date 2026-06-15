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
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy_v2.tt import qwen36_delta_attention as _galaxy_delta_attention_mod
from models.demos.qwen3_6_galaxy_v2.tt.qwen35_chunk_delta_rule_ops import (
    chunk_gated_delta_rule_ttnn,
    create_chunk_masks,
    l2_norm_ttnn,
)
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_delta_attention import _causal_conv1d_fir_mesh
from models.demos.qwen3_6_galaxy_v2.tt.ttnn_delta_rule_ops_fp32 import (
    _fp32_compute_cfg_hifi4,
    recurrent_gated_delta_rule_ttnn_fp32,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    _recurrent_read_query_program_config,
)

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
        # Lower-fidelity kernel for the bulk input/output projections (qkvz, ba,
        # out). The recurrent state + norm stay HiFi4/fp32; the projections feed
        # bf8 weights and tolerate HiFi2 without fp32 dest-acc (cheaper MACs).
        self.compute_kernel_proj = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
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

        # --- V3 fused tt-lang recurrent decode kernel ------------------------
        # Ports the galaxy fused-kernel recurrent core (1 GenericOp launch /
        # layer, ~6us) in place of the ~47-op pure-ttnn recurrent core. Gated
        # ON by default; falls back to the pure-ttnn path on build failure
        # (missing emitted kernels etc.).
        #
        # KERNELS_DIR is the SAME emitted-kernels dir the galaxy uses (the
        # kernels live next to the galaxy module). V_HEADS / HEAD_DIM are the
        # per-chip head counts (identical to galaxy per-row); TILE=32;
        # V_TILES = head_dim // 32 = 4.
        self._REC3_KERNELS_DIR = (
            Path(_galaxy_delta_attention_mod.__file__).resolve().parent / "kernels" / "recurrent_delta_rule_v3"
        )
        self._REC3_TILE = 32
        self._REC3_HEAD_DIM = self.head_dim  # 128
        self._REC3_K_TILES = self.head_dim // self._REC3_TILE  # 4
        self._REC3_V_TILES = self.head_dim // self._REC3_TILE  # 4
        self._REC3_V_HEADS = self.n_v_per_chip  # 6
        self._REC3_NUM_TENSORS = 8
        self._REC3_KERNEL_TENSOR_INDICES = [
            [],
            [5, 4, 2, 1, 0, 3],  # noc reader: beta, decay, k_col, q, state, v
            [7, 6],  # noc writer: o, state_out
        ]
        self._REC3_NUM_CBS = 10
        self._REC3_CB_PAGE = 4096
        self._REC3_CB_TOTAL = 8192

        # V3 fused recurrent: default OFF. The kernel is validated standalone
        # (PCC 1.0 on BH), but the in-model launch via _launch_recurrent_v3_kernel
        # segfaults — two suspects to clear before enabling: (1) the class
        # program-builder _launch_recurrent_v3_kernel vs the test's _run_kernel
        # (only the latter was validated standalone), (2) the core range
        # (2,0)-(5,5) may hit a harvested core on some BH chip (standalone passed
        # with (0,0)-(3,5)). Opt-in via QWEN36_GDN_V3_RECURRENT=1 for debugging.
        self.use_v3_recurrent = os.environ.get("QWEN36_GDN_V3_RECURRENT", "0") == "1"
        self._recurrent_v3_kernel_state = None
        if self.use_v3_recurrent:
            try:
                self._recurrent_v3_kernel_state = self._build_recurrent_v3_kernel_state()
            except Exception as e:  # missing kernel files, build error, etc.
                import logging

                logging.getLogger(__name__).warning(
                    "TtQwen36GDNAttention: V3 fused recurrent kernel unavailable "
                    "(%s); falling back to pure-ttnn recurrent decode path.",
                    e,
                )
                self.use_v3_recurrent = False
                self._recurrent_v3_kernel_state = None

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

    # ------------------------------------------------------------------
    # V3 fused tt-lang recurrent decode kernel (ported verbatim from galaxy
    # TtQwen36DeltaAttention; self.n_v_per_row -> self.n_v_per_chip via the
    # _REC3_V_HEADS attribute). DO NOT diverge from the galaxy math.
    # ------------------------------------------------------------------

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
        assert H == self._REC3_V_HEADS, f"V3 H={H} != n_v_per_chip={self._REC3_V_HEADS}"
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
        ck = self.compute_kernel_proj

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
            compute_kernel_config=self.compute_kernel_proj,
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

        use_v3 = self.use_v3_recurrent and self._recurrent_v3_kernel_state is not None

        if use_v3:
            # 3'. V3 fused kernel wants the NON-transposed [B, T, H, D] layout
            #     (it does the l2norm + scale + transpose internally). z_h stays
            #     [B, T, n_v, D] for the norm.
            q_h = ttnn.reshape(q_conv, [B, T, self.n_k_per_chip, self.head_dim])
            k_h = ttnn.reshape(k_conv, [B, T, self.n_k_per_chip, self.head_dim])
            v_h = ttnn.reshape(v_conv, [B, T, self.n_v_per_chip, self.head_dim])
            z_h = ttnn.reshape(z, [B, T, self.n_v_per_chip, self.head_dim])
            q_conv.deallocate(True)
            k_conv.deallocate(True)
            v_conv.deallocate(True)

            # 4'. beta and g → [B, T, n_v] (the V3 method transposes internally).
            beta, g = self._compute_beta_g(b, a)
            b.deallocate(True)
            a.deallocate(True)
            beta = ttnn.reshape(beta, [B, T, self.n_v_per_chip])
            g = ttnn.reshape(g, [B, T, self.n_v_per_chip])

            # 5'. GQA expand q, k on the head dim (dim 2 in the [B,T,H,D] layout).
            q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h, head_dim_axis=2)
            q_h.deallocate(True)
            k_h.deallocate(True)

            # CRITICAL Blackhole gotcha: the V3 method does ttnn.transpose(q,1,2)
            # on its input BEFORE up-casting to fp32. Transposing a bf16 tensor
            # whose head dim is sub-tile (H=6 padded to a 32-tile) CORRUPTS the
            # data on Blackhole (proven bug). So up-cast q/k/v to fp32 HERE,
            # before calling the V3 method, so the in-method transpose runs on
            # fp32 (safe). beta/g may stay bf16 — the method casts them.
            def _cast_fp32(t):
                if t.dtype == ttnn.float32:
                    return t
                t_new = ttnn.typecast(t, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                if t_new is not t:
                    t.deallocate(True)
                return t_new

            q_exp = _cast_fp32(q_exp)
            k_exp = _cast_fp32(k_exp)
            v_h = _cast_fp32(v_h)

            # 6'. Fused V3 recurrent decode. State written IN PLACE — new_state
            #     aliases self.dn_state_buffer (the post-call writeback copy is
            #     skipped below to avoid a self-copy).
            core_out, new_state = self.recurrent_gated_delta_rule_tt_lang_v3_decode(
                q=q_exp,
                k=k_exp,
                v=v_h,
                beta=beta,
                g=g,
                initial_state=self.dn_state_buffer,
            )
        else:
            # 3. Per-head layout — FUSED-HEADS [B, H, T, D] (pre_transposed). At
            #    decode T==1 the [B, H, 1, D] and [B, 1, H, D] layouts share identical
            #    contiguous memory, so this reshape is bit-identical to the
            #    [B, T, H, D] form but lets the recurrent core run pre_transposed:
            #    it SKIPS its 5 in-core transposes. That avoids the Blackhole bf16
            #    sub-tile transpose corruption (H=6 padded to a 32-tile) WITHOUT the
            #    fp32 up-cast — so q/k/v stay bf16 and the recurrent matmuls run bf16
            #    (QWEN36_DN_RECUR_BF16_IN, default on) instead of fp32. (Matches the
            #    galaxy use_dn_fused_heads decode path.)
            q_h = ttnn.reshape(q_conv, [B, self.n_k_per_chip, T, self.head_dim])
            k_h = ttnn.reshape(k_conv, [B, self.n_k_per_chip, T, self.head_dim])
            v_h = ttnn.reshape(v_conv, [B, self.n_v_per_chip, T, self.head_dim])
            z_h = ttnn.reshape(z, [B, T, self.n_v_per_chip, self.head_dim])
            q_conv.deallocate(True)
            k_conv.deallocate(True)
            v_conv.deallocate(True)

            # 4. beta and g → [B, n_v, T] (pre_transposed layout; bit-identical at T=1)
            beta, g = self._compute_beta_g(b, a)
            b.deallocate(True)
            a.deallocate(True)
            beta = ttnn.reshape(beta, [B, self.n_v_per_chip, T])
            g = ttnn.reshape(g, [B, self.n_v_per_chip, T])

            # 5. GQA expand q, k on the head dim (now dim 1)
            q_exp, k_exp = self._gqa_expand_q_k(q_h, k_h, head_dim_axis=1)
            q_h.deallocate(True)
            k_h.deallocate(True)

            # 6. Recurrent delta rule — pre_transposed=True (bf16 q/k/v, no in-core transpose).
            core_out, new_state = recurrent_gated_delta_rule_ttnn_fp32(
                q=q_exp,
                k=k_exp,
                v=v_h,
                beta=beta,
                g=g,
                initial_state=self.dn_state_buffer,
                device=self.mesh_device,
                pre_transposed=True,
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

        # 9. In-place state writeback (trace-safe). In the V3 path the kernel
        #    already wrote state in place (new_state IS dn_state_buffer), so the
        #    copy would be a self-copy — skip it (mirrors galaxy forward_decode).
        if not use_v3:
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
