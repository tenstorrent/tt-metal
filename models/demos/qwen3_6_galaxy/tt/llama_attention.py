# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B Gated Attention block for BH GLX 8×4 mesh.

Forked from models/demos/llama3_70b_galaxy/tt/llama_attention.py and adapted for:
1. Output gate: q_proj output is [Q|gate] fused per-head (2×head_dim per head).
   After reshape: query, gate split → apply sigmoid(gate) * attn_out before WO.
2. Partial RoPE: rotary_dim=64 of head_dim=256 via Qwen36RopeSetup.apply_partial_rope.
   (NOT the fused full-head-dim kernel.)
3. Zero-centered QK-norm: q_norm/k_norm weight baked with +1.0 shift.
4. Pad-and-slice for head count: native n_q=24, n_kv=4 padded to 64/8 to fit
   existing Qwen3-32B program configs.
5. head_dim=256 vs 128: reshape constants re-derived for head_dim=256.

Weight layout (per-head interleaved Q+gate in q_proj):
    q_proj.weight: [n_q * head_dim * 2, H] = [12288, 5120]
    Per-head interleave: rows [Q_h0(256) | gate_h0(256) | Q_h1(256) | gate_h1(256) | ...]

Pad-and-slice strategy (matching OLMo pattern):
    - n_q native=24 → padded=64 heads (zero-pad extra head rows in wq and wgate)
    - n_kv native=4 → padded=8 heads (zero-pad extra rows in wk, wv)
    - WO input dim: native 6144 → padded 16384 (zero-pad WO input cols beyond 6144)
    - Padded heads produce zero SDPA output → correct final output

Data flow (prefill):
    x [B, T, H] (replicated or sharded)
    → xqkvg [B, T, (64+64+8+8)*256 = 36864] via fused wqkvg matmul
    → split: q[B,T,64,256], gate[B,T,64,256], k[B,T,8,256], v[B,T,8,256]
    → QK-norm per-head (zero-centered weight = 1+w baked in)
    → transpose to [B, n_heads, T, hd]
    → partial RoPE on Q and K (first 64 dims rotated)
    → KV cache fill
    → GQA expand k,v: 8 → 64 heads
    → SDPA with causal mask → [B, 64, T, 256]
    → transpose + flatten → [B, T, 64*256=16384]
    → sigmoid(gate) * attn_out  (gate: [B, T, 16384])
    → WO [16384, 5120] → [B, T, 5120]
    → all_reduce across mesh rows (cluster_axis=0)

Standalone test usage (replicated input):
    attn = TtQwen36GatedAttention(mesh_device, args, state_dict, layer_num=3)
    out_tt = attn.forward_prefill(x_tt, rot_mats=(cos_tt, sin_tt), kv_cache=None)
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

TILE = 32


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pad_rows(t: torch.Tensor, target_rows: int, value: float = 0.0) -> torch.Tensor:
    """Pad tensor along dim=0 (rows) to target_rows."""
    current = t.shape[0]
    if current >= target_rows:
        return t
    pad_size = target_rows - current
    return torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=value)


def _pad_cols(t: torch.Tensor, target_cols: int, value: float = 0.0) -> torch.Tensor:
    """Pad tensor along last dim (cols) to target_cols."""
    current = t.shape[-1]
    if current >= target_cols:
        return t
    return torch.nn.functional.pad(t, (0, target_cols - current), value=value)


def _make_qknorm_weight_tt(weight_torch: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Create zero-centered QK-norm weight on device.

    Applies (1+w) and stores as [1, 1, head_dim//32, 32] tile-aligned row-major,
    replicated across all mesh devices.
    """
    w = 1.0 + weight_torch.float()
    dim = w.numel()
    assert dim % TILE == 0, f"head_dim={dim} must be tile-aligned"
    w_4d = w.reshape(1, 1, dim // TILE, TILE)
    return ttnn.from_torch(
        w_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


# ---------------------------------------------------------------------------
# Main attention class
# ---------------------------------------------------------------------------


class TtQwen36GatedAttention(LightweightModule):
    """Qwen3.6-27B Gated Multi-Head Attention for BH GLX 8×4 mesh.

    Key differences from TtLlamaAttention (Qwen3-32B):
    - Output gate fused with Q projection (per-head interleaved layout)
    - Partial RoPE (rotary_dim=64, not full head_dim=256)
    - Zero-centered QK-norm weights (baked +1 shift)
    - Native head counts: n_q=24, n_kv=4 (no padding in forward path)

    Weight layout (fused QKVG):
        wqkvg: [Q_native | gate_native | K_native | V_native].T
               = [14336, 5120] → stored as [5120, 14336] for ttnn.linear
    WO weight:
        wo: [H, n_q * head_dim] = [5120, 6144] → stored as [6144, 5120]

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        Full 8×4 BH GLX mesh.
    args : TtQwen36ModelArgs
        Model configuration.
    state_dict : dict
        State dict for THIS layer's self_attn block. Expected keys:
        q_proj.weight [12288, 5120], k_proj.weight [1024, 5120],
        v_proj.weight [1024, 5120], o_proj.weight [5120, 6144],
        q_norm.weight [256], k_norm.weight [256]
    layer_num : int
        Layer index (for logging/debug only).
    rope_setup : Qwen36RopeSetup | None
        Optional rope setup for partial RoPE application.
    dtype : ttnn.DataType
        Activation dtype. Default bfloat16.
    kv_cache_max_seq_len : int
        Maximum KV cache sequence length.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        args: TtQwen36ModelArgs,
        state_dict: dict,
        layer_num: int = 0,
        rope_setup=None,
        dtype=ttnn.bfloat16,
        kv_cache_max_seq_len: int = 512,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype
        self.rope_setup = rope_setup
        self.cluster_shape = args.cluster_shape  # [8, 4]

        # Native dimensions
        self.n_q = args.n_heads  # 24
        self.n_kv = args.n_kv_heads  # 4
        self.head_dim = args.head_dim  # 256
        self.hidden_size = args.dim  # 5120
        self.rotary_dim = args.rope_dim  # 64
        self.eps = args.norm_eps  # 1e-6

        # GQA ratio (native)
        self.gqa_ratio = self.n_q // self.n_kv  # 6

        # Scale
        self.scale = self.head_dim**-0.5

        # KV cache params
        self.kv_cache_max_seq_len = kv_cache_max_seq_len
        self.max_batch_size = args.max_batch_size

        # Compute kernels
        self.compute_kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Build weights and KV cache
        self._build_weights(state_dict)
        self._build_kv_cache()

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------

    def _build_weights(self, sd: dict):
        """Prepare and upload all weights to device.

        q_proj.weight layout (HF Qwen3Next convention):
            [Q_h0(hd) | gate_h0(hd) | Q_h1(hd) | gate_h1(hd) | ...]
            shape: [n_q * 2 * head_dim, H] = [12288, 5120]

        We de-interleave into separate Q and gate blocks, then fuse as:
            wqkvg = [Q_native | gate_native | K_native | V_native].T
        shape on device (2D): [H=5120, (n_q+n_q+n_kv+n_kv)*hd] = [5120, 14336]

        WO weight is stored as native [n_q*hd, H] = [6144, 5120].
        """
        hd = self.head_dim  # 256
        n_q = self.n_q  # 24
        n_kv = self.n_kv  # 4
        H = self.hidden_size  # 5120

        # 1. De-interleave Q and gate from q_proj.weight
        q_proj_w = sd["q_proj.weight"]  # [12288, 5120]
        expected_q = (n_q * 2 * hd, H)
        assert q_proj_w.shape == expected_q, f"q_proj.weight: expected {expected_q}, got {q_proj_w.shape}"
        # Reshape to [n_q, 2, hd, H] then split on dim=1
        q_2hd = q_proj_w.reshape(n_q, 2, hd, H)
        wq_native = q_2hd[:, 0, :, :].reshape(n_q * hd, H)  # [6144, 5120]
        wgate_native = q_2hd[:, 1, :, :].reshape(n_q * hd, H)  # [6144, 5120]

        wk_native = sd["k_proj.weight"]  # [1024, 5120]
        wv_native = sd["v_proj.weight"]  # [1024, 5120]

        # 2. Fuse: [Q | gate | K | V] then transpose — all native sizes
        # Total = (n_q + n_q + n_kv + n_kv) * hd = (24+24+4+4)*256 = 14336
        wqkvg = torch.cat([wq_native, wgate_native, wk_native, wv_native], dim=0)  # [14336, 5120]
        # TTNN linear(x, W): x=[B,T,H], W=[H, out_dim=14336]
        # Use 2D [H, out_dim] to keep output rank same as input
        wqkvg_T = wqkvg.T.contiguous()  # [5120, 14336]

        self.wqkvg = ttnn.from_torch(
            wqkvg_T,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # 3. WO weight: [H, n_q * hd] = [5120, 6144] → store as [n_q*hd, H] = [6144, 5120]
        wo_native = sd["o_proj.weight"]  # [5120, 6144]
        expected_wo = (H, n_q * hd)
        assert wo_native.shape == expected_wo, f"o_proj.weight: expected {expected_wo}, got {wo_native.shape}"
        # TTNN linear(x, W): x=[B,T,n_q*hd=6144], W=[6144, H=5120]
        wo_T = wo_native.T.contiguous()  # [6144, 5120]

        self.wo = ttnn.from_torch(
            wo_T,
            device=self.mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # 4. QK-norm weights (zero-centered: bake +1 shift)
        self.q_norm_w = _make_qknorm_weight_tt(sd["q_norm.weight"], self.mesh_device)
        self.k_norm_w = _make_qknorm_weight_tt(sd["k_norm.weight"], self.mesh_device)

    # ------------------------------------------------------------------
    # KV cache construction
    # ------------------------------------------------------------------

    def _build_kv_cache(self):
        """Allocate on-device KV cache tensors.

        Shape: [max_batch_size, n_kv, max_seq_len, head_dim]
        Replicated across all mesh devices.
        """
        cache_k = torch.zeros(
            self.max_batch_size,
            self.n_kv,
            self.kv_cache_max_seq_len,
            self.head_dim,
        )
        cache_v = torch.zeros(
            self.max_batch_size,
            self.n_kv,
            self.kv_cache_max_seq_len,
            self.head_dim,
        )
        self.layer_past = [
            ttnn.from_torch(
                kv,
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for kv in [cache_k, cache_v]
        ]

    # ------------------------------------------------------------------
    # Partial RoPE application
    # ------------------------------------------------------------------

    def _apply_partial_rope(
        self,
        x_tt: ttnn.Tensor,
        cos_tt: ttnn.Tensor,
        sin_tt: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply partial RoPE (first rotary_dim=64 dims) to x_tt.

        Delegates to rope_setup if provided, otherwise implements inline.
        x_tt: [..., head_dim=256]
        cos_tt, sin_tt: [..., rotary_dim=64]
        Returns: same shape as x_tt.
        """
        if self.rope_setup is not None:
            return self.rope_setup.apply_partial_rope(x_tt, cos_tt, sin_tt)

        # Inline implementation (slice → rotate → concat)
        rd = self.rotary_dim  # 64
        hd = self.head_dim  # 256
        shape = list(x_tt.shape)
        ndim = len(shape)

        # Slice rotated and pass-through parts
        b_rot = [0] * ndim
        e_rot = shape[:]
        e_rot[-1] = rd

        b_pass = [0] * ndim
        e_pass = shape[:]
        b_pass[-1] = rd

        x_rot = ttnn.slice(x_tt, b_rot, e_rot)
        x_pass = ttnn.slice(x_tt, b_pass, e_pass)

        # rotate_half: cat([-x2, x1])
        half = rd // 2
        sr = list(x_rot.shape)
        b_x1 = [0] * ndim
        e_x1 = sr[:]
        e_x1[-1] = half
        b_x2 = [0] * ndim
        e_x2 = sr[:]
        b_x2[-1] = half

        x1 = ttnn.slice(x_rot, b_x1, e_x1)
        x2 = ttnn.slice(x_rot, b_x2, e_x2)
        neg_x2 = ttnn.neg(x2)
        rh = ttnn.concat([neg_x2, x1], dim=-1)
        x1.deallocate(True)
        x2.deallocate(True)
        neg_x2.deallocate(True)

        # x_rot * cos + rh * sin
        rc = ttnn.multiply(x_rot, cos_tt)
        rs = ttnn.multiply(rh, sin_tt)
        x_rotated = ttnn.add(rc, rs)
        x_rot.deallocate(True)
        rh.deallocate(True)
        rc.deallocate(True)
        rs.deallocate(True)

        # Concat [rotated | pass-through]
        out = ttnn.concat([x_rotated, x_pass], dim=-1)
        x_rotated.deallocate(True)
        x_pass.deallocate(True)
        return out

    # ------------------------------------------------------------------
    # Forward: prefill
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple,
        user_id: int = 0,
        kv_cache=None,
        page_table=None,
        batch_size: int = 1,
        current_pos=None,
    ) -> ttnn.Tensor:
        """Prefill forward pass.

        Args:
            x: Input tensor. Acceptable shapes:
               - [B, 1, T, H] (standard 4D sharded)
               - [B, T, H]    (3D)
               All chips receive the same tensor (replicated or full).
            rot_mats: (cos_tt, sin_tt) from Qwen36RopeSetup.get_cos_sin_for_prefill.
                      cos_tt shape: [1, 1, T, 64], replicated.
            user_id: Batch slot for KV cache fill (0..max_batch_size-1).
            kv_cache: External (k, v) tuple or None to use internal layer_past.
            page_table: Paged attention table (not used in standalone test path).
            batch_size: Number of users (typically 1 for test path).
            current_pos: Not used for prefill.

        Returns:
            Output tensor, same leading shape as input, last dim H=5120.
        """
        cos_tt, sin_tt = rot_mats

        # Normalize to 3D [B, T, H]
        orig_shape = list(x.shape)
        is_4d = len(orig_shape) == 4
        if is_4d:
            B, _, T, H = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
            x_3d = x

        hd = self.head_dim  # 256
        n_q = self.n_q  # 24
        n_kv = self.n_kv  # 4
        q_dim = n_q * hd  # 6144
        g_dim = n_q * hd  # 6144
        k_dim = n_kv * hd  # 1024
        v_dim = n_kv * hd  # 1024
        total = q_dim + g_dim + k_dim + v_dim  # 14336

        # ------------------------------------------------------------------
        # 1. QKVgate projection: x [B, T, H] @ wqkvg [H, 14336] → [B, T, 14336]
        # ------------------------------------------------------------------
        xqkvg = ttnn.linear(
            x_3d,
            self.wqkvg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi2,
        )

        # ------------------------------------------------------------------
        # 2. Split Q, gate, K, V (native sizes)
        # ------------------------------------------------------------------
        q_flat = ttnn.slice(xqkvg, [0, 0, 0], [B, T, q_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat = ttnn.slice(xqkvg, [0, 0, q_dim], [B, T, q_dim + g_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim + g_dim], [B, T, q_dim + g_dim + k_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v_flat = ttnn.slice(xqkvg, [0, 0, q_dim + g_dim + k_dim], [B, T, total], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xqkvg.deallocate(True)

        # Reshape to [B, T, n_heads, hd]
        q_h = ttnn.reshape(q_flat, [B, T, n_q, hd])
        # Keep gate_flat as [B, T, g_dim] for now (apply after SDPA)
        k_h = ttnn.reshape(k_flat, [B, T, n_kv, hd])
        v_h = ttnn.reshape(v_flat, [B, T, n_kv, hd])
        q_flat.deallocate(True)
        k_flat.deallocate(True)
        v_flat.deallocate(True)

        # ------------------------------------------------------------------
        # 3. QK-norm (per head_dim, zero-centered weight = 1+w baked in)
        # ------------------------------------------------------------------
        q_normed = ttnn.rms_norm(q_h, weight=self.q_norm_w, epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_normed = ttnn.rms_norm(k_h, weight=self.k_norm_w, epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_h.deallocate(True)
        k_h.deallocate(True)

        # ------------------------------------------------------------------
        # 4. Transpose to [B, n_heads, T, hd]
        # ------------------------------------------------------------------
        q_t = ttnn.transpose(q_normed, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_t = ttnn.transpose(k_normed, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_t = ttnn.transpose(v_h, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_normed.deallocate(True)
        k_normed.deallocate(True)
        v_h.deallocate(True)

        # ------------------------------------------------------------------
        # 5. Partial RoPE on Q and K
        # ------------------------------------------------------------------
        q_rot = self._apply_partial_rope(q_t, cos_tt, sin_tt)
        k_rot = self._apply_partial_rope(k_t, cos_tt, sin_tt)
        q_t.deallocate(True)
        k_t.deallocate(True)

        # ------------------------------------------------------------------
        # 6. KV cache fill
        # ------------------------------------------------------------------
        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]

        # k_rot: [B, n_kv, T, hd]  v_t: [B, n_kv, T, hd]
        ttnn.fill_cache(keys_cache, k_rot, user_id % max(self.max_batch_size, 1))
        ttnn.fill_cache(values_cache, v_t, user_id % max(self.max_batch_size, 1))

        # ------------------------------------------------------------------
        # 7. GQA expand K, V: n_kv → n_q heads (native ratio = n_q // n_kv = 6)
        # ------------------------------------------------------------------
        gqa = self.gqa_ratio  # 6
        k_exp = ttnn.repeat_interleave(k_rot, gqa, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_exp = ttnn.repeat_interleave(v_t, gqa, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        # ------------------------------------------------------------------
        # 8. SDPA with causal mask
        # ------------------------------------------------------------------
        # Build causal mask on CPU and move to device
        causal = torch.zeros(B, 1, T, T, dtype=torch.bfloat16)
        causal = causal.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), float("-inf"))
        mask_tt = ttnn.from_torch(
            causal,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_exp,
            v_exp,
            is_causal=False,
            attn_mask=mask_tt,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_hifi4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_tt.deallocate(True)
        k_exp.deallocate(True)
        v_exp.deallocate(True)
        q_rot.deallocate(True)

        # ------------------------------------------------------------------
        # 9. Output gate: sigmoid(gate) * attn_out
        # ------------------------------------------------------------------
        # attn_out: [B, n_q, T, hd]
        # gate_flat: [B, T, g_dim=n_q*hd=6144]

        # Transpose attn_out to [B, T, n_q, hd] then flatten to [B, T, g_dim]
        attn_t = ttnn.transpose(attn_out, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)
        attn_flat = ttnn.reshape(attn_t, [B, T, q_dim])
        attn_t.deallocate(True)

        # gate_flat: [B, T, g_dim=6144] — sigmoid then multiply
        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat.deallocate(True)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_flat.deallocate(True)
        gate_sig.deallocate(True)

        # ------------------------------------------------------------------
        # 10. WO projection: [B, T, n_q*hd=6144] @ [6144, H=5120]
        # ------------------------------------------------------------------
        dense_out = ttnn.linear(
            gated,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi2,
        )
        gated.deallocate(True)

        return dense_out

    # ------------------------------------------------------------------
    # Forward: decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats: tuple,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """Decode step forward pass.

        Args:
            x: [B, 1, 1, H] or [B, 1, H] (T=1 decode token).
            current_pos: ttnn tensor [max_batch_size] int32 with current positions.
            rot_mats: (cos_tt, sin_tt) from Qwen36RopeSetup.get_cos_sin_for_decode.
                      cos_tt: [1, 1, 1, 64], replicated.
            page_table: Paged attention table or None.
            kv_cache: External (k, v) or None to use internal layer_past.

        Returns:
            [B, 1, H] bfloat16.
        """
        cos_tt, sin_tt = rot_mats

        orig_shape = list(x.shape)
        is_4d = len(orig_shape) == 4
        if is_4d:
            B, _, T, H = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
            x_3d = x

        assert T == 1, f"Decode expects T=1, got T={T}"

        hd = self.head_dim  # 256
        n_q = self.n_q  # 24
        n_kv = self.n_kv  # 4
        q_dim = n_q * hd  # 6144
        g_dim = n_q * hd  # 6144
        k_dim = n_kv * hd  # 1024
        v_dim = n_kv * hd  # 1024
        total = q_dim + g_dim + k_dim + v_dim  # 14336

        # QKVgate projection
        xqkvg = ttnn.linear(
            x_3d,
            self.wqkvg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi2,
        )

        # Split
        q_flat = ttnn.slice(xqkvg, [0, 0, 0], [B, T, q_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat = ttnn.slice(xqkvg, [0, 0, q_dim], [B, T, q_dim + g_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim + g_dim], [B, T, q_dim + g_dim + k_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v_flat = ttnn.slice(xqkvg, [0, 0, q_dim + g_dim + k_dim], [B, T, total], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xqkvg.deallocate(True)

        q_h = ttnn.reshape(q_flat, [B, T, n_q, hd])
        k_h = ttnn.reshape(k_flat, [B, T, n_kv, hd])
        v_h = ttnn.reshape(v_flat, [B, T, n_kv, hd])
        q_flat.deallocate(True)
        k_flat.deallocate(True)
        v_flat.deallocate(True)

        # QK-norm
        q_normed = ttnn.rms_norm(q_h, weight=self.q_norm_w, epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_normed = ttnn.rms_norm(k_h, weight=self.k_norm_w, epsilon=self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_h.deallocate(True)
        k_h.deallocate(True)

        # Transpose to [B, n_heads, T, hd]
        q_t = ttnn.transpose(q_normed, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_t = ttnn.transpose(k_normed, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_t = ttnn.transpose(v_h, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q_normed.deallocate(True)
        k_normed.deallocate(True)
        v_h.deallocate(True)

        # Partial RoPE
        q_rot = self._apply_partial_rope(q_t, cos_tt, sin_tt)
        k_rot = self._apply_partial_rope(k_t, cos_tt, sin_tt)
        q_t.deallocate(True)
        k_t.deallocate(True)

        # KV cache update
        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]

        # current_pos is either an int or a TTNN tensor [max_batch_size] int32.
        # ttnn.update_cache accepts an integer update_index.
        if isinstance(current_pos, int):
            cur_pos_int = current_pos
        else:
            # Gather from all devices (replicated) then take first element
            cur_pos_cpu = ttnn.to_torch(
                current_pos,
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
            )
            cur_pos_int = int(cur_pos_cpu[0].item())

        # Update KV cache at position cur_pos_int for batch 0
        # k_rot: [B=1, n_kv=4, T=1, hd=256] — matches cache [1, 4, 512, 256] layout
        ttnn.update_cache(keys_cache, k_rot, cur_pos_int, batch_offset=0)
        ttnn.update_cache(values_cache, v_t, cur_pos_int, batch_offset=0)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        # Slice KV cache to [0 .. cur_pos_int] for attention
        # Cache shape: [1, n_kv, max_seq_len, hd] → slice seq_len to cur_pos+1
        T_kv = cur_pos_int + 1
        # Pad to nearest TILE for SDPA
        T_kv_pad = ((T_kv + TILE - 1) // TILE) * TILE
        k_cached = ttnn.slice(keys_cache, [0, 0, 0, 0], [B, n_kv, T_kv_pad, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_cached = ttnn.slice(
            values_cache, [0, 0, 0, 0], [B, n_kv, T_kv_pad, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # GQA expand K, V: n_kv → n_q heads (native ratio = n_q // n_kv = 6)
        gqa = self.gqa_ratio  # 6
        k_exp = ttnn.repeat_interleave(k_cached, gqa, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_exp = ttnn.repeat_interleave(v_cached, gqa, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_cached.deallocate(True)
        v_cached.deallocate(True)

        # Decode SDPA: q [B, n_q, T=1, hd] attends to k/v [B, n_q, T_kv, hd]
        # No mask needed: causal is satisfied by attending only to 0..cur_pos
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_exp,
            v_exp,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_hifi4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k_exp.deallocate(True)
        v_exp.deallocate(True)
        q_rot.deallocate(True)

        # Output gate
        # attn_out: [B, n_q, T, hd] → transpose → [B, T, n_q, hd] → flatten
        attn_t = ttnn.transpose(attn_out, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_out.deallocate(True)
        attn_flat = ttnn.reshape(attn_t, [B, T, q_dim])
        attn_t.deallocate(True)

        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat.deallocate(True)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_flat.deallocate(True)
        gate_sig.deallocate(True)

        # WO projection
        dense_out = ttnn.linear(
            gated,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi2,
        )
        gated.deallocate(True)

        return dense_out

    # ------------------------------------------------------------------
    # Unified forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats=None,
        user_id: int = 0,
        mode: str = "decode",
        page_table=None,
        kv_cache=None,
        batch_size: int = 1,
    ) -> ttnn.Tensor:
        """Dispatch to forward_prefill or forward_decode."""
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id=user_id,
                kv_cache=kv_cache,
                page_table=page_table,
                batch_size=batch_size,
            )
        else:
            return self.forward_decode(
                x,
                current_pos,
                rot_mats,
                page_table=page_table,
                kv_cache=kv_cache,
            )
