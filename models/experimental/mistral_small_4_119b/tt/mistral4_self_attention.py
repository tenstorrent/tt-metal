# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral4 Multi-head Latent Attention (MLA) — prefill mode.

Architecture (all on device, no PyTorch fallback for compute):

  Q path:
    hidden  →[q_a_proj]→ q_latent (Q_LORA_RANK)
            →[q_a_norm]→
            →[q_b_proj]→ [N_HEADS, HEAD_DIM]
            split → q_nope [N_HEADS, QK_NOPE_HEAD_DIM]
                  + q_rope [N_HEADS, QK_ROPE_HEAD_DIM]
            RoPE(q_rope) → q_rope_rotated
            q = concat(q_nope, q_rope_rotated)

  KV path:
    hidden  →[kv_a_proj]→ kv_combined (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
            split → kv_latent [KV_LORA_RANK]
                  + k_rope_raw [QK_ROPE_HEAD_DIM]
            kv_latent →[kv_a_norm]→
                      →[kv_b_proj]→ [N_HEADS, KV_B_PER_HEAD]
                      split → k_nope [N_HEADS, QK_NOPE_HEAD_DIM]
                            + v      [N_HEADS, V_HEAD_DIM]
            RoPE(k_rope_raw) → k_rope_rotated
            k_rope_expanded = broadcast k_rope_rotated to N_HEADS
            k = concat(k_nope, k_rope_expanded)

  Attention:
    SDPA(q, k, v, is_causal=True)
    concat_heads → [seq, N_HEADS * V_HEAD_DIM]
    →[o_proj]→ hidden

Sharding strategy for 2-device mesh [1, 2] (P300 × 2):
  All attention weights are *replicated* for the initial bring-up.
  Both devices compute identical attention outputs; only device-0
  output is used for logit accuracy, but both are in sync for the
  residual stream used by the MoE layer.
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.constants import (
    HEAD_DIM,
    KV_A_PROJ_OUT,
    KV_B_PROJ_OUT_PER_HEAD,
    KV_LORA_RANK,
    N_HEADS,
    NORM_EPS,
    Q_LORA_RANK,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    V_HEAD_DIM,
)


def _torch_for_ttnn_upload(w: torch.Tensor, scale_inv: torch.Tensor | None = None) -> torch.Tensor:
    """Convert a weight tensor to bfloat16 for TTNN upload, dequantizing FP8 if needed.

    scale_inv may be scalar () or per-expert [N]; reshaped to broadcast correctly.
    """
    if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        w = w.to(torch.float32)
        if scale_inv is not None:
            s = scale_inv.to(torch.float32)
            while s.dim() < w.dim():
                s = s.unsqueeze(-1)
            w = w * s
    return w.to(torch.bfloat16).contiguous()


def _load_weight(
    state_dict: dict,
    key: str,
    transpose: bool,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
    mesh_mapper=None,
    transform_fn=None,
) -> ttnn.Tensor:
    """Load a weight from state_dict to TTNN device with optional transpose.
    Automatically dequantizes FP8 weights using the companion weight_scale_inv key."""
    scale_inv = state_dict.get(key.replace(".weight", ".weight_scale_inv"))
    w = _torch_for_ttnn_upload(state_dict[key], scale_inv)
    if transpose:
        w = w.T.contiguous()
    if transform_fn is not None:
        w = transform_fn(w)
    # Ensure 4D for TILE layout
    while w.dim() < 2:
        w = w.unsqueeze(0)
    mapper = mesh_mapper if mesh_mapper is not None else ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _load_norm_weight(
    state_dict: dict,
    key: str,
    dim: int,
    mesh_device: ttnn.MeshDevice,
) -> ttnn.Tensor:
    """
    Load RMSNorm ``weight`` for ``ttnn.rms_norm`` with TILE activations.

    ROW_MAJOR gamma must end in tile width (``ttnn.TILE_SIZE``); see
    ``LayerNormDeviceOperation::validate_on_program_cache_miss`` (gamma last dim
    == tile width, volume aligns with input last dim).
    """
    tw = ttnn.TILE_SIZE
    if dim % tw != 0:
        raise ValueError(f"RMS norm hidden dim {dim} must be divisible by ttnn.TILE_SIZE ({tw})")
    w = _torch_for_ttnn_upload(state_dict[key]).reshape(1, 1, dim // tw, tw)
    return ttnn.as_tensor(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _deinterleave_q_b_proj(w: torch.Tensor) -> torch.Tensor:
    """
    Convert rope columns of q_b_proj from interleaved [r0,i0,r1,i1,…] to
    half-split [r0,r1,…,i0,i1,…] to match apply_rotary_pos_emb_interleave.

    w: [Q_LORA_RANK, N_HEADS * HEAD_DIM]  (already transposed)
    """
    in_dim = w.shape[0]
    w3 = w.reshape(in_dim, N_HEADS, HEAD_DIM)
    nope = w3[:, :, :QK_NOPE_HEAD_DIM]
    rope = w3[:, :, QK_NOPE_HEAD_DIM:]  # [in, H, rope_dim]
    rope = rope.reshape(in_dim, N_HEADS, QK_ROPE_HEAD_DIM // 2, 2)
    rope = rope.permute(0, 1, 3, 2).contiguous()  # [in, H, 2, rope_dim//2]
    rope = rope.reshape(in_dim, N_HEADS, QK_ROPE_HEAD_DIM)
    return torch.cat([nope, rope], dim=-1).reshape(in_dim, N_HEADS * HEAD_DIM).contiguous()


def _deinterleave_kv_a_proj(w: torch.Tensor) -> torch.Tensor:
    """
    Convert k_rope columns of kv_a_proj from interleaved to half-split.

    w: [HIDDEN_SIZE, KV_A_PROJ_OUT]  (already transposed)
    The last QK_ROPE_HEAD_DIM columns produce k_rope_raw.
    """
    nope_cols = w[:, :KV_LORA_RANK]
    rope_cols = w[:, KV_LORA_RANK:]  # [H, rope_dim]
    rope_cols = rope_cols.reshape(-1, QK_ROPE_HEAD_DIM // 2, 2)
    rope_cols = rope_cols.permute(0, 2, 1).contiguous()  # [H, 2, rope_dim//2]
    rope_cols = rope_cols.reshape(-1, QK_ROPE_HEAD_DIM)
    return torch.cat([nope_cols, rope_cols], dim=-1).contiguous()


def _apply_rope_ttnn(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    seq_len: int,
    n_heads: int,
    rope_dim: int,
) -> ttnn.Tensor:
    """
    Apply rotary position embeddings in pure TTNN (standard half-split variant).

    Args:
        x:       [1, n_heads, seq_len, rope_dim]  — already in interleaved layout
        cos/sin: [1, 1, seq_len, rope_dim]  (broadcast over heads)
        Returns: [1, n_heads, seq_len, rope_dim]
    """
    half = rope_dim // 2
    # rotate_half: concat([-x2, x1]) where x1=x[..,:half], x2=x[..,half:]
    x1 = ttnn.slice(x, [0, 0, 0, 0], [1, n_heads, seq_len, half])
    x2 = ttnn.slice(x, [0, 0, 0, half], [1, n_heads, seq_len, rope_dim])
    x_rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)

    out = ttnn.add(
        ttnn.multiply(x, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.multiply(x_rot, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(x_rot)
    return out


class TtMistral4Attention(LightweightModule):
    """
    MLA attention for Mistral-Small-4 (prefill + decode, replicated weights).

    All weights are replicated across all mesh devices; both devices
    execute identical computation.  This is intentional for the initial
    bring-up: the dominant memory cost is the expert weights (MoE), not
    the relatively small attention projections.

    KV cache convention (for decode): [1, N_HEADS, max_seq_len, dim] — batch-first.
    Both update_cache_for_token_ and fill_cache_for_user_ validate padded_shape()[0]==1.
    SDPA decode expects Q=[1, B, NH, DH] and K/V=[1, NH, S, DH] in the same batch-first layout.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_prefix: str,
        compute_kernel_config=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.qk_nope_head_dim = QK_NOPE_HEAD_DIM
        self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        self.kv_lora_rank = KV_LORA_RANK
        self.kv_a_proj_out = KV_A_PROJ_OUT
        self.kv_b_per_head = KV_B_PROJ_OUT_PER_HEAD
        self.scale = 1.0 / math.sqrt(self.head_dim)

        if compute_kernel_config is None:
            compute_kernel_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        self.compute_kernel_config = compute_kernel_config

        p = layer_prefix + "self_attn."

        # ── Replicated weights (small bottleneck projections) ──────────────
        # HF stores [out, in]; we transpose → [in, out] for TTNN matmul
        self.q_a_proj = _load_weight(
            state_dict,
            p + "q_a_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
        )  # [HIDDEN_SIZE, Q_LORA_RANK]

        self.q_a_norm = _load_norm_weight(
            state_dict,
            p + "q_a_layernorm.weight",
            Q_LORA_RANK,
            mesh_device,
        )  # [1, 1, Q_LORA_RANK / TILE, TILE]

        self.q_b_proj = _load_weight(
            state_dict,
            p + "q_b_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            transform_fn=_deinterleave_q_b_proj,
        )  # [Q_LORA_RANK, N_HEADS * HEAD_DIM]

        self.kv_a_proj = _load_weight(
            state_dict,
            p + "kv_a_proj_with_mqa.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            transform_fn=_deinterleave_kv_a_proj,
        )  # [HIDDEN_SIZE, KV_A_PROJ_OUT]

        self.kv_a_norm = _load_norm_weight(
            state_dict,
            p + "kv_a_layernorm.weight",
            KV_LORA_RANK,
            mesh_device,
        )  # [1, 1, KV_LORA_RANK / TILE, TILE]

        self.kv_b_proj = _load_weight(
            state_dict,
            p + "kv_b_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
        )  # [KV_LORA_RANK, KV_B_PROJ_OUT_TOTAL]

        self.o_proj = _load_weight(
            state_dict,
            p + "o_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
        )  # [N_HEADS * V_HEAD_DIM, HIDDEN_SIZE]

    # ------------------------------------------------------------------
    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: tuple = None,
    ) -> ttnn.Tensor:
        """
        Prefill forward.

        Args:
            x:        [1, 1, seq_len, HIDDEN_SIZE] replicated on all devices
            cos/sin:  [1, 1, seq_len, QK_ROPE_HEAD_DIM]  (from HF rotary, on device)
            kv_cache: optional (k_cache, v_cache) — if provided, filled in-place
        Returns:
            [1, 1, seq_len, HIDDEN_SIZE] replicated on all devices
        """
        seq_len = x.shape[2]

        # ── Q projection ──────────────────────────────────────────────────
        q_latent = ttnn.linear(
            x,
            self.q_a_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, Q_LORA_RANK]

        q_latent = ttnn.rms_norm(
            q_latent,
            weight=self.q_a_norm,
            epsilon=NORM_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q = ttnn.linear(
            q_latent,
            self.q_b_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, N_HEADS * HEAD_DIM]
        ttnn.deallocate(q_latent)

        q = ttnn.reshape(q, [1, seq_len, self.n_heads, self.head_dim])
        q = ttnn.transpose(q, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # [1, N_HEADS, seq, HEAD_DIM]

        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, self.n_heads, seq_len, self.qk_nope_head_dim])
        q_rope = ttnn.slice(q, [0, 0, 0, self.qk_nope_head_dim], [1, self.n_heads, seq_len, self.head_dim])
        ttnn.deallocate(q)

        q_rope_rotated = _apply_rope_ttnn(q_rope, cos, sin, seq_len, self.n_heads, self.qk_rope_head_dim)
        ttnn.deallocate(q_rope)

        q_full = ttnn.concat([q_nope, q_rope_rotated], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_rope_rotated)
        # q_full: [1, N_HEADS, seq, HEAD_DIM]

        # ── KV projection ─────────────────────────────────────────────────
        kv_combined = ttnn.linear(
            x,
            self.kv_a_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, KV_A_PROJ_OUT]

        kv_latent = ttnn.slice(kv_combined, [0, 0, 0, 0], [1, 1, seq_len, self.kv_lora_rank])
        k_rope_raw = ttnn.slice(kv_combined, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len, self.kv_a_proj_out])
        ttnn.deallocate(kv_combined)

        kv_latent_normed = ttnn.rms_norm(
            kv_latent,
            weight=self.kv_a_norm,
            epsilon=NORM_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Keep kv_latent alive; will be reused for kv_b_proj

        kv = ttnn.linear(
            kv_latent_normed,
            self.kv_b_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, KV_B_PROJ_OUT_TOTAL]
        ttnn.deallocate(kv_latent)
        ttnn.deallocate(kv_latent_normed)

        kv = ttnn.reshape(kv, [1, seq_len, self.n_heads, self.kv_b_per_head])
        kv = ttnn.transpose(kv, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # [1, N_HEADS, seq, KV_B_PER_HEAD=192]

        k_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, self.n_heads, seq_len, self.qk_nope_head_dim])
        v = ttnn.slice(kv, [0, 0, 0, self.qk_nope_head_dim], [1, self.n_heads, seq_len, self.kv_b_per_head])
        ttnn.deallocate(kv)

        k_rope_rotated = _apply_rope_ttnn(k_rope_raw, cos, sin, seq_len, 1, self.qk_rope_head_dim)
        # [1, 1, seq, QK_ROPE_HEAD_DIM]
        ttnn.deallocate(k_rope_raw)

        # Broadcast k_rope to all heads by repeating along dim=1
        k_rope_expanded = ttnn.concat(
            [k_rope_rotated] * self.n_heads,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, N_HEADS, seq, QK_ROPE_HEAD_DIM]
        ttnn.deallocate(k_rope_rotated)

        k_full = ttnn.concat([k_nope, k_rope_expanded], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_rope_expanded)
        # k_full: [1, N_HEADS, seq, HEAD_DIM]

        # Fill KV cache if provided (prefill with cache)
        if kv_cache is not None:
            self.fill_kv_cache(k_full, v, kv_cache)

        # ── Scaled dot-product attention ───────────────────────────────────
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_full,
            k_full,
            v,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, N_HEADS, seq, V_HEAD_DIM]
        ttnn.deallocate(q_full)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v)

        # ── Output projection ──────────────────────────────────────────────
        # attn_out is [1, N_HEADS, seq, V_HEAD_DIM]; transpose to [1, seq, N_HEADS, V_HEAD_DIM]
        # before reshaping so head features for the same position are contiguous.
        attn_out_t = ttnn.transpose(attn_out, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        attn_flat = ttnn.reshape(
            attn_out_t, [1, 1, seq_len, self.n_heads * self.v_head_dim]
        )  # [1, 1, seq, N_HEADS * V_HEAD_DIM = 4096]
        ttnn.deallocate(attn_out_t)

        out = ttnn.linear(
            attn_flat,
            self.o_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, HIDDEN_SIZE]
        ttnn.deallocate(attn_flat)

        return out

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def allocate_kv_cache(self, max_seq_len: int) -> tuple:
        """
        Pre-allocate K and V cache tensors on device (zeroed, DRAM, replicated).

        Cache shape: [1, N_HEADS, padded_seq, dim] — batch-first.
        padded_seq is max_seq_len rounded up to the nearest 32 because
        scaled_dot_product_attention_decode requires k_chunk_size % 32 == 0,
        and get_chunk_size(S) can only return a multiple of 32 if S itself is.
        """
        padded_seq = ((max_seq_len + 31) // 32) * 32
        k_cache = ttnn.as_tensor(
            torch.zeros(1, self.n_heads, padded_seq, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        v_cache = ttnn.as_tensor(
            torch.zeros(1, self.n_heads, padded_seq, self.v_head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return k_cache, v_cache

    def fill_kv_cache(self, k_full: ttnn.Tensor, v: ttnn.Tensor, kv_cache: tuple) -> None:
        """
        Fill KV cache in-place from prefill K/V tensors.

        k_full and v are already [1, N_HEADS, seq, dim] (batch-first),
        which is exactly what fill_cache_for_user_ expects.
        """
        k_cache, v_cache = kv_cache
        ttnn.kv_cache.fill_cache_for_user_(k_cache, k_full, 0)
        ttnn.kv_cache.fill_cache_for_user_(v_cache, v, 0)

    # ------------------------------------------------------------------
    # Decode forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: tuple,
        current_pos: int,
    ) -> ttnn.Tensor:
        """
        Single-token decode step.

        Args:
            x:           [1, 1, 1, HIDDEN_SIZE] replicated on all devices
            cos/sin:     [1, 1, 1, QK_ROPE_HEAD_DIM] for position current_pos
            kv_cache:    (k_cache, v_cache) each [1, N_HEADS, max_seq_len, dim] (batch-first)
            current_pos: cache slot to write the new K/V token into
        Returns:
            [1, 1, 1, HIDDEN_SIZE]
        """
        k_cache, v_cache = kv_cache

        # ── Q path (seq_len=1) ─────────────────────────────────────────
        q_latent = ttnn.linear(
            x,
            self.q_a_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_latent = ttnn.rms_norm(
            q_latent,
            weight=self.q_a_norm,
            epsilon=NORM_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.linear(
            q_latent,
            self.q_b_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_latent)

        q = ttnn.reshape(q, [1, self.n_heads, 1, self.head_dim])
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, self.n_heads, 1, self.qk_nope_head_dim])
        q_rope = ttnn.slice(q, [0, 0, 0, self.qk_nope_head_dim], [1, self.n_heads, 1, self.head_dim])
        ttnn.deallocate(q)

        q_rope_rotated = _apply_rope_ttnn(q_rope, cos, sin, 1, self.n_heads, self.qk_rope_head_dim)
        ttnn.deallocate(q_rope)
        q_full = ttnn.concat([q_nope, q_rope_rotated], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_rope_rotated)
        # q_full: [1, N_HEADS, 1, HEAD_DIM]

        # ── KV path (seq_len=1) ────────────────────────────────────────
        kv_combined = ttnn.linear(
            x,
            self.kv_a_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kv_latent = ttnn.slice(kv_combined, [0, 0, 0, 0], [1, 1, 1, self.kv_lora_rank])
        k_rope_raw = ttnn.slice(kv_combined, [0, 0, 0, self.kv_lora_rank], [1, 1, 1, self.kv_a_proj_out])
        ttnn.deallocate(kv_combined)

        kv_latent_normed = ttnn.rms_norm(
            kv_latent,
            weight=self.kv_a_norm,
            epsilon=NORM_EPS,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        kv = ttnn.linear(
            kv_latent_normed,
            self.kv_b_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(kv_latent)
        ttnn.deallocate(kv_latent_normed)

        kv = ttnn.reshape(kv, [1, self.n_heads, 1, self.kv_b_per_head])
        k_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, self.n_heads, 1, self.qk_nope_head_dim])
        v_new = ttnn.slice(kv, [0, 0, 0, self.qk_nope_head_dim], [1, self.n_heads, 1, self.kv_b_per_head])
        ttnn.deallocate(kv)

        k_rope_rotated = _apply_rope_ttnn(k_rope_raw, cos, sin, 1, 1, self.qk_rope_head_dim)
        ttnn.deallocate(k_rope_raw)
        k_rope_expanded = ttnn.concat([k_rope_rotated] * self.n_heads, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_rope_rotated)
        k_full = ttnn.concat([k_nope, k_rope_expanded], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_rope_expanded)
        # k_full: [1, N_HEADS, 1, HEAD_DIM],  v_new: [1, N_HEADS, 1, V_HEAD_DIM]

        # ── Update KV cache at current_pos ─────────────────────────────
        # k_full/v_new are already [1, N_HEADS, 1, dim] — batch-first, no transpose needed.
        # update_cache_for_token_ validates padded_shape()[0]==1 and [1]==cache[1].
        ttnn.kv_cache.update_cache_for_token_(k_cache, k_full, current_pos)
        ttnn.kv_cache.update_cache_for_token_(v_cache, v_new, current_pos)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_new)

        # ── Decode SDPA ────────────────────────────────────────────────
        # scaled_dot_product_attention_decode expects Q=[1, B, NH, DH].
        # q_full is [1, N_HEADS, 1, HEAD_DIM]; transpose dims 1&2 → [1, 1, N_HEADS, HEAD_DIM].
        q_decode = ttnn.transpose(q_full, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_full)

        cur_pos_tensor = ttnn.as_tensor(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B=1, NH=N_HEADS, V_HEAD_DIM]
        ttnn.deallocate(q_decode)
        ttnn.deallocate(cur_pos_tensor)

        # [1, 1, N_HEADS, V_HEAD_DIM] → transpose(1,2) → [1, N_HEADS, 1, V_HEAD_DIM]
        # → reshape → [1, 1, 1, N_HEADS * V_HEAD_DIM] for o_proj
        attn_out_t = ttnn.transpose(attn_out, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # ── Output projection ──────────────────────────────────────────
        attn_flat = ttnn.reshape(attn_out_t, [1, 1, 1, self.n_heads * self.v_head_dim])
        ttnn.deallocate(attn_out_t)
        out = ttnn.linear(
            attn_flat,
            self.o_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_flat)
        return out
