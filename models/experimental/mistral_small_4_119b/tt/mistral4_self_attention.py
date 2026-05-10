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


def _torch_for_ttnn_upload(w: torch.Tensor) -> torch.Tensor:
    """Convert a PyTorch weight tensor to bfloat16 for TTNN upload."""
    return w.to(torch.bfloat16).contiguous()


def _load_weight(
    state_dict: dict,
    key: str,
    transpose: bool,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
    mesh_mapper=None,
) -> ttnn.Tensor:
    """Load a weight from state_dict to TTNN device with optional transpose."""
    w = _torch_for_ttnn_upload(state_dict[key])
    if transpose:
        w = w.T.contiguous()
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


def _apply_rope_ttnn(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    seq_len: int,
    n_heads: int,
    rope_dim: int,
) -> ttnn.Tensor:
    """
    Apply rotary position embeddings in pure TTNN.

    Args:
        x:       [1, n_heads, seq_len, rope_dim]
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
    MLA attention for Mistral-Small-4 (prefill mode, replicated weights).

    All weights are replicated across all mesh devices; both devices
    execute identical computation.  This is intentional for the initial
    bring-up: the dominant memory cost is the expert weights (MoE), not
    the relatively small attention projections.
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
        )  # [Q_LORA_RANK, N_HEADS * HEAD_DIM]

        self.kv_a_proj = _load_weight(
            state_dict,
            p + "kv_a_proj_with_mqa.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
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
    ) -> ttnn.Tensor:
        """
        Args:
            x:       [1, 1, seq_len, HIDDEN_SIZE] replicated on all devices
            cos/sin: [1, 1, seq_len, QK_ROPE_HEAD_DIM]  (from HF rotary, on device)
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

        q = ttnn.reshape(q, [1, self.n_heads, seq_len, self.head_dim])
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

        kv = ttnn.reshape(kv, [1, self.n_heads, seq_len, self.kv_b_per_head])
        # [1, N_HEADS, seq, KV_B_PER_HEAD=192]

        k_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, self.n_heads, seq_len, self.qk_nope_head_dim])
        v = ttnn.slice(kv, [0, 0, 0, self.qk_nope_head_dim], [1, self.n_heads, seq_len, self.kv_b_per_head])
        ttnn.deallocate(kv)

        # Apply RoPE to the single-head k_rope
        k_rope_rotated = _apply_rope_ttnn(
            k_rope_raw, cos, sin, seq_len, 1, self.qk_rope_head_dim
        )  # [1, 1, seq, QK_ROPE_HEAD_DIM]
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
        attn_flat = ttnn.reshape(
            attn_out, [1, 1, seq_len, self.n_heads * self.v_head_dim]
        )  # [1, 1, seq, N_HEADS * V_HEAD_DIM = 4096]
        ttnn.deallocate(attn_out)

        out = ttnn.linear(
            attn_flat,
            self.o_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, HIDDEN_SIZE]
        ttnn.deallocate(attn_flat)

        return out
