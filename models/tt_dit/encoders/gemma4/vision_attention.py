# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 vision attention (Option A: head_dim padded from 72 → 96 for tile alignment).

Mirrors ``Gemma4VisionAttention`` from ``transformers.models.gemma4.modeling_gemma4``:
standard MHA (``num_kv_heads == num_attention_heads`` — no GQA), QK + V norm,
multidim RoPE applied per spatial-dim chunk, bidirectional attention.

Differences vs text attention:
  * Always has a separate ``v_proj`` (no K=V quirk).
  * Q/K/V projections are padded on the head_dim axis (72 → ``head_dim_padded``)
    by zero-filling the trailing channels of the weight (i.e. each head's last
    ``head_dim_padded - head_dim`` output channels are zero).
  * Per-head RMSNorm weights for Q/K are scaled by ``sqrt(head_dim_padded /
    head_dim)`` and zero-padded on the trailing channels, so the normalized
    output matches HF in the real channels.
  * V norm has ``with_scale=False`` (no weight to fold the correction into),
    so we explicitly multiply the v-norm output by ``sqrt(head_dim_padded /
    head_dim)`` to compensate the RMS denominator.
  * o_proj input is also padded on the head_dim axis (zero columns at the padded
    positions, matching the zero outputs of SDPA there).
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.linear import ColParallelLinear, RowParallelLinear
from ...layers.module import Module
from ...layers.normalization import RMSNorm
from ...parallel.config import DiTParallelConfig

TILE = ttnn.TILE_SIZE


class Gemma4VisionAttention(Module):
    """Vision encoder self-attention (multidim RoPE, head_dim padded)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        head_dim_padded: int,
        rms_norm_eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.head_dim_padded = head_dim_padded
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device

        tp_factor = parallel_config.tensor_parallel.factor
        assert num_attention_heads % tp_factor == 0
        assert head_dim_padded % TILE == 0, f"head_dim_padded={head_dim_padded} must be tile-aligned"
        assert head_dim_padded >= head_dim

        self.num_local_heads = num_attention_heads // tp_factor

        # Pre-compute the V-norm correction factor: sqrt(head_dim_padded / head_dim).
        # Applied multiplicatively on v_norm's output to undo the larger denominator from padded zeros.
        self._v_scale = math.sqrt(head_dim_padded / head_dim)

        col_kwargs = dict(
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.q_proj = ColParallelLinear(hidden_size, num_attention_heads * head_dim_padded, **col_kwargs)
        self.k_proj = ColParallelLinear(hidden_size, num_attention_heads * head_dim_padded, **col_kwargs)
        self.v_proj = ColParallelLinear(hidden_size, num_attention_heads * head_dim_padded, **col_kwargs)

        # o_proj: row-parallel from (num_heads * head_dim_padded) → hidden_size.
        self.o_proj = RowParallelLinear(
            num_attention_heads * head_dim_padded,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Per-head RMSNorm over head_dim_padded (last axis is fully local).
        norm_kwargs = dict(
            embedding_dim=head_dim_padded,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.q_norm = RMSNorm(**norm_kwargs)
        self.k_norm = RMSNorm(**norm_kwargs)
        self.v_norm = RMSNorm(
            embedding_dim=head_dim_padded,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
        )

        # SDPA + compute kernel configs (matches gemma3 / Wan conventions).
        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        # Accuracy-first compute config (see comments in encoders/gemma4/attention.py).
        # ``packer_l1_acc=False`` mirrors tt_dit LayerNorm's default: avoids the packer's
        # low-precision L1 accumulation over deep reductions (head_dim_padded=96 here).
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Pad HF projections + Q/K norm weights along the head_dim axis."""
        H = self.num_attention_heads
        D = self.head_dim
        Dp = self.head_dim_padded
        Hidden = self.hidden_size
        if Dp == D:
            return  # No padding needed.

        scale = math.sqrt(Dp / D)

        # Q/K/V projections: HF weight is [num_heads * head_dim, hidden_size]. Pad output dim per head.
        for name in ("q_proj", "k_proj", "v_proj"):
            w = state.get(f"{name}.weight")
            if w is None:
                continue
            # [H*D, Hidden] → [H, D, Hidden] → pad → [H, Dp, Hidden] → [H*Dp, Hidden]
            w = w.reshape(H, D, Hidden)
            w = torch.nn.functional.pad(w, (0, 0, 0, Dp - D))  # pad dim -2 (D) on the right
            state[f"{name}.weight"] = w.reshape(H * Dp, Hidden)

        # o_proj: HF weight [Hidden, H * D]. Pad input dim per head.
        w = state.get("o_proj.weight")
        if w is not None:
            w = w.reshape(Hidden, H, D)
            w = torch.nn.functional.pad(w, (0, Dp - D))  # pad dim -1 (D) on the right
            state["o_proj.weight"] = w.reshape(Hidden, H * Dp)

        # Q/K norms: scale by sqrt(Dp/D) and zero-pad.
        for name in ("q_norm", "k_norm"):
            w = state.get(f"{name}.weight")
            if w is None:
                continue
            w = w * scale
            w = torch.nn.functional.pad(w, (0, Dp - D))
            state[f"{name}.weight"] = w

    def _apply_rope_multidim(
        self,
        x: ttnn.Tensor,
        cos_half: ttnn.Tensor,
        sin_half: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply multidim RoPE per spatial-dim chunk.

        x:          ``[B, H_local, P, head_dim_padded]``
        cos_half:   ``[B, 1, P, head_dim_padded / 2]`` (layout: cos_x_half, cos_y_half, pad=1)
        sin_half:   same layout (cos_y_half region → sin_y_half, pad=0)

        Returns ``[B, H_local, P, head_dim_padded]``.

        Per HF: head_dim is split into ``ndim=2`` chunks of ``head_dim/2`` (=36) each; 1-D
        RoPE is applied within each chunk. Padding chunk (channels ``[head_dim :
        head_dim_padded]``) is identity.

        With tt_dit's half-dim convention, each chunk's rotation reduces to:

            x_chunk = x[..., chunk_start : chunk_start + chunk_width]
            x1, x2 = first/second half of x_chunk (each chunk_width/2)
            cos_c = cos_half[..., halfchunk_start : halfchunk_start + chunk_width/2]
            sin_c = sin_half[..., halfchunk_start : halfchunk_start + chunk_width/2]
            out1 = x1*cos_c - x2*sin_c;  out2 = x2*cos_c + x1*sin_c
            chunk_out = concat(out1, out2, -1)

        The pad chunk is passed through unchanged (its values are zero anyway).
        """
        D = self.head_dim
        Dp = self.head_dim_padded
        chunk = D // 2  # 36
        half_chunk = chunk // 2  # 18

        # Two real chunks: x-dim then y-dim.
        out_chunks: list[ttnn.Tensor] = []
        for c in range(2):
            cs = c * chunk
            x_chunk = x[..., cs : cs + chunk]
            cos_c = cos_half[..., c * half_chunk : (c + 1) * half_chunk]
            sin_c = sin_half[..., c * half_chunk : (c + 1) * half_chunk]
            x1 = x_chunk[..., :half_chunk]
            x2 = x_chunk[..., half_chunk:]
            out1 = ttnn.subtract(ttnn.multiply(x1, cos_c), ttnn.multiply(x2, sin_c))
            out2 = ttnn.add(ttnn.multiply(x2, cos_c), ttnn.multiply(x1, sin_c))
            out_chunks.append(ttnn.concat([out1, out2], dim=-1))

        # Pass-through pad chunk (if any).
        if Dp > D:
            out_chunks.append(x[..., D:Dp])

        return ttnn.concat(out_chunks, dim=-1)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Args:
            hidden_states:   replicated [1, B, num_patches, hidden_size] (tt_dit ``1BND`` convention).
            cos, sin:        [B, 1, num_patches, head_dim_padded / 2] (from Gemma4VisionRotaryEmbedding)
            attention_mask:  bf16 additive mask, optional. None → unmasked (bidirectional).

        Returns:
            replicated [1, B, num_patches, hidden_size].
        """
        # Read batch/patch from trailing 3 dims so both 3D and 4D ``1BND`` inputs work.
        B, P = hidden_states.shape[-3], hidden_states.shape[-2]
        H_local = self.num_local_heads
        Dp = self.head_dim_padded

        # Projections (column-parallel over head axis). Input is replicated per the
        # docstring — no ``parallel_config`` argument, so ColParallelLinear runs its
        # ``minimal_matmul`` path (each device against its local weight shard).
        # ``compute_kernel_config=self.compute_config`` (HiFi4) overrides Linear's default HiFi2.
        q = self.q_proj(hidden_states, compute_kernel_config=self.compute_config)
        k = self.k_proj(hidden_states, compute_kernel_config=self.compute_config)
        v = self.v_proj(hidden_states, compute_kernel_config=self.compute_config)

        # Reshape and permute to (B, H_local, P, Dp).
        q = ttnn.permute(ttnn.reshape(q, (B, P, H_local, Dp)), (0, 2, 1, 3))
        k = ttnn.permute(ttnn.reshape(k, (B, P, H_local, Dp)), (0, 2, 1, 3))
        v = ttnn.permute(ttnn.reshape(v, (B, P, H_local, Dp)), (0, 2, 1, 3))

        # Per-head RMSNorm (Dp last axis; padded entries pre-zeroed by their projections,
        # weights pre-scaled & zero-padded by _prepare_torch_state).
        # Pass HiFi4/fp32 compute config — RMSNorm's default is unset (device default HiFi2
        # bf16-acc), which loses precision squaring bf16 head_dim values.
        q = self.q_norm(q, compute_kernel_config=self.compute_config)
        k = self.k_norm(k, compute_kernel_config=self.compute_config)
        v = self.v_norm(v, compute_kernel_config=self.compute_config)
        # v_norm has no scale, so apply the sqrt(Dp/D) correction manually.
        v = ttnn.multiply(v, self._v_scale)

        # Multidim RoPE on Q and K (not on V).
        q = self._apply_rope_multidim(q, cos, sin)
        k = self._apply_rope_multidim(k, cos, sin)

        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        # Bidirectional SDPA (no causal mask). scale=1.0 (HF Gemma 4 vision sets self.scaling=1.0).
        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=attention_mask,
            scale=1.0,
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute_config,
        )

        # (B, H_local, P, Dp) → (B, P, H_local * Dp) 3D → unsqueeze → 4D 1BND → o_proj.
        attn = ttnn.transformer.concatenate_heads(attn)
        attn = ttnn.unsqueeze(attn, 0)
        out = self.o_proj(attn, compute_kernel_config=self.compute_config)
        # RowParallelLinear returns TP-fractured on the output dim; gather to replicated.
        if self.parallel_config.tensor_parallel.factor > 1:
            out = self.ccl_manager.all_gather_persistent_buffer(
                out, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        return out
