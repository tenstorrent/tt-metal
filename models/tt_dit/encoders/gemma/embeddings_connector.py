# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Embeddings Connector — projects Gemma hidden states to DiT dimensions.

Pipeline: Gemma hidden states (49 layers × 3840) → aggregate_embed → connector → embeddings
- Video: aggregate_embed(188160→4096) + 8 transformer blocks at 4096
- Audio: aggregate_embed(188160→2048) + 2 transformer blocks at 2048

Reference: ltx_core.text_encoders.gemma.embeddings_connector
"""

from __future__ import annotations

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList, Parameter
from ...utils.substate import pop_substate, rename_substate


def _rms_norm(x: ttnn.Tensor, eps: float = 1e-6) -> ttnn.Tensor:
    """Parameter-free RMS normalization matching ltx_core.utils.rms_norm."""
    # x / sqrt(mean(x^2) + eps)
    x_sq = ttnn.multiply(x, x)
    mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
    rms = ttnn.rsqrt(ttnn.add(mean_sq, eps))
    return ttnn.multiply(x, rms)


class ConnectorBlock(Module):
    """Single transformer block for the embeddings connector (pre-norm, self-attn + FF)."""

    def __init__(self, dim: int, ff_dim: int, num_heads: int, eps: float, mesh_device, ccl_manager, parallel_config):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.eps = eps
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        # Norms are parameter-free RMS norms (matching reference ltx_core.utils.rms_norm)
        # No learnable weight — implemented as function calls in forward()

        # Self-attention (Q, K, V, O projections)
        col_kwargs = {"bias": True, "mesh_device": mesh_device, "mesh_axis": tp_axis}
        self.to_q = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_k = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_v = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_out = Linear(dim, dim, bias=True, mesh_device=mesh_device)

        # Feed-forward (GELU gated)
        self.ff1 = ColParallelLinear(
            dim, ff_dim, bias=True, activation_fn="gelu", mesh_device=mesh_device, mesh_axis=tp_axis
        )
        self.ff2 = RowParallelLinear(
            ff_dim, dim, bias=True, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl_manager
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state):
        rename_substate(state, "attn1.to_q", "to_q")
        rename_substate(state, "attn1.to_k", "to_k")
        rename_substate(state, "attn1.to_v", "to_v")
        rename_substate(state, "attn1.to_out.0", "to_out")
        rename_substate(state, "ff.net.0.proj", "ff1")
        rename_substate(state, "ff.net.2", "ff2")
        # Remove norms we handle ourselves and unused keys
        pop_substate(state, "attn1.q_norm")
        pop_substate(state, "attn1.k_norm")
        pop_substate(state, "attn1.to_gate_logits")

    def forward(self, x, rope_cos=None, rope_sin=None):
        # Self-attention with residual
        residual = x
        x = _rms_norm(x, self.eps)
        q = self.to_q(x, compute_kernel_config=self.compute_config)
        k = self.to_k(x, compute_kernel_config=self.compute_config)
        v = self.to_v(x, compute_kernel_config=self.compute_config)

        # Apply INTERLEAVED RoPE to Q and K BEFORE head split.
        # TP all_gather → host RoPE → re-shard. Connector is 1024 tokens so overhead is small.
        if rope_cos is not None and rope_sin is not None:
            from models.tt_dit.models.transformers.ltx.rope_ltx import _apply_interleaved_rotary_emb

            tp = self.parallel_config.tensor_parallel.factor
            tp_axis = self.parallel_config.tensor_parallel.mesh_axis
            if tp > 1:
                q = self.ccl_manager.all_gather(q, dim=2, mesh_axis=tp_axis, use_hyperparams=True)
                k = self.ccl_manager.all_gather(k, dim=2, mesh_axis=tp_axis, use_hyperparams=True)

            q_host = ttnn.to_torch(ttnn.get_device_tensors(q)[0]).float()
            k_host = ttnn.to_torch(ttnn.get_device_tensors(k)[0]).float()
            q_host = _apply_interleaved_rotary_emb(q_host, rope_cos.float(), rope_sin.float())
            k_host = _apply_interleaved_rotary_emb(k_host, rope_cos.float(), rope_sin.float())

            q = ttnn.from_torch(
                q_host.bfloat16(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )
            k = ttnn.from_torch(
                k_host.bfloat16(), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
            )

            # Re-shard for TP head split
            if tp > 1:
                q = ttnn.mesh_partition(q, dim=2, cluster_axis=tp_axis)
                k = ttnn.mesh_partition(k, dim=2, cluster_axis=tp_axis)

        tp = self.parallel_config.tensor_parallel.factor
        n_local_heads = self.num_heads // tp

        # Split heads
        B, S = q.shape[0], q.shape[1]
        q = ttnn.reshape(q, (B, S, n_local_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (B, S, n_local_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (B, S, n_local_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Ensure DRAM for SDPA
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)

        grid = self.mesh_device.compute_with_storage_grid_size()
        sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,
        )
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=sdpa_config,
            compute_kernel_config=self.compute_config,
        )
        attn_out = ttnn.transformer.concatenate_heads(attn_out)
        attn_out = ttnn.unsqueeze(attn_out, 0)

        if tp > 1:
            attn_out = self.ccl_manager.all_gather(
                attn_out,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        attn_out = self.to_out(attn_out, compute_kernel_config=self.compute_config)
        attn_out = ttnn.squeeze(attn_out, 0)
        x = attn_out + residual

        # FF with residual
        residual = x
        x = _rms_norm(x, self.eps)
        x = self.ff1(x, compute_kernel_config=self.compute_config)
        # ff2 is RowParallel: reduce_scatter internally → each device has dim/TP
        x = self.ff2(x, compute_kernel_config=self.compute_config)
        x = ttnn.unsqueeze(x, 0)
        if tp > 1:
            x = self.ccl_manager.all_gather(
                x,
                dim=3,
                mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
                use_hyperparams=True,
            )
        x = ttnn.squeeze(x, 0)
        return x + residual


class EmbeddingsConnector(Module):
    """Embeddings connector: aggregate_embed + transformer blocks.

    Takes stacked Gemma hidden states (B, seq, 49*3840) and produces
    embeddings at the target dim (4096 for video, 2048 for audio).
    """

    def __init__(
        self,
        *,
        input_dim: int,  # 188160 = 49 * 3840
        output_dim: int,  # 4096 (video) or 2048 (audio)
        num_blocks: int,  # 8 (video) or 2 (audio)
        num_heads: int = 32,
        ff_mult: int = 4,
        num_learnable_registers: int = 128,
        eps: float = 1e-6,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_learnable_registers = num_learnable_registers
        self.mesh_device = mesh_device

        # Aggregate embed: Linear(188160 → output_dim)
        self.aggregate_embed = Linear(input_dim, output_dim, bias=True, mesh_device=mesh_device)

        # Learnable registers (replace padding tokens)
        if num_learnable_registers > 0:
            self.learnable_registers = Parameter(
                total_shape=[num_learnable_registers, output_dim],
                device=mesh_device,
                dtype=ttnn.bfloat16,
            )

        # Transformer blocks
        ff_dim = output_dim * ff_mult
        self.transformer_1d_blocks = ModuleList(
            ConnectorBlock(output_dim, ff_dim, num_heads, eps, mesh_device, ccl_manager, parallel_config)
            for _ in range(num_blocks)
        )

        # Final norm is parameter-free RMS norm (matching reference)

    def _prepare_torch_state(self, state):
        # aggregate_embed comes from a separate prefix in the checkpoint
        # (handled by the caller, not here)
        pass

    def forward(self, stacked_hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            stacked_hidden_states: (B, seq, 49*3840) stacked Gemma hidden states

        Returns:
            (B, seq, output_dim) embeddings ready for DiT
        """
        # Project to target dim
        x = self.aggregate_embed(stacked_hidden_states)

        # Run through transformer blocks
        for block in self.transformer_1d_blocks:
            x = block(x)

        # Final norm (parameter-free RMS norm)
        x = _rms_norm(x)
        return x
