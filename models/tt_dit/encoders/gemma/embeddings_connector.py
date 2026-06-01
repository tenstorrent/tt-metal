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

import torch

import ttnn

from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import RMSNorm
from ...utils.substate import rename_substate


def _rms_norm_cc(mesh_device) -> ttnn.DeviceComputeKernelConfig:
    """HiFi4 + fp32 dest-acc compute config for the parameter-free RMS norms (matches the
    fidelity the DiT norms use; the native kernel's default fidelity costs ~1e-3 PCC)."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


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

        # FSDP: shard weights on the sequence-parallel axis (gathered per-op).
        sp = parallel_config.sequence_parallel
        fsdp_mesh_axis = sp.mesh_axis if (sp is not None and sp.factor > 1) else None

        # Norms are parameter-free RMS norms (matching reference ltx_core.utils.rms_norm)
        # No learnable weight — implemented as function calls in forward()

        # Self-attention (Q, K, V, O projections)
        col_kwargs = {"bias": True, "mesh_device": mesh_device, "mesh_axis": tp_axis}
        if fsdp_mesh_axis is not None:
            col_kwargs["fsdp_mesh_axis"] = fsdp_mesh_axis
            col_kwargs["ccl_manager"] = ccl_manager
        self.to_q = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_k = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_v = ColParallelLinear(dim, dim, **col_kwargs)
        self.to_out = Linear(dim, dim, bias=True, mesh_device=mesh_device)

        # Gemma-style QK normalization over the full inner dim, applied before RoPE
        # (reference Attention uses torch.nn.RMSNorm(inner_dim): raw weight, no +1).
        self.q_norm = RMSNorm(dim, norm_eps=eps, bias=False, mesh_device=mesh_device)
        self.k_norm = RMSNorm(dim, norm_eps=eps, bias=False, mesh_device=mesh_device)
        # Per-head gated attention: gates = 2*sigmoid(to_gate_logits(x)), applied to attn output.
        # dtype=float32 routes the matmul through HiFi4 + fp32 dest acc to match the host fp32
        # baseline (mirrors attention_ltx; the gate is precision-sensitive over 8 blocks).
        self.to_gate_logits = Linear(dim, num_heads, bias=True, dtype=ttnn.float32, mesh_device=mesh_device)

        # Feed-forward (GELU gated)
        self.ff1 = ColParallelLinear(
            dim,
            ff_dim,
            bias=True,
            activation_fn="gelu",
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            **({"fsdp_mesh_axis": fsdp_mesh_axis, "ccl_manager": ccl_manager} if fsdp_mesh_axis is not None else {}),
        )
        self.ff2 = RowParallelLinear(
            ff_dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.rmsnorm_cc = _rms_norm_cc(mesh_device)

    def _prepare_torch_state(self, state):
        rename_substate(state, "attn1.to_q", "to_q")
        rename_substate(state, "attn1.to_k", "to_k")
        rename_substate(state, "attn1.to_v", "to_v")
        rename_substate(state, "attn1.to_out.0", "to_out")
        rename_substate(state, "ff.net.0.proj", "ff1")
        rename_substate(state, "ff.net.2", "ff2")
        # QK norms and the gated-attention gate (reference Attention uses these).
        rename_substate(state, "attn1.q_norm", "q_norm")
        rename_substate(state, "attn1.k_norm", "k_norm")
        rename_substate(state, "attn1.to_gate_logits", "to_gate_logits")

        # SPLIT-rotation checkpoint → INTERLEAVED rotary_embedding_llama: permute Q/K
        # output channels per head so the SPLIT pair (i, i+D/2) lands at adjacent
        # interleaved slots (2i, 2i+1). q_norm/k_norm share the channel layout. Mirrors
        # attention_ltx; without it interleaved RoPE on the unpermuted layout gives PCC ~0.09.
        D = self.head_dim
        perm = torch.empty(D, dtype=torch.long)
        perm[0::2] = torch.arange(D // 2)
        perm[1::2] = torch.arange(D // 2, D)

        def _permute_qk(t: torch.Tensor) -> torch.Tensor:
            rest = t.shape[1:]
            return t.reshape(self.num_heads, D, *rest).index_select(1, perm).reshape(self.num_heads * D, *rest)

        for key in ("to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias", "q_norm.weight", "k_norm.weight"):
            if key in state:
                state[key] = _permute_qk(state[key])

    def forward(self, x, rope_cos=None, rope_sin=None, trans_mat=None):
        # Self-attention with residual
        residual = x
        x = ttnn.experimental.dit_rms_norm_unary_fused(
            x, weight=None, epsilon=self.eps, compute_kernel_config=self.rmsnorm_cc
        )
        attn_in = x  # normed input, reused for the per-head gate
        q = self.to_q(x, compute_kernel_config=self.compute_config)
        k = self.to_k(x, compute_kernel_config=self.compute_config)
        v = self.to_v(x, compute_kernel_config=self.compute_config)

        tp = self.parallel_config.tensor_parallel.factor
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        apply_rope = rope_cos is not None and rope_sin is not None

        # Gemma QK-norm over the full inner dim, BEFORE RoPE (raw-weight RMS, matching the
        # reference torch.nn.RMSNorm). Done on device — the q/k channels and the q_norm/k_norm
        # weights were permuted at load (SPLIT→INTERLEAVED), so RoPE below is the fast
        # interleaved rotary_embedding_llama kernel, no host round-trip. QK-norm runs over the
        # full inner dim, so TP gathers to full then re-shards for the head split.
        if apply_rope:
            if tp > 1:
                q = self.ccl_manager.all_gather(q, dim=2, mesh_axis=tp_axis, use_hyperparams=True)
                k = self.ccl_manager.all_gather(k, dim=2, mesh_axis=tp_axis, use_hyperparams=True)
            q = self.q_norm(q, compute_kernel_config=self.rmsnorm_cc)
            k = self.k_norm(k, compute_kernel_config=self.rmsnorm_cc)
            if tp > 1:
                q = ttnn.mesh_partition(q, dim=2, cluster_axis=tp_axis)
                k = ttnn.mesh_partition(k, dim=2, cluster_axis=tp_axis)

        n_local_heads = self.num_heads // tp

        # Split heads
        B, S = q.shape[0], q.shape[1]
        q = ttnn.reshape(q, (B, S, n_local_heads, self.head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (B, S, n_local_heads, self.head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (B, S, n_local_heads, self.head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Interleaved RoPE on the head-split Q/K (cos/sin/trans_mat prepared by the caller).
        if apply_rope:
            q = ttnn.experimental.rotary_embedding_llama(
                q, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.compute_config
            )
            k = ttnn.experimental.rotary_embedding_llama(
                k, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.compute_config
            )

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
        attn_out = ttnn.squeeze(attn_out, 0)  # (B, T, H*D), full

        # Per-head gated attention: gates = 2*sigmoid(to_gate_logits(attn_in)), each head's
        # output scaled by its gate. Done on device (matmul accumulates in fp32 via
        # compute_config); the gate is a smooth scalar in (0,2) so bf16 sigmoid is adequate.
        # No compute_kernel_config override: the fp32 weight selects HiFi4 (matching attention_ltx).
        gate_logits = self.to_gate_logits(attn_in)  # (B,T,H)
        gates = ttnn.multiply(ttnn.sigmoid(gate_logits), 2.0)  # (B,T,H)
        b, t = attn_out.shape[0], attn_out.shape[1]
        ao = ttnn.reshape(attn_out, (b, t, self.num_heads, self.head_dim))
        ao = ttnn.multiply(ao, ttnn.reshape(gates, (b, t, self.num_heads, 1)))
        attn_out = ttnn.reshape(ao, (b, t, self.num_heads * self.head_dim))

        attn_out = self.to_out(attn_out, compute_kernel_config=self.compute_config)
        x = attn_out + residual

        # FF with residual
        residual = x
        x = ttnn.experimental.dit_rms_norm_unary_fused(
            x, weight=None, epsilon=self.eps, compute_kernel_config=self.rmsnorm_cc
        )
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
        self.eps = eps
        self.rmsnorm_cc = _rms_norm_cc(mesh_device)

        # aggregate_embed lives in GemmaFeatureExtractor now (mirrors the reference
        # FeatureExtractorV2 boundary); this connector consumes the projected features.

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

    def forward(self, features: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            features: (B, seq, output_dim) aggregate_embed output from GemmaFeatureExtractor

        Returns:
            (B, seq, output_dim) embeddings ready for DiT
        """
        # Run through transformer blocks (no RoPE on this convenience path)
        x = features
        for block in self.transformer_1d_blocks:
            x = block(x)

        # Final norm (parameter-free RMS norm)
        x = ttnn.experimental.dit_rms_norm_unary_fused(
            x, weight=None, epsilon=self.eps, compute_kernel_config=self.rmsnorm_cc
        )
        return x
