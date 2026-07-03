# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native Cosmos3VLTextMoTDecoderLayer.

Composes the native joint-attention + dual MLPs + 4 RMSNorms + 2
residual adds into a single on-device decoder layer. Mirrors the
reference `Cosmos3VLTextMoTDecoderLayer`:

    und_norm = input_layernorm(und_seq)
    gen_norm = input_layernorm_moe_gen(gen_seq)
    und_attn, gen_attn = self_attn(und_norm, gen_norm, rotary_emb)
    res_und = und_seq + und_attn
    res_gen = gen_seq + gen_attn
    mlp_und = mlp(post_attention_layernorm(res_und))
    mlp_gen = mlp_moe_gen(post_attention_layernorm_moe_gen(res_gen))
    return (res_und + mlp_und, res_gen + mlp_gen)

Why this matters: with Phase 1 tt-symbiote + native attention/MLP, the
RMSNorms and residual adds stay on host PyTorch, forcing ~12
host↔device roundtrips per decoder layer. With 64 layers and 20+
denoise steps that's ~15k roundtrips per generate. Fusing everything
on-device cuts it to ~1 roundtrip per layer (only the layer boundary).

Tensor layout:
  - Inputs (und_seq, gen_seq): replicated `[1, 1, N, hidden_size]`.
  - All intermediates stay on device, replicated across the mesh
    (RMSNorm is per-chip, attention/MLP do their own internal sharding
    and gather back to replicated at the layer boundary).
  - Outputs: replicated `[1, 1, N, hidden_size]`.

mRoPE cos/sin are still pre-computed on host and passed in as tuples —
that's a separate task (Phase 2 item 2). The forward signature matches
the reference's `forward(und_seq, gen_seq, rotary_emb)` where
`rotary_emb = (cos_und, sin_und, cos_gen, sin_gen)`.
"""

from __future__ import annotations

import os

import ttnn

from ....layers.module import Module
from ....layers.normalization import DistributedRMSNorm, RMSNorm
from ....parallel.config import DiTParallelConfig, ParallelFactor
from .attention import Cosmos3JointAttention
from .mlp import Cosmos3VLTextMLP

_FRACTURED_TP = os.environ.get("TT_COSMOS3_FRACTURED_TP") in ("1", "true", "True")


def _default_parallel_config() -> DiTParallelConfig:
    return DiTParallelConfig(
        cfg_parallel=ParallelFactor(1, 0),
        tensor_parallel=ParallelFactor(1, 1),
        sequence_parallel=ParallelFactor(1, 0),
    )


class Cosmos3VLTextMoTDecoderLayer(Module):
    """Dual-pathway decoder layer for the Cosmos3 MoT trunk."""

    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig | None = None,
        ccl_manager=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if parallel_config is None:
            parallel_config = _default_parallel_config()

        self.hidden_size = hidden_size
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        tp_factor = parallel_config.tensor_parallel.factor

        if _FRACTURED_TP and tp_factor > 1:
            norm_kw = {
                "embedding_dim": hidden_size,
                "norm_eps": rms_norm_eps,
                "norm_elementwise_affine": True,
                "bias": False,
                "mesh_axis": tp_axis,
                "mesh_device": mesh_device,
                "ccl_manager": ccl_manager,
            }
            NormClass = DistributedRMSNorm
        else:
            norm_kw = {
                "embedding_dim": hidden_size,
                "norm_eps": rms_norm_eps,
                "norm_elementwise_affine": True,
                "bias": False,
                "mesh_device": mesh_device,
                "dtype": dtype,
            }
            NormClass = RMSNorm
        attn_kw = {
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "attention_bias": attention_bias,
            "rms_norm_eps": rms_norm_eps,
            "mesh_device": mesh_device,
            "parallel_config": parallel_config,
            "ccl_manager": ccl_manager,
            "dtype": dtype,
        }
        mlp_kw = {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "mesh_device": mesh_device,
            "parallel_config": parallel_config,
            "ccl_manager": ccl_manager,
            "dtype": dtype,
        }

        self.input_layernorm = NormClass(**norm_kw)
        self.input_layernorm_moe_gen = NormClass(**norm_kw)
        self.self_attn = Cosmos3JointAttention(**attn_kw)
        self.post_attention_layernorm = NormClass(**norm_kw)
        self.post_attention_layernorm_moe_gen = NormClass(**norm_kw)
        self.mlp = Cosmos3VLTextMLP(**mlp_kw)
        self.mlp_moe_gen = Cosmos3VLTextMLP(**mlp_kw)

        if _FRACTURED_TP and tp_factor > 1:
            object.__setattr__(self.self_attn, "_fractured_tp", True)
            self._fractured_tp = True
        else:
            self._fractured_tp = False

    def forward(
        self,
        und_seq: ttnn.Tensor,
        gen_seq: ttnn.Tensor,
        cos_und: ttnn.Tensor,
        sin_und: ttnn.Tensor,
        cos_gen: ttnn.Tensor,
        sin_gen: ttnn.Tensor,
        logical_n_gen: int | None = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run one decoder layer entirely on device.

        Args:
            und_seq: Replicated `[1, 1, N_und, hidden_size]`.
            gen_seq: Replicated at sp=1, sp-sharded at sp>1.
            cos_und, sin_und, cos_gen, sin_gen: Match the corresponding seq tensor's layout.
            logical_n_gen: Unpadded N_gen, required when sp_factor>1.

        At sp>1, gen tensors stay sp-sharded across the entire layer — RMSNorm,
        residual add, and MLP are per-token, so they compose with sp-sharding
        without any cross-token collectives.
        """
        # NOTE: `self.self_attn(...)` and `self.mlp(...)` return persistent ping-pong buffers
        # from ccl_manager when tp_factor > 1. Do NOT `ttnn.deallocate` their return values —
        # the ccl_manager owns those buffers, and freeing them corrupts the cache for the next
        # CCL op of the same shape (e.g. a later layer's attention or MLP all-gather). Same
        # constraint applies to RMSNorm outputs that feed directly into a sub-module that
        # might return a persistent buffer. To stay correct without micro-managing which
        # specific tensors are persistent, we don't manually deallocate inside the layer.
        # Python GC + the ccl_manager's ping-pong rotation handle memory reuse.
        und_norm = self.input_layernorm(und_seq)
        gen_norm = self.input_layernorm_moe_gen(gen_seq)

        und_attn, gen_attn = self.self_attn(
            und_norm, gen_norm, cos_und, sin_und, cos_gen, sin_gen, logical_n_gen=logical_n_gen
        )

        residual_und = ttnn.add(und_seq, und_attn)
        residual_gen = ttnn.add(gen_seq, gen_attn)

        mlp_und_in = self.post_attention_layernorm(residual_und)
        mlp_gen_in = self.post_attention_layernorm_moe_gen(residual_gen)

        mlp_und_out = self.mlp(mlp_und_in, fractured_tp=self._fractured_tp)
        mlp_gen_out = self.mlp_moe_gen(mlp_gen_in, fractured_tp=self._fractured_tp)

        und_out = ttnn.add(residual_und, mlp_und_out)
        gen_out = ttnn.add(residual_gen, mlp_gen_out)

        return und_out, gen_out
