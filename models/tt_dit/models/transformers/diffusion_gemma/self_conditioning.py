# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from ....layers.linear import ColParallelLinear, RowParallelLinear
from ....layers.module import Module
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig

TILE = ttnn.TILE_SIZE


class DiffusionGemmaSelfConditioning(Module):
    """Gated-MLP self-conditioning block prepended to the decoder.

    Reference: transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaSelfConditioning.

        normed = pre_norm(self_conditioning_signal)
        sc     = down_proj( act_fn(gate_proj(normed)) * up_proj(normed) )
        return post_norm(inputs_embeds + sc)

    The input embeddings are passed through replicated; the gated MLP is megatron-style
    TP (column-parallel on the gate/up, row-parallel on the down). Output is replicated.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
        parallel_config: DiTParallelConfig,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        tp_factor = parallel_config.tensor_parallel.factor
        # Tile alignment: per-device intermediate dim must be a multiple of 32.
        # DiffusionGemma uses intermediate_size=2112 → divisible by 32 only for tp∈{1,2}.
        # Larger TP factors will require padding handled by a higher-level wrapper.
        assert (intermediate_size // tp_factor) % TILE == 0, (
            f"intermediate_size ({intermediate_size}) / tp_factor ({tp_factor}) "
            f"must be tile-aligned ({TILE}); add output padding if larger TP is needed."
        )
        assert hidden_size % TILE == 0

        self.parallel_config = parallel_config
        self.mesh_device = mesh_device

        # pre_norm: weight, no bias. Input is replicated, normed over hidden_size → use plain RMSNorm.
        self.pre_norm = RMSNorm(
            embedding_dim=hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        # post_norm: no scale (with_scale=False in HF) → norm_elementwise_affine=False.
        self.post_norm = RMSNorm(
            embedding_dim=hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
        )

        col_kwargs = dict(
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.gate_proj = ColParallelLinear(hidden_size, intermediate_size, activation_fn="gelu_tanh", **col_kwargs)
        self.up_proj = ColParallelLinear(hidden_size, intermediate_size, **col_kwargs)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

    def forward(
        self,
        inputs_embeds: ttnn.Tensor,
        self_conditioning_signal: ttnn.Tensor,
        *,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """
        inputs_embeds, self_conditioning_signal: replicated [B, canvas_length, hidden_size].
        Output: replicated [B, canvas_length, hidden_size].
        """
        normed = self.pre_norm(self_conditioning_signal, compute_kernel_config=compute_kernel_config)
        gate = self.gate_proj(normed, parallel_config=self.parallel_config, compute_kernel_config=compute_kernel_config)
        up = self.up_proj(normed, parallel_config=self.parallel_config, compute_kernel_config=compute_kernel_config)
        gated = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        sc_signal = self.down_proj(gated, compute_kernel_config=compute_kernel_config)
        ttnn.deallocate(gated)
        combined = ttnn.add(inputs_embeds, sc_signal)
        ttnn.deallocate(sc_signal)
        out = self.post_norm(combined, compute_kernel_config=compute_kernel_config)
        ttnn.deallocate(combined)
        return out
