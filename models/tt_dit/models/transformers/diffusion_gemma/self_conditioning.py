# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn

from ....layers.feedforward import GatedMLP
from ....layers.module import Module
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig


class DiffusionGemmaSelfConditioning(Module):
    """Self-conditioning block prepended to the decoder.

    Reference: ``transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaSelfConditioning``::

        normed = pre_norm(self_conditioning_signal)
        sc     = down_proj( act_fn(gate_proj(normed)) * up_proj(normed) )
        return post_norm(inputs_embeds + sc)

    The gated body is the shared ``GatedMLP`` primitive.
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
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager

        self.pre_norm = RMSNorm(
            embedding_dim=hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.gated_mlp = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.post_norm = RMSNorm(
            embedding_dim=hidden_size,
            norm_eps=rms_norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
        )

    def _prepare_torch_state(self, state: dict) -> None:
        """HF stores ``gate_proj``/``up_proj``/``down_proj`` as direct children of
        ``DiffusionGemmaSelfConditioning``; we nest them under ``gated_mlp``. Re-prefix."""
        for name in ("gate_proj", "up_proj", "down_proj"):
            for k in list(state.keys()):
                if k == f"{name}.weight" or k.startswith(f"{name}."):
                    state[f"gated_mlp.{k}"] = state.pop(k)

    def forward(self, inputs_embeds: ttnn.Tensor, self_conditioning_signal: ttnn.Tensor) -> ttnn.Tensor:
        normed = self.pre_norm(self_conditioning_signal)
        sc_signal = self.gated_mlp(normed)
        ttnn.deallocate(normed)
        # GatedMLP emits TP-fractured hidden; gather so the residual add against
        # replicated ``inputs_embeds`` (and the post_norm reduction) sees the full hidden dim.
        # Use ``dim=-1`` so this works whether the caller feeds a 3D [B,S,H] tensor (as the
        # decoder does — embed_tokens output before the 4D unsqueeze) or a 4D [1,B,S,H].
        if self.parallel_config.tensor_parallel.factor > 1:
            sc_signal = self.ccl_manager.all_gather_persistent_buffer(
                sc_signal, dim=-1, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        combined = ttnn.add(inputs_embeds, sc_signal)
        ttnn.deallocate(sc_signal)
        out = self.post_norm(combined)
        ttnn.deallocate(combined)
        return out
