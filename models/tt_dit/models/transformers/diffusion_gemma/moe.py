# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin tt_dit wrapper around the demos/gemma4 MoE block.

Each DiffusionGemma encoder/decoder layer runs a 128-expert, top-8 MoE alongside
its dense MLP. The expert kernel (sparse_matmul over expert slots), router (RMSNorm
→ proj → softmax → topk → scatter), and weight handling are already implemented in
``models/demos/gemma4/tt/{moe,router,experts}.py``. This module:

  1. Adapts tt_dit-style construction (``DiTParallelConfig`` + ``mesh_device``) to
     the demos/gemma4 ``MeshConfig`` + ``CCLManager`` the inner classes expect.
  2. Adapts shapes at the boundary: tt_dit layers pass ``[B, S, H]`` tensors;
     demos/gemma4 wants ``[1, 1, B*S, H]``.
  3. Takes the full HF state-dict at construction time (no separate load step,
     matching demos/gemma4's ``__init__``-time weight loading pattern).

Prefill-only path: encoder prompt and decoder canvas always run S ≥ 32 in
DiffusionGemma's use case (no autoregressive single-token decode at the
sub-layer level — the diffusion sampler replaces that).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager as Gemma4CCLManager
from models.demos.gemma4.tt.moe import MoEBlock

from ....layers.module import Module
from ....parallel.config import DiTParallelConfig

TILE = ttnn.TILE_SIZE


@dataclass
class _HFConfigShim:
    """Subset of HF Gemma 4 text config consumed by demos/gemma4 MoE."""

    hidden_size: int
    num_experts: int
    top_k_experts: int
    moe_intermediate_size: int
    rms_norm_eps: float


class DiffusionGemmaMoE(Module):
    """Per-layer Mixture-of-Experts wrapper.

    The wrapper takes two pre-norm inputs (matching the demos/gemma4 ``MoEBlock``
    API): the router input (raw residual; the demos/gemma4 router applies its own
    internal RMSNorm) and the expert input (already pre-norm-ed by the caller via
    ``pre_feedforward_layernorm_2``).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k_experts: int,
        moe_intermediate_size: int,
        rms_norm_eps: float,
        state_dict: dict | None,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        expert_dtype: ttnn.DataType = ttnn.bfloat16,
        router_dtype: ttnn.DataType = ttnn.bfloat16,
        tensor_cache_path: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device

        tp_factor = parallel_config.tensor_parallel.factor
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mesh_shape = tuple(mesh_device.shape)

        # Construct demos/gemma4 MeshConfig from our DiTParallelConfig.
        # Prefer the explicit expert_parallel slot when set; otherwise default to "the
        # full non-tp axis" (replicated experts collapse to ep=1 if that axis is 1).
        if parallel_config.expert_parallel is not None:
            ep_factor = parallel_config.expert_parallel.factor
        else:
            ep_factor = mesh_shape[1 - tp_axis] if mesh_shape[1 - tp_axis] > 1 else 1
        mesh_config = MeshConfig(
            mesh_shape=mesh_shape,
            decode=ModeConfig(tp=tp_factor, ep=ep_factor, sp=1),
            prefill=ModeConfig(tp=tp_factor, ep=ep_factor, sp=1),
            tp_axis=tp_axis,
        )

        # demos/gemma4 has its own CCLManager. Construct one from our mesh device.
        gemma4_ccl_manager = Gemma4CCLManager(
            mesh_device=mesh_device,
            num_links=num_links,
            topology=topology,
        )

        hf_config = _HFConfigShim(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            moe_intermediate_size=moe_intermediate_size,
            rms_norm_eps=rms_norm_eps,
        )

        self._moe = MoEBlock(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            ccl_manager=gemma4_ccl_manager,
            mesh_config=mesh_config,
            dtype=expert_dtype,
            router_dtype=router_dtype,
            tensor_cache_path=tensor_cache_path,
        )

    def forward(self, router_input: ttnn.Tensor, expert_input: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            router_input:  replicated ``[B, S, hidden_size]`` (raw residual, no pre-norm).
            expert_input:  replicated ``[B, S, hidden_size]`` (already pre_feedforward_layernorm_2'd).

        Returns:
            replicated ``[B, S, hidden_size]``.
        """
        B, S = router_input.shape[0], router_input.shape[1]
        H = self.hidden_size
        assert (B * S) % TILE == 0, f"B*S = {B}*{S} = {B * S} must be tile-aligned for the prefill MoE path."

        # demos/gemma4 wants [1, 1, M, H] where M = B*S.
        router_in_1_1_M_H = ttnn.reshape(router_input, (1, 1, B * S, H))
        expert_in_1_1_M_H = ttnn.reshape(expert_input, (1, 1, B * S, H))

        out_1_1_M_H = self._moe(router_in_1_1_M_H, expert_in_1_1_M_H)
        return ttnn.reshape(out_1_1_M_H, (B, S, H))
