# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thin tt_dit wrapper around the demos/gemma4 MoE block.

Adapts tt_dit's DiTParallelConfig + [B, S, H] convention to demos/gemma4's
MeshConfig + [1, 1, B*S, H] MoEBlock. Prefill-only path (diffusion sampler
replaces sub-layer autoregressive decode).
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager as Gemma4CCLManager
from models.demos.gemma4.tt.moe import MoEBlock

from ....layers.module import Module
from ....parallel.config import DiTParallelConfig
from ....utils.tensor import local_device_to_torch


@dataclass
class _HFConfigShim:
    """Subset of HF Gemma 4 text config consumed by demos/gemma4 MoE."""

    hidden_size: int
    num_experts: int
    top_k_experts: int
    moe_intermediate_size: int
    rms_norm_eps: float


class DiffusionGemmaMoE(Module):
    """Per-layer MoE wrapper. Takes raw residual (router applies its own RMSNorm) and
    pre_feedforward_layernorm_2'd expert input, matching demos/gemma4's MoEBlock API.
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
        log_routing_histogram: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device
        # When True, pulls the dense_routing tensor to host and logs per-expert token assignment counts.
        self.log_routing_histogram = log_routing_histogram

        tp_factor = parallel_config.tensor_parallel.factor
        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mesh_shape = tuple(mesh_device.shape)

        # Build demos/gemma4 MeshConfig from our DiTParallelConfig. EP defaults to the
        # non-tp axis; collapses to 1 if that axis is size 1 (replicated experts).
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
            router_input:  replicated ``[1, B, S, H]`` (raw residual).
            expert_input:  replicated ``[1, B, S, H]`` (already pre_feedforward_layernorm_2'd).

        Returns replicated ``[1, B, S, H]``.
        """
        # Read batch/seq from the trailing 3 dims so both 3D and 4D 1BND inputs work.
        B, S = router_input.shape[-3], router_input.shape[-2]
        H = self.hidden_size

        # sparse_matmul empirically hangs at non-multiple-of-128 M values (M=32/160/192/224
        # hang; M=128/256/1024/4096 work). Pad up to the next multiple of 128 (floor 128).
        M = B * S
        _STRIDE = 128
        M_padded = max(_STRIDE, ((M + _STRIDE - 1) // _STRIDE) * _STRIDE)
        pad_rows = M_padded - M

        router_in_1_1_M_H = ttnn.reshape(router_input, (1, 1, M, H))
        expert_in_1_1_M_H = ttnn.reshape(expert_input, (1, 1, M, H))

        if pad_rows > 0:
            router_in_1_1_M_H = ttnn.pad(router_in_1_1_M_H, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)
            expert_in_1_1_M_H = ttnn.pad(expert_in_1_1_M_H, [(0, 0), (0, 0), (0, pad_rows), (0, 0)], value=0.0)

        # Inline what MoEBlock.__call__ does (router → experts) so we can capture the
        # dense_routing tensor for the diagnostic histogram without an extra router pass.
        dense_routing = self._moe.router(router_in_1_1_M_H)
        if self.log_routing_histogram:
            self._log_routing_histogram(dense_routing, real_rows=M)
        out_1_1_Mp_H = self._moe.experts(expert_in_1_1_M_H, dense_routing)

        if pad_rows > 0:
            out_1_1_Mp_H = ttnn.slice(out_1_1_Mp_H, [0, 0, 0, 0], [1, 1, M, H])

        return ttnn.reshape(out_1_1_Mp_H, (1, B, S, H))

    def _log_routing_histogram(self, dense_routing: ttnn.Tensor, real_rows: int) -> None:
        """Diagnostic: log per-expert token counts from an existing dense_routing tensor."""
        routing_host = local_device_to_torch(dense_routing).to("cpu")
        if routing_host.ndim == 4:
            routing_host = routing_host.squeeze(0).squeeze(0)  # [M_padded, num_experts]
        routing_host = routing_host[:real_rows]  # drop padded rows
        counts = (routing_host != 0).sum(dim=0).tolist()
        total = sum(counts)
        expected = real_rows * self.top_k_experts
        selected = sum(1 for c in counts if c > 0)
        summary = ", ".join(f"e{e}:{c}" for e, c in enumerate(counts) if c > 0)
        logger.info(
            f"[MoE routing] tokens={real_rows} top_k={self.top_k_experts} "
            f"total_assignments={total} (expected {expected})"
        )
        logger.info(
            f"[MoE routing] per-expert counts (nonzero only): {summary} .... "
            f"selected_experts = {selected} / {self.num_experts}"
        )
