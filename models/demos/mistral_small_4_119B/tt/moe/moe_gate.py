# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MoE router / gate (HF ``Mistral4TopkRouter``), analogous to DeepSeek ``moe_gate.MoEGate``.

DeepSeek reference: ``models/demos/deepseek_v3/tt/moe_gate.py``.

This module exposes DeepSeek-style TT hooks while preserving the Mistral4 routing
math from HF ``Mistral4MoE.route_tokens_to_experts``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.modeling_mistral4 import Mistral4TopkRouter

import ttnn


class TtMistral4MoEGate(Mistral4TopkRouter):
    """Top‑k expert router for Mistral-4 MoE layers."""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: Any,
        prefix: str = "",
    ) -> dict[str, Any]:
        del hf_config
        (state_dict,) = state_dicts
        assert state_dict is not None
        output_path.mkdir(parents=True, exist_ok=True)

        gate_weight = state_dict[f"{prefix}weight"].detach().cpu()
        if hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() == 1:
            weight_mapper = None
        else:
            weight_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        gate_weight_ttnn = ttnn.from_torch(
            gate_weight.unsqueeze(0).unsqueeze(0).contiguous(),
            device=mesh_device,
            mesh_mapper=weight_mapper,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        return {
            "gate_proj": {"input_tensor_b": gate_weight_ttnn},
            # Host fallback path uses this tensor to avoid depending on full TT op coverage.
            "gate_weight_torch": gate_weight,
        }

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: Any, ccl: Any) -> dict[str, Any]:
        del hf_config, ccl
        return {"mesh_device": mesh_device}

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        mode: str,
        topk_fallback: bool = True,
        use_bitonic_sort: bool = True,
    ) -> dict[str, Any]:
        del use_bitonic_sort
        del mode
        # Keep gate outputs in L1 for both decode/prefill in bring-up path.
        memory_config = ttnn.L1_MEMORY_CONFIG
        return {
            "mesh_device": mesh_device,
            "input_memory_config": memory_config,
            "output_memory_config": memory_config,
            "topk_fallback": topk_fallback,
            "n_group": hf_config.n_group,
            "n_routed_experts": hf_config.n_routed_experts,
            "topk_group": hf_config.topk_group,
            "top_k": hf_config.num_experts_per_tok,
            "norm_topk_prob": hf_config.norm_topk_prob,
            "routed_scaling_factor": hf_config.routed_scaling_factor,
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = True,
        use_bitonic_sort: bool = True,
    ) -> dict[str, Any]:
        return cls.model_config(hf_config, mesh_device, "decode", topk_fallback, use_bitonic_sort)

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = True,
        use_bitonic_sort: bool = True,
    ) -> dict[str, Any]:
        return cls.model_config(hf_config, mesh_device, "prefill", topk_fallback, use_bitonic_sort)

    @classmethod
    def _route_tokens_to_experts(
        cls, router_logits: torch.Tensor, cfg: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs = router_logits.softmax(-1)
        n_group = int(cfg["n_group"])
        n_routed_experts = int(cfg["n_routed_experts"])
        topk_group = int(cfg["topk_group"])
        top_k = int(cfg["top_k"])

        group_scores = probs.view(-1, n_group, n_routed_experts // n_group).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1).expand(-1, n_group, n_routed_experts // n_group).reshape(-1, n_routed_experts)
        )
        scores_for_choice = probs.masked_fill(~score_mask.bool(), 0.0)

        topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
        topk_weights = probs.gather(1, topk_indices)
        if bool(cfg["norm_topk_prob"]):
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * float(cfg["routed_scaling_factor"])
        return topk_weights, topk_indices

    @classmethod
    def topk_fallback_op(
        cls,
        input: ttnn.Tensor,
        *,
        mesh_device: ttnn.Device,
        k: int,
        dim: int,
        largest: bool = True,
        sorted: bool = False,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if mesh_device.get_num_devices() == 1:
            torch_input = ttnn.to_torch(input)
        else:
            torch_input = ttnn.to_torch(
                input,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                ),
            )
        torch_scores, torch_indices = torch.topk(torch_input, k=k, dim=dim, largest=largest, sorted=sorted)
        if mesh_device.get_num_devices() == 1:
            mapper = None
        else:
            mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
        tt_scores = ttnn.from_torch(
            torch_scores,
            device=mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_indices = ttnn.from_torch(
            torch_indices.to(torch.int32),
            device=mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        return tt_scores, tt_indices

    @classmethod
    def linear_fallback_op(
        cls,
        input_tensor: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        *,
        mesh_device: ttnn.Device,
        dtype: ttnn.DataType,
        memory_config: ttnn.MemoryConfig,
        transpose_b: bool = True,
    ) -> ttnn.Tensor:
        if mesh_device.get_num_devices() == 1:
            torch_input = ttnn.to_torch(input_tensor)
            torch_weight = ttnn.to_torch(input_tensor_b)[0, 0]
        else:
            torch_input = ttnn.to_torch(
                input_tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                ),
            )
            torch_weight = ttnn.to_torch(
                input_tensor_b,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=tuple(mesh_device.shape)),
            )[0, 0]
        torch_input_2d = torch_input.view(-1, torch_input.shape[-1])
        torch_weight_2d = torch_weight if transpose_b else torch_weight.T
        torch_output_2d = F.linear(torch_input_2d, torch_weight_2d)
        torch_output = torch_output_2d.view(*torch_input.shape[:-1], torch_output_2d.shape[-1])

        if mesh_device.get_num_devices() == 1:
            out_mapper = None
        else:
            out_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
        return ttnn.from_torch(
            torch_output,
            device=mesh_device,
            mesh_mapper=out_mapper,
            dtype=dtype,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        assert x.memory_config() == cfg["input_memory_config"]
        mesh_device = cfg["mesh_device"]
        weight = cfg["gate_weight_torch"]

        if mesh_device.get_num_devices() == 1:
            x_torch = ttnn.to_torch(x)
        else:
            x_torch = ttnn.to_torch(
                x,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                ),
            )
        while x_torch.dim() > 3 and x_torch.shape[1] == 1:
            x_torch = x_torch.squeeze(1)
        x_flat = x_torch.reshape(-1, x_torch.shape[-1]).to(torch.float32)
        router_logits = F.linear(x_flat, weight.to(torch.float32))

        topk_weights_torch, topk_indices_torch = cls._route_tokens_to_experts(router_logits, cfg)
        topk_weights_torch = topk_weights_torch.to(torch.bfloat16).view(1, 1, x_flat.shape[0], -1)
        topk_indices_torch = topk_indices_torch.to(torch.int32).view(1, 1, x_flat.shape[0], -1)

        if mesh_device.get_num_devices() == 1:
            mapper = None
        else:
            mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape))
        topk_weights = ttnn.from_torch(
            topk_weights_torch,
            device=mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.bfloat16,
            memory_config=cfg["output_memory_config"],
            layout=ttnn.TILE_LAYOUT,
        )
        topk_indices = ttnn.from_torch(
            topk_indices_torch,
            device=mesh_device,
            mesh_mapper=mapper,
            dtype=ttnn.uint16,
            memory_config=cfg["output_memory_config"],
            layout=ttnn.TILE_LAYOUT,
        )
        return topk_weights, topk_indices

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)
