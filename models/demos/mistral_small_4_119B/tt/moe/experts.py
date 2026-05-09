# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Routed expert weights + grouped matmul path (HF ``Mistral4NaiveMoe``).

Analogous to DeepSeek ``experts.Experts`` (``models/demos/deepseek_v3/tt/experts.py``).

Mistral packs gate/up and down projections as 3D tensors per expert; HF applies
``use_experts_implementation`` for grouped-MM dispatch.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4NaiveMoe

import ttnn


class TtMistral4Experts(Mistral4NaiveMoe):
    """Routed MoE experts (same forward as HF until ``ttnn`` expert kernels exist)."""

    def __init__(self, config: Mistral4Config) -> None:
        nn.Module.__init__(self)
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.config = config
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_intermediate_size * 2, config.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size, config.moe_intermediate_size)
        )
        self.act_fn = nn.SiLU()

    @classmethod
    def _experts_impl_config(cls, config: Mistral4Config) -> None:
        if getattr(config, "_experts_implementation", None) in (None, ""):
            try:
                config._experts_implementation = "grouped_mm"
            except (AttributeError, TypeError):
                pass

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: Any,
    ) -> dict[str, Any]:
        del hf_config
        (state_dict,) = state_dicts
        assert state_dict is not None
        output_path.mkdir(parents=True, exist_ok=True)

        gate_up = state_dict["gate_up_proj"].detach().cpu()
        down = state_dict["down_proj"].detach().cpu()

        if hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() == 1:
            weight_mapper = None
        else:
            weight_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        gate_up_ttnn = ttnn.from_torch(
            gate_up.unsqueeze(0).contiguous(),
            device=mesh_device,
            mesh_mapper=weight_mapper,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        down_ttnn = ttnn.from_torch(
            down.unsqueeze(0).contiguous(),
            device=mesh_device,
            mesh_mapper=weight_mapper,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        return {
            "w_gate_up_experts": {"input_tensor_b": gate_up_ttnn},
            "w_down_experts": {"input_tensor_b": down_ttnn},
            "experts_state_torch": {"gate_up_proj": gate_up, "down_proj": down},
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
    ) -> dict[str, Any]:
        memory_config = ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        return {
            "mesh_device": mesh_device,
            "mistral_hf_config": hf_config,
            "input_memory_config": memory_config,
            "output_memory_config": memory_config,
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> dict[str, Any]:
        return cls.model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> dict[str, Any]:
        return cls.model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def _bridge_host_forward(
        cls,
        x: ttnn.Tensor,
        top_k_index: ttnn.Tensor,
        top_k_weights: ttnn.Tensor,
        cfg: dict[str, Any],
    ) -> ttnn.Tensor:
        mesh_device = cfg["mesh_device"]
        hf_cfg = cfg["mistral_hf_config"]
        assert isinstance(hf_cfg, Mistral4Config)
        experts_state = cfg["experts_state_torch"]

        if mesh_device.get_num_devices() == 1:
            x_torch = ttnn.to_torch(x)
            topk_idx_torch = ttnn.to_torch(top_k_index)
            topk_w_torch = ttnn.to_torch(top_k_weights)
        else:
            x_torch = ttnn.to_torch(
                x,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                ),
            )
            topk_idx_torch = ttnn.to_torch(
                top_k_index,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                ),
            )
            topk_w_torch = ttnn.to_torch(
                top_k_weights,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                ),
            )

        while x_torch.dim() > 3 and x_torch.shape[1] == 1:
            x_torch = x_torch.squeeze(1)
        while topk_idx_torch.dim() > 3 and topk_idx_torch.shape[1] == 1:
            topk_idx_torch = topk_idx_torch.squeeze(1)
        while topk_w_torch.dim() > 3 and topk_w_torch.shape[1] == 1:
            topk_w_torch = topk_w_torch.squeeze(1)

        x_flat = x_torch.reshape(-1, x_torch.shape[-1]).to(torch.float32)
        idx_flat = topk_idx_torch.reshape(-1, topk_idx_torch.shape[-1]).to(torch.int64)
        w_flat = topk_w_torch.reshape(-1, topk_w_torch.shape[-1]).to(torch.float32)

        mcfg = deepcopy(hf_cfg)
        cls._experts_impl_config(mcfg)
        ref = Mistral4NaiveMoe(mcfg).eval().to(torch.float32)
        ref.load_state_dict(experts_state, strict=False)
        with torch.no_grad():
            y = ref(x_flat, idx_flat, w_flat)
        y = y.to(torch.bfloat16).view(1, 1, x_flat.shape[0], y.shape[-1])

        if mesh_device.get_num_devices() == 1:
            out_mapper = None
        else:
            out_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
        return ttnn.from_torch(
            y,
            device=mesh_device,
            mesh_mapper=out_mapper,
            dtype=ttnn.bfloat16,
            memory_config=cfg["output_memory_config"],
            layout=ttnn.TILE_LAYOUT,
        )

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        top_k_index: ttnn.Tensor,
        top_k_weights: ttnn.Tensor,
        cfg: dict[str, Any],
    ) -> ttnn.Tensor:
        return cls._bridge_host_forward(x, top_k_index, top_k_weights, cfg)

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        top_k_index: ttnn.Tensor,
        top_k_weights: ttnn.Tensor,
        cfg: dict[str, Any],
    ) -> ttnn.Tensor:
        return cls._bridge_host_forward(x, top_k_index, top_k_weights, cfg)
