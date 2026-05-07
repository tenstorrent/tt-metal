# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared expert MLP inside ``Mistral4MoE`` (HF ``Mistral4MLP`` with widened intermediate).

Analogous to DeepSeek ``mlp/shared_expert.SharedExpert``
(``models/demos/deepseek_v3/tt/mlp/shared_expert.py``).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4MLP

import ttnn


class TtMistral4SharedExpert(Mistral4MLP):
    """One shared MLP applied to every token; width matches ``n_shared_experts`` in HF MoE."""

    def __init__(self, config: Mistral4Config) -> None:
        super().__init__(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )

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

        gate = state_dict["gate_proj.weight"].detach().cpu()
        up = state_dict["up_proj.weight"].detach().cpu()
        down = state_dict["down_proj.weight"].detach().cpu()

        gate_ttnn = ttnn.from_torch(
            gate.unsqueeze(0).contiguous(),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        up_ttnn = ttnn.from_torch(
            up.unsqueeze(0).contiguous(),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        down_ttnn = ttnn.from_torch(
            down.unsqueeze(0).contiguous(),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        return {
            "w_gate_shared_expert": {"input_tensor_b": gate_ttnn},
            "w_up_shared_expert": {"input_tensor_b": up_ttnn},
            "w_down_shared_expert": {"input_tensor_b": down_ttnn},
            "shared_expert_state_torch": {
                "gate_proj.weight": gate,
                "up_proj.weight": up,
                "down_proj.weight": down,
            },
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
    def _bridge_host_forward(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> ttnn.Tensor:
        mesh_device = cfg["mesh_device"]
        hf_cfg = cfg["mistral_hf_config"]
        assert isinstance(hf_cfg, Mistral4Config)
        mlp_state = cfg["shared_expert_state_torch"]

        x_torch = ttnn.to_torch(
            x,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        while x_torch.dim() > 3 and x_torch.shape[1] == 1:
            x_torch = x_torch.squeeze(1)
        x_flat = x_torch.reshape(-1, x_torch.shape[-1]).to(torch.float32)

        mcfg = deepcopy(hf_cfg)
        ref = cls(mcfg).eval().to(torch.float32)
        ref.load_state_dict(mlp_state, strict=False)
        with torch.no_grad():
            y = ref(x_flat)
        y = y.to(torch.bfloat16).view(1, 1, x_flat.shape[0], y.shape[-1])

        return ttnn.from_torch(
            y,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=cfg["output_memory_config"],
            layout=ttnn.TILE_LAYOUT,
        )

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        cfg: dict[str, Any],
        handle_tensor_parallel: bool = False,
    ) -> ttnn.Tensor:
        del handle_tensor_parallel
        return cls._bridge_host_forward(x, cfg)

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        cfg: dict[str, Any],
        handle_tensor_parallel: bool = False,
    ) -> ttnn.Tensor:
        del handle_tensor_parallel
        return cls._bridge_host_forward(x, cfg)
