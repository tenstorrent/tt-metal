# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    FromWeightConfig,
    LinearConfig,
    ModelDecodeConfig,
    ModelPrefillConfig,
    MulConfig,
    WeightConfig,
)


class Expert(AbstractModule):
    """
    Expert layer for Mixture-of-Experts (MoE) models.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        """
        MOE expert layer running on 1 device.
        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """

        # Get the weights of exerpt from the state dict
        # breakpoint()
        torch_weight_w1 = state_dict["gate_proj.weight"]
        torch_weight_w2 = state_dict["down_proj.weight"]
        torch_weight_w3 = state_dict["up_proj.weight"]

        torch_weight_w1 = torch_weight_w1.transpose(-2, -1)
        torch_weight_w2 = torch_weight_w2.transpose(-2, -1)
        torch_weight_w3 = torch_weight_w3.transpose(-2, -1)

        weight_config = {}

        def add_weight_config(
            torch_weight,
            our_name,
            kwarg_name,
            dtype,
            mem_config,
            layout,
            mesh_mapper=None,
        ):
            ttnn_weight = ttnn.as_tensor(
                torch_weight,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=layout,
                memory_config=mem_config,
            )
            weight_file_path = output_path / f"{our_name}.{kwarg_name}.weight"
            ttnn.dump_tensor(weight_file_path, ttnn_weight)
            ttnn.deallocate(ttnn_weight)

            # Add to weight config
            weight_config[our_name] = {kwarg_name: str(weight_file_path)}

        add_weight_config(
            torch_weight_w1,
            "w1",
            "input_tensor_b",
            dtype=ttnn.bfloat16,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        add_weight_config(
            torch_weight_w2,
            "w2",
            "input_tensor_b",
            dtype=ttnn.bfloat16,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        add_weight_config(
            torch_weight_w3,
            "w3",
            "input_tensor_b",
            dtype=ttnn.bfloat16,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        return weight_config

    @staticmethod
    def decode_model_config(hf_config, mesh_device) -> ModelDecodeConfig:
        """Generate decode operator configuration for this embedding layer.
        Same as prefill mode for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        config = {"mode": "decode"}
        # Expert configuration for decode mode
        config["w1"] = LinearConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=None, input_tensor_b=FromWeightConfig()
        )

        config["w2"] = LinearConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=None, input_tensor_b=FromWeightConfig()
        )

        config["w3"] = LinearConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=None, input_tensor_b=FromWeightConfig()
        )

        config["mul"] = MulConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

        config["input_memory_config"] = ttnn.DRAM_MEMORY_CONFIG
        config["output_memory_config"] = ttnn.DRAM_MEMORY_CONFIG

        return config

    @staticmethod
    def prefill_model_config(hf_config, mesh_device) -> ModelPrefillConfig:
        """Prefill model config for an RMSNorm with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """
        config = Expert.decode_model_config(hf_config, mesh_device)
        config["mode"] = "prefill"

        return config

    def __init__(self, hf_config, mesh_device):
        """
        Initializes the Expert layer.
        """

        super().__init__(hf_config, mesh_device)
        self.hf_config = hf_config
        self.mesh_device = mesh_device

    @classmethod
    def forward(cls, x, cfg, mesh_device):
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """

        if cfg["mode"] == "decode":
            return cls.forward_decode(x, cfg, mesh_device)
        elif cfg["mode"] == "prefill":
            return cls.forward_prefill(x, cfg, mesh_device)
        else:
            raise ValueError(f"Invalid mode: {cfg['mode']}. Expected 'decode' or 'prefill'.")

    @classmethod
    def forward_decode(cls, x, cfg, mesh_device):
        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])
        ttnn.deallocate(x)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        # Down projection
        output = ttnn.linear(activated, **cfg["w2"])
        ttnn.deallocate(activated)

        return output

    @classmethod
    def forward_prefill(cls, x, cfg, mesh_device):
        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])
        ttnn.deallocate(x)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        # Down projection
        output = ttnn.linear(activated, **cfg["w2"])
        ttnn.deallocate(activated)

        return output
