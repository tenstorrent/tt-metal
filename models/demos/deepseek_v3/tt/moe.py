from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import LinearConfig, MulConfig, WeightsConfig
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_HIFI2_FP16


class Expert(AbstractModule):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightsConfig:
        """DRAM-sharded weights split 1D across all wormholes

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """

        # Get the weights of exerpt from the state dict
        torch_weight_gate_proj = state_dict["gate_proj"]
        torch_weight_up_proj = state_dict["up_proj"]
        torch_weight_down_proj = state_dict["down_proj"]

        # Convert to TTNN tensor with 1D sharding across final dimension
        ttnn_weight_gate_proj = ttnn.as_tensor(
            torch_weight_gate_proj,
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[-2, -1], mesh_shape=list(mesh_device.shape)),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_weight_up_proj = ttnn.as_tensor(
            torch_weight_up_proj,
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[-2, -1], mesh_shape=list(mesh_device.shape)),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn_weight_down_proj = ttnn.as_tensor(
            torch_weight_down_proj,
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[-2, -1], mesh_shape=list(mesh_device.shape)),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return {
            "gate_proj": WeightStub.from_weight(ttnn_weight_gate_proj, output_path / "gate_proj.weight"),
            "up_proj": WeightStub.from_weight(ttnn_weight_up_proj, output_path / "up_proj.weight"),
            "down_proj": WeightStub.from_weight(ttnn_weight_down_proj, output_path / "down_proj.weight"),
        }

    @staticmethod
    def prefill_model_config(hf_config, mesh_device):
        """Prefill model config for an RMSNorm with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """
        config = {"mode": "prefill"}

        return config

    @staticmethod
    def decode_model_config(hf_config, mesh_device):
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
        config["gate_proj"] = LinearConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        )

        config["up_proj"] = LinearConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        )

        config["down_proj"] = LinearConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        )

        config["mul"] = MulConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

        return config

    def __init__(self, hf_config, mesh_device):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        """
        Ref:
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

        """
        super().__init__(hf_config, mesh_device)
        self.hf_config = hf_config
        self.mesh_device = mesh_device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.
        Decode is very straightforward but prefill reshapes and has dynamic program configs
        so we implement forward as two functions for clarity.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.

        DeepseekV3MLP(
              (gate_proj): Linear(in_features=7168, out_features=2048, bias=False)
              (up_proj): Linear(in_features=7168, out_features=2048, bias=False)
              (down_proj): Linear(in_features=2048, out_features=7168, bias=False)
              (act_fn): SiLU()
            )
        """

        if cfg["mode"] == "decode":
            return self._forward_decode(x, cfg, mesh_device)
        else:
            assert cfg["mode"] == "prefill"
            return self._forward_prefill(x, cfg, mesh_device)

    def _forward_decode(self, x, cfg, mesh_device):
        print("Forward Decode")

        """
         return self.w2(F.silu(self.w1(x)) * self.w3(x))
         return self.w2(F.silu(w1_out) * self.w3(x))

        w1_out = w1(x)
        w1_silu = silu(w1_out)
        w3_out = w3(x)
        output = w2(w1_silu * w3_out)

        """
        print(f"x = {x.shape}")
        gate_proj_out = ttnn.linear(x, **cfg["gate_proj"])
        up_proj_out = ttnn.linear(x, **cfg["up_proj"])
        ttnn.deallocate(x)

        print(f"gate_proj_out = {gate_proj_out.shape}")
        print(f"up_proj_out = {up_proj_out.shape}")

        # Apply activation and multiply
        activated = ttnn.mul(gate_proj_out, up_proj_out, **cfg["mul"])
        ttnn.deallocate(gate_proj_out)
        ttnn.deallocate(up_proj_out)

        print(f"activated = {activated.shape}")

        # Down projection
        output = ttnn.linear(activated, **cfg["down_proj"])
        ttnn.deallocate(activated)

        print(f"output = {output.shape}")

        return output

    def _forward_prefill(self, x, cfg, mesh_device):
        print("Forward Prefill not implemented yet")
        return x
