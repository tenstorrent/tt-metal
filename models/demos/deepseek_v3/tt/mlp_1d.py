# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path
from typing import Any

import torch
from attr import dataclass
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherConfig,
    AllReduceConfig,
    LinearConfig,
    MeshDeviceStub,
    ModelConfig,
    MulConfig,
    OpConfigBase,
    RunConfig,
    WeightConfig,
    WeightStub,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
    COMPUTE_KERNEL_CONFIG_LOFI,
    TILE_SIZE,
    dram_shard_core_grid_for_k_and_n,
    dram_sharded_matmul_config,
    dram_sharded_weight_config,
    find_prefill_grid,
    matmul_config,
)
from models.utility_functions import is_blackhole


class MLP1D(AbstractModel):
    """Example MLP module with 1D tensor parallelism based on TTT code.

    THIS IS NOT THE MLP WE WILL USE FOR DEEPSEEK-R1.
    (it also doesn't work at the moment).
    IT IS JUST TO SHOW HOW A REAL SUBMODULE AND TEST WOULD LOOK.

    Typical usage by a caller would be split between convertting torch weights to ttnn weights and running those weights.

    Weight conversion one-off:
    - Use MLP_1D.convert_weights to convert PyTorch weights to TTNN format and save to disk

    At run-time:
    1. Call MLP_1D.prefill_model_config and MLP_1D.decode_model_config to generate static model configs
    2. Create prefill and decode RunConfigs with the model configs and the path to the weights to load into it
    3. Call MLP_1D.forward to run the model with each RunConfig as needed

    A RunConfig is a dict with everything each ttnn op needs to run except the input tensor, e.g.
    you can run ttnn.linear(x, **cfg["w1"]) and it will expand with the weights and program configs etc.
    This keeps the forward pass clean and readable.

    Both convert_weights and the model configs are static methods and can be called without instantiating the class.
    This functional design makes it easy to re-use them in other models if we want to, without having to subclass or
    instantiate it; the class is essentially a namespace for them.

    Keep the constructor as empty as you can. A good use of it is to set up ttnn tensors that are not weights,
    e.g. kv_cache, or as in this example dynamic program configs for prefill.
    """

    @dataclass
    class MLPProgramConfigData(OpConfigBase):
        """Data class for the data for generating the PC for ttnn.linear."""

        dim: int
        hidden_dim: int
        num_devices: int

    MAX_BATCH_SIZE = TILE_SIZE
    DRAM_SHARD_GRID_WIDTH = 8
    PREFILL_ROWS = 8

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to keyword tensor arguments and their tensor stubs
        """

        assert cls.is_device_supported(mesh_device)
        return {
            models_name: {
                "input_tensor_b": WeightStub.from_weight(
                    cls.convert_weight(
                        hf_config,
                        state_dict[f"{hf_name}.weight"],
                        state_dict[f"{hf_name}.weight_scale_inv"],
                        mesh_shard_dims,
                        output_path,
                        mesh_device,
                    ),
                    output_path / f"{models_name}.input_tensor_b",
                )
            }
            for hf_name, models_name, mesh_shard_dims in [
                ("gate_proj", "w1", (0, 1)),
                ("down_proj", "w2", (1, 0)),
                ("up_proj", "w3", (0, 1)),
            ]
        }

    @classmethod
    def convert_weight(
        cls,
        hf_config: Any,
        quantized_weight_tensor: torch.Tensor,
        scale_inv_tensor: torch.Tensor,
        mesh_shard_dims: tuple[int, int],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> ttnn.Tensor:
        """
        Convert the quantized weight tensor to a format suitable for TTNN.

        Args:
            hf_config: The Hugging Face configuration object.
            quantized_weight_tensor: The quantized weight tensor.
            scale_inv_tensor: The scale inverse tensor.
            output_path: The path to save the converted weight file.
            mesh_device: The mesh device to use for the conversion.

        Returns:
            The path to the converted weight file.
        """
        torch_weight_tensor = dequantize(
            quantized_weight_tensor, scale_inv_tensor, hf_config.quantization_config.weight_block_size
        ).permute(
            1, 0
        )  # In torch the weights are in (out_features, in_features) format
        in_features, out_features = torch_weight_tensor.shape
        device_shape = tuple(mesh_device.shape)
        return ttnn.from_torch(
            torch_weight_tensor,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=ttnn.MeshDevice(mesh_device),
            memory_config=dram_sharded_weight_config(
                in_features // device_shape[mesh_shard_dims[0]],
                out_features // device_shape[mesh_shard_dims[1]],
                mesh_device.dram_grid_size,
            ),
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=mesh_shard_dims, mesh_shape=device_shape),
        )

    @staticmethod
    def is_device_supported(mesh_device: ttnn.Device) -> bool:
        """
        Check if the device is supported for MLP operations.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return tuple(mesh_device.shape) == (1, 8)

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelConfig:
        """Prefill model config for a module with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device
            kwargs: Additional arguments for specifying the model configuration

        Returns:
            Dict containing operator configurations for prefill mode
        """
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_devices = mesh_device.get_num_devices()

        config: ModelConfig = {"mode": "prefill"}

        # Maximum rows to process at once in prefill mode
        config["max_rows"] = 512 if is_blackhole() else 1024

        # Program configs are dynamically generated in __init__ based on sequence length
        # See __init__ function for implementation details
        config["w1"] = config["w2"] = config["w3"] = LinearConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2_FP16,
        )

        config["mul"] = MulConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

        config["all_reduce"] = AllReduceConfig(
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=1,
            num_all_gather_links=1,
            topology=ttnn.Topology.Ring,
            dtype=ttnn.bfloat8_b,
            use_composite=dim >= 8192,  # Use composite for larger models
            mesh_device=MeshDeviceStub(tuple(mesh_device.shape)),
        )

        config["pc_gen"] = MLP1D.MLPProgramConfigData(dim=dim, hidden_dim=hidden_dim, num_devices=num_devices)

        return config

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelConfig:
        """Generate decode operator configuration for this MLP layer.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        # Extract dimensions from HF config
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_devices = mesh_device.get_num_devices()

        config: ModelConfig = {"mode": "decode"}

        # Decode mode configurations
        config["w1"] = config["w3"] = LinearConfig(
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=dram_sharded_matmul_config(dim, hidden_dim, mesh_device),
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,  # FP16 accumulation saves L1
        )

        tile_padded_batch_rows = TILE_SIZE * int(math.ceil(MLP1D.MAX_BATCH_SIZE / TILE_SIZE))
        mlp2_core_grid = dram_shard_core_grid_for_k_and_n(hidden_dim // num_devices, dim)
        config["w2_reshard"] = AllGatherConfig(
            memory_config=ttnn.create_sharded_memory_config(
                (
                    tile_padded_batch_rows,
                    hidden_dim // num_devices // mlp2_core_grid.num_cores,
                ),
                mlp2_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            ),
            mesh_device=MeshDeviceStub(tuple(mesh_device.shape)),
        )
        config["w2"] = LinearConfig(
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=dram_sharded_matmul_config(hidden_dim // num_devices, dim, mesh_device),
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
        )

        # Activation configurations
        config["mul"] = MulConfig(
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )

        # All-reduce configuration for multi-device synchronization
        config["all_reduce"] = AllReduceConfig(
            cluster_axis=0,
            dim=3,
            num_reduce_scatter_links=1,
            num_all_gather_links=1,
            topology=ttnn.Topology.Ring if num_devices == 8 else ttnn.Topology.Linear,
            dtype=ttnn.bfloat8_b,
            # FIXME: From tt_transformers/tt/mlp.py, surely >= and not ==?
            # FIXME: Why this value and not e.g. 7*1024?
            use_composite=dim == 8192,  # Use composite for larger models
            mesh_device=MeshDeviceStub(tuple(mesh_device.shape)),
        )

        config["pc_gen"] = MLP1D.MLPProgramConfigData(dim=dim, hidden_dim=hidden_dim, num_devices=num_devices)

        return config

    @classmethod
    def _get_w1_w3_pc(cls, seq_len: int, dim: int, hidden_dim: int, num_devices: int) -> Any:
        """Get the program config for w1 and w3 linear layers based on sequence length."""
        mlp_1_3_grid = find_prefill_grid(cls.PREFILL_ROWS, dim // TILE_SIZE)
        n_w1_w3 = hidden_dim // num_devices  # weights are 1d sharded across devices
        return matmul_config(
            m=seq_len,
            k=dim // num_devices,
            n=n_w1_w3,
            grid_size=mlp_1_3_grid,
            per_core_N=math.ceil(n_w1_w3 / (cls.MAX_BATCH_SIZE * cls.DRAM_SHARD_GRID_WIDTH)),
        )

    @classmethod
    def _get_w2_pc(cls, seq_len: int, dim: int, hidden_dim: int, num_devices: int) -> Any:
        mlp2_grid = find_prefill_grid(cls.PREFILL_ROWS, hidden_dim // TILE_SIZE)
        n_w2 = dim
        return matmul_config(
            m=seq_len,
            k=hidden_dim // num_devices,
            n=n_w2,
            grid_size=mlp2_grid,
            per_core_N=math.ceil(n_w2 / (cls.MAX_BATCH_SIZE * cls.DRAM_SHARD_GRID_WIDTH)),
        )

    @classmethod
    def _forward_prefill(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Forward pass of the MLP.

        Prefill mode we reshape to respect cfg["max_rows"] and generate program configs from the seq-len lambda.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after MLP computation
        """
        seq_len = x.shape[-2]

        # Handle large sequence lengths
        if seq_len > cfg["max_rows"]:
            # Reshape input to process in chunks
            original_shape = x.shape
            num_chunks = seq_len // cfg["max_rows"]
            x = ttnn.reshape(x, [1, num_chunks, cfg["max_rows"], -1])

            # Get current sequence length for program config
            current_seq_len = x.shape[-2]
        else:
            current_seq_len = seq_len
            original_shape = None

        # Gate and up projections with dynamic program configs
        w1_out = ttnn.linear(x, program_config=cls._get_w1_w3_pc(current_seq_len, **cfg["pc_gen"]), **cfg["w1"])
        w3_out = ttnn.linear(x, program_config=cls._get_w1_w3_pc(current_seq_len, **cfg["pc_gen"]), **cfg["w3"])
        ttnn.deallocate(x)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # Down projection with dynamic program configs
        output = ttnn.linear(activated, program_config=cls._get_w2_pc(current_seq_len, **cfg["pc_gen"]), **cfg["w2"])
        ttnn.deallocate(activated)

        # All-reduce across devices to sum partial results
        output = ttnn.all_reduce(output, **cfg["all_reduce"])

        # Reshape output to expected format if we reshaped the input
        if original_shape is not None:
            output = ttnn.reshape(output, original_shape)

        return output

    @classmethod
    def _forward_decode(cls, x: ttnn.Tensor, cfg: RunConfig) -> ttnn.Tensor:
        """Straightforward forward pass for decode mode"""
        # Gate and up projections
        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])
        ttnn.deallocate(x)

        # Apply activation and multiply
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        # w2 may use a different core grid, this is a no-op if they already match
        activated = ttnn.to_memory_config(activated, **cfg["w2_reshard"])

        # Down projection
        output = ttnn.linear(activated, **cfg["w2"])
        ttnn.deallocate(activated)

        # All-reduce across devices to sum partial results
        return ttnn.all_reduce(output, **cfg["all_reduce"])
