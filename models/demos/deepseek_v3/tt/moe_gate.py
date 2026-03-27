# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.deepseek_moe_gate.op import DeepseekMoeGateSingleCore
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    BinaryOpConfig,
    FromWeightConfig,
    LinearConfig,
    LinearFallbackConfig,
    MeshDeviceStub,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    get_dequantized_tensor,
    shard_and_save,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MoEGate(AbstractModule):
    """MoE gate module from DeepSeek-R1.
    See the `AbstractModule` docstring for usage info.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
        prefix: str = "",
    ) -> WeightConfig:
        (state_dict,) = state_dicts
        assert state_dict is not None
        gate_weight = get_dequantized_tensor(state_dict, f"{prefix}weight")
        score_correction_bias = get_dequantized_tensor(
            state_dict, f"{prefix}e_score_correction_bias", dtype=torch.float32
        )
        return {
            "gate_proj": {
                "input_tensor_b": shard_and_save(
                    output_path / f"gate_proj.input_tensor_b",
                    gate_weight.T.unsqueeze(0).unsqueeze(0),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )
            },
            "add_score_correction_bias": {
                "input_tensor_b": shard_and_save(
                    output_path / f"e_score_correction_bias.input_tensor_b",
                    score_correction_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0).reshape(1, 16, 16).transpose(1, 2),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.float32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            },
        }

    @classmethod
    def create_shared_state(
        cls,
        mesh_device: ttnn.Device,
    ) -> ModelState:
        """Create input_indices, output_indices and output_tensor for each MoE layer

        Args:
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelState containing input_indices, output_indices and output_tensor for each MoE layer
        """
        ttnn_output_tensor = ttnn.zeros(
            shape=(1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_output_indices = ttnn.zeros(
            shape=(1, 32, 32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_input_indices = ttnn.arange(
            start=0,
            end=16 * 16,
            step=1,
            dtype=ttnn.int32,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn_input_indices = ttnn.unsqueeze(ttnn_input_indices, dim=0)
        ttnn_input_indices = ttnn.reshape(ttnn_input_indices, (1, 16, 16))
        ttnn_input_indices = ttnn.transpose(ttnn_input_indices, dim1=-2, dim2=-1)
        ttnn_input_indices = ttnn.typecast(ttnn_input_indices, dtype=ttnn.uint16)
        ttnn_input_indices = ttnn.to_layout(ttnn_input_indices, ttnn.ROW_MAJOR_LAYOUT)
        return {
            "gate_routing": {
                "ttnn_output_tensor": ttnn_output_tensor,
                "ttnn_input_indices": ttnn_input_indices,
                "ttnn_output_indices": ttnn_output_indices,
            },
            "mesh_device": mesh_device,
        }

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        mode: str,
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            mode: "decode" or "prefill"
        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG

            return {
                "gate_proj": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
                ),
                "add_score_correction_bias": BinaryOpConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    dtype=ttnn.bfloat16,
                ),
                "linear_fallback": False,
                "linear_fallback_config": LinearFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=ttnn.bfloat16,
                ),
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
                "input_output_shard_shape": (32, 32),
                "token_shape": (16, 16),
                "routed_scaling_factor": hf_config.routed_scaling_factor,
                "eps": 1e-20,
                "enable_sigmoid": True,
                "mode": mode,
            }
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG

            return {
                "gate_proj": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
                ),
                "add_score_correction_bias": BinaryOpConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    dtype=ttnn.bfloat16,
                ),
                "linear_fallback": False,
                "linear_fallback_config": LinearFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=ttnn.bfloat16,
                ),
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
                "input_output_shard_shape": (32, 32),
                "token_shape": (16, 16),
                "routed_scaling_factor": hf_config.routed_scaling_factor,
                "eps": 1e-20,
                "enable_sigmoid": True,
                "mode": mode,
            }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        assert x.memory_config() == cfg["input_memory_config"]

        # Gate projections
        if cfg["linear_fallback"]:
            logits = cls.linear_fallback_op(x, **cfg["linear_fallback_config"], **cfg["gate_proj"])
        else:
            logits = ttnn.linear(x, **cfg["gate_proj"])

        mesh_device = cfg["mesh_device"]
        num_experts = 256
        assert num_experts == 256, "num_experts should be 256"
        total_batch_size = logits.shape[2]

        # create the shard spec and memory config for the input, logits and output
        grid = mesh_device.compute_with_storage_grid_size()
        num_device_cores = grid.x * grid.y
        start_index = 0
        eps = cfg["eps"]
        scaling_factor = cfg["routed_scaling_factor"]
        enable_sigmoid = cfg["enable_sigmoid"]
        if cfg["mode"] == "decode":
            assert (
                total_batch_size <= num_device_cores
            ), "total_batch_size should be less than or equal to num_device_cores for decode mode"
        # in order to save time, we need to reuse bias, input_tensor, output indices and output tensor
        # we pad the logits to make the shape of above three are always the same
        num_iters = (total_batch_size + num_device_cores - 1) // num_device_cores
        padding_shape = (num_iters - (total_batch_size % num_iters)) % num_iters
        batch_size_per_iter = (total_batch_size + padding_shape) // num_iters
        if padding_shape != 0:
            logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, padding_shape), (0, 0)], 0)
        core_grid = ttnn.num_cores_to_corerangeset(
            batch_size_per_iter,
            ttnn.CoreCoord(grid.x, grid.y),
            row_wise=True,
        )
        input_output_shard_shape = cfg["input_output_shard_shape"]
        reshaped_input_shape = (batch_size_per_iter, *cfg["token_shape"])
        # currently we cannot convert the tile size of logits and input indices to 16*16 which is required by the original op,
        # but the memory layout is the same since the length is 256
        input_output_shard_spec = ttnn.ShardSpec(
            core_grid,
            input_output_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_output_shard_spec
        )
        # create the bias tensor
        scores_correction_bias = cfg["add_score_correction_bias"]["input_tensor_b"]
        scores_correction_bias = ttnn.repeat(scores_correction_bias, ttnn.Shape((batch_size_per_iter, 1, 1)))
        scores_correction_bias = ttnn.to_layout(
            scores_correction_bias, ttnn.TILE_LAYOUT, memory_config=input_output_mem_config
        )

        # create the output tensor, input indices and output indices
        ttnn_output_tensor = cfg["gate_routing"]["ttnn_output_tensor"]
        ttnn_output_tensor = ttnn.repeat(ttnn_output_tensor, (batch_size_per_iter, 1, 1))
        ttnn_output_tensor = ttnn.to_memory_config(ttnn_output_tensor, memory_config=input_output_mem_config)

        ttnn_input_indices = cfg["gate_routing"]["ttnn_input_indices"]
        ttnn_input_indices = ttnn.repeat(ttnn_input_indices, (batch_size_per_iter, 1, 1))
        ttnn_input_indices = ttnn.to_layout(ttnn_input_indices, ttnn.TILE_LAYOUT, memory_config=input_output_mem_config)

        ttnn_output_indices = cfg["gate_routing"]["ttnn_output_indices"]
        ttnn_output_indices = ttnn.repeat(ttnn_output_indices, (batch_size_per_iter, 1, 1))
        ttnn_output_indices = ttnn.to_memory_config(ttnn_output_indices, memory_config=input_output_mem_config)

        # we can only have one token per core at a time
        # this while loop is designed to handle the huge batch size (4096)
        topk_experts_weights_list = []
        topk_experts_indices_list = []
        for start_index in range(0, total_batch_size + padding_shape, batch_size_per_iter):
            cur_logits = logits[:, :, start_index : start_index + batch_size_per_iter, :]
            cur_logits = ttnn.reshape(cur_logits, reshaped_input_shape)  # maybe remove this
            cur_logits = ttnn.to_memory_config(cur_logits, memory_config=input_output_mem_config)

            topk_experts_weights, topk_experts_indices = DeepseekMoeGateSingleCore.op(
                # why dram
                cur_logits,
                scores_correction_bias,
                ttnn_output_tensor,
                ttnn_input_indices,
                ttnn_output_indices,
                eps,
                scaling_factor,
                enable_sigmoid,
            )
            topk_experts_indices = ttnn.typecast(
                topk_experts_indices, dtype=ttnn.int32
            )  # remove this after above op outputs to L1
            topk_experts_weights = ttnn.to_memory_config(topk_experts_weights, memory_config=ttnn.L1_MEMORY_CONFIG)
            topk_experts_indices = ttnn.to_memory_config(topk_experts_indices, memory_config=ttnn.L1_MEMORY_CONFIG)
            if cfg["mode"] == "prefill":
                topk_experts_weights_list.append(topk_experts_weights)
                topk_experts_indices_list.append(topk_experts_indices)
            ttnn.deallocate(cur_logits)

        ttnn.deallocate(logits)

        if cfg["mode"] == "prefill":
            topk_experts_weights = ttnn.concat(topk_experts_weights_list, dim=0)
            topk_experts_indices = ttnn.concat(topk_experts_indices_list, dim=0)
        # here we only take the 1x8  out of 32x32
        topk_experts_weights = topk_experts_weights[:total_batch_size, 0, :8]
        topk_experts_indices = topk_experts_indices[:total_batch_size, 0, :8]
        topk_experts_weights = ttnn.view(topk_experts_weights, (1, 1, total_batch_size, 8))
        topk_experts_indices = ttnn.view(topk_experts_indices, (1, 1, total_batch_size, 8))
        # remove below two
        topk_experts_indices = ttnn.to_layout(topk_experts_indices, ttnn.TILE_LAYOUT)
        topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.uint16)

        return topk_experts_weights, topk_experts_indices

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def linear_fallback_op(
        cls,
        input_tensor: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        mesh_device: ttnn.Device,
        dtype: ttnn.DataType,
        memory_config: ttnn.MemoryConfig,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """Linear fallback operation using torch.nn.functional.linear"""
        # convert ttnn mesh tensors to torch tensors
        logger.info(f"linear_fallback_op: input shape: {input_tensor.shape}, weight shape: {input_tensor_b.shape}")

        torch_input = ttnn.to_torch(
            input_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].unsqueeze(0)

        torch_weight = ttnn.to_torch(
            input_tensor_b,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0][0]

        torch_input_2d = torch_input.squeeze(0).squeeze(0)  # [seq_len, hidden_dim]
        torch_weight_2d = torch_weight.T  # [output_dim, hidden_dim]

        # use torch linear: input @ weight.T
        torch_output = torch.nn.functional.linear(torch_input_2d, torch_weight_2d)

        # Restore dimensions and convert back to ttnn
        torch_output = torch_output.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, output_dim]

        ttnn_output = ttnn.from_torch(
            torch_output,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
            dtype=dtype,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        return ttnn_output


"""
{
    "MatmulDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 73506.26349206349,
            "MIN": 71781.0,
            "MAX": 73988.0,
            "STD": 471.5722824890904
        }
    },
    "ReshapeViewDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 8085.6796875,
            "MIN": 7099.0,
            "MAX": 9457.0,
            "STD": 843.3255091918757
        }
    },
    "RepeatDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 4423.990625,
            "MIN": 4195.0,
            "MAX": 10900.0,
            "STD": 797.7202328439369
        }
    },
    "PermuteDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 5226.18125,
            "MIN": 5124.0,
            "MAX": 5435.0,
            "STD": 65.01182645003836
        }
    },
    "TilizeWithValPaddingDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 6114.472916666667,
            "MIN": 5371.0,
            "MAX": 6980.0,
            "STD": 620.465963432937
        }
    },
    "InterleavedToShardedDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 2974.9109375,
            "MIN": 1954.0,
            "MAX": 3895.0,
            "STD": 701.5092775543458
        }
    },
    "UntilizeWithUnpaddingDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 6708.809375,
            "MIN": 6595.0,
            "MAX": 6959.0,
            "STD": 80.32756940883785
        }
    },
    "GenericOpDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 3719.746875,
            "MIN": 3668.0,
            "MAX": 3860.0,
            "STD": 30.416569527192046
        }
    },
    "TypecastDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 4483.453125,
            "MIN": 1626.0,
            "MAX": 7611.0,
            "STD": 2824.5118637624482
        }
    },
    "ShardedToInterleavedDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 3702.403125,
            "MIN": 3548.0,
            "MAX": 3757.0,
            "STD": 30.671769164965365
        }
    },
    "SliceDeviceOperation": {
        "DEVICE KERNEL": {
            "AVG": 2748.6828125,
            "MIN": 2616.0,
            "MAX": 2851.0,
            "STD": 23.708275627626826
        }
    }
}
"""
