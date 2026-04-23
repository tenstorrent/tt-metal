# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, MeshDeviceStub
from models.demos.deepseek_v3.utils.config_helpers import get_dequantized_tensor, shard_and_save
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)

SEND_CORES = (0, 3, 6, 9)
RECV_CORES = (1, 2, 4, 5, 7, 8, 10, 11)


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
        gate_weight = get_dequantized_tensor(state_dict, f"{prefix}weight").transpose(0, 1).unsqueeze(0)
        score_correction_bias = get_dequantized_tensor(
            state_dict, f"{prefix}e_score_correction_bias", dtype=torch.bfloat16
        ).unsqueeze(0)
        with torch.no_grad():
            score_correction_bias -= torch.mean(score_correction_bias)
        # maybe divide weight's mean here

        def prepare_w_tensor(torch_w, torch_bias, L, K, N, ring2cores):
            """
            Prepare the w tensor and bias tensor by padding and reordering tiles.

            Args:
                torch_w: Weight tensor of shape (L, K, N)
                torch_bias: Bias tensor of shape (L, N)
                L: Number of layers
                K: Input dimension
                N: Output dimension
                ring2cores: Dictionary mapping ring position to (core_coord, dram_bank_id, send_flag)

            Returns:
                torch_w: Tensor of shape (L, K, N)
            """
            Kt, Nt = math.ceil(K / ttnn.TILE_SIZE), math.ceil(N / ttnn.TILE_SIZE)
            # 8 cores get 2/3rd of K tiles and 1 N tile -> Type 1 (send flag is 0)
            # 4 cores get 1/3rd of K tiles and 2 N tiles -> Type 2 (send flag is 1)
            # Every third core is of type 2.
            w_tile_view = torch_w.view(L, Kt, ttnn.TILE_SIZE, Nt, ttnn.TILE_SIZE)

            # For the 8 cores, we append values from the bias tensor at the end, so it can be read in the
            # same DRAM transaction, optimally without any additional overhead.
            bias_tile_view = torch_bias.view(L, Nt, ttnn.TILE_SIZE)

            each_shard = []

            current_N_tile = 0
            for ring_pos in range(len(ring2cores)):
                _, _, send_flag = ring2cores[ring_pos]

                if send_flag:
                    # Type 2: Last 72 K tiles for 2 N tiles
                    first_chunk = w_tile_view[:, -72:, :, current_N_tile, :]
                    second_chunk = w_tile_view[:, -72:, :, current_N_tile + 1, :]

                    # Interleave the two chunks, one tile each on width dimension.
                    interleaved = torch.stack([first_chunk, second_chunk], dim=3)

                    # Reshape to interleave: (L, E, K, Nt * 2 * ttnn.TILE_SIZE) = (L, E, K, 4096)
                    # The order will be: w0_chunk_0, w1_chunk_0, w0_chunk_1, w1_chunk_1, ...
                    interleaved_chunks = interleaved.view(L, 72, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE)

                    # Since we want the shard height to be same on all 12 cores, we add some padding here.
                    padding = torch.zeros(L, 5, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE, dtype=torch_w.dtype)
                    torch_w_with_padding = torch.cat([interleaved_chunks, padding], dim=1)
                    each_shard.append(torch_w_with_padding)
                else:
                    # Type 1: First 2 * 76 K tiles for 1 N tile
                    all_tiles = w_tile_view[:, : 2 * 76, :, current_N_tile, :]

                    # Separate the even and odd tiles.
                    even_tiles = all_tiles[:, ::2, :, :]
                    odd_tiles = all_tiles[:, 1::2, :, :]
                    interleaved = torch.cat([even_tiles, odd_tiles], dim=-1)

                    # Put one each of even and odd tiles in width dimension.
                    all_tiles = interleaved.reshape(L, 76, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE)

                    # Create the bias tile with zero padding.
                    bias_tile = torch.zeros((L, 1, ttnn.TILE_SIZE, 2 * ttnn.TILE_SIZE), dtype=torch_bias.dtype)

                    # Add data from bias tensor to the bias tile.
                    bias_tile[:, 0, 0, : ttnn.TILE_SIZE] = bias_tile_view[:, current_N_tile, :]

                    # Append the bias tensor to the end of the all tiles.
                    w_bias_tile = torch.cat([all_tiles, bias_tile], dim=1)
                    each_shard.append(w_bias_tile)
                    current_N_tile += 1

            torch_w_all_banks = torch.stack(each_shard, dim=0)
            return torch_w_all_banks

        in0_core_coords = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)

        core2dram = {}
        for dram_bank_id, core_coords in enumerate(in0_core_coords):
            core2dram[core_coords] = dram_bank_id

        in0_num_cores = len(in0_core_coords)

        # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
        in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

        ring2cores = {}
        for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
            # key: ring_pos, value: (core_coord, dram_bank_id, send_flag)
            ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in SEND_CORES else 0)

        L = gate_weight.shape[0]
        K = gate_weight.shape[1]
        N = gate_weight.shape[2]
        torch_w_reorders = prepare_w_tensor(gate_weight, score_correction_bias, L, K, N, ring2cores)

        in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords_sorted]
        in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

        # --------------------------------------------------------------------------
        # Constants
        # --------------------------------------------------------------------------
        in0_dtype = ttnn.bfloat16
        w_dtype = ttnn.bfloat16
        num_dram_banks = len(in0_core_coords)

        dram_core_coords = [ttnn.CoreCoord(core2dram[in0_core_coord], 0) for in0_core_coord in in0_core_coords_sorted]
        dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
        dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

        # ------------------------------------------------------------------------
        # Create DRAM shard spec for w
        # Tensor shape: (L, K, N) -> Sharded across N cores
        # ------------------------------------------------------------------------
        w_shard_height = L * (76 + 1) * ttnn.TILE_SIZE
        w_shard_width = 2 * ttnn.TILE_SIZE

        w_shard_spec = ttnn.ShardSpec(
            dram_core_range_set, (w_shard_height, w_shard_width), ttnn.ShardOrientation.ROW_MAJOR
        )

        w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

        return {
            "weights": {
                "input_tensor_b": shard_and_save(
                    output_path / f"weights.input_tensor_b",
                    torch_w_reorders,
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=w_mem_config,
                    layout=ttnn.TILE_LAYOUT,
                )
            },
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelState:
        """Create input_indices, output_indices and output_tensor for each MoE layer

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelState containing input_indices, output_indices and output_tensor for each MoE layer
        """
        in0_core_coords = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)
        in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords_sorted]
        in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)
        output_shard_spec = ttnn.ShardSpec(in0_core_range_set, (32, ttnn.TILE_SIZE), ttnn.ShardOrientation.ROW_MAJOR)
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )
        output_tensor = ttnn.empty(
            shape=(32, 12 * ttnn.TILE_SIZE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=output_mem_config,
        )

        return {
            "gate_routing": {
                "ttnn_output_tensor": output_tensor,
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
                "weights": {
                    "input_tensor_b": FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                },
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
                "input_output_shard_shape": (32, 32),
                "token_shape": (16, 16),
                "routed_scaling_factor": hf_config.routed_scaling_factor,
                "enable_sigmoid": True,
                "mode": mode,
            }
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG

            return {
                "weights": {
                    "input_tensor_b": FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                },
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
                "input_output_shard_shape": (32, 32),
                "token_shape": (16, 16),
                "routed_scaling_factor": hf_config.routed_scaling_factor,
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
        mode = cfg["mode"]
        if mode == "prefill":
            output_list = []
        else:
            assert x.shape[2] <= 32, "Decode mode only supports 32 tokens"
        assert "Core should be the same"
        weight_tensor = cfg["weights"]["input_tensor_b"]
        output_tensor = cfg["gate_routing"]["ttnn_output_tensor"]

        in0_core_coords = cfg["mesh_device"].get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)
        in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords_sorted]
        in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)
        batch_size_per_iter = 32
        in0_shard_spec = ttnn.ShardSpec(
            grid=in0_core_range_set,
            shard_shape=(batch_size_per_iter, x.shape[3]),
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        input_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
        )
        x = ttnn.view(x, (1, x.shape[2], x.shape[3]))
        padding_shape = (batch_size_per_iter - x.shape[1] % batch_size_per_iter) % batch_size_per_iter
        if padding_shape > 0:
            x = ttnn.pad(x, padding=((0, 0), (0, padding_shape), (0, 0)), value=0.0)

        if mode == "prefill":
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        for start_index in range(0, x.shape[1], batch_size_per_iter):
            x_chunk = x[:, start_index : start_index + batch_size_per_iter, :]
            if mode == "prefill":
                x_chunk = ttnn.to_memory_config(x_chunk, ttnn.L1_MEMORY_CONFIG)
            x_chunk = ttnn.repeat(x_chunk, (12, 1, 1))
            x_chunk = ttnn.to_memory_config(x_chunk, input_sharded_mem_config)

            ttnn.experimental.deepseek.moe.moe_gate_mm(
                x_chunk,
                w_tensor=weight_tensor,
                output_tensor=output_tensor,
                layer_id=1,
                column_id=1,
            )
            ttnn.deallocate(x_chunk)

            if mode == "prefill":
                output_list.append(ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG))
        if mode == "decode":
            output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        else:
            output_tensor = ttnn.concat(output_list, 0)
            output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.view(output_tensor, (-1, batch_size_per_iter, output_tensor.shape[-1]))
        assert SEND_CORES == (0, 3, 6, 9), "SEND_CORES should be (0, 3, 6, 9)"
        weight_tensor = ttnn.reshape(output_tensor, (output_tensor.shape[0], batch_size_per_iter, 4, -1))
        valid_indices = weight_tensor.shape[-1] // 3
        weight_tensor = weight_tensor[:, :, :, valid_indices:]
        f1_scores = ttnn.reshape(weight_tensor, (weight_tensor.shape[0], weight_tensor.shape[1], -1, ttnn.TILE_SIZE))[
            :, 3, :, :16
        ]
        f2_scores = ttnn.reshape(weight_tensor, (weight_tensor.shape[0], weight_tensor.shape[1], -1, ttnn.TILE_SIZE))[
            :, 4, :, :16
        ]
        topk_experts_weights = ttnn.concat(
            [f1_scores, f2_scores],
            dim=-1,
            memory_config=ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
        )
        topk_experts_weights = ttnn.transpose(topk_experts_weights, -1, -2)
        topk_experts_indices = ttnn.transpose(output_tensor[:, 8:16, -32:], -1, -2)
        topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.uint16)
        topk_experts_indices = ttnn.view(topk_experts_indices, (1, 1, -1, topk_experts_indices.shape[-1]))
        topk_experts_indices = ttnn.bitwise_right_shift(topk_experts_indices, 7)
        # or ttnn.logical_right_shift(topk_experts_indices, 7)
        topk_experts_weights = ttnn.view(topk_experts_weights, (1, 1, -1, topk_experts_weights.shape[-1]))
        if padding_shape > 0:
            output_tensor = output_tensor[:-padding_shape, :]
        if mode == "prefill":
            topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.int32)
            topk_experts_indices = ttnn.to_memory_config(topk_experts_indices, ttnn.DRAM_MEMORY_CONFIG)
            topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.uint16)
        return topk_experts_weights, topk_experts_indices

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)
