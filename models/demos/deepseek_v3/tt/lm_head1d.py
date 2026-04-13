# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import math
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    SEQ_LEN_CHUNK_SIZE,
    even_int_div,
    get_dequantized_tensor,
    shard_and_save,
)
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class LMHead1D(AbstractModule):
    """TT implementation of Language model head for Deepseek V3."""

    @classmethod
    def _get_model_dims_from_cfg(cls, hf_config: PretrainedConfig) -> tuple[int, int]:
        """Get the dimensions of the model from the HuggingFace config.

        Args:
            hf_config: HuggingFace model configuration object.

        Returns:
            Tuple containing the hidden dimension and vocab_size of the LMHHead.
        """

        return hf_config.hidden_size, hf_config.vocab_size

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor], ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert len(state_dicts) == 1, "Only one non-padding state dict is expected for LMHead conversion"
        (state_dict,) = state_dicts

        hidden_dim, vocab_size = cls._get_model_dims_from_cfg(hf_config)

        weight_tensor = get_dequantized_tensor(state_dict, "weight").permute(
            1, 0
        )  # In torch the weights are in (out_features, in_features) format
        assert weight_tensor.shape == (hidden_dim, vocab_size)

        return {
            "linear": {
                "input_tensor_b": shard_and_save(
                    output_path / "linear.input_tensor_b",
                    weight_tensor,
                    shard_dims=(None, -1),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
        }

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelDecodeConfig:
        """Generate model configuration for this module.

        Args:
            hf_config: HuggingFace model configuration.
            mesh_device: Mesh device whose column count (`shape[1]`) defines tensor-parallel
                vocab sharding for the LM head program config.
        """
        hidden_dim, vocab_size = cls._get_model_dims_from_cfg(hf_config)
        tile_size = 32
        mesh_cols = mesh_device.shape[1]
        if vocab_size % mesh_cols != 0:
            raise ValueError(
                f"LMHead1D.decode_model_config requires vocab_size ({vocab_size}) to be divisible by "
                f"mesh_device.shape[1] ({mesh_cols})."
            )
        if hidden_dim % tile_size != 0:
            raise ValueError(
                f"LMHead1D.decode_model_config requires hidden_dim ({hidden_dim}) to be a multiple of "
                f"tile_size ({tile_size})."
            )
        n_per_device = vocab_size // mesh_cols
        if n_per_device % tile_size != 0:
            raise ValueError(
                f"LMHead1D.decode_model_config requires per-device vocab shard size ({n_per_device}) to be "
                f"a multiple of tile_size ({tile_size}). Computed from vocab_size ({vocab_size}) and "
                f"mesh_device.shape[1] ({mesh_cols})."
            )
        K_tiles = hidden_dim // tile_size
        N_tiles = n_per_device // tile_size
        if N_tiles == 0:
            raise ValueError(
                "LMHead1D.decode_model_config requires N_tiles >= 1 (per-device vocab must span at least "
                f"one tile in N); got N_tiles=0 from n_per_device={n_per_device}, tile_size={tile_size}."
            )

        # 1D multicast: broadcast small decode activation to all cores,
        # each core computes a slice of the output columns (N dimension).
        grid_size = mesh_device.compute_with_storage_grid_size()
        num_cores = grid_size.x * grid_size.y
        per_core_N = math.ceil(N_tiles / num_cores)
        if per_core_N == 0:
            raise ValueError(
                "LMHead1D.decode_model_config requires per_core_N >= 1 for matmul subblocking; "
                f"got per_core_N=0 (N_tiles={N_tiles}, num_cores={num_cores})."
            )

        in0_block_w = 32
        while K_tiles % in0_block_w != 0:
            in0_block_w //= 2

        out_subblock_w = min(per_core_N, 4)
        while out_subblock_w > 1 and per_core_N % out_subblock_w != 0:
            out_subblock_w -= 1

        program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_size.x, grid_size.y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            per_core_M=1,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        return {
            "linear": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
                program_config=program_config,
            ),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
            "input_memory_config": ttnn.L1_MEMORY_CONFIG,
            "output_memory_config": ttnn.L1_MEMORY_CONFIG,
        }

    @classmethod
    def prefill_model_config(cls, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Generate model configuration for this module."""
        # Construct the config
        return {
            "linear": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

    @classmethod
    def create_state(cls, mesh_device: ttnn.Device, ccl: CCL) -> ModelState:
        # Store CCL object for runtime semaphore initialization
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
        }

    @staticmethod
    def _fwd_linear(x: ttnn.Tensor, cfg: dict) -> ttnn.Tensor:
        """Pure compute for the lm_head linear projection (no deallocation)."""
        return ttnn.linear(x, **cfg["linear"])

    @staticmethod
    def _fwd_prefill(x: ttnn.Tensor, cfg: dict, deallocate_inputs: bool = True) -> ttnn.Tensor:
        """Prefill compute: chunk, linear, de-chunk.

        Args:
            deallocate_inputs: When True (production), eagerly frees input
                tensors to avoid holding the original and padded/chunked
                copies simultaneously.  Set to False in perf-measurement
                loops that reuse the input across iterations.
        """
        _, _, seq_len, _ = x.shape
        original_seq_len = seq_len

        pad_rows = 0
        if seq_len > SEQ_LEN_CHUNK_SIZE:
            if seq_len % SEQ_LEN_CHUNK_SIZE != 0:
                pad_rows = SEQ_LEN_CHUNK_SIZE - (seq_len % SEQ_LEN_CHUNK_SIZE)
                x_padded = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_rows), (0, 0)), value=0.0)
                if deallocate_inputs:
                    ttnn.deallocate(x)
                x = x_padded
                seq_len += pad_rows
            x = ttnn.reshape(x, [1, even_int_div(seq_len, SEQ_LEN_CHUNK_SIZE), SEQ_LEN_CHUNK_SIZE, -1])

        output = ttnn.linear(x, **cfg["linear"])

        if deallocate_inputs:
            ttnn.deallocate(x)

        _, num_chunks, _, output_dim = output.shape
        if num_chunks > 1:
            output = ttnn.reshape(output, [1, 1, -1, output_dim])
            if pad_rows > 0:
                output = ttnn.slice(output, [0, 0, 0, 0], [1, 1, original_seq_len, output_dim])

        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        output = cls._fwd_linear(x, cfg)

        ttnn.deallocate(x)

        assert output.memory_config() == cfg["output_memory_config"]

        return output

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        output = cls._fwd_prefill(x, cfg)

        assert output.memory_config() == cfg["output_memory_config"]

        return output
