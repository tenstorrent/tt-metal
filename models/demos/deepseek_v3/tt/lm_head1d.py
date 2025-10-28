# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


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
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_LOFI, shard_and_save
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

        weight_tensor = state_dict["weight"].permute(
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
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
        }

    @classmethod
    def _model_config(
        cls,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig | ModelDecodeConfig:
        """Generate model configuration for this module."""
        # Construct the config
        return {
            "linear": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
            ),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Linear,
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

    @classmethod
    def decode_model_config(cls, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        return cls._model_config(mesh_device)

    @classmethod
    def prefill_model_config(cls, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        return cls._model_config(mesh_device)

    @classmethod
    def create_state(cls, mesh_device: ttnn.Device, ccl: CCL) -> ModelState:
        # Store CCL object for runtime semaphore initialization
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
        }

    @classmethod
    def _forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"

        output = ttnn.linear(x, **cfg["linear"])

        assert output.memory_config() == cfg["output_memory_config"]

        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls._forward(x, cfg)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls._forward(x, cfg)
