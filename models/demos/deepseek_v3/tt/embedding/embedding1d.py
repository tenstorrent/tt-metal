# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import EmbeddingConfig, FromWeightConfig, OpConfigBase, SavedWeight
from models.demos.deepseek_v3.utils.config_helpers import shard_and_save
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class Embedding1D(AbstractModule):
    """Embedding module with 1D tensor parallel compatible interface.
    Provides base implementation used by Embedding2D via inheritance."""

    @classmethod
    def _embedding_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        memory_config: ttnn.MemoryConfig,
        output_dtype: ttnn.DataType,
    ) -> dict[str, OpConfigBase]:
        # Config dict for the embedding op and IO memory configs
        return {
            "embedding": EmbeddingConfig(
                weight=FromWeightConfig(mesh_device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            ),
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }

    @classmethod
    def _forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig | RunDecodeConfig) -> ttnn.Tensor:
        # Perform embedding lookup, releasing input afterwards
        out = ttnn.embedding(x, **cfg["embedding"])
        ttnn.deallocate(x)
        return out

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, ttnn.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        # Expect single state_dict with key "weight"
        (state_dict,) = state_dicts
        assert state_dict is not None, "Embedding1D.convert_weights expects one state dict"
        weight = state_dict["weight"]  # [vocab_size, hidden_size]

        return {
            "embedding": {
                "weight": SavedWeight(
                    path=shard_and_save(
                        output_path / "embedding.weight",
                        weight,
                        shard_dims=(None, None),
                        mesh_device=mesh_device,
                        dtype=ttnn.bfloat8_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ).path
                )
            }
        }

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelPrefillConfig:
        return cls._embedding_config(
            hf_config=hf_config,
            mesh_device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_dtype=ttnn.bfloat16,
        )

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelDecodeConfig:
        return cls._embedding_config(
            hf_config=hf_config,
            mesh_device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_dtype=ttnn.bfloat16,
        )

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL) -> ModelState:
        # Keep a minimal state with mesh device and CCL handle for consistency
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
        }

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"
        out = cls._forward(x, cfg)
        assert out.memory_config() == cfg["output_memory_config"]
        return out

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        assert x.memory_config() == cfg["input_memory_config"], f"{x.memory_config()} != {cfg['input_memory_config']}"
        out = cls._forward(x, cfg)
        assert out.memory_config() == cfg["output_memory_config"]
        return out
