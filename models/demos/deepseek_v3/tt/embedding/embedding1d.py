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
    EmbeddingConfig,
    FromWeightConfig,
    MeshDeviceStub,
    OpConfigBase,
    TypecastConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    USERS_PER_ROW,
    even_int_div,
    find_largest_divisor,
    shard_and_save,
)
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    WeightConfig,
)


class Embedding1D(AbstractModule):
    """Embedding module with 1D tensor parallelism from TTT code.
    Uses DRAM-sharded weights split 1D across all wormholes"""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        # Check that there is only one state dict
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"Embedding expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts

        # Get the embedding weight from the state dict (in the full model: model.embed_tokens.weight)
        torch_weight = state_dict["weight"]

        # Split the last dim in 2 so that it can be sharded across the mesh
        assert (
            torch_weight.shape[-1] == hf_config.hidden_size
        ), "Embedding size does not match the hf_config hidden size"
        assert (
            torch_weight.shape[-1] % mesh_device.get_num_devices() == 0
        ), "Embedding weight last dimension must be divisible by the number of devices"

        # Save to disk with standard naming - "embedding" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "embedding": {
                "weight": shard_and_save(
                    output_path / "embedding.weight",
                    # Convert to TTNN tensor with 1D sharding across final dimension
                    torch_weight.reshape(
                        hf_config.vocab_size,
                        mesh_device.shape[1],
                        mesh_device.shape[0],
                        even_int_div(hf_config.hidden_size, mesh_device.get_num_devices()),
                    ),
                    shard_dims=(2, 1),
                    mesh_device=mesh_device,
                    remove_dims=True,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
        }

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelPrefillConfig:
        """Prefill model config for an embedding with 1D tensor parallelism.
        Same as decode. Does not specify a mode because we override forward to handle both.

        Returns:
            Dict containing operator configurations for prefill mode
        """

        return cls._embedding_config(
            hf_config, mesh_device, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16
        )  # RMSNorm does not support float32 in prefill

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelDecodeConfig:
        """Generate decode operator configuration for this embedding layer.
        Same as prefill. Does not specify a mode because we override forward to handle both.

        Returns:
            Dict containing operator configurations for decode mode
        """
        _, num_sharding_devices = mesh_device.shape
        num_width_shard_tiles = ttnn.core.divup(
            even_int_div(hf_config.hidden_size, num_sharding_devices), ttnn.TILE_SIZE
        )
        num_sharding_cores = find_largest_divisor(num_width_shard_tiles, mesh_device.core_grid.num_cores)
        memory_config = ttnn.create_sharded_memory_config(
            shape=(
                USERS_PER_ROW,
                num_width_shard_tiles * ttnn.TILE_SIZE,
            ),
            core_grid=ttnn.num_cores_to_corerangeset(
                num_sharding_cores, ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y)
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return cls._embedding_config(hf_config, mesh_device, memory_config, ttnn.float32)

    @classmethod
    def _embedding_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        memory_config: ttnn.MemoryConfig,
        output_dtype: ttnn.DataType,
    ) -> dict[str, OpConfigBase]:
        """Config for the Embedding module."""
        assert (
            hf_config.hidden_size % ttnn.TILE_SIZE == 0
        ), "Hidden dimension must be divisible by TILE_SIZE"  # TODO: remove this restriction once all gather async supports subtile gathering

        return {
            "embedding": EmbeddingConfig(
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),  # matched to the path in the WeightConfig
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            ),
            "typecast": TypecastConfig(dtype=output_dtype),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=0,
                dim=-1,
                topology=ttnn.Topology.Linear,
                # memory_config=memory_config, # TODO: uncomment once all gather async segfault is solved (Issue #26672)
            ),
        }

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL) -> dict[str, ttnn.Tensor]:
        """Create the state for the embedding module."""
        # Store CCL object for runtime semaphore initialization
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
        }

    @classmethod
    def forward_prefill(cls, x, cfg):
        return cls._forward(x, cfg)

    @classmethod
    def forward_decode(cls, x, cfg):
        return cls._forward(x, cfg)

    @classmethod
    def _forward(cls, x, cfg):
        assert len(x.shape) == 3, "Ids tensor must be 3D: [1, 1, batch]"

        # TODO: remove this padding once all gather async supports subtile gathering
        # Add padding so that the batch dimension is divisible by TILE_SIZE
        _, _, original_seq_len = x.shape
        if original_seq_len % ttnn.TILE_SIZE == 0:
            embeddings = ttnn.embedding(x, **cfg["embedding"])
        else:
            x_padded = ttnn.pad(x, [(0, 0), (0, 0), (0, ttnn.TILE_SIZE - original_seq_len % ttnn.TILE_SIZE)], 0)
            embeddings = ttnn.embedding(x_padded, **cfg["embedding"])
            ttnn.deallocate(x_padded)

        embeddings = ttnn.unsqueeze(embeddings, 0)

        embeddings_tc = ttnn.typecast(embeddings, **cfg["typecast"])
        ttnn.deallocate(embeddings)

        # CCL runtime initialization in execution order
        ccl = cfg["ccl"]

        embeddings_ag = ttnn.experimental.all_gather_async(
            embeddings_tc, **ccl.populate_all_gather_runtime_args(cfg["all_gather"])
        )
        ttnn.deallocate(embeddings_tc)

        assert len(embeddings_ag.shape) == 4
        if embeddings_ag.shape[-2] == original_seq_len:
            return embeddings_ag

        assert embeddings_ag.shape[-2] > original_seq_len

        embeddings_ag_slice = embeddings_ag[:, :, :original_seq_len, :]
        ttnn.deallocate(embeddings_ag)

        return embeddings_ag_slice
