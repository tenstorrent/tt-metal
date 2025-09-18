# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import cast

import torch
import ttnn.experimental
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
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
    MAX_BATCH_SIZE,
    even_int_div,
    find_largest_divisor,
    save_and_get_path,
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
        ), f"Embedding1D expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = cast(tuple[dict[str, torch.Tensor]], state_dicts)

        # Get the embedding weight from the state dict (in the full model: model.embed_tokens.weight)
        torch_weight = state_dict["weight"]

        # Split the last dim in 2 so that it can be sharded across the mesh
        assert (
            torch_weight.shape[-1] == hf_config.hidden_size
        ), "Embedding size does not match the hf_config hidden size"
        assert (
            torch_weight.shape[-1] % mesh_device.get_num_devices() == 0
        ), "Embedding weight last dimension must be divisible by the number of devices"
        torch_weight = torch_weight.reshape((*torch_weight.shape[:-1], mesh_device.shape[1], -1))

        # Convert to TTNN tensor with 1D sharding across final dimension
        ttnn_weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, (2, 1)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn_weight = ttnn.reshape(
            ttnn_weight, (hf_config.vocab_size, even_int_div(hf_config.hidden_size, mesh_device.get_num_devices()))
        )  # Remove the extra dimension added for sharding

        # Save to disk with standard naming - "embedding" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {"embedding": {"weight": save_and_get_path(output_path / "embedding.weight", ttnn_weight)}}

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelPrefillConfig:
        """Prefill model config for an embedding with 1D tensor parallelism.
        Same as decode. Does not specify a mode because we override forward to handle both.

        Returns:
            Dict containing operator configurations for prefill mode
        """

        return cls._embedding_config(hf_config, mesh_device, ttnn.DRAM_MEMORY_CONFIG)

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
                MAX_BATCH_SIZE,
                num_width_shard_tiles * ttnn.TILE_SIZE,
            ),
            core_grid=ttnn.num_cores_to_corerangeset(
                num_sharding_cores, ttnn.CoreCoord(mesh_device.core_grid.x, mesh_device.core_grid.y)
            ),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        return cls._embedding_config(hf_config, mesh_device, memory_config)

    @classmethod
    def _embedding_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, memory_config: ttnn.MemoryConfig
    ) -> dict[str, OpConfigBase]:
        """Config for the Embedding1D module."""
        assert (
            hf_config.hidden_size % ttnn.TILE_SIZE == 0
        ), "Hidden dimension must be divisible by TILE_SIZE"  # TODO: remove this restriction once all gather async supports subtile gathering

        return {
            "embedding": EmbeddingConfig(
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),  # matched to the path in the WeightConfig
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            ),
            "typecast": TypecastConfig(dtype=ttnn.float32),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=0,
                dim=-1,
                topology=ttnn.Topology.Linear,
                # memory_config=memory_config, # TODO: uncomment once all gather async segfault is solved (Issue #26672)
            ),
        }

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D) -> dict[str, ttnn.Tensor]:
        """Create the state for the embedding module."""
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "all_gather": {
                "multi_device_global_semaphore": ccl.get_gather_sem(0),
                "num_links": ccl.get_max_links(0),
            },
        }

    @classmethod
    def forward_prefill(cls, x, cfg):
        assert len(x.shape) == 3, "Ids tensor must be 3D: [1, 1, batch]"
        assert (
            x.shape[-1] % ttnn.TILE_SIZE == 0
        ), "Batch dimension must be divisible by TILE_SIZE for decode"  # TODO: remove this restriction once all gather async supports subtile gathering

        embeddings = ttnn.embedding(x, **cfg["embedding"])
        embeddings = ttnn.unsqueeze(embeddings, 0)

        embeddings_ag = ttnn.experimental.all_gather_async(embeddings, **cfg["all_gather"])
        ttnn.deallocate(embeddings)

        return embeddings_ag

    @classmethod
    def forward_decode(cls, x, cfg):
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

        embeddings_ag = ttnn.experimental.all_gather_async(embeddings_tc, **cfg["all_gather"])
        ttnn.deallocate(embeddings_tc)

        assert len(embeddings_ag.shape) == 4
        if embeddings_ag.shape[-2] == original_seq_len:
            return embeddings_ag

        assert embeddings_ag.shape[-2] > original_seq_len

        embeddings_ag_slice = embeddings_ag[:, :, :original_seq_len, :]
        ttnn.deallocate(embeddings_ag)

        return embeddings_ag_slice
