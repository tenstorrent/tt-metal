# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device-side token embedding (1D hidden sharding + optional all-gather).

Token ids are expected as a 3D tensor ``[1, 1, seq_len]`` (row layout).

- **Single device** (``mesh_device.get_num_devices() == 1``): full embedding table is replicated on
  the mesh, ``ttnn.embedding`` runs locally, and **no** ``all_gather_async`` is used. This avoids
  initializing **fabric** (which would otherwise require healthy inter-chip Ethernet links and can
  time out on e.g. Blackhole bring-up with a 1×1 mesh).

- **Multi-device**: weights are reshaped to ``[vocab, mesh_w, mesh_h, hidden / num_devices]`` and
  sharded like the shared DeepSeek-style embedding path; ``hidden_size`` must divide
  ``mesh_device.get_num_devices()``, then **all_gather_async** stitches the hidden dimension (needs
  fabric + ``CCL``).
"""

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.mistral_small_4_119B.tt_utils.abstract_module import AbstractModule
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import (
    AllGatherAsyncConfig,
    EmbeddingConfig,
    FromWeightConfig,
    MeshDeviceStub,
    TypecastConfig,
)
from models.demos.mistral_small_4_119B.tt_utils.config_helpers import (
    even_int_div,
    get_dequantized_tensor,
    shard_and_save,
)
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    WeightConfig,
)


def _use_fabric_gather(mesh_device: ttnn.MeshDevice) -> bool:
    return mesh_device.get_num_devices() > 1


class Mistral4Embedding1D(AbstractModule):
    """``embed_tokens`` on device: lookup, then (if multi-device) all-gather on the hidden axis."""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"Embedding expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts

        torch_weight = get_dequantized_tensor(state_dict, "weight")

        assert torch_weight.shape[-1] == hf_config.hidden_size, "Embedding size does not match hf_config.hidden_size"

        if not _use_fabric_gather(mesh_device):
            return {
                "embedding": {
                    "weight": ttnn.from_torch(
                        torch_weight,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    )
                }
            }

        assert (
            torch_weight.shape[-1] % mesh_device.get_num_devices() == 0
        ), "Embedding weight last dimension must be divisible by the number of devices"

        return {
            "embedding": {
                "weight": shard_and_save(
                    output_path / "embedding.weight",
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
        assert hf_config.hidden_size % ttnn.TILE_SIZE == 0, "Hidden dimension must be divisible by TILE_SIZE"

        cfg: ModelPrefillConfig = {
            "embedding": EmbeddingConfig(
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            ),
        }
        if _use_fabric_gather(mesh_device):
            cfg["typecast"] = TypecastConfig(dtype=ttnn.bfloat16)
            cfg["all_gather"] = AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=0,
                dim=-1,
            )
        return cfg

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelDecodeConfig:
        assert hf_config.hidden_size % ttnn.TILE_SIZE == 0, "Hidden dimension must be divisible by TILE_SIZE"

        if not _use_fabric_gather(mesh_device):
            return {
                "embedding": EmbeddingConfig(
                    weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                ),
            }

        return {
            "embedding": EmbeddingConfig(
                weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            ),
            "typecast": TypecastConfig(dtype=ttnn.float32),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=0,
                dim=-1,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            ),
        }

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, ccl: CCL | None = None) -> dict:
        del hf_config
        gather = _use_fabric_gather(mesh_device)
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl if gather else None,
            "use_fabric_gather": gather,
        }

    @staticmethod
    def _fwd_embedding(x: ttnn.Tensor, cfg: dict, original_seq_len: int) -> ttnn.Tensor:
        if original_seq_len % ttnn.TILE_SIZE == 0:
            return ttnn.embedding(x, **cfg["embedding"])
        x_padded = ttnn.pad(x, [(0, 0), (0, 0), (0, ttnn.TILE_SIZE - original_seq_len % ttnn.TILE_SIZE)], 0)
        embeddings = ttnn.embedding(x_padded, **cfg["embedding"])
        ttnn.deallocate(x_padded)
        return embeddings

    @staticmethod
    def _fwd_all_gather_embedding(x: ttnn.Tensor, cfg: dict, ccl: CCL) -> ttnn.Tensor:
        return ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["all_gather"]))

    @classmethod
    def forward_prefill(cls, x, cfg):
        return cls._forward(x, cfg)

    @classmethod
    def forward_decode(cls, x, cfg):
        return cls._forward(x, cfg)

    @classmethod
    def _forward(cls, x, cfg):
        assert len(x.shape) == 3, "Ids tensor must be 3D: [1, 1, seq_len]"

        _, _, original_seq_len = x.shape
        embeddings = cls._fwd_embedding(x, cfg, original_seq_len)

        embeddings = ttnn.unsqueeze(embeddings, 0)

        if not cfg.get("use_fabric_gather", True):
            assert len(embeddings.shape) == 4
            if embeddings.shape[-2] == original_seq_len:
                return embeddings
            assert embeddings.shape[-2] > original_seq_len
            out = embeddings[:, :, :original_seq_len, :]
            ttnn.deallocate(embeddings)
            return out

        ccl = cfg["ccl"]
        assert ccl is not None, "Multi-device embedding requires a CCL instance in run config state"

        embeddings_tc = ttnn.typecast(embeddings, **cfg["typecast"])
        ttnn.deallocate(embeddings)

        embeddings_ag = cls._fwd_all_gather_embedding(embeddings_tc, cfg, ccl)
        ttnn.deallocate(embeddings_tc)

        assert len(embeddings_ag.shape) == 4
        if embeddings_ag.shape[-2] == original_seq_len:
            return embeddings_ag

        assert embeddings_ag.shape[-2] > original_seq_len

        embeddings_ag_slice = embeddings_ag[:, :, :original_seq_len, :]
        ttnn.deallocate(embeddings_ag)

        return embeddings_ag_slice
