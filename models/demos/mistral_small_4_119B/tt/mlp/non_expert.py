# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    MulConfig,
    ReduceScatterAsyncMinimalConfig,
)
from models.demos.mistral_small_4_119B.tt_utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    dram_sharded_weight_config,
    even_int_div,
    get_state_dicts,
    shard_and_save,
)
from models.demos.mistral_small_4_119B.tt_utils.run_config import MESH_DEVICE_STATE_DICT_KEY


class NonExpert:
    """TT-native dense MLP (`Mistral4MLP`) for decoder dense layers.

    Implements gate/up/down projections on device with TP all-gather and reduce-scatter,
    following the DeepSeek `NonExpert`/`MLPDequant` execution pattern.
    """

    WEIGHT_TORCH_DTYPE = torch.bfloat16
    WEIGHT_DTYPE = ttnn.bfloat16

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> dict[str, Any]:
        return {
            models_name: {
                "input_tensor_b": cls.convert_metaweight(
                    output_path / f"{models_name}.input_tensor_b",
                    get_state_dicts(
                        state_dicts,
                        f"{hf_name}.weight",
                        shape=(out_features, in_features),
                        dtype=cls.WEIGHT_TORCH_DTYPE,
                    ),
                    mesh_device,
                    is_w2,
                ),
            }
            for hf_name, models_name, is_w2 in [
                ("gate_proj", "w1", False),
                ("down_proj", "w2", True),
                ("up_proj", "w3", False),
            ]
            for in_features, out_features in [cls.get_weight_shape(hf_config, is_w2)]
        }

    @classmethod
    def convert_metaweight(
        cls,
        path: Path,
        torch_metaweight_tensor: torch.Tensor,
        mesh_device: ttnn.MeshDevice,
        is_w2: bool,
    ) -> ttnn.Tensor:
        torch_metaweight_tensor = torch_metaweight_tensor.transpose(2, 1).contiguous()
        num_shards, per_device_in_features, per_device_out_features = torch_metaweight_tensor.shape
        mp, tp = mesh_device.shape
        assert num_shards == mp, "Number of mesh rows does not match weight shards"

        if is_w2:
            per_device_in_features = even_int_div(per_device_in_features, tp)
            mesh_sharded_dim = 1
        else:
            per_device_out_features = even_int_div(per_device_out_features, tp)
            mesh_sharded_dim = 2

        return shard_and_save(
            path,
            torch_metaweight_tensor,
            shard_dims=(0, mesh_sharded_dim),
            mesh_device=mesh_device,
            remove_dims=(True, False),
            dtype=cls.WEIGHT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram_sharded_weight_config(
                per_device_in_features,
                per_device_out_features,
                mesh_device.dram_grid_size(),
            ),
        )

    @classmethod
    def get_weight_shape(cls, hf_config: PretrainedConfig, is_w2: bool) -> tuple[int, int]:
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        if is_w2:
            return hidden_dim, dim
        return dim, hidden_dim

    @classmethod
    def _model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> dict[str, Any]:
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        return {
            "hidden_size": dim,
            "tp_size": int(mesh_device.shape[1]),
            "all_gather": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=1,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "w1": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            "w2": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            "w3": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            "mul": MulConfig(
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            ),
            "reduce_scatter_async": ReduceScatterAsyncMinimalConfig(
                cluster_axis=1,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "input_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "linear_dims": {"dim": dim, "hidden_dim": hidden_dim},
        }

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
    ) -> dict[str, Any]:
        del fabric_config
        return cls._model_config(hf_config, mesh_device)

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
    ) -> dict[str, Any]:
        del batch_size_per_row, fabric_config
        return cls._model_config(hf_config, mesh_device)

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> dict[str, Any]:
        del hf_config
        return {MESH_DEVICE_STATE_DICT_KEY: mesh_device, "ccl": ccl}

    @classmethod
    def _forward_impl(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> ttnn.Tensor:
        ccl = cfg["ccl"]
        dim = int(cfg["hidden_size"])
        tp = int(cfg["tp_size"])
        expected_sharded_width = even_int_div(dim, tp)

        if int(x.shape[-1]) == expected_sharded_width:
            x = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["all_gather"]))

        w1_out = ttnn.linear(x, **cfg["w1"])
        w3_out = ttnn.linear(x, **cfg["w3"])
        activated = ttnn.mul(w1_out, w3_out, **cfg["mul"])
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)

        w2_out = ttnn.linear(activated, **cfg["w2"])
        ttnn.deallocate(activated)

        if tp > 1 and int(w2_out.shape[-1]) == dim:
            output = ttnn.experimental.reduce_scatter_minimal_async(
                w2_out, **ccl.populate_reduce_scatter_runtime_args(cfg["reduce_scatter_async"])
            )
            ttnn.deallocate(w2_out)
            return output

        return w2_out

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> ttnn.Tensor:
        return cls._forward_impl(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: dict[str, Any]) -> ttnn.Tensor:
        return cls._forward_impl(x, cfg)
