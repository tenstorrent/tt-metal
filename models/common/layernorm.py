# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode

TILE = 32
SHARD_HEIGHT = TILE


class LayerNorm(LightweightModule):
    """
    LayerNorm with the same call pattern as models.common.rmsnorm.RMSNorm.

    This allows plugging into DistributedNorm without special casing.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        layer_num=None,
        state_dict_prefix=None,
        weight_cache_path=None,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        weight_dtype=ttnn.bfloat16,
        is_distributed=None,
        eps: float = 1e-5,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        fp32_dest_acc_en=True,
    ):
        super().__init__()
        self.device = device
        self.eps = eps
        self.is_distributed = is_distributed

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
            bias_name = f"{state_dict_prefix}{weight_key}.bias"
        else:
            if layer_num is None:
                weight_name = f"{weight_key}.weight"
                bias_name = f"{weight_key}.bias"
            else:
                weight_name = f"layers.{layer_num}.{weight_key}.weight"
                bias_name = f"layers.{layer_num}.{weight_key}.bias"

        torch_weight = state_dict[weight_name].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])
        torch_bias = state_dict[bias_name].unsqueeze(0).view(1, 1, dim).expand([1, SHARD_HEIGHT, dim])

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        cache_name = lambda n: None if weight_cache_path is None else weight_cache_path / n

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name(weight_name),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )
        self.bias = ttnn.as_tensor(
            torch_bias,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name(bias_name),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if self.is_distributed:
            self.weight_distributed = ttnn.as_tensor(
                torch_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=cache_name(weight_name + "_distributed"),
                mesh_mapper=(
                    ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))
                    if is_mesh_device
                    else None
                ),
            )
            self.bias_distributed = ttnn.as_tensor(
                torch_bias,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=cache_name(bias_name + "_distributed"),
                mesh_mapper=(
                    ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))
                    if is_mesh_device
                    else None
                ),
            )

        self.sharded_program_config = sharded_program_config
        self.sharded_output_config = sharded_output_config
        self.output_mem_config = output_mem_config
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def forward(self, x, mode: Mode | str, in_sharded=False, out_sharded=False, norm_config=None):
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError:
                raise ValueError(f"Invalid mode: {mode}")
        elif not isinstance(mode, Mode):
            raise ValueError(f"Invalid mode: {mode}")

        sharded_program_config = norm_config.get("sharded_program_config") if norm_config else None
        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        output_mem_config = norm_config.get("output_mem_config") if norm_config else None

        distributed = self.is_distributed and self.is_distributed(mode)
        weight = self.weight_distributed if distributed else self.weight
        bias = self.bias_distributed if distributed else self.bias

        if in_sharded:
            assert not distributed, "Distributed LayerNorm does not support sharded inputs"
        else:
            assert not out_sharded, "Non-sharded LayerNorm cannot output a sharded tensor"

        program_config = sharded_program_config if in_sharded else None
        memory_config = sharded_output_config if out_sharded else None

        x = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=weight,
            bias=bias,
            program_config=program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(x)

        if output_mem_config is not None:
            x = ttnn.to_memory_config(x, output_mem_config)
        return x
