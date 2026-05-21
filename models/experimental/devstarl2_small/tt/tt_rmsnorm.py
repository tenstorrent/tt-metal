# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# RMSNorm with optional simplified_rms path (Mistral-Small / Devstral vision).

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
SHARD_HEIGHT = TILE  # rms_norm expects shard height == one tile


class RMSNorm(LightweightModule):
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
        eps: float = 1e-05,
        add_unit_offset=False,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        ccl_topology=ttnn.Topology.Ring,
        simplified_rms=False,
    ):
        super().__init__()
        self.eps = eps
        self.is_distributed = is_distributed
        self.ccl_topology = ccl_topology

        if state_dict_prefix:
            weight_name = f"{state_dict_prefix}{weight_key}.weight"
        else:
            if layer_num is None:
                weight_name = f"{weight_key}.weight"
            else:
                weight_name = f"layers.{layer_num}.{weight_key}.weight"

        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )

        if add_unit_offset:
            torch_weight = torch_weight + 1.0

        cache_name = None if weight_cache_path is None else weight_cache_path / weight_name

        is_mesh_device = isinstance(device, ttnn._ttnn.multi_device.MeshDevice)

        self.weight = ttnn.as_tensor(  # gamma last dim must be TILE (32)
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT if weight_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT,
            memory_config=weight_memory_config,
            cache_file_name=cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if self.is_distributed:
            self.weight_distributed = ttnn.as_tensor(
                torch_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=weight_memory_config,
                cache_file_name=cache_name,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, 2), mesh_shape=list(device.shape))
                if is_mesh_device
                else None,
            )

        self.sharded_output_config = sharded_output_config
        self.sharded_program_config = sharded_program_config
        self.output_mem_config = output_mem_config

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.simplified_rms = simplified_rms

    def forward(
        self,
        x: ttnn.Tensor,
        mode,
        in_sharded=False,
        out_sharded=False,
        memory_config=None,
    ) -> ttnn.Tensor:
        program_config = self.sharded_program_config if in_sharded else None
        if memory_config is None:
            memory_config = self.sharded_output_config if out_sharded else self.output_mem_config
        if (
            memory_config is not None
            and memory_config.buffer_type == ttnn.BufferType.L1
            and x.memory_config().buffer_type != ttnn.BufferType.L1
        ):
            x = ttnn.to_memory_config(x, memory_config)
        elif (
            memory_config is not None
            and memory_config.buffer_type == ttnn.BufferType.DRAM
            and x.memory_config().buffer_type == ttnn.BufferType.L1
        ):
            x = ttnn.to_memory_config(x, memory_config)
        distributed = self.is_distributed and self.is_distributed(mode)
        norm = (
            self._simplified_rmsnorm
            if self.simplified_rms
            else self._distributed_rmsnorm
            if distributed
            else ttnn.rms_norm
        )

        weight = self.weight_distributed if distributed else self.weight

        if in_sharded and distributed:
            raise ValueError("Distributed RMSNorm does not support sharded inputs")
        if not in_sharded and out_sharded:
            raise ValueError("Non-sharded version of RMSNorm cannot output a sharded tensor")

        x = norm(
            x,
            epsilon=self.eps,
            weight=weight,
            program_config=program_config,
            memory_config=memory_config,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        if in_sharded and not out_sharded:
            return ttnn.sharded_to_interleaved(x)
        else:
            return x

    def _simplified_rmsnorm(
        self, inp, epsilon=None, weight=None, program_config=None, memory_config=None, compute_kernel_config=None
    ):
        inp = ttnn.sharded_to_interleaved(inp, ttnn.DRAM_MEMORY_CONFIG)
        xnorm = ttnn.pow(inp, 2)
        xnorm = ttnn.mean(xnorm, dim=-1, keepdim=True)
        xnorm = ttnn.rsqrt(xnorm + epsilon)
        xnorm = ttnn.multiply(inp, xnorm)
        gamma = ttnn.reshape(weight, [1, 1, -1])
        output = ttnn.multiply(xnorm, gamma)

        if memory_config is not None:
            output = ttnn.to_memory_config(output, memory_config)

        ttnn.deallocate(xnorm)
        ttnn.deallocate(gamma)

        return output

    def _distributed_rmsnorm(
        self, inp, epsilon=None, weight=None, program_config=None, memory_config=None, compute_kernel_config=None
    ):
        if program_config is not None:
            raise ValueError("Distributed RMSNorm does not support sharded inputs")
        if memory_config is not None:
            raise ValueError("Distributed RMSNorm does not support sharded outputs")

        tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=1,
            topology=self.ccl_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=weight,
            compute_kernel_config=compute_kernel_config,
        )
        tt_stats.deallocate(True)

        return tt_out
