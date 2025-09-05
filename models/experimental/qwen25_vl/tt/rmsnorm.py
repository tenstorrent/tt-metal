"""
This is a modified RMSNorm implementation for Qwen-VL-7B.

It's based on the existing RMSNorm in models/common/rmsnorm.py,
with slight changes to support the bf8 data type.
"""


import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


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
        eps: float = 1e-06,
        sharded_program_config=None,
        sharded_output_config=None,
        output_mem_config=None,
        ccl_topology=ttnn.Topology.Ring,
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

        cache_name = None if weight_cache_path is None else weight_cache_path / weight_name

        # Compatibility with models that don't use mesh devices (e.g. single-chip Mistral-7b)
        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            # cache_file_name=cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        )

        if self.is_distributed:
            self.weight_distributed = ttnn.as_tensor(
                torch_weight,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=weight_memory_config,
                # cache_file_name=cache_name,
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

    def forward(self, x: ttnn.Tensor, mode="decode", in_sharded=False, out_sharded=False) -> ttnn.Tensor:
        # If input is sharded do sharded RMSNorm and optionally return sharded output
        program_config = self.sharded_program_config if in_sharded else None
        memory_config = self.sharded_output_config if out_sharded else None
        distributed = self.is_distributed and self.is_distributed(mode)
        norm = self._distributed_rmsnorm
        weight = self.weight_distributed if distributed else self.weight

        if in_sharded:
            assert not distributed, "Distributed RMSNorm does not support sharded inputs"
        else:
            assert not out_sharded, "Non-sharded version of RMSNorm cannot output a sharded tensor"

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

    def _distributed_rmsnorm(
        self, inp, epsilon=None, weight=None, program_config=None, memory_config=None, compute_kernel_config=None
    ):
        inp = ttnn.sharded_to_interleaved(inp, ttnn.DRAM_MEMORY_CONFIG)

        xnorm = ttnn.pow(inp, 2)

        xnorm = ttnn.mean(xnorm, dim=-1, keepdim=True)

        xnorm = ttnn.rsqrt(xnorm + epsilon)

        xnorm = ttnn.multiply(inp, xnorm)

        weight = ttnn.reshape(weight, [1, 1, -1])

        output = ttnn.multiply(xnorm, (weight), use_legacy=False)

        if memory_config is not None:
            output = ttnn.to_memory_config(output, memory_config)

        ttnn.deallocate(xnorm)
        ttnn.deallocate(weight)

        return output
