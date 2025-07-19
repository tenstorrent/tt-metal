# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
SHARD_HEIGHT = TILE  # Current ttnn.rms_norm implementation requires shard height to be a single tile


class RMSNorm(LightweightModule):
    """
    RMSNorm supporting replication over a MeshDevice and sharding within devices.

    This class implements a Root Mean Square Normalization (RMSNorm) that can be
    distributed across multiple devices and cores. If the `device` parameter is a
    MeshDevice, the weights and computations are replicated across all devices in
    the mesh. Expects an interleaved input tensor, can optionally output a sharded tensor.

    Args:
        device: The device or MeshDevice on which to perform the computations.
        state_dict: The state dictionary containing the model parameters.
        dim: Input dimension (e.g. model hidden dimension size).
        layer_num: The layer number to determine the weight key in the state dictionary.
        weight_key: The key for retrieving the weight from the state dictionary.
        weight_cache_path: Optional path for caching the tilized weights.
        weight_memory_config: Configuration for the weight memory, default is DRAM_MEMORY_CONFIG.
        weight_dtype: The data type for the tensors, bfp8_b hits >0.999 PCC in the models we tested.
        model_config: Optional configuration dictionary for the model.
        eps (float): Small value to avoid division by zero in normalization, default is 1e-05.

    If model_config is provided, it must specify SHARDED_NORM_INPUT_MEMCFG, SHARDED_NORM_PRGM_CFG
    and SHARDED_NORM_OUTPUT_MEMCFG. If not provided, default configurations will be generated.
    """

    def __init__(
        self,
        device,
        dim,
        state_dict,
        weight_key,
        tt_ccl=None,
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
    ):
        super().__init__()
        self.device = device
        self.tt_ccl = tt_ccl
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

        # Add offset before caching
        if add_unit_offset:
            torch_weight = torch_weight + 1.0

        cache_name = None if weight_cache_path is None else weight_cache_path / weight_name

        # Compatibility with models that don't use mesh devices (e.g. single-chip Mistral-7b)
        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        self.weight = ttnn.as_tensor(
            torch_weight,
            device=device,
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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

    def forward(self, x: ttnn.Tensor, mode, in_sharded=False, out_sharded=False) -> ttnn.Tensor:
        # If input is sharded do sharded RMSNorm and optionally return sharded output
        program_config = self.sharded_program_config if in_sharded else None
        memory_config = self.sharded_output_config if out_sharded else None
        distributed = self.is_distributed and self.is_distributed(mode)
        norm = self._distributed_rmsnorm if distributed else ttnn.rms_norm
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
        assert program_config is None, "Distributed RMSNorm does not support sharded inputs"
        assert memory_config is None, "Distributed RMSNorm does not support sharded outputs"
        assert self.tt_ccl is not None, "Distributed RMSNorm requires tt_ccl"

        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
        # AllGather stats
        tt_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=self.ccl_topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=self.tt_ccl.worker_sub_device_id,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            tt_stats,
            epsilon=epsilon,
            weight=weight,
            compute_kernel_config=compute_kernel_config,
        )
        tt_stats.deallocate(True)

        return tt_out
