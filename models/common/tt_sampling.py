# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import fields, replace
from typing import List

import torch

import ttnn

logger = logging.getLogger(__name__)
from models.common.lightweightmodule import LightweightModule


class TTSampling(LightweightModule):
    """
    On-device sampling module supporting top-k, top-p, and temperature-based sampling.

    This class implements high-performance on-device sampling that can work across different
    model implementations by accepting configuration parameters rather than assuming specific
    args structures.

    Multi-device sampling works by partitioning the vocabulary across devices. Each device
    computes top-k locally on its vocabulary partition, then all-gather operations combine
    the results across devices to perform global top-k selection before final sampling.

    Args:
        mesh_device: The device or MeshDevice for computations
        tt_ccl: CCL object for distributed operations (supports both line_all_gather and tt_all_gather)
        vocab_size: Vocabulary size of the model
        padded_vocab_size: Padded vocabulary size (must be divisible by num devices)
        max_batch_size: Maximum batch size supported
        max_top_k: Maximum number of top-k tokens to consider
        cluster_shape: Shape of the device cluster (rows, cols)
        sub_core_grids: Sub-core grid configuration for operations
        sub_core_grid_topk: Sub-core grid configuration specifically for top-k operations
        start_core: Starting core coordinate for sampling operations
        num_gather_links: Number of links to use for all-gather operations (optional)
        sampling_memory_config: Memory configuration for sampling tensors (optional)
        k, p, temp: Initial sampling parameters (tensors of size max_batch_size)

    Note:
        Uses persistent buffers when CCL supports line_all_gather (llama3_70b_galaxy),
        otherwise uses standard all_gather where the CCL API handles memory allocation (tt-transformers).
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        k=None,
        p=None,
        temp=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        # Multi-step reduction is supported only on single device
        self.multi_step_reduction = list(mesh_device.shape) == [1, 1]
        self.tt_ccl = tt_ccl
        self.vocab_size = args.vocab_size
        self.padded_vocab_size = getattr(args, "padded_vocab_size", None)
        self.max_batch_size = 32
        self.max_top_k = getattr(args, "max_top_k", 32)
        self.cluster_shape = args.cluster_shape
        self.sub_core_grids = getattr(args, "sub_core_grids", None)
        self.sub_core_grid_topk = getattr(args, "sub_core_grid_topk", None)
        self.start_core = getattr(args, "start_core", ttnn.CoreCoord(0, 0))

        if hasattr(args, "model_config") and "GALAXY_NUM_LINKS" in args.model_config:
            # Calculate num_gather_links based on model config
            max_num_gather_links = args.model_config["GALAXY_NUM_LINKS"]
            self.num_gather_links = (
                args.max_top_k // 32 if args.max_top_k // 32 <= max_num_gather_links else max_num_gather_links
            )
        else:
            self.num_gather_links = 1
        if hasattr(args, "model_config") and "DECODE_SAMPLING_INPUT_MEMCFG" in args.model_config:
            self.sampling_memory_config = args.model_config["DECODE_SAMPLING_INPUT_MEMCFG"]
        else:
            self.sampling_memory_config = ttnn.DRAM_MEMORY_CONFIG

        # Set defaults for sampling parameters if not provided
        # Default: k=1 (top-1), p=0 (effectively argmax), temp=1 (no temperature scaling)
        # When p=0, the sampling operation will select the token with highest probability (argmax)
        if k is None:
            k = torch.ones(self.max_batch_size)
        if p is None:
            p = torch.zeros(self.max_batch_size)
        if temp is None:
            temp = torch.ones(self.max_batch_size)

        # Create sampling parameter tensors on device
        self.k_tensor = ttnn.from_torch(
            k,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
        )
        self.p_tensor = ttnn.from_torch(
            p,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
        )
        self.temp_tensor = ttnn.from_torch(
            temp,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
        )

        # Create device offset indices for global indexing
        self._create_indices_tensors()
        self.warmup_done = False

    def _create_indices_tensors(self):
        """Create the indices tensors needed for distributed top-k operations."""
        # Create indices tensor for device offsets
        # For multi-step reduction, we use reduce over 2 steps in a single device
        num_devices_in_mesh = 2 if self.multi_step_reduction else max(self.cluster_shape[0], self.cluster_shape[1])
        indices_device_offsets = torch.ones(
            1, 1, self.max_batch_size, self.max_top_k * num_devices_in_mesh, dtype=torch.int64
        )
        per_device_vocab_size = (
            self.vocab_size // num_devices_in_mesh
            if self.cluster_shape[0] * self.cluster_shape[1] <= 8
            else self.padded_vocab_size // num_devices_in_mesh
        )
        for device_id in range(num_devices_in_mesh):
            indices_device_offsets[:, :, :, device_id * self.max_top_k : (device_id + 1) * self.max_top_k] = (
                device_id * per_device_vocab_size
            )
        self.tt_indices_device_offsets = ttnn.from_torch(
            indices_device_offsets,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Create local indices tensor for top-k operations
        indices_tensor_torch = torch.zeros(1, 1, self.max_batch_size, per_device_vocab_size, dtype=torch.int32)
        for i in range(per_device_vocab_size):
            indices_tensor_torch[:, :, :, i] = i
        self.tt_indices_tensor = ttnn.from_torch(
            indices_tensor_torch,
            dtype=ttnn.uint16,
            layout=ttnn.Layout.TILE,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _perform_all_gather(self, tensor, dim, cluster_axis, memory_config, num_links, buffer_key=None, dtype=None):
        """Flexible all-gather that works with both CCL implementations."""
        if self.cluster_shape[0] * self.cluster_shape[1] == 32:
            # Use line_all_gather with persistent buffer support
            return self.tt_ccl.line_all_gather(
                tensor,
                dim=dim,
                cluster_axis=cluster_axis,
                memory_config=memory_config,
                num_links=num_links,
                buffer_key=buffer_key,
            )
        else:
            # Use tt_all_gather
            cluster_axis = None
            num_links = 1
            tt_logits = ttnn.all_gather(
                tensor,
                dim=dim,
                num_links=num_links,
                memory_config=tensor.memory_config(),
                cluster_axis=cluster_axis,
                topology=ttnn.Topology.Linear,
            )
            return tt_logits

    def reset_params(self, k, p, temp):
        """Update sampling parameters (k, p, temperature) dynamically."""
        self.k_tensor_new = ttnn.from_torch(
            torch.tensor(k),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.p_tensor_new = ttnn.from_torch(
            torch.tensor(p),
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        self.temp_tensor_new = ttnn.from_torch(
            torch.tensor(temp),
            device=None,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        ttnn.copy_host_to_device_tensor(self.k_tensor_new, self.k_tensor)
        ttnn.copy_host_to_device_tensor(self.p_tensor_new, self.p_tensor)
        ttnn.copy_host_to_device_tensor(self.temp_tensor_new, self.temp_tensor)

    def forward(
        self,
        x: ttnn.Tensor,
        seed: int = 0,
        tt_out_tok: ttnn.Tensor = None,
    ):
        """
        Perform on-device sampling on logits tensor.
        The logits are sharded over the devices in the cluster.
        We perform local top-k on each device, then all-gather the top-k values and indices across all devices.
        We then convert the gathered values and indices to the appropriate format, add the device offsets to get the global vocabulary indices,
        and perform the actual sampling with top-k, top-p, and temperature.

        Args:
            x: Input logits tensor
            seed: Random seed for sampling
            tt_out_tok: Optional output tensor to write results to

        Returns:
            Sampled token indices tensor
        """
        # Warmup needed for this issue: https://github.com/tenstorrent/tt-metal/issues/30289
        if self.warmup_done is False:
            self.warmup_done = True
            self.forward(x, seed=42, tt_out_tok=tt_out_tok)

        # Convert to bfloat16 for top-k operations (typecast is no-op if already bfloat16)
        x_bf16 = ttnn.typecast(x, dtype=ttnn.bfloat16, sub_core_grids=self.sub_core_grids)

        if self.multi_step_reduction:
            x_bf16_list = ttnn.split(x_bf16, x_bf16.shape[-1] // 2, dim=3)
            indices_tensor_list = ttnn.split(self.tt_indices_tensor, self.tt_indices_tensor.shape[-1] // 2, dim=3)
            topk_values_list = []
            topk_indices_list = []

            for i in range(len(x_bf16_list)):
                topk_values, topk_indices = ttnn.topk(
                    x_bf16_list[i],
                    k=self.max_top_k,
                    dim=-1,
                    sub_core_grids=self.sub_core_grid_topk,
                    indices_tensor=indices_tensor_list[i],
                )
                topk_values_list.append(topk_values)
                topk_indices_list.append(topk_indices)
                x_bf16_list[i].deallocate()
                indices_tensor_list[i].deallocate()

            topk_values_gathered_bf16_interleaved = ttnn.concat(topk_values_list, dim=3)
            topk_indices_gathered = ttnn.concat(topk_indices_list, dim=3)

            for i in range(len(topk_indices_list)):
                ttnn.deallocate(topk_values_list[i])
                ttnn.deallocate(topk_indices_list[i])

        else:
            # Perform local top-k on each device
            topk_values, topk_indices = ttnn.topk(
                x_bf16,
                k=self.max_top_k,
                dim=-1,
                sub_core_grids=self.sub_core_grid_topk,
                indices_tensor=self.tt_indices_tensor,
            )

            # Gather top-k values across all devices
            topk_values_gathered = self._perform_all_gather(
                topk_values,
                dim=3,
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=self.num_gather_links,
                buffer_key="SAMPLING_VALUES",
            )

            ttnn.deallocate(topk_values)

            # Convert gathered values to appropriate format
            if self.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
                topk_values_gathered_bf16 = ttnn.to_memory_config(
                    topk_values_gathered,
                    memory_config=self.sampling_memory_config,
                    dtype=ttnn.bfloat16,
                )
                topk_values_gathered_bf16_interleaved = ttnn.to_memory_config(
                    topk_values_gathered_bf16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ttnn.deallocate(topk_values_gathered_bf16)
            else:
                topk_values_gathered_bf16_interleaved = topk_values_gathered

            # Gather top-k indices across all devices
            topk_indices_gathered = self._perform_all_gather(
                topk_indices,
                dim=3,
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                num_links=self.num_gather_links,
                buffer_key="SAMPLING_INDICES",
                dtype=ttnn.uint16,
            )
            ttnn.deallocate(topk_indices)

        # Convert indices to appropriate data types

        topk_indices_gathered_int32 = ttnn.typecast(
            topk_indices_gathered, dtype=ttnn.int32, sub_core_grids=self.sub_core_grids
        )

        if self.sampling_memory_config != ttnn.DRAM_MEMORY_CONFIG:
            topk_indices_gathered_int32_sharded = ttnn.to_memory_config(
                topk_indices_gathered_int32, self.sampling_memory_config
            )
            ttnn.deallocate(topk_indices_gathered_int32)
        else:
            topk_indices_gathered_int32_sharded = topk_indices_gathered_int32

        # Add device offsets to get global vocabulary indices
        topk_global_indices = ttnn.add(
            self.tt_indices_device_offsets, topk_indices_gathered_int32_sharded, dtype=ttnn.int32
        )

        ttnn.deallocate(topk_indices_gathered_int32_sharded)

        topk_global_indices_interleaved = ttnn.to_memory_config(topk_global_indices, ttnn.DRAM_MEMORY_CONFIG)

        # Untilize indices for sampling operation
        topk_global_indices_interleaved_untilised = ttnn.untilize(
            topk_global_indices_interleaved, use_multicore=True, sub_core_grids=self.sub_core_grids
        )
        ttnn.deallocate(topk_global_indices_interleaved)

        # Perform the actual sampling with top-k, top-p, and temperature
        tt_out_tok = ttnn.sampling(
            topk_values_gathered_bf16_interleaved,
            topk_global_indices_interleaved_untilised,
            k=self.k_tensor,
            p=self.p_tensor,
            temp=self.temp_tensor,
            seed=seed,
            sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                self.start_core, self.max_batch_size, self.sub_core_grids, row_wise=True
            )
            if self.sub_core_grids is not None
            else None,
            output_tensor=tt_out_tok,
        )

        ttnn.deallocate(topk_values_gathered_bf16_interleaved)
        ttnn.deallocate(topk_global_indices_interleaved_untilised)

        return tt_out_tok


def clamp(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    return value


def format_sampling_params(sampling_params):
    """
    Format sampling parameters to a dictionary.
    """
    if not isinstance(sampling_params.temperature, List):
        # convert all sampling_params to lists
        update_dict = {field.name: [getattr(sampling_params, field.name)] for field in fields(sampling_params)}
        sampling_params = replace(sampling_params, **update_dict)

    # Must pad sampling_params to max_batch_size
    default_params = {"temp": 0.0, "p": 1.0, "k": 1}
    target_len = self.max_batch_size
    for name, tensor in zip(
        ("temp", "p", "k"), (sampling_params.temperature, sampling_params.top_p, sampling_params.top_k)
    ):
        current_len = len(tensor)
        if current_len < target_len:
            tensor.extend([default_params[name]] * (target_len - current_len))

    # We must clamp top-p in range [0.0, 1.0)
    # Cannot rely on external SamplingParams to be clamped
    TOP_P_MIN = 0.0
    # TOP_P_MAX is 0.99 instead of 1.0 to ensure numerical stability in cumulative probability calculations
    # A value of 1.0 can cause floating point precision issues when comparing cumulative probabilities
    TOP_P_MAX = 1.0

    for i, (top_p, temp) in enumerate(zip(sampling_params.top_p, sampling_params.temperature)):
        # Clamp top-p
        clamped_top_p = clamp(top_p, TOP_P_MIN, TOP_P_MAX)
        if clamped_top_p != top_p:
            sampling_params.top_p[i] = clamped_top_p

        # Process temperature
        if temp == 0:
            sampling_params.temperature[i] = 1.0
            sampling_params.top_k[i] = 1
        else:
            sampling_params.temperature[i] = 1 / temp
    return sampling_params
