# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect

import torch
from loguru import logger

import ttnn
from models.common.sampling._utils import filter_none


class LogProbsCalculator:
    """
    Class to calculate log-probs for a given logits tensor and indices tensor.

    Args:
        mesh_device: MeshDevice to use for all-gather operations
        sub_core_grids: Sub-core grid configuration for operations (optional)
        tt_ccl: CCL object for distributed operations (optional)
        batch_size: Maximum batch size for log-probs calculation (default: 32)
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        sub_core_grids: ttnn.CoreRangeSet = None,
        tt_ccl=None,
        batch_size: int = 32,
    ):
        self.global_max = None
        self.global_exp_sum = None
        self.mesh_device = mesh_device
        self.enable_log_probs = False  # default to False
        self.cluster_shape = list(mesh_device.shape)
        self.sub_core_grids = sub_core_grids
        self.tt_ccl = tt_ccl
        self.batch_size = batch_size
        self.common_args = filter_none(
            {
                "sub_core_grids": sub_core_grids,
            }
        )

        # CCL introspection (same pattern as TTSampling)
        self._line_all_gather = getattr(self.tt_ccl, "line_all_gather", None)
        self._line_all_gather_supports_buffer_key = False
        if callable(self._line_all_gather):
            try:
                sig = inspect.signature(self._line_all_gather)
                params = sig.parameters
                self._line_all_gather_supports_buffer_key = "buffer_key" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError):
                logger.warning("Unable to inspect line_all_gather signature; assuming no buffer_key support.")

        num_devices = self.mesh_device.get_num_devices()

        # Determine the TP dimension: for 2D meshes, logits are sharded across
        # the larger dimension (TP). For 1D or single-device, use all devices.
        if self.cluster_shape[0] > 1 and self.cluster_shape[1] > 1:
            # 2D mesh: TP axis is the larger dimension
            tp_axis = 0 if self.cluster_shape[0] >= self.cluster_shape[1] else 1
            num_devices_for_sharding = self.cluster_shape[tp_axis]
            self._all_gather_cluster_axis = tp_axis
        elif num_devices > 1:
            # 1D mesh
            num_devices_for_sharding = num_devices
            self._all_gather_cluster_axis = None
        else:
            # Single device
            num_devices_for_sharding = num_devices
            self._all_gather_cluster_axis = None

        self.num_devices_for_sharding = num_devices_for_sharding

        # Create mask tensor with shape (num_devices_for_sharding, batch_size)
        # Each row will have device_id starting from 0 to num_devices_for_sharding - 1
        mask_tensor = torch.arange(num_devices_for_sharding).unsqueeze(1).expand(num_devices_for_sharding, batch_size)

        # Choose mesh mapper based on mesh topology
        if self.cluster_shape[0] > 1 and self.cluster_shape[1] > 1:
            # 2D mesh: shard along the TP axis
            dims = (0, None) if self._all_gather_cluster_axis == 0 else (None, 0)
            mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=self.cluster_shape)
        elif num_devices > 1:
            # 1D mesh
            mesh_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)

        self.mask = ttnn.as_tensor(
            mask_tensor,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            preprocess=lambda x: x.to(torch.bfloat16),
            mesh_mapper=mesh_mapper,
        )
        self.output_tensor = ttnn.as_tensor(
            torch.ones(1, 1, 1, batch_size),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _perform_all_gather(self, tensor: ttnn.Tensor, dim: int, num_links: int, buffer_key: str = None):
        if callable(self._line_all_gather):
            kwargs = {
                "dim": dim,
                "num_links": num_links,
                "memory_config": tensor.memory_config(),
                "cluster_axis": self._all_gather_cluster_axis,
            }
            if self._line_all_gather_supports_buffer_key and buffer_key is not None:
                kwargs["buffer_key"] = buffer_key
            return self._line_all_gather(tensor, **kwargs)

        return ttnn.all_gather(
            tensor,
            dim=dim,
            num_links=num_links,
            memory_config=tensor.memory_config(),
            cluster_axis=self._all_gather_cluster_axis,
            topology=ttnn.Topology.Linear,
        )

    def set_log_probs_mode(self, enable_log_probs: bool | list[bool] = False):
        if isinstance(enable_log_probs, list):
            # If any of the users in the batch have log_probs enabled,
            # then whole batch will run log-probs calculation
            enable_log_probs_result = any(enable_log_probs)
        else:
            enable_log_probs_result = enable_log_probs

        self.enable_log_probs = enable_log_probs_result

    def _compute_global_stats(
        self,
        logits_tensor: ttnn.Tensor,
    ):
        """
        To calculate log-probs, we need to calculate the global max and global sum(exp(logits - global_max)) for each chip.
        This is done by all-gathering the max and sum(exp(logits - global_max)) for each chip and then taking the max and sum of the gathered tensors.
        log-prob formula: log-prob(x) = logits(x) - global_max - log(sum(exp(logits - global_max)))

        Args:
            logits_tensor (ttnn.Tensor): Logits as model output (1, 1, batch_size, vocab_size_per_device)
        """
        # Calculate local max
        local_max_tensor = ttnn.max(logits_tensor, dim=-1, keepdim=True, **self.common_args)

        gathered_max_tensors = self._perform_all_gather(
            local_max_tensor,
            dim=1,
            num_links=1,
            buffer_key="LOGPROBS_MAX_REDUCTION",
        )
        # Convert to ROW_MAJOR_LAYOUT due to memory clobbering which affects all ttnn.reshape ops with TILE_LAYOUT
        gathered_max_tensors = ttnn.to_layout(gathered_max_tensors, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        D = self.num_devices_for_sharding
        B = gathered_max_tensors.shape[2]
        gathered_max_tensors = ttnn.reshape(gathered_max_tensors, (1, 1, D, B), **self.common_args)
        gathered_max_tensors = ttnn.to_layout(gathered_max_tensors, ttnn.TILE_LAYOUT, **self.common_args)

        self.global_max = ttnn.max(gathered_max_tensors, dim=2, keepdim=True, **self.common_args)

        global_max_to_subtract = ttnn.to_layout(self.global_max, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        global_max_to_subtract = ttnn.reshape(global_max_to_subtract, (1, 1, B, 1), **self.common_args)
        global_max_to_subtract = ttnn.to_layout(global_max_to_subtract, ttnn.TILE_LAYOUT, **self.common_args)

        # Calculate stable local sum-exp using subtract of global-max from each local logit
        subtracted_tensor = ttnn.subtract(logits_tensor, global_max_to_subtract, **self.common_args)
        exp_tensor = ttnn.exp(subtracted_tensor, **self.common_args)
        sum_exp_tensor = ttnn.sum(exp_tensor, dim=-1, keepdim=True, **self.common_args)

        gathered_sum_exp_tensors = self._perform_all_gather(
            sum_exp_tensor,
            dim=1,
            num_links=1,
            buffer_key="LOGPROBS_SUM_EXP_REDUCTION",
        )
        gathered_sum_exp_tensors = ttnn.to_layout(gathered_sum_exp_tensors, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        B_sum = gathered_sum_exp_tensors.shape[2]
        gathered_sum_exp_tensors = ttnn.reshape(gathered_sum_exp_tensors, (1, 1, D, B_sum), **self.common_args)
        gathered_sum_exp_tensors = ttnn.to_layout(gathered_sum_exp_tensors, ttnn.TILE_LAYOUT, **self.common_args)

        self.global_exp_sum = ttnn.sum(gathered_sum_exp_tensors, dim=2, keepdim=True, **self.common_args)

    def _prepare_relevant_logits(self, logits_tensor: ttnn.Tensor, global_idx_tensor: ttnn.Tensor):
        """
        Prepare global idx tensor with correct values on all devices.
        """
        size_per_device = logits_tensor.shape[-1]

        # convert global_idx_tensor to ttnn.TILE_LAYOUT
        global_idx_tilized_tensor = ttnn.to_layout(global_idx_tensor, ttnn.TILE_LAYOUT, **self.common_args)

        # TODO: Raise an issue on this since for UINT_32 ttnn.div produces incorrect output (all zeros)
        global_idx_tilized_tensor = ttnn.typecast(global_idx_tilized_tensor, ttnn.float32, **self.common_args)

        # Get chip_id for each user based on global_idx values in global_idx_tensor
        chip_ids_tensor = ttnn.div(
            global_idx_tilized_tensor,
            size_per_device,
            rounding_mode="floor",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self.common_args,
        )

        # Get local index for each user based on global_idx values in global_idx_tensor
        remainder_tensor = ttnn.remainder(
            global_idx_tilized_tensor,
            size_per_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **self.common_args,
        )

        # Convert remainder_tensor to int32
        remainder_tensor = ttnn.typecast(remainder_tensor, ttnn.uint32, **self.common_args)
        # convert to ROW_MAJOR_LAYOUT due to memory clobbering which affects all ttnn.reshape ops with TILE_LAYOUT
        remainder_tensor = ttnn.to_layout(remainder_tensor, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        batch_vol = remainder_tensor.shape[2] * remainder_tensor.shape[3]
        remainder_tensor = ttnn.reshape(remainder_tensor, (1, 1, batch_vol, 1), **self.common_args)
        remainder_tensor = ttnn.to_layout(remainder_tensor, ttnn.TILE_LAYOUT, **self.common_args)

        # Get logits for each user on each chip based on local index
        selected_logits_tensor = ttnn.gather(logits_tensor, dim=3, index=remainder_tensor, **self.common_args)

        # convert to ROW_MAJOR_LAYOUT due to memory clobbering which affects all ttnn.reshape ops with TILE_LAYOUT
        selected_logits_tensor = ttnn.to_layout(selected_logits_tensor, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        batch_vol_s = selected_logits_tensor.shape[2] * selected_logits_tensor.shape[3]
        selected_logits_tensor = ttnn.reshape(selected_logits_tensor, (1, 1, 1, batch_vol_s), **self.common_args)
        selected_logits_tensor = ttnn.to_layout(selected_logits_tensor, ttnn.TILE_LAYOUT, **self.common_args)
        # Compare mask to chip_ids tensor and select correct positions for each user on all chips inplace
        ttnn.eq_(chip_ids_tensor, self.mask, **self.common_args)

        # Multiply selected_logits_tensor with chip_ids_tensor to get expected logits for each user
        selected_logits_tensor = ttnn.multiply(selected_logits_tensor, chip_ids_tensor, **self.common_args)

        # All gather logits across all devices
        selected_logits_tensor = self._perform_all_gather(
            selected_logits_tensor,
            dim=1,
            num_links=1,
            buffer_key="LOGPROBS_LOGITS",
        )

        selected_logits_tensor = ttnn.to_layout(selected_logits_tensor, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        D_s = self.num_devices_for_sharding
        B_g = selected_logits_tensor.shape[3]
        selected_logits_tensor = ttnn.reshape(selected_logits_tensor, (1, 1, D_s, B_g), **self.common_args)
        selected_logits_tensor = ttnn.to_layout(selected_logits_tensor, ttnn.TILE_LAYOUT, **self.common_args)

        # Apply sum over device dimension to get logits for each user on all chips
        selected_logits_tensor = ttnn.sum(selected_logits_tensor, dim=2, keepdim=True, **self.common_args)

        return selected_logits_tensor

    def _calculate_log_probs(self, sampled_logits_tensor: ttnn.Tensor):
        """
        Calculate log-probs for a given logits tensor with formula:
        log-prob(x) = logits(x) - global_max - log(global_exp_sum)
        """
        out = ttnn.subtract(sampled_logits_tensor, self.global_max, **self.common_args)
        log_global_exp_sum = ttnn.log(self.global_exp_sum, **self.common_args)
        # Subtract and put result to self.output_tensor
        ttnn.subtract(out, log_global_exp_sum, output_tensor=self.output_tensor, **self.common_args)

    def calculate_log_probs(
        self,
        logits_tensor: ttnn.Tensor,
        indices_tensor: ttnn.Tensor,
    ):
        """
        Calculate log-probs for a given logits tensor and indices tensor.
        """
        if not self.enable_log_probs:
            return self.output_tensor

        if self.num_devices_for_sharding < 2:
            return self.output_tensor

        # Calculating log-probs requires bfloat16 precision for near-stable sum-exp calculation
        if logits_tensor.dtype == ttnn.bfloat8_b:
            logits_tensor = ttnn.typecast(logits_tensor, ttnn.bfloat16, **self.common_args)

        # Compute global max and global sum(exp(logits - global_max)) for each chip
        self._compute_global_stats(logits_tensor)

        # Prepare relevant logits for each user on each chip
        relevant_logits = self._prepare_relevant_logits(logits_tensor, indices_tensor)

        # Calculate log-probs for each user on each chip and stores in self.output_tensor
        self._calculate_log_probs(relevant_logits)

        return self.output_tensor
