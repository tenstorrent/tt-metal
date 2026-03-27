# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.sampling._utils import filter_none

# Maximum number of top logprobs that can be requested (OpenAI API limit)
MAX_TOP_LOGPROBS = 20
# Number of top logprobs computed on device (gathered top-k from all devices)
DEVICE_TOP_K = 32


@dataclass
class LogProbsResult:
    """Result of log-probs calculation for a batch.

    Contains logprobs and global indices for the gathered top-k tokens across all
    devices.  The sampled token is always part of the gathered top-k (it was selected
    from them by ttnn.sampling), so its logprob can be looked up by matching its
    index in ``topk_indices``.

    Attributes:
        topk_logprobs: Optional ttnn.Tensor of shape (1, 1, batch_size, DEVICE_TOP_K)
            containing logprobs for the gathered top-k tokens.
        topk_indices: Optional ttnn.Tensor of shape (1, 1, batch_size, DEVICE_TOP_K)
            containing global vocabulary indices for the gathered top-k tokens.
        topk_logprobs_host: Optional ttnn.Tensor on host of shape (1, 1, batch_size, DEVICE_TOP_K)
            containing logprobs for the gathered top-k tokens after moving to host.
        topk_indices_host: Optional ttnn.Tensor on host of shape (1, 1, batch_size, DEVICE_TOP_K)
            containing global vocabulary indices for the gathered top-k tokens after moving to host.
    """

    topk_logprobs: Optional[ttnn.Tensor]
    topk_indices: Optional[ttnn.Tensor]
    topk_logprobs_host: Optional[ttnn.Tensor]
    topk_indices_host: Optional[ttnn.Tensor]

    def cpu(self, blocking: bool = True) -> "LogProbsResult":
        """Transfer device tensors to host CPU.

        Args:
            blocking: If True, wait for the host tensor to be ready before returning.

        Returns:
            A new LogProbsResult so that each iteration gets its own host
            tensor references (avoids race with trace-captured singletons).
        """
        return LogProbsResult(
            topk_logprobs=None,
            topk_indices=None,
            topk_logprobs_host=self.topk_logprobs.cpu(blocking=blocking),
            topk_indices_host=self.topk_indices.cpu(blocking=blocking),
        )

    def extract_user(self, user_batch_idx: int) -> "LogProbsResult":
        """Extract a single user's top-K logprobs from a batched host result.

        Args:
            user_batch_idx: Index of the user within the batch dimension.

        Returns:
            A new LogProbsResult with host tensors sliced to shape (1, DEVICE_TOP_K).
        """
        lp_torch = ttnn.to_torch(ttnn.get_device_tensors(self.topk_logprobs_host)[0])
        idx_torch = ttnn.to_torch(ttnn.get_device_tensors(self.topk_indices_host)[0])
        return LogProbsResult(
            topk_logprobs=None,
            topk_indices=None,
            topk_logprobs_host=lp_torch[0, 0, user_batch_idx, :].unsqueeze(0),
            topk_indices_host=idx_torch[0, 0, user_batch_idx, :].unsqueeze(0),
        )

    def to_torch_pair(self):
        """Convert host tensors to a (logprobs, indices) pair of torch tensors.

        Returns:
            Tuple of (logprobs_tensor, indices_tensor), each of shape (DEVICE_TOP_K,).
        """
        lp = self.topk_logprobs_host
        idx = self.topk_indices_host
        if isinstance(lp, torch.Tensor):
            return lp.reshape(-1)[:DEVICE_TOP_K].float(), idx.reshape(-1)[:DEVICE_TOP_K].int()
        return ttnn.to_torch(lp).reshape(-1)[:DEVICE_TOP_K].float(), ttnn.to_torch(idx).reshape(-1)[:DEVICE_TOP_K].int()


def reformat_logprobs(output_log_probs, batch_size):
    """Convert a list of per-user logprobs into the format expected by vLLM.

    Args:
        output_log_probs: List of LogProbsResult, torch.Tensor, float, int, or None per user.
        batch_size: Total batch size.

    Returns:
        For new path (LogProbsResult): tuple of (topk_lp[B, DEVICE_TOP_K], topk_idx[B, DEVICE_TOP_K])
        For old path: flat tensor of shape [B]
    """
    has_logprobs_result = any(isinstance(lp, LogProbsResult) for lp in output_log_probs)
    if has_logprobs_result:
        all_lp = torch.zeros(batch_size, DEVICE_TOP_K, dtype=torch.float32)
        all_idx = torch.zeros(batch_size, DEVICE_TOP_K, dtype=torch.int32)
        for i, lp in enumerate(output_log_probs):
            if isinstance(lp, LogProbsResult) and lp.topk_logprobs_host is not None:
                all_lp[i], all_idx[i] = lp.to_torch_pair()
        return (all_lp, all_idx)
    else:
        flat_lp = torch.ones(batch_size, dtype=torch.float32)
        for i, lp in enumerate(output_log_probs):
            if lp is not None and isinstance(lp, (torch.Tensor, float, int)):
                flat_lp[i] = float(lp)
        return flat_lp


class LogProbsCalculator:
    """
    Class to calculate log-probs for a given logits tensor and indices tensor.

    Supports two modes:
    - Old mode (backward compat): calculate_log_probs() returns single sampled-token logprob
    - New mode (gpt-oss-120b): calculate_topk_log_probs() returns top-32 logprobs + indices

    Args:
        mesh_device: MeshDevice to use for all-gather operations
        sub_core_grids: Sub-core grid configuration for operations (optional)
        tt_ccl: CCL object for distributed operations (optional)
        batch_size: Maximum batch size for log-probs calculation (default: 32)
        use_topk_logprobs: If True, allocate tensors for new top-K path instead of old path
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        sub_core_grids: ttnn.CoreRangeSet = None,
        tt_ccl=None,
        batch_size: int = 32,
        use_topk_logprobs: bool = False,
    ):
        self.global_max = None
        self.global_exp_sum = None
        self.mesh_device = mesh_device
        self.enable_log_probs = False  # default to False
        # Per-user boolean array tracking which users have logprobs enabled
        self.logprobs_enabled = [False] * batch_size
        # Per-user integer array tracking how many top logprobs each user requested (0-20)
        self.num_logprobs = [0] * batch_size
        # Flag: True when at least one user needs top-k logprobs (num_logprobs > 0)
        self.topk_logprobs_needed = False
        self.cluster_shape = list(mesh_device.shape)
        self.sub_core_grids = sub_core_grids
        self.tt_ccl = tt_ccl
        self.batch_size = batch_size
        self._use_topk_logprobs = use_topk_logprobs
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

        # Initialize tensors based on mode
        # Old path tensors (backward compat for non-gpt-oss models)
        self.mask = None
        self.output_tensor = None
        # New path tensors (gpt-oss-120b top-K logprobs)
        self.topk_logprobs_output = None
        self.topk_indices_output = None

        if use_topk_logprobs:
            # New path: allocate top-K output tensors
            self.topk_logprobs_output = ttnn.as_tensor(
                torch.zeros(1, 1, batch_size, DEVICE_TOP_K),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            self.topk_indices_output = ttnn.as_tensor(
                torch.zeros(1, 1, batch_size, DEVICE_TOP_K, dtype=torch.int32),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            # Old path: allocate mask and output tensors
            mask_tensor = (
                torch.arange(num_devices_for_sharding).unsqueeze(1).expand(num_devices_for_sharding, batch_size)
            )

            if self.cluster_shape[0] > 1 and self.cluster_shape[1] > 1:
                dims = (0, None) if self._all_gather_cluster_axis == 0 else (None, 0)
                mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=self.cluster_shape)
            elif num_devices > 1:
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

    def set_log_probs_mode(
        self,
        enable_log_probs: bool | list[bool] = False,
        num_logprobs: int | list[int] | None = None,
        empty_slots: list[int] | None = None,
    ):
        """Set logprobs mode for the current batch.

        Args:
            enable_log_probs: Boolean or per-user boolean list. If any user has logprobs
                enabled, the entire batch runs logprobs computation.
            num_logprobs: Integer or per-user integer list (0-20). Specifies how many
                top logprobs to return per user. 0 means sampled token logprob only.
                Values > 0 trigger top-k logprobs computation on device.
            empty_slots: Optional list of batch indices at which to apply the new
                logprobs settings. When provided, only those positions are updated
                and the rest of the batch retains its previous values.
        """
        if empty_slots is not None:
            # Partial update: only modify the specified batch positions
            if isinstance(enable_log_probs, list):
                for i, slot in enumerate(empty_slots):
                    self.logprobs_enabled[slot] = enable_log_probs[i]
            else:
                for slot in empty_slots:
                    self.logprobs_enabled[slot] = enable_log_probs

            if num_logprobs is not None:
                if isinstance(num_logprobs, list):
                    for i, slot in enumerate(empty_slots):
                        self.num_logprobs[slot] = num_logprobs[i]
                else:
                    for slot in empty_slots:
                        self.num_logprobs[slot] = num_logprobs
        else:
            # Full batch update
            if isinstance(enable_log_probs, list):
                self.logprobs_enabled = list(enable_log_probs)
            else:
                self.logprobs_enabled = [enable_log_probs] * self.batch_size

            if num_logprobs is not None:
                if isinstance(num_logprobs, list):
                    self.num_logprobs = list(num_logprobs)
                else:
                    self.num_logprobs = [num_logprobs] * self.batch_size
            else:
                self.num_logprobs = [0] * self.batch_size

        # Recompute derived flags from the full arrays
        self.enable_log_probs = any(self.logprobs_enabled)
        # Top-K computation is needed whenever logprobs are enabled (even with
        # num_logprobs=0) because the sampled token's logprob is extracted from
        # the top-K results in the new path.
        self.topk_logprobs_needed = self.enable_log_probs

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
        ttnn.deallocate(local_max_tensor)
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
        ttnn.deallocate(global_max_to_subtract)
        ttnn.deallocate(subtracted_tensor)
        sum_exp_tensor = ttnn.sum(exp_tensor, dim=-1, keepdim=True, **self.common_args)
        ttnn.deallocate(exp_tensor)

        gathered_sum_exp_tensors = self._perform_all_gather(
            sum_exp_tensor,
            dim=1,
            num_links=1,
            buffer_key="LOGPROBS_SUM_EXP_REDUCTION",
        )
        gathered_sum_exp_tensors = ttnn.to_layout(gathered_sum_exp_tensors, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        ttnn.deallocate(sum_exp_tensor)
        B_sum = gathered_sum_exp_tensors.shape[2]
        gathered_sum_exp_tensors = ttnn.reshape(gathered_sum_exp_tensors, (1, 1, D, B_sum), **self.common_args)
        gathered_sum_exp_tensors = ttnn.to_layout(gathered_sum_exp_tensors, ttnn.TILE_LAYOUT, **self.common_args)

        self.global_exp_sum = ttnn.sum(gathered_sum_exp_tensors, dim=2, keepdim=True, **self.common_args)
        ttnn.deallocate(gathered_sum_exp_tensors)

    def _is_supported(self):
        """Check if logprobs computation is supported on this device configuration."""
        num_devices = self.mesh_device.get_num_devices()
        if num_devices not in (8, 32):
            return False
        if self.num_devices_for_sharding < 2:
            return False
        return True

    # -----------------------------------------------------------------------
    # Old path (backward compat for non-gpt-oss models)
    # -----------------------------------------------------------------------

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
        Returns None if log-probs are not requested, not supported, or the device count is not 8 or 32.
        (Old path — backward compat for non-gpt-oss models)
        """
        if not self.enable_log_probs:
            return None

        if not self._is_supported():
            return None

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

    # -----------------------------------------------------------------------
    # New path (gpt-oss-120b top-K logprobs)
    # -----------------------------------------------------------------------

    def _calculate_topk_log_probs_from_values(self, topk_values: ttnn.Tensor):
        """Compute logprobs for gathered top-k values using pre-computed global stats.

        Applies the log-softmax formula: logprob = logit - global_max - log(global_exp_sum)
        to each of the gathered top-k values.

        Args:
            topk_values: Gathered top-k values tensor of shape (1, 1, batch_size, num_topk).
        """
        B = topk_values.shape[2]

        # Reshape global_max from (1,1,1,B) to (1,1,B,1) for broadcasting with (1,1,B,K)
        global_max_bcast = ttnn.to_layout(self.global_max, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        global_max_bcast = ttnn.reshape(global_max_bcast, (1, 1, B, 1), **self.common_args)
        global_max_bcast = ttnn.to_layout(global_max_bcast, ttnn.TILE_LAYOUT, **self.common_args)

        # Compute log(global_exp_sum) and reshape for broadcasting
        log_global_exp_sum = ttnn.log(self.global_exp_sum, **self.common_args)
        log_global_exp_sum_bcast = ttnn.to_layout(log_global_exp_sum, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        log_global_exp_sum_bcast = ttnn.reshape(log_global_exp_sum_bcast, (1, 1, B, 1), **self.common_args)
        log_global_exp_sum_bcast = ttnn.to_layout(log_global_exp_sum_bcast, ttnn.TILE_LAYOUT, **self.common_args)
        ttnn.deallocate(log_global_exp_sum)

        # Apply log-softmax formula: logprob = logit - global_max - log(global_exp_sum)
        ttnn.subtract(topk_values, global_max_bcast, output_tensor=self.topk_logprobs_output, **self.common_args)
        ttnn.subtract(
            self.topk_logprobs_output,
            log_global_exp_sum_bcast,
            output_tensor=self.topk_logprobs_output,
            **self.common_args,
        )
        ttnn.deallocate(global_max_bcast)
        ttnn.deallocate(log_global_exp_sum_bcast)

        return self.topk_logprobs_output

    def calculate_topk_log_probs(
        self,
        logits_tensor: ttnn.Tensor,
        topk_values: ttnn.Tensor,
        topk_global_indices: ttnn.Tensor,
        sub_core_grid_topk: ttnn.CoreRangeSet = None,
    ) -> LogProbsResult | None:
        """Calculate logprobs for the gathered top-k tokens in a single pass.

        Args:
            logits_tensor: Full logits tensor, sharded across devices.
            topk_values: Gathered top-k values from all devices. Shape: (1,1,B,256).
            topk_global_indices: Global vocabulary indices for the gathered top-k. Shape: (1,1,B,256).
            sub_core_grid_topk: Sub-core grid for topk operation.

        Returns:
            LogProbsResult with top-32 logprobs and indices, or None if disabled/unsupported.
        """
        if not self.enable_log_probs:
            return None

        if not self._is_supported():
            return None

        # Ensure bfloat16 precision for numerical stability
        if logits_tensor.dtype == ttnn.bfloat8_b:
            logits_tensor = ttnn.typecast(logits_tensor, ttnn.bfloat16, **self.common_args)

        # Compute global stats — large intermediates allocated and freed here
        self._compute_global_stats(logits_tensor)

        # Narrow 256 -> 32 topk values and indices
        topk_local_values, topk_local_indices = ttnn.topk(
            topk_values,
            k=DEVICE_TOP_K,
            dim=-1,
            sub_core_grids=sub_core_grid_topk,
        )

        # ttnn.gather requires uint32 indices in TILE_LAYOUT
        topk_local_indices = ttnn.typecast(topk_local_indices, ttnn.uint32, **self.common_args)
        topk_local_indices = ttnn.to_layout(topk_local_indices, ttnn.ROW_MAJOR_LAYOUT, **self.common_args)
        topk_local_indices = ttnn.to_layout(topk_local_indices, ttnn.TILE_LAYOUT, **self.common_args)
        ttnn.gather(
            topk_global_indices, dim=-1, index=topk_local_indices, out=self.topk_indices_output, **self.common_args
        )
        ttnn.deallocate(topk_local_indices)

        # Ensure topk_values is bfloat16 for consistent computation
        if topk_local_values.dtype != ttnn.bfloat16:
            topk_local_values = ttnn.typecast(topk_local_values, ttnn.bfloat16, **self.common_args)

        # Single-pass logprob computation for all gathered top-k tokens
        self._calculate_topk_log_probs_from_values(topk_local_values)
        ttnn.deallocate(topk_local_values)

        return LogProbsResult(
            topk_logprobs=self.topk_logprobs_output,
            topk_indices=self.topk_indices_output,
            topk_logprobs_host=None,
            topk_indices_host=None,
        )

    # -----------------------------------------------------------------------
    # Host transfer helpers
    # -----------------------------------------------------------------------

    def _build_mesh_composer(self):
        """Build the appropriate mesh composer for transferring tensors from device to host."""
        if self.cluster_shape[0] > 1 and self.cluster_shape[1] > 1:
            return ttnn.ConcatMesh2dToTensor(
                self.mesh_device,
                dims=(0, 1),
                mesh_shape=self.cluster_shape,
            )
        else:
            return ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)

    def transfer_logprobs_to_host(
        self,
        log_probs_result: LogProbsResult | None,
        sampled_token_ids: torch.Tensor,
        num_logprobs_per_user: list[int] | None = None,
    ) -> list[dict | None]:
        """Move logprobs from device to host and build per-user response objects.

        tt-metal path only (standalone demos/tests). vLLM does NOT call this.

        Args:
            log_probs_result: LogProbsResult from calculate_topk_log_probs.
            sampled_token_ids: Host tensor of sampled token IDs, shape (batch_size,).
            num_logprobs_per_user: Per-user count of top logprobs to return (0-20).
                If None, uses self.num_logprobs.

        Returns:
            List of length batch_size. Each element is None for users with
            logprobs disabled, otherwise a dict with returned_token and top_logprobs.
        """
        if log_probs_result is None:
            return [None] * self.batch_size

        if num_logprobs_per_user is None:
            num_logprobs_per_user = self.num_logprobs

        mesh_composer = self._build_mesh_composer()

        topk_logprobs_host = ttnn.to_torch(
            log_probs_result.topk_logprobs_host
            if log_probs_result.topk_logprobs_host is not None
            else log_probs_result.topk_logprobs,
            mesh_composer=mesh_composer,
        )
        topk_indices_host = ttnn.to_torch(
            log_probs_result.topk_indices_host
            if log_probs_result.topk_indices_host is not None
            else log_probs_result.topk_indices,
            mesh_composer=mesh_composer,
        )
        # Remove replicas
        topk_logprobs_host = topk_logprobs_host[0, 0, ...].float()
        topk_indices_host = topk_indices_host[0, 0, ...].to(torch.int32)

        results: list[dict | None] = []
        for user_idx in range(self.batch_size):
            if not self.logprobs_enabled[user_idx]:
                results.append(None)
                continue

            sampled_id = int(sampled_token_ids[user_idx].item())
            user_logprobs = topk_logprobs_host[user_idx]
            user_indices = topk_indices_host[user_idx]

            # Extract sampled token logprob by matching its ID in the top-k
            match_mask = user_indices == sampled_id
            if match_mask.any():
                sampled_logprob = float(user_logprobs[match_mask][0].item())
            else:
                logger.warning(f"Sampled token {sampled_id} not found in top-k for user {user_idx}")
                sampled_logprob = float("nan")

            n = num_logprobs_per_user[user_idx] if user_idx < len(num_logprobs_per_user) else 0

            results.append(
                {
                    "returned_token": {
                        "token_idx": sampled_id,
                        "logprob": sampled_logprob,
                    },
                    "top_logprobs": {
                        "token_indices": user_indices[:n].tolist(),
                        "logprobs": user_logprobs[:n].tolist(),
                        "num_logprobs": n,
                        "logprobs_formatted": False,
                    },
                }
            )

        return results
