# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from torch import Tensor

import ttnn


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class LogProbsCalculator:
    """
    Class to calculate log-probs for a given logits tensor and indices tensor.

    Args:
        mesh_device: MeshDevice to use for all-gather operations
        vocab_size: Vocabulary size
    """

    def __init__(self, vocab_size: int, mesh_device: ttnn.MeshDevice):
        self.global_max = None
        self.global_exp_sum = None
        self.vocab_size = vocab_size
        self.mesh_device = mesh_device
        self.enable_log_probs = False  # default to False

        # Create mask for each user on each chip.
        batch_size = 32

        num_devices = self.mesh_device.get_num_devices()
        # TODO: Test this for 6U Galaxy
        # Enforce working on 8 devices instead of all 32 since logits are sharded across 8 devices
        num_devices_for_sharding = 8 if num_devices == 32 else num_devices

        # Create mask tensor with shape (num_devices, batch_size)
        # Each row will have device_id starting from 0 to num_devices - 1
        mask_tensor = torch.arange(num_devices_for_sharding).unsqueeze(1).expand(num_devices_for_sharding, batch_size)

        if self.mesh_device.get_num_devices() == 32:
            self.cluster_shape = (8, 4)
            mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=self.cluster_shape)
        elif self.mesh_device.get_num_devices() == 8:
            mesh_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        else:
            # TODO: Implement method for other number of devices, for now replicate tensor to mesh to support all the ops
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

    def set_log_probs_mode(self, enable_log_probs: bool | list[bool] = False):
        if isinstance(enable_log_probs, list):
            # If any of the users in the batch have log_probs enabled,
            # then whole batch will run log-probs calculation
            enable_log_probs_result = any(enable_log_probs)
        else:
            enable_log_probs_result = enable_log_probs

        self.enable_log_probs = enable_log_probs_result

    def compute_global_stats(
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
        if not self.enable_log_probs:
            return

        if self.mesh_device.get_num_devices() != 8:
            self.global_max = None
            self.global_exp_sum = None
            return
        # Calculate local max
        local_max_tensor = ttnn.max(logits_tensor, dim=-1, keepdim=True)

        # All-gather local max to get global max
        gathered_max_tensors = ttnn.all_gather(
            local_max_tensor,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=None,
            topology=ttnn.Topology.Linear,
        )
        self.global_max = ttnn.max(gathered_max_tensors, dim=-1, keepdim=True)

        # Calculate stable local sum-exp using subtract of global-max from each local logit
        subtracted_tensor = ttnn.subtract(logits_tensor, self.global_max)
        sum_exp_tensor = ttnn.sum(ttnn.exp(subtracted_tensor), dim=-1, keepdim=True)

        # All-gather stable local sum-exp to get global sum-exp
        gathered_sum_exp_tensors = ttnn.all_gather(
            sum_exp_tensor,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=None,
            topology=ttnn.Topology.Linear,
        )
        self.global_exp_sum = ttnn.sum(gathered_sum_exp_tensors, dim=-1, keepdim=True)

        # reshape global_max and global_exp_sum to support same output shape as sampling output -> (1, 1, 1, 32)
        self.global_max = ttnn.reshape(self.global_max, (1, 1, 1, 32))
        self.global_exp_sum = ttnn.reshape(self.global_exp_sum, (1, 1, 1, 32))

    def prepare_relevant_logits(self, logits_tensor: ttnn.Tensor, global_idx_tensor: ttnn.Tensor):
        """
        Prepare global idx tensor with correct values on all devices.
        """
        if not self.enable_log_probs:
            return logits_tensor

        if self.mesh_device.get_num_devices() != 8:
            # TODO: Implement method for Llama 3.70b Galaxy with sub_core_grid support for all the ops
            return None

        if not self.enable_log_probs:
            return logits_tensor

        size_per_device = logits_tensor.shape[-1]

        # convert global_idx_tensor to ttnn.TILE_LAYOUT
        global_idx_tilized_tensor = ttnn.to_layout(global_idx_tensor, ttnn.TILE_LAYOUT)

        # TODO: Raise an issue on this since for UINT_32 ttnn.div produces incorrect output (all zeros)
        global_idx_tilized_tensor = ttnn.typecast(global_idx_tilized_tensor, ttnn.float32)

        # Get chip_id for each user based on global_idx values in global_idx_tensor
        chip_ids_tensor = ttnn.div(
            global_idx_tilized_tensor, size_per_device, round_mode="floor", memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Get local index for each user based on global_idx values in global_idx_tensor
        remainder_tensor = ttnn.remainder(
            global_idx_tilized_tensor, size_per_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # Convert remainder_tensor to int32
        remainder_tensor = ttnn.reshape(ttnn.typecast(remainder_tensor, ttnn.uint32), (1, 1, 32, 1))

        # Get logits for each user on each chip based on local index
        selected_logits_tensor = ttnn.gather(logits_tensor, dim=3, index=remainder_tensor)

        selected_logits_tensor = ttnn.reshape(selected_logits_tensor, (1, 1, 1, 32))
        # Compare mask to chip_ids tensor and select correct positions for each user on all chips inplace
        ttnn.eq_(chip_ids_tensor, self.mask)

        # Multiply selected_logits_tensor with chip_ids_tensor to get expected logits for each user
        selected_logits_tensor = ttnn.multiply(selected_logits_tensor, chip_ids_tensor)

        # Use ttnn.all_gather to get logits across all devices
        selected_logits_tensor = ttnn.all_gather(
            selected_logits_tensor,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=None,
            topology=ttnn.Topology.Linear,
        )

        # Apply sum over device dimension to get logits for each user on all chips
        selected_logits_tensor = ttnn.sum(selected_logits_tensor, dim=2, keepdim=True)

        return selected_logits_tensor

    def calculate_log_probs(self, logits_tensor: ttnn.Tensor):
        """
        Calculate log-probs for a given logits tensor.
        """
        if not self.enable_log_probs:
            return self.output_tensor

        if self.mesh_device.get_num_devices() != 8:
            # TODO: Currently not implemented for 32 devices due sub_core_grid not supported for all the ops
            return self.output_tensor

        if not self.enable_log_probs:
            return self.output_tensor

        if self.global_max is None or self.global_exp_sum is None:
            raise ValueError("Global max or global exp sum is not calculated yet. Call compute_global_stats first.")

        # Calculate log-probs with formula:
        # logits_tensor - self.global_max - ttnn.log(self.global_exp_sum)
        out = ttnn.subtract(logits_tensor, self.global_max)
        out = ttnn.subtract(out, ttnn.log(self.global_exp_sum))

        return out
