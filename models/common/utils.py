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

    def compute_global_stats(
        self,
        logits_tensor: ttnn.Tensor,
    ):
        """
        To calculate log-probs, we need to calculate the global max and global sum(exp(logits - global_max)) for each chip.
        This is done by all-gathering the max and sum(exp(logits - global_max)) for each chip and then taking the max and sum of the gathered tensors.
        log-prob formula: log-prob(x) = logits(x) - global_max - log(sum(exp(logits - global_max)))

        Args:
            logits_tensor (ttnn.Tensor): Logits as model output (batch_size, vocab_size)
        """
        # Calculate local max
        local_max_tensor = ttnn.max(logits_tensor, dim=-1, keepdim=True)
        # All-gather local max to get global max
        print(f"mesh_device: {self.mesh_device}")
        gathered_max_tensors = ttnn.all_gather(
            local_max_tensor,
            dim=3,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=None,
            topology=ttnn.Topology.Linear,
        )
        self.global_max = ttnn.max(gathered_max_tensors, dim=-1)

        # Calculate stable local sum-exp using subtract of global-max from each local logit
        subtracted_tensor = ttnn.subtract(logits_tensor, self.global_max)
        sum_exp_tensor = ttnn.sum(ttnn.exp(subtracted_tensor), dim=-1)
        # All-gather stable local sum-exp to get global sum-exp
        gathered_sum_exp_tensors = ttnn.all_gather(
            sum_exp_tensor, dim=-1, mesh_device=self.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.global_exp_sum = ttnn.sum(gathered_sum_exp_tensors, dim=-1)

    def calculate_log_probs(self, logits_tensor: ttnn.Tensor):
        """
        Calculate log-probs for a given logits tensor and indices tensor.
        """
        if self.global_max is None or self.global_exp_sum is None:
            raise ValueError("Global max or global exp sum is not calculated yet. Call compute_global_stats first.")

        # Calculate log-probs with formula:
        # logits_tensor - self.global_max - ttnn.log(self.global_exp_sum)

        out = ttnn.subtract(logits_tensor, self.global_max)
        out = ttnn.subtract(out, ttnn.log(self.global_exp_sum))

        return out
