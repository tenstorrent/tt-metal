# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
All-to-all communication primitives for EP MoE on P150a/Wormhole.

The P150a has 4 chips connected via mesh NoC.
All-to-all scatter/gather is implemented using ttnn's mesh communication
or fallback to CPU-based token sharding when mesh ops are unavailable.

Scatter: active_chip tokens → distributed to all chips based on expert mapping
Gather: results from all chips → aggregated back to active chip
"""

import torch


class AllToAllScatterGather:
    """
    All-to-all communication for EP MoE token dispatch/aggregate.

    On Wormhole (P150a), 4 chips share a coherent memory space via NoC.
    We can use ttnn's L1-to-L1 or DRAM-to-DRAM transfers for inter-chip communication.
    """

    def __init__(self, device, ep_config, hidden_size: int):
        self.device = device
        self.ep_config = ep_config
        self.hidden_size = hidden_size
        self.ep_size = ep_config.ep_size

    def scatter(self, x: torch.Tensor, topk_indices: torch.Tensor,
                topk_weights: torch.Tensor):
        """
        Scatter tokens to expert-owning chips.

        Args:
            x: [S, H] input tokens
            topk_indices: [S, K] expert indices
            topk_weights: [S, K] gating weights

        Returns:
            dict: {chip_id: (tokens, weights, local_expert_indices)}
        """
        S, H = x.shape
        K = topk_indices.shape[1]

        # Expand to [S*K] individual (token, expert) pairs
        flat_experts = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        flat_tokens = x.repeat_interleave(K, dim=0)

        # Determine which chip owns each expert
        chip_ids = (flat_experts // self.ep_config.experts_per_device)

        results = {}
        for cid in range(self.ep_size):
            mask = chip_ids == cid
            if mask.sum() == 0:
                results[cid] = (
                    torch.zeros(0, H, dtype=torch.bfloat16),
                    torch.zeros(0, 1, dtype=torch.bfloat16),
                    torch.zeros(0, dtype=torch.long),
                )
                continue

            # Extract tokens destined for this chip
            tokens = flat_tokens[mask]
            weights = flat_weights[mask].unsqueeze(-1)
            local_expert_idx = flat_experts[mask] - cid * self.ep_config.experts_per_device

            results[cid] = (tokens, weights, local_expert_idx)

        return results

    def gather(self, ep_outputs: dict, topk_indices: torch.Tensor,
               topk_weights: torch.Tensor, S: int, K: int, H: int) -> torch.Tensor:
        """
        Gather expert outputs and aggregate by gating weights.

        Args:
            ep_outputs: {chip_id: [local_S, H]}
            topk_indices: [S, K]
            topk_weights: [S, K]
            S, K, H: dimensions

        Returns:
            [S, H] aggregated output (weighted sum)
        """
        output = torch.zeros(S * K, H, dtype=torch.bfloat16)

        flat_experts = topk_indices.reshape(-1)
        chip_ids = flat_experts // self.ep_config.experts_per_device

        for cid, chip_output in ep_outputs.items():
            if chip_output.numel() == 0:
                continue
            mask = chip_ids == cid
            output[mask] = chip_output

        # Reshape back to [S, K, H] and weight by gating
        output = output.reshape(S, K, H)
        weighted = output * topk_weights.unsqueeze(-1)
        return weighted.sum(dim=1)  # [S, H]

    def scatter_device(self, x_tt, topk_indices, topk_weights):
        """
        Device-side scatter using ttnn mesh ops (for Wormhole).

        This is the performance-critical path for EP MoE.
        Uses ttnn's L1-to-L1 NoC transfers when available.
        Falls back to CPU scatter when mesh ops are unavailable.
        """
        # TODO: Implement with ttnn's mesh_noC ops when available
        # For now, use CPU scatter as fallback
        import ttnn
        x_cpu = ttnn.to_torch(x_tt)
        return self.scatter(x_cpu, topk_indices, topk_weights)

    def gather_device(self, ep_outputs, topk_indices, topk_weights, S, K, H):
        """
        Device-side gather using ttnn mesh ops (for Wormhole).
        """
        # TODO: Implement with ttnn's mesh_noC ops
        # Flatten CPU outputs and gather
        cpu_outputs = {}
        for cid, out in ep_outputs.items():
            import ttnn
            if isinstance(out, torch.Tensor):
                cpu_outputs[cid] = out
            else:
                cpu_outputs[cid] = ttnn.to_torch(out)
        return self.gather(cpu_outputs, topk_indices, topk_weights, S, K, H)
