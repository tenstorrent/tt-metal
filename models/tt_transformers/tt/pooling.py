# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule


class MeanPooling(LightweightModule):
    """
    Mean pooling layer for text embedding models.

    This layer performs mean pooling over the sequence dimension to create
    fixed-size embeddings from variable-length sequences.
    """

    def __init__(self, mesh_device, args, tt_ccl=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.hidden_size = args.dim
        self.tt_ccl = tt_ccl

    def forward(self, hidden_states: ttnn.Tensor, attention_mask=None) -> ttnn.Tensor:
        """
        Apply mean pooling to the hidden states.

        Args:
            hidden_states: ttnn.Tensor of shape [batch, 1, seq_len, hidden_dim]
            attention_mask: Optional ttnn.Tensor of shape [batch, 1, seq_len, 1]
                           1 for tokens to include, 0 for tokens to exclude (padding)

        Returns:
            ttnn.Tensor of shape [batch, 1, 1, hidden_dim] containing pooled embeddings
        """
        # Input shape: [batch, 1, seq_len, hidden_dim]

        # For multi-device, gather the sharded input before pooling
        if self.args.is_multichip and self.tt_ccl is not None:
            cluster_axis = 0 if self.args.is_galaxy else None
            num_links = 2 if self.args.is_galaxy else 1
            hidden_states = ttnn.experimental.all_gather_async(
                hidden_states,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
                num_links=num_links,
                topology=self.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                cluster_axis=cluster_axis,
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        # For multi-device, ensure we use DRAM memory config
        memory_config = ttnn.DRAM_MEMORY_CONFIG

        if attention_mask is not None:
            # Apply attention mask: multiply hidden states by mask to zero out padding tokens
            # attention_mask shape: [batch, 1, seq_len, 1] -> expand to [batch, 1, seq_len, hidden_dim]
            attention_mask_expanded = ttnn.unsqueeze_to_4D(attention_mask)  # [batch, 1, seq_len, 1]
            attention_mask_expanded = ttnn.repeat(
                attention_mask_expanded, [1, 1, 1, self.hidden_size], memory_config=memory_config
            )  # [batch, 1, seq_len, hidden_dim]

            # Apply mask
            masked_hidden_states = ttnn.mul(hidden_states, attention_mask_expanded, memory_config=memory_config)

            # Compute sum over sequence dimension (dim=2)
            sum_embeddings = ttnn.sum(
                masked_hidden_states, dim=2, keepdim=True, memory_config=memory_config
            )  # [batch, 1, 1, hidden_dim]

            # Compute sum of attention mask values for normalization
            mask_sum = ttnn.sum(attention_mask, dim=2, keepdim=True, memory_config=memory_config)  # [batch, 1, 1, 1]

            # Add small epsilon to avoid division by zero
            eps = 1e-9
            mask_sum = ttnn.add(mask_sum, eps, memory_config=memory_config)

            # Expand mask_sum to match embedding dimensions
            mask_sum_expanded = ttnn.repeat(
                mask_sum, [1, 1, 1, self.hidden_size], memory_config=memory_config
            )  # [batch, 1, 1, hidden_dim]

            # Compute mean by dividing sum by count
            pooled_embeddings = ttnn.div(sum_embeddings, mask_sum_expanded, memory_config=memory_config)
        else:
            # Simple mean pooling without attention mask
            pooled_embeddings = ttnn.mean(
                hidden_states, dim=2, keepdim=True, memory_config=memory_config
            )  # [batch, 1, 1, hidden_dim]

        return pooled_embeddings
