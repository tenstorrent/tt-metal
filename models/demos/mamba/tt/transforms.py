# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib as ttl
import torch


class MambaSsmBlockTransformer:
    def __init__(self, device, batch_size, hidden_size, latent_size):
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        repeat_interleave_mask = torch.ones(1, 1, batch_size, latent_size)
        self.repeat_interleave_mask = ttnn.from_torch(
            repeat_interleave_mask,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

        repeat_mask = torch.ones(1, 1, batch_size, hidden_size)
        self.repeat_mask = ttnn.from_torch(
            repeat_mask,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def repeat_interleave(self, x, memory_config):
        """
        This function implements an SSM-specific repeat_interleave operation needed to transform
        the SSM block input (X) from (B, 2E) to (B, 2EN) so that it can be multiplied with delta*B.

        """
        assert x.shape == (
            1,
            1,
            self.batch_size,
            self.hidden_size,
        ), f"Expected repeat_interleave input to be (1, 1, B, 2E) (was {x.shape})"
        return ttl.operations.primary.transformers.ssm_eltwise_mul(
            self.repeat_interleave_mask, x, output_mem_config=memory_config
        )

    def repeat(self, x, memory_config):
        """
        This function implements an SSM-specific repeat operation needed to transform the C
        value from (B, N) to (B, 2EN) where N is the latent size (32) and E is the
        up project size (2560).
        """
        assert x.shape == (
            1,
            1,
            self.batch_size,
            self.latent_size,
        ), f"Expected repeat input to be (1, 1, B, N) (was {x.shape})"
        return ttl.operations.primary.transformers.ssm_eltwise_mul(x, self.repeat_mask, output_mem_config=memory_config)
