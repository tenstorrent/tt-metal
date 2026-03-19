from math import log

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


class TTNNSinusoidalPositionEmbedding(LightweightModule):
    def __init__(self, device, num_channels, max_timescale=10000):
        super().__init__()

        if num_channels % 2 != 0:
            raise ValueError("Embedding dimension must be even")

        self.device = device
        self.num_channels = num_channels
        self.max_timescale = max_timescale
        self.half_dim = num_channels // 2

        # Exact PyTorch formula
        log_timescale_increment = log(max_timescale) / (self.half_dim - 1)

        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(self.half_dim, dtype=torch.float32))

        # Store in TTNN
        self.inv_timescales = ttnn.from_torch(
            inv_timescales,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, timesteps):
        """
        timesteps: (N,)  -> same as torch.arange(length)
        output: (N, D)
        """

        # (N, 1)
        timesteps = ttnn.unsqueeze(timesteps, -1)

        # (1, D/2)
        inv_timescales = ttnn.unsqueeze(self.inv_timescales, 0)

        # scaled_time = position * inv_timescales
        scaled_time = ttnn.multiply(timesteps, inv_timescales, use_legacy=False)

        # sin + cos
        emb_sin = ttnn.sin(scaled_time)
        emb_cos = ttnn.cos(scaled_time)

        # concatenate
        emb = ttnn.concat([emb_sin, emb_cos], dim=-1)

        return emb
