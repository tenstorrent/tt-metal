import torch.nn as nn

import ttnn


class ApplyRotaryPosEmb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, x, cos, sin):
        num_heads = x.shape[-3]
        # Broadcast cos and sin to the number of heads based on Q or K
        cos_bcast = ttnn.repeat(cos, (num_heads, 1, 1))
        sin_bcast = ttnn.repeat(sin, (num_heads, 1, 1))
        first_half, second_half = ttnn.chunk(x, 2, -1)
        first_ = first_half * cos_bcast - second_half * sin_bcast
        second_ = second_half * cos_bcast + first_half * sin_bcast
        return ttnn.concat((first_, second_), dim=-1)
