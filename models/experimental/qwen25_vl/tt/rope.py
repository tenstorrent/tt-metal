"""
This is the vision rotary embedding implementation for Qwen-VL-7B.

The existing RotarySetup(models/tt_transformers/tt/rope.py) in tt_transformers can't be used here,
as Qwen-VL uses a different logic for applying rotary embeddings.
This version is implemented specifically to match Qwen's design.
"""


import torch
import ttnn


class TTQwen2_5_VisionRotaryEmbedding:
    def __init__(self, device, dim: int, theta: float = 10000.0, mode="decode"):
        self.dim = dim
        self.theta = theta
        self.device = device

        arange_indices = ttnn.arange(start=0, end=dim, step=2, device=device)
        arange_indices = ttnn.to_layout(arange_indices, ttnn.TILE_LAYOUT)
        exponent = ttnn.div(arange_indices, dim)
        pow_result = ttnn.pow(theta, exponent)
        recip = ttnn.reciprocal(pow_result)
        self.inv_freq = ttnn.multiply(recip, 1.0)

    def __call__(self, seqlen: int):
        tt_seq = ttnn.arange(end=seqlen, device=self.device)
        tt_seq = ttnn.to_torch(tt_seq)
        tt_inv_freq = ttnn.to_torch(self.inv_freq)
        tt_freqs = torch.outer(tt_seq, tt_inv_freq)
        tt_freqs = ttnn.from_torch(
            tt_freqs,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(self.inv_freq)

        return tt_freqs
