# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gated Attention -- standard softmax attention with a query-derived output gate.

Ordinary GQA attention, plus two Qwen-specific twists:
  * the Q projection is 2x wide; the second half is a sigmoid gate applied to the
    attention output
  * RoPE rotates only the first 64 of each head's 256 dims (partial rotary)
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36GatedAttention(LightweightModule):
    """
    Gated full attention. 16 of the 64 layers.

    Dimensions:
        D          = 5120     hidden size
        n_heads    = 24       query heads
        n_kv_heads = 4        key/value heads (GQA, 6 query heads per kv head)
        head_dim   = 256
        rotary_dim = 64       only the first 64 dims of each head are rotated

    Weights (torch shapes, before the [out, in] -> [in, out] transpose):
        q_proj  [12288, 5120]   = 24 * 256 * 2   <- 2x wide: query || gate
        k_proj  [ 1024, 5120]   =  4 * 256
        v_proj  [ 1024, 5120]   =  4 * 256
        o_proj  [ 5120, 6144]   6144 = 24 * 256
        q_norm  [256]           per-head RMSNorm
        k_norm  [256]

    Shapes through the forward:
        x        [B, T, D]
        q||gate  [B, T, 12288]  -> q [B, T, 24, 256], gate [B, T, 6144]
        k, v     [B, T, 4, 256]
        (q/k norm, then RoPE on dims :64 of each head)
        sdpa     [B, 24, T, 256]        k/v repeated 6x for GQA
        out      [B, T, 6144]  * sigmoid(gate)
        output   [B, T, D]

    To implement:
        1. q_proj, split into query and gate halves
        2. q_norm / k_norm  (zero-centered: x * rsqrt(...) * (1 + w))
        3. RoPE on the first 64 dims, pass the rest through untouched
        4. repeat k/v 6x, then SDPA with scale = 256 ** -0.5
        5. multiply by sigmoid(gate)
        6. o_proj
      Later: cos/sin arguments, and a KV cache (this layer type has one; DeltaNet
      layers do not).
    """

    def __init__(self, device, layer_idx: int):
        self.device = device
        self.layer_idx = layer_idx

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T, D] -> [B, T, D]."""
        return x
