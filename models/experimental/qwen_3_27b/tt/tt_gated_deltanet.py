# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gated DeltaNet -- linear attention with a gated delta rule.

The important structural difference from softmax attention: instead of a KV cache
that grows with sequence length, this layer carries a FIXED-SIZE recurrent state
[B, 48, 128, 128] plus a small conv state. Memory per token is O(1).

Two algorithms compute the same thing:
  * prefill (T > 1): chunked scan over the sequence
  * decode  (T = 1): single recurrent step

Reference: transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5GatedDeltaNet
(class at :371, forward at :437).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36GatedDeltaNet(LightweightModule):
    """
    Gated DeltaNet linear attention. 48 of the 64 layers.

    Dimensions:
        D            = 5120   hidden size
        n_qk_heads   = 16     query/key heads      -> key_dim   = 16 * 128 = 2048
        n_v_heads    = 48     value heads          -> value_dim = 48 * 128 = 6144
        head_dim     = 128    for q, k and v alike
        conv_kernel  = 4      depthwise causal conv1d over q||k||v (10240 channels)

    Weights (torch shapes, before the [out, in] -> [in, out] transpose). Note the
    projections are only PARTLY fused -- qkv together, z / a / b separate:
        in_proj_qkv    [10240, 5120]   = 2048 + 2048 + 6144   (q, k, v)
        in_proj_z      [ 6144, 5120]   output gate
        in_proj_b      [   48, 5120]   beta  (write strength), one per value head
        in_proj_a      [   48, 5120]   alpha (decay),          one per value head
        conv1d.weight  [10240, 1, 4]   depthwise -- NO bias (nn.Conv1d bias=False)
        A_log          [48]
        dt_bias        [48]
        norm.weight    [128]           gated RMSNorm over head_v_dim
        out_proj       [ 5120, 6144]

    The qkv split is a PLAIN contiguous split -- torch.split(mixed, [2048, 2048,
    6144], dim=-1). No per-head interleaving. (Qwen3-Next fused all of q/k/v/z
    into one head-grouped weight and needed a de-interleave; Qwen3.5 does not.)

    Shapes through the forward:
        x         [B, T, D]
        mixed_qkv [B, T, 10240]  -> transpose -> [B, 10240, T] for the conv
        conv+SiLU [B, 10240, T]  depthwise, padding = K-1 then truncate to T
        split     q, k [B, T, 2048] -> [B, T, 16, 128] -> repeat_interleave(3)
                                                       -> [B, T, 48, 128]
                  v      [B, T, 6144] -> [B, T, 48, 128]
        z         [B, T, 6144]   -> [B, T, 48, 128]
        b, a      [B, T, 48]
        delta rule -> o [B, T, 48, 128], recurrent state [B, 48, 128, 128]
        o = gated_rmsnorm(o, gate=z)
        output    [B, T, D]

    To implement:
        1. project qkv (one matmul), z, b, a
        2. depthwise causal conv1d + SiLU on the 10240-wide qkv
           - padding = kernel-1 = 3, then truncate the output back to T. That
             truncation IS the causality.
        3. split [2048, 2048, 6144]; reshape to heads
        4. repeat_interleave q and k 3x to match the 48 value heads
        5. beta = sigmoid(b)
           g    = -exp(A_log) * softplus(a + dt_bias)      <- compute in fp32
        6. delta rule (chunked for prefill, recurrent for decode).
           NOTE: the reference passes use_qk_l2norm_in_kernel=True -- q and k are
           L2-normalized INSIDE the kernel, so our version must do it explicitly.
        7. gated norm: normalize o, scale by weight, THEN multiply by silu(z)
           (order matters -- see the note in tt_rms_norm.py)
        8. out_proj
      Later: carry recurrent_state + conv_state across calls for decode.
    """

    def __init__(self, device, layer_idx: int):
        self.device = device
        self.layer_idx = layer_idx

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, T, D] -> [B, T, D]."""
        return x
