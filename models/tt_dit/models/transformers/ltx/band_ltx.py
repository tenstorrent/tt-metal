# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Temporal-band self-attention pattern for LTX video tokens (frame-major order). Torch only."""

import torch


def temporal_band_mask(n_pad: int, n_real: int, tokens_per_frame: int, window: int) -> torch.Tensor:
    """(n_pad, n_pad) additive bf16 mask: 0 where |frame(q) - frame(k)| <= window and k < n_real.

    Padded query rows see every real key so no row is all -inf (softmax NaN).
    """
    mask = torch.full((n_pad, n_pad), float("-inf"), dtype=torch.bfloat16)
    num_frames = -(-n_real // tokens_per_frame)
    for f in range(num_frames):
        q0, q1 = f * tokens_per_frame, min((f + 1) * tokens_per_frame, n_real)
        k0 = max(f - window, 0) * tokens_per_frame
        k1 = min((f + window + 1) * tokens_per_frame, n_real)
        mask[q0:q1, k0:k1] = 0
    mask[n_real:, :n_real] = 0
    return mask
