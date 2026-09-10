# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""conv3d blocking stubs for the MiniMax-H3 audio VAE.

``_FP32_BLOCKINGS`` in ``utils/conv3d.py`` is tuned for the **LTX** vocoder's channel
schedule (1536 -> 768 -> ... -> 24). H3's is different at both ends -- the decoder runs
1024 -> 512 -> ... -> **8** and the encoder 64 -> ... -> 2048 -- so every H3 audio conv
misses the table and falls back to ``(32, 32, 1, 1, 1)``. For a 1-D conv over a long time
axis (a 10 s clip decodes to 324000 samples) a ``T_out_block`` of 1 means one output frame
per work unit, and the conv3d DRAM slicer cannot find a workable configuration at all.

These are **stubs**, shaped after the LTX audio entries which have the same ``(k, 1, 1)``
form: ``C_out_block`` 32, ``H``/``W`` 1, and a ``T_out_block`` in the range LTX's tuned
values occupy (4-64). ``bruteforce_conv3d_sweep.py`` is the tool for real tuning, and that
is a performance-pass job -- these exist only so the correctness gates can run.
"""

from __future__ import annotations

from ....layers.audio_ops import DEFAULT_MAX_C_IN_BLOCK
from ....utils.conv3d import _FP32_BLOCKINGS, aligned_channels

# 16 overshoots L1 by 1.26x (1979264 B against 1572864 B) at the widest audio convs.
T_OUT_BLOCK = 8
C_OUT_BLOCK = 32

# ``C_in_block`` is not only a performance knob: conv3d accumulates one partial sum per block over the
# reduction and those partials are rounded, so a narrower block is measurably less accurate *in isolation*
# -- on `conv_pre` (Cin 2048 x k 7) the error falls monotonically, 2.40e-03 at 32, 1.86e-03 at 128,
# 1.63e-03 at 512, and with operand splitting on, 512 is 1.48x better than 128.
#
# **That does not transfer end to end, so 128 stays.** Measured with every other lever on: 128 gives
# 3.20 % chain RMSE, 256 gives 3.31 % (no better), and 512 fails outright
# (`program.cpp:1706`). The reason is structural -- the chain is dominated by the 126 AMP convs at
# Cin 8-512, where the block cannot widen anyway, so only `conv_pre` and `dec_in_proj` would gain and
# there are two of them against 126. Pass ``max_c_in_block`` to re-sweep. A non-default cap changes
# the prepared weight *bytes* (`prepare_conv3d_weight_state` blocks by ``C_in_block``) with an
# unchanged file set, which is why `weights_variant` folds it into the device-weight cache key.


def _c_in_block(in_channels: int, max_c_in_block: int = DEFAULT_MAX_C_IN_BLOCK) -> int:
    """Largest 32-multiple <= ``max_c_in_block`` that divides ``in_channels`` evenly.

    The kernel requires ``C_in_block`` to be a multiple of the tile width and to divide the
    padded input channel count.
    """
    aligned = aligned_channels(in_channels)
    for block in range(min(max_c_in_block, aligned) // 32 * 32, 0, -32):
        if aligned % block == 0:
            return block
    return 32


def h3_audio_channel_widths(
    *,
    encoder_dim: int = 64,
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5),
    latent_dim: int = 2048,
    latent_channels: int = 32,
    decoder_dim: int = 1024,
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2),
) -> set[tuple[int, int]]:
    """Every ``(in_channels, out_channels)`` pair the audio VAE's convolutions use."""
    pairs: set[tuple[int, int]] = set()

    # Encoder: conv_in, then per stage three residual units at dim//2 plus a strided
    # channel-doubling conv.
    pairs.add((aligned_channels(1), encoder_dim))
    dim = encoder_dim
    for stride in encoder_rates:
        inner = dim
        dim *= 2
        pairs.add((inner, inner))  # residual unit k7 and k1
        pairs.add((inner, dim))  # strided conv
    pairs.add((dim, latent_dim))

    # Decoder: conv_pre, the transposed upsamplers' inner convs, the AMP blocks, conv_post.
    pairs.add((latent_channels, latent_dim))  # dec_in_proj
    pairs.add((latent_dim, decoder_dim))  # conv_pre
    channels = decoder_dim
    for _ in decoder_rates:
        nxt = channels // 2
        pairs.add((channels, nxt))  # upsampler inner conv
        pairs.add((aligned_channels(nxt), aligned_channels(nxt)))  # AMP convs
        channels = nxt
    pairs.add((aligned_channels(channels), aligned_channels(1)))  # conv_post -> mono
    return pairs


def register_h3_audio_blockings(*, max_c_in_block: int = DEFAULT_MAX_C_IN_BLOCK, **config) -> int:
    """Seed ``_FP32_BLOCKINGS`` for every H3 audio conv shape. Returns the number added.

    ``setdefault``, so a swept value that later lands in ``conv3d.py`` wins over these.
    Kernels cover every size the model uses: 1 and 3 (projections), 4/7/8/9/10/11 (AMP
    blocks, strided encoder convs and transposed upsamplers).
    """
    kernels = (1, 3, 4, 7, 8, 9, 10, 11)
    added = 0
    for in_channels, out_channels in h3_audio_channel_widths(**config):
        for kernel in kernels:
            key = (aligned_channels(in_channels), max(32, out_channels), (kernel, 1, 1))
            blocking = (_c_in_block(in_channels, max_c_in_block), C_OUT_BLOCK, T_OUT_BLOCK, 1, 1)
            existing = _FP32_BLOCKINGS.get(key)
            if existing is None:
                _FP32_BLOCKINGS[key] = blocking
                added += 1
            elif existing[2] > T_OUT_BLOCK:
                # A few of H3's (C_in, C_out, kernel) triples coincide with LTX's -- e.g.
                # (128, 64, (4,1,1)) is LTX's ups[3] -- and LTX's tuned T_out_block of 32 is
                # sized for its tensors, not H3's: at H3's widths it overshoots L1 (1979264 B
                # against 1572864 B). Cap the temporal block for the shapes H3 actually uses
                # rather than deferring to an entry tuned for a different model.
                _FP32_BLOCKINGS[key] = (existing[0], existing[1], T_OUT_BLOCK, existing[3], existing[4])
                added += 1
    return added
