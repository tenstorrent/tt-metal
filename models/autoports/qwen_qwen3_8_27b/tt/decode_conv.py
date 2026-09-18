# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pack independent one-token convolutions into the existing native kernel."""

import ttnn


def packed_decode_conv(row_qkv, history, taps, widths):
    """Return [B,32,C] Q/K/V; only the first time row is live.

    B must be a multiple of eight. Each four-row group contains that user's
    complete three-row history followed by its current token. The convolution
    at row 4*i+3 therefore has exactly the original four operands. Earlier
    outputs may cross user boundaries and are discarded. Padding rows are zero;
    the caller's zero beta/log-decay makes their GDN recurrence an identity.
    """
    batch, _, channels = row_qkv.shape
    joined = ttnn.concat([history, row_qkv[:, :1, :]], dim=1)
    joined = ttnn.reshape(joined, [1, batch * 4, channels])
    outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        joined,
        history[:1],
        *taps,
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
    )
    result = []
    for output, width in zip(outputs, widths):
        output = ttnn.reshape(ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT), [batch, 4, width])
        output = output[:, 3:4, :]
        # Tilization already zero-pads each user's single row to a tile. Expose
        # those physical rows directly instead of writing a padded RM buffer
        # to DRAM and then reading it back for a separate layout conversion.
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        result.append(ttnn.reshape(output, [batch, 32, width], output.padded_shape))
    return result
