# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pack independent one-token convolutions into the existing native kernel."""

import ttnn


def make_actual_start(device):
    """Allocate the replicated zero chronology metadata required for non-SP execution."""
    import torch

    mesh_kwargs = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if hasattr(device, "get_num_devices") else {}
    return ttnn.from_torch(
        torch.zeros(1, dtype=torch.int64),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        **mesh_kwargs,
    )


def packed_prefill_conv(row_qkv, history, taps, widths, actual_start):
    """Pack independent tile-aligned sequences with a 32-row history guard.

    Only each guard's last three rows matter: the first retained output uses
    exactly this user's history, and later outputs use only its own input.
    Discard the complete guard tile, including any cross-user convolution.
    """
    batch, length, channels = row_qkv.shape
    guard = ttnn.pad(history, [(0, 0), (29, 0), (0, 0)], 0.0)
    joined = ttnn.concat([guard, row_qkv], dim=1)
    joined = ttnn.reshape(joined, [1, batch * (length + 32), channels])
    outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        joined,
        history[:1],
        *taps,
        *widths,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=256),
        actual_start=actual_start,
        predecessor_carry=history[:1],
    )
    return [ttnn.reshape(output, [batch, length + 32, width])[:, 32:, :] for output, width in zip(outputs, widths)]


def packed_decode_conv(row_qkv, history, taps, widths, actual_start):
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
        actual_start=actual_start,
        predecessor_carry=history[:1],
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
