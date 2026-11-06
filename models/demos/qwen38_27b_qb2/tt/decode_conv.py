# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batched one-token convolutions through the native kernel's per-row histories."""

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


def packed_decode_conv(row_qkv, history, taps, widths, actual_start):
    """Return [B,32,C] Q/K/V; only the first time row is live.

    Each user's current token is one row of a [1, B, C] input, and the [B, 3, C]
    history gives each row its own three preceding tokens, so the kernel
    convolves the B users independently in one call.
    """
    batch, _, channels = row_qkv.shape
    rows = ttnn.reshape(row_qkv[:, :1, :], [1, batch, channels])
    outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        rows,
        history,
        *taps,
        *widths,
        # Per-row staging work scales with the block width, so narrow blocks spread it over more cores.
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=32),
        actual_start=actual_start,
        predecessor_carry=history,
    )
    result = []
    for output, width in zip(outputs, widths):
        output = ttnn.reshape(ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT), [batch, 1, width])
        # Tilization zero-pads each user's single row to a tile. Expose those
        # physical rows directly instead of writing a padded RM buffer to DRAM
        # and then reading it back for a separate layout conversion.
        output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)
        result.append(ttnn.reshape(output, [batch, 32, width], output.padded_shape))
    return result
