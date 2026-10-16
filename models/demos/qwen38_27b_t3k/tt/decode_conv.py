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


def row_selector(device, batch):
    """One-hot reading row 4*i+3 of a packed convolution result into row 0 of user i's tile.

    Every other row is zero, matching what tilizing a single live row per user produced.
    bfloat16 keeps the product exact: each output row is one input row scaled by 1.0.

    The users are stacked along rows rather than carried as a batch dimension. A batched
    matmul splits over N tiles and walks the batch serially, which caps it at width/32 cores;
    stacked, the same work fills the grid.
    """
    import torch

    mesh_kwargs = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if hasattr(device, "get_num_devices") else {}
    host = torch.zeros(batch * 32, batch * 4, dtype=torch.float32)
    for user in range(batch):
        host[user * 32, user * 4 + 3] = 1.0
    return ttnn.from_torch(
        host,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        **mesh_kwargs,
    )


def packed_decode_conv(row_qkv, history, taps, widths, actual_start, selector, compute_kernel_config=None):
    """Return [B,32,C] Q/K/V; only the first time row is live.

    B must be a multiple of eight. Each four-row group contains that user's
    complete three-row history followed by its current token. The convolution
    at row 4*i+3 therefore has exactly the original four operands. Earlier
    outputs may cross user boundaries and are discarded. Padding rows are zero;
    the caller's zero beta/log-decay makes their GDN recurrence an identity.

    `selector` must be `row_selector(device, batch)`.
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
    # The one-hot read never leaves tile layout. Reaching the live rows through row-major
    # instead costs a tilize per output, and tilize splits work by tile row, so eight users
    # mean eight cores however wide the projection is. Both reshapes are views: the row count
    # is tile-aligned and the width never changes.
    return [
        ttnn.reshape(
            ttnn.matmul(
                selector,
                ttnn.reshape(output, [batch * 4, width]),
                compute_kernel_config=compute_kernel_config,
            ),
            [batch, 32, width],
        )
        for output, width in zip(outputs, widths)
    ]
