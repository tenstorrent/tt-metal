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

    The matrix is [batch*32, batch*4]: users stacked along rows, so the matmul's M is the whole
    user stack and its output is read back as one [batch, 32, width] tile tensor per projection.
    Carrying the users as a batch dimension instead measures slower on this mesh.

    The selector's 1.0 and 0.0 are exact in bfloat16 and each output row sums one input row with
    zeros, so a finite result is exact at any fidelity at or above the bf16 default. Non-finite
    values do not survive: the rows this discards are convolved across user boundaries, and
    0 * inf is NaN, so one overflow anywhere in the packed result reaches every user.
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


def packed_decode_conv(row_qkv, history, taps, widths, actual_start, selector):
    """Return [B,32,C] Q/K/V; only the first time row is live.

    B must be a multiple of eight. Each four-row group contains that user's
    complete three-row history followed by its current token. The convolution
    at row 4*i+3 therefore has exactly the original four operands. Earlier
    outputs may cross user boundaries and are discarded. Padding rows are zero;
    the caller's zero beta/log-decay makes their GDN recurrence an identity.

    `selector` is `row_selector(device, batch)`. The caller owns it because it is a persistent
    buffer: built once per batch width before any trace is captured, and read by every layer on
    every step. Building it here would allocate beside a live trace.
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
    # Rows 1..31 of each user's tile stay zero; the chunk-32 scan downstream reads the whole
    # tile. No compute config: multiplying by exactly 1.0 gains nothing above the bf16 default.
    # Both reshapes are metadata only — the first keeps its last two dimensions, the second
    # splits a tile-aligned row count.
    return [
        ttnn.reshape(
            ttnn.matmul(selector, ttnn.reshape(output, [batch * 4, width])),
            [batch, 32, width],
        )
        for output, width in zip(outputs, widths)
    ]
