# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — regime R1' (alignment = m_non_aligned): rows per device Mq = S/G not a multiple of 32.

Destination q's quarter starts at row q*Mq, mid-tile; the port reader and the local-quarter reader re-tile it.
Checks: the golden INPUTS shapes against the reference, a multi-batch case, and bit-exact row provenance through an
identity weight (the re-tile copies stored bytes; row q*Mq lands at row 0 of the destination's first tile).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.all_to_all_matmul.helpers import (
    _tolerance_for,
    check_output,
    create_ttnn_input_tensor,
    output_shape,
    pytorch_all_to_all_matmul,
    read_back,
    replicate,
)
from ttnn.operations.all_to_all_matmul import all_to_all_matmul, default_compute_kernel_config

CLUSTER_AXIS = 1
G = 4


def _call(mesh_device, x, w):
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(replicate(w, mesh_device), mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return all_to_all_matmul(
        tx,
        tw,
        cluster_axis=CLUSTER_AXIS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        compute_kernel_config=default_compute_kernel_config(),
    )


SHAPES = [
    pytest.param((1, 24, 160, 128), (1, 96, 128, 512), id="kimi_heads_40_rows"),  # Mq 40: 2 tile rows, shifts 0/8/16/24
    pytest.param((1, 8, 96, 128), (1, 32, 128, 512), id="24_rows"),  # Mq 24 < one tile
    pytest.param((1, 8, 48, 64), (1, 32, 64, 128), id="12_rows_input_1p5_tiles"),  # S 48 not tile-aligned
    pytest.param((2, 4, 200, 64), (1, 16, 64, 128), id="multi_batch_50_rows"),  # B 2, quarter spans 3 source rows
]


@pytest.mark.parametrize("x_shape,w_shape", SHAPES)
def test_all_to_all_matmul_non_aligned(mesh_device, x_shape, w_shape):
    assert tuple(mesh_device.shape)[CLUSTER_AXIS] == G
    torch.manual_seed(42)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *x_shape), dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn(w_shape, dtype=torch.float32) / w_shape[-2] ** 0.5).to(torch.bfloat16)
    out = _call(mesh_device, x, w)
    check_output(
        out,
        pytorch_all_to_all_matmul(x, w, CLUSTER_AXIS),
        shape=output_shape(x_shape, w_shape, G),
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        group=G,
        tolerance=_tolerance_for(ttnn.bfloat16, True),
        fp32_dest_acc_en=True,
    )


@pytest.mark.parametrize("B,Hl,S,K", [(1, 8, 160, 64), (2, 4, 48, 32)], ids=["40_rows", "b2_12_rows"])
def test_non_aligned_row_provenance_bit_exact(mesh_device, B, Hl, S, K):
    """Identity weight + closed-form rows (even columns: the source row index; odd columns: source chip / head /
    batch id; all exact small integers in bf16): the output equals the exchanged rows bit for bit."""
    H = G * Hl
    rows, cols = tuple(mesh_device.shape)
    x = torch.zeros((rows, cols, B, Hl, S, K), dtype=torch.float32)
    x[..., 0::2] = torch.arange(S, dtype=torch.float32).view(S, 1) + 1
    for c in range(cols):
        for j in range(Hl):
            for b in range(B):
                x[:, c, b, j, :, 1::2] = -(1 + c * Hl * B + j * B + b)
    x = x.to(torch.bfloat16)
    w = torch.eye(K, dtype=torch.float32).expand(1, H, K, K).contiguous().to(torch.bfloat16)
    out = _call(mesh_device, x, w)
    got = read_back(out, (rows, cols))
    want = pytorch_all_to_all_matmul(x, w, CLUSTER_AXIS).to(torch.bfloat16)
    assert got.shape == want.shape, f"{got.shape} vs {want.shape}"
    bad = (got.view(torch.int16) != want.view(torch.int16)).any(dim=-1)  # (R, C, B, H, Mq)
    where = [
        f"dev{c} head{h} b{b}: rows {torch.nonzero(bad[0, c, b, h]).flatten().tolist()} "
        f"got {got[0, c, b, h, :, 0].tolist()} / {got[0, c, b, h, :, 1].tolist()[:2]}"
        for c in range(cols)
        for b in range(B)
        for h in range(H)
        if bad[0, c, b, h].any()
    ]
    assert not where, "re-tiled rows were not moved bit-exactly:\n" + "\n".join(where[:12])
