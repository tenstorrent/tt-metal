# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only (PLAN.md section 21.3, before any kernel of the index query's assembly exists): the piece arithmetic of
kernels/attn_idx_rows.hpp, compiled on the host, against the layouts of the two pieces it joins:
  in    what the chips land: chip h's 4 row tiles of head h's query (fused/projections layout.to_row_tiles), and
        chip 22's head-weight row tile
  out   what the scan reads: a user's query as the 4 tiles of ``Scan.set_query`` in the rows form
        (ScanConfig(query_rows=True): [32 heads x 128 dims] BF16, TILE layout: ttnn's own tiling of a host tensor),
        and the 16 multiplier words ``Scan.set_query`` puts into the step page (unfolded: the head weights)
for 8 and for 16 token rows.

    python -m pytest models/experimental/hy4_preview/fused/layer_attn/tests/test_idx_rows_host.py
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from models.experimental.hy4_preview.fused.projections import layout as L

HERE = Path(__file__).parent
HEADS, DIM = 32, 128


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    exe = tmp_path_factory.mktemp("idx_rows") / "host_idx_rows"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", "-Wall", "-Werror", "-o", str(exe), str(HERE / "host_idx_rows.cpp")], check=True
    )
    return exe


def bf16_words(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint32).copy()


def standard_tiles(x: torch.Tensor) -> np.ndarray:
    """[32, 32 n] BF16 values -> the words of its n standard tiles (4 faces of [16, 16] each, row-major inside a
    face: rows 0..15 x columns 0..15 | rows 0..15 x columns 16..31 | rows 16..31 x 0..15 | rows 16..31 x 16..31)."""
    n = x.shape[1] // 32
    faces = x.reshape(2, 16, n, 2, 16).permute(2, 0, 3, 1, 4)  # [tile, row half, column half, 16, 16]
    return bf16_words(faces.reshape(-1))


@pytest.mark.parametrize("rows", [8, 16])
def test_assembly(binary, tmp_path, rows):
    g = torch.Generator().manual_seed(rows)
    q = (torch.randn(rows, HEADS, DIM, generator=g) * 3).to(torch.bfloat16).float()  # [token row, head, dim]
    w = torch.randn(rows, HEADS, generator=g).to(torch.bfloat16).float()
    landing = np.concatenate([bf16_words(L.to_row_tiles(q[:, h])) for h in range(HEADS)])
    weights = bf16_words(L.to_row_tiles(w))
    assert landing.size == HEADS * 4 * 16 * rows and weights.size == 16 * rows
    (tmp_path / "in.bin").write_bytes(landing.tobytes() + weights.tobytes())
    out = subprocess.run(
        [str(binary), str(rows), str(tmp_path / "in.bin"), str(tmp_path / "out.bin")], capture_output=True, text=True
    )
    assert out.returncode == 0, (out.returncode, out.stderr)
    got = np.fromfile(tmp_path / "out.bin", dtype=np.uint32).reshape(rows, 2048 + 16)
    for u in range(rows):
        want = standard_tiles(q[u])  # [32 heads, 128 dims]: the rows form of the scan's query
        assert np.array_equal(got[u, :2048], want), (rows, u, int((got[u, :2048] != want).sum()))
        bits = w[u].to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF  # (Scan.set_query's packing)
        packed = (bits[0::2] | (bits[1::2] << 16)).numpy().astype(np.uint32)
        assert np.array_equal(got[u, 2048:], packed), (rows, u)
    # ttnn's own tiling of a host tensor agrees with standard_tiles (values through a TILE-layout round trip)
    import ttnn

    t = ttnn.from_torch(q[:1].reshape(1, 1, HEADS, DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    assert torch.equal(ttnn.to_torch(t).float().reshape(HEADS, DIM), q[0])
    print(
        f"HY4_MK8A idx rows host, {rows} token rows: {rows} queries of 256 pieces each and their 32 multipliers as the scan takes them"
    )
