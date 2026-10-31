# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.18 (WINDOW_KV_FORMAT=bfp8_tiles): the window region's neighbour send (tt/window_send.py) on the spec's whole
mesh, byte for byte against the host: chip q < N - 1's last 4 region pages land in chip q + 1's first 4, the last
chip's kept tail in chip 0's first 4, the last chip's last 4 pages in its kept tail, every other byte unchanged.
Three calls on two regions and two tails (the prepared program with its io addresses patched), host time a
call. No model.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tt/perf_window_tiles.py -s
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.tt.attention import WINDOW_BLOCK_BYTES
from models.demos.deepseek_v41_flash_d_p.tt.window_send import PAGES, WindowSend


def _put(mesh, t: torch.Tensor) -> ttnn.Tensor:
    """[R, C, pages, bytes] uint8 -> one [1, 1, pages, bytes] per chip (R, C)."""
    return ttnn.from_torch(
        t,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, 1)),
    )


def _get(t: ttnn.Tensor, R: int, C: int) -> torch.Tensor:
    shards = [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]
    return torch.stack(shards).reshape(R, C, *shards[0].shape[-2:])


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [harness.device_params(spec())], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("q_rows", [160, 640])  # 8x4 / 2x4 at chunk 5120 (the region: 9 / 24 pages)
def test_window_send(mesh_device, q_rows):
    mesh = mesh_device
    R, C = tuple(mesh.shape)
    n = R * C
    pages = (128 + q_rows) // 32
    send = WindowSend(mesh, pages, WINDOW_BLOCK_BYTES)
    print(f"window send {R}x{C}: {pages} pages, {send.seg} B x {send.nseg} packets a page, routes {send.routes}")
    gen = torch.Generator().manual_seed(18)
    rand = lambda *s: torch.randint(0, 256, s, generator=gen, dtype=torch.uint8)  # noqa: E731
    regions = [_put(mesh, rand(R, C, pages, WINDOW_BLOCK_BYTES)) for _ in range(2)]
    tails = [_put(mesh, rand(R, C, PAGES, WINDOW_BLOCK_BYTES)) for _ in range(2)]
    walls = []
    for call in range(3):
        reg, tail = regions[call % 2], tails[(call // 2) % 2]
        r0, t0 = _get(reg, R, C).reshape(n, pages, -1), _get(tail, R, C).reshape(n, PAGES, -1)
        ttnn.synchronize_device(mesh)
        w = time.perf_counter()
        send(reg, tail)
        walls.append((time.perf_counter() - w) * 1e6)
        ttnn.synchronize_device(mesh)
        r1, t1 = _get(reg, R, C).reshape(n, pages, -1), _get(tail, R, C).reshape(n, PAGES, -1)
        want = r0.clone()
        for q in range(n):
            src = t0[n - 1] if q == 0 else r0[q - 1, pages - PAGES :]
            want[q, :PAGES] = src
        want_t = t0.clone()
        want_t[n - 1] = r0[n - 1, pages - PAGES :]
        bad = [q for q in range(n) if not torch.equal(r1[q], want[q])]
        bad_t = [q for q in range(n) if not torch.equal(t1[q], want_t[q])]
        assert not bad and not bad_t, f"call {call}: region wrong on chips {bad}, tail wrong on chips {bad_t}"
    print(f"window send OK, 3 calls; host enqueue a call {[round(x) for x in walls]} us")
