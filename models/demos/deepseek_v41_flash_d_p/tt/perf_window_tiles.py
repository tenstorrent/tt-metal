# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.18 (WINDOW_KV_FORMAT=bfp8_tiles): the window region's neighbour send (tt/window_send.py) on the spec's whole
mesh, byte for byte against the host: chip q < N - 1's last 4 region pages land in chip q + 1's first 4, the last
chip's kept tail in chip 0's first 4, the last chip's last 4 pages in its kept tail, every other byte unchanged.
Three calls on two regions and two tails (the prepared program with its io addresses patched), host time a
call. No model. P.26 (WINDOW_KV_FORMAT=bf16_tiles): the same on a bf16 ROW_MAJOR region (one row a page, the carry
128 row pages, ``fmt`` bf16), and the window region's memory a chip in both formats (bytes of the buffers).

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
    """[R, C, pages, bytes] uint8 (or [R, C, rows, 512] bf16) -> one [1, 1, pages, bytes] per chip (R, C)."""
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16 if t.dtype == torch.bfloat16 else ttnn.uint8,
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
@pytest.mark.parametrize("fmt", ["bfp8", "bf16"])  # bfp8_tiles (P.18) / bf16_tiles (P.26)
def test_window_send(mesh_device, q_rows, fmt):
    mesh = mesh_device
    R, C = tuple(mesh.shape)
    n = R * C
    gen = torch.Generator().manual_seed(18)
    if fmt == "bf16":  # one 512-wide bf16 row a page
        pages, cp, width = 128 + q_rows, 128, 512
        send = WindowSend(mesh, pages, width * 2, cp)
        rand = lambda *s: torch.randn(s, generator=gen).to(torch.bfloat16)  # noqa: E731
    else:
        pages, cp, width = (128 + q_rows) // 32, PAGES, WINDOW_BLOCK_BYTES
        send = WindowSend(mesh, pages, WINDOW_BLOCK_BYTES)
        rand = lambda *s: torch.randint(0, 256, s, generator=gen, dtype=torch.uint8)  # noqa: E731
    print(
        f"window send {fmt} {R}x{C}: {pages} pages, {send.seg} B x {send.nseg} packets a page, batch {send.batch}, "
        f"routes {send.routes}"
    )
    regions = [_put(mesh, rand(R, C, pages, width)) for _ in range(2)]
    tails = [_put(mesh, rand(R, C, cp, width)) for _ in range(2)]
    nbytes = lambda t: int(t.shape[-2]) * int(t.shape[-1]) * (2 if t.dtype == ttnn.bfloat16 else 1)  # noqa: E731
    print(
        f"window region memory a chip ({fmt}, q_rows {q_rows}): region {nbytes(regions[0])} B a layer "
        f"({40 * nbytes(regions[0]) / 1e6:.2f} MB for 40 layers), kept tail {nbytes(tails[0])} B a layer a slot"
    )
    PAGES_ = cp
    walls, syncs = [], []
    for call in range(3):
        reg, tail = regions[call % 2], tails[(call // 2) % 2]
        r0, t0 = _get(reg, R, C).reshape(n, pages, -1), _get(tail, R, C).reshape(n, PAGES_, -1)
        ttnn.synchronize_device(mesh)
        w = time.perf_counter()
        send(reg, tail)
        walls.append((time.perf_counter() - w) * 1e6)
        ttnn.synchronize_device(mesh)
        syncs.append((time.perf_counter() - w) * 1e6)
        r1, t1 = _get(reg, R, C).reshape(n, pages, -1), _get(tail, R, C).reshape(n, PAGES_, -1)
        want = r0.clone()
        for q in range(n):
            src = t0[n - 1] if q == 0 else r0[q - 1, pages - PAGES_ :]
            want[q, :PAGES_] = src
        want_t = t0.clone()
        want_t[n - 1] = r0[n - 1, pages - PAGES_ :]
        bad = [q for q in range(n) if not torch.equal(r1[q], want[q])]
        bad_t = [q for q in range(n) if not torch.equal(t1[q], want_t[q])]
        assert not bad and not bad_t, f"call {call}: region wrong on chips {bad}, tail wrong on chips {bad_t}"
    print(
        f"window send {fmt} OK, 3 calls; host enqueue a call {[round(x) for x in walls]} us, enqueue + sync "
        f"{[round(x) for x in syncs]} us"
    )
