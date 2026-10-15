# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cost of the prefill kv-table latent FIFO shift (tt/prefill_sparse.py attend_dyn, in 16384-row pieces) at long-context table sizes: ms per shift (one layer, one chunk)
for L rows of 512 bf16 and a chunk of Cc latents. Prints FS lines. Env: FS_L (comma list of table rows), FS_CC (default 32 = chunk 128 / ratio 4)."""

import os
import time

import pytest

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import GATE_ROWS


def shift(kt, lat, L, Cc):
    for a in range(0, L - Cc, GATE_ROWS):
        b = min(L - Cc, a + GATE_ROWS)
        piece = ttnn.slice(kt, [0, 0, a + Cc, 0], [1, 1, b + Cc, 512])
        ttnn.experimental.slice_write(piece, kt, [0, 0, a, 0], [1, 1, b, 512], [1, 1, 1, 1])
        ttnn.deallocate(piece)
    ttnn.experimental.slice_write(lat, kt, [0, 0, L - Cc, 0], [1, 1, L, 512], [1, 1, 1, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_fifo_shift(mesh_device):
    md = mesh_device
    Cc = int(os.environ.get("FS_CC", "32"))
    for L in [int(x) for x in os.environ.get("FS_L", "32768,131072,262144,524288").split(",")]:
        kt = ttnn.zeros([1, 1, L, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
        lat = ttnn.zeros([1, 1, Cc, 512], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=md)
        shift(kt, lat, L, Cc)
        ttnn.synchronize_device(md)
        t = time.perf_counter()
        for _ in range(3):
            shift(kt, lat, L, Cc)
        ttnn.synchronize_device(md)
        ms = (time.perf_counter() - t) / 3 * 1e3
        print(
            f"FS L={L} rows ({L * 1024 / 2**20:.0f} MiB): shift {ms:.2f} ms ({2 * L * 1024 / ms / 1e6:.0f} GB/s read+write)",
            flush=True,
        )
        ttnn.deallocate(kt)
