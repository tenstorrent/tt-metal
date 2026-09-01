# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""One signposted, timed warm vsa_sdpa streaming run at the median 15s shard (tracy-compatible)."""

import time

from tracy import signpost

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0

from .test_vsa_sdpa_perf import make_inputs


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
def test_vsa_sdpa_stream_profile(device):
    args = make_inputs(device, s_local=14464, n_blocks=1808, row_blocks=197, dense_rows=0, order="topk")
    q, k, v, idx, counts, flops = args
    out = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts, streaming=True)  # compile
    ttnn.synchronize_device(device)
    signpost("start")
    t0 = time.perf_counter()
    for _ in range(8):
        out = ttnn.transformer.vsa_sdpa(q, k, v, idx, counts, streaming=True)
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) / 8 * 1e3
    signpost("stop")
    grid = device.compute_with_storage_grid_size()
    peak = grid.x * grid.y * 4096 * 1.35e9 / 2
    print(f"\nSTREAM_ONE {ms:.3f} ms   util {flops / (ms * 1e-3) / peak * 100:.2f} %")
    ttnn.deallocate(out)
