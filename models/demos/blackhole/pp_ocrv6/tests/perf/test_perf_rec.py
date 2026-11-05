# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.common import DET_L1_SMALL_SIZE, REC_HEIGHT, REC_WIDTH, load_torch_rec
from models.demos.blackhole.pp_ocrv6.tt.ttnn_rec import TtPPOCRv6SmallRec


@pytest.mark.parametrize("device_params", [{"l1_small_size": DET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch", [1, 8])
def test_perf_rec(device, batch, iterations=20):
    tt_model = TtPPOCRv6SmallRec(device, load_torch_rec())
    host_x = ttnn.from_torch(torch.rand(batch, REC_HEIGHT, REC_WIDTH, 3) * 2 - 1, dtype=ttnn.bfloat16)

    out = tt_model(host_x, batch, REC_HEIGHT, REC_WIDTH)
    ttnn.synchronize_device(device)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        ttnn.to_torch(tt_model(host_x, batch, REC_HEIGHT, REC_WIDTH))
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    median = times[len(times) // 2]
    print(
        f"rec batch {batch} end-to-end (host in, host out): median {median:.2f} ms "
        f"({median / batch:.2f} ms per line), min {times[0]:.2f} ms"
    )
