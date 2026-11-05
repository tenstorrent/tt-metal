# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch

import ttnn
from models.demos.blackhole.pp_ocrv6.common import DET_L1_SMALL_SIZE, load_torch_det
from models.demos.blackhole.pp_ocrv6.tt.ttnn_det import TtPPOCRv6SmallDet


@pytest.mark.parametrize("device_params", [{"l1_small_size": DET_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("size", [640])
def test_perf_det(device, size, iterations=20):
    tt_model = TtPPOCRv6SmallDet(device, load_torch_det())
    host_x = ttnn.from_torch(torch.randn(1, size, size, 3), dtype=ttnn.bfloat16)

    out, _, _ = tt_model(host_x, 1, size, size)
    ttnn.synchronize_device(device)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        out, _, _ = tt_model(host_x, 1, size, size)
        ttnn.to_torch(out)
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    print(
        f"det {size}x{size} end-to-end (host in, host out): median {times[len(times) // 2]:.2f} ms, min {times[0]:.2f} ms"
    )
