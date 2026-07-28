# SPDX-License-Identifier: Apache-2.0
"""Device-time measurement of the [32,32] <-> [1,32] tile bridge, for the profiler.

Wall-clock timing of these conversions turned out to be dispatch-bound (inbound and
outbound produced identical times, which cannot be real), so the only trustworthy
number comes from the device profiler. Run under tracy:

    python3 -m tracy -p -r -v --op-support-count 5000 -o <out> -m \
      "pytest .../bench_tile_convert_prof.py -s -q"

Deliberately few iterations: the earlier attempt fired ~40k ops and crashed the
profiler. ITERS per shape is enough to get a stable median from the CSV.

Shapes are gemma2-9B decode: K=3584 into QKV/FF1/FF3, N=14336 out of FF1/FF3,
N=8192 out of QKV, K=4096 into WO, N=3584 out of WO and FF2.
"""
import torch
from loguru import logger

import ttnn

ITERS = 20
INBOUND_WIDTHS = [3584, 4096, 14336]
OUTBOUND_WIDTHS = [3584, 8192, 14336]


def test_convert_ops(device):
    for w in INBOUND_WIDTHS:
        src = ttnn.from_torch(
            torch.randn(1, 1, 32, w),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for _ in range(ITERS):
            out = ttnn.untilize_with_unpadding(src, [0, 0, 0, w - 1])
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        logger.info(f"MARK inbound width={w}")
        ttnn.deallocate(src)

    for w in OUTBOUND_WIDTHS:
        row = ttnn.from_torch(
            torch.randn(1, 1, 1, w),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for _ in range(ITERS):
            out = ttnn.tilize_with_val_padding(row, [1, 1, 32, w], 0.0)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        logger.info(f"MARK outbound width={w}")
        ttnn.deallocate(row)
