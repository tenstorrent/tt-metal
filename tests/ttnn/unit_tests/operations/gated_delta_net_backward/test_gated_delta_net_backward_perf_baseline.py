# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warm-cache latency baseline + no-regression guard set for
gated_delta_net_backward.

Why it exists: the refinement queue (`op_requirements.md`) files *measured*
perf refinements, and every one of them is gated on "no regression across the
config-spanning guard set".  This file IS that guard set — one representative
per distinct kernel path:

    path                                   representative
    ------------------------------------   -------------------------------
    num_v_blocks == 1, chunk 32, square    (1,512,2,64,64) c32   (16 chunks)
    num_v_blocks == 1, chunk 64, wide_v    (1,128,2,64,128) c64
    num_v_blocks  > 1, chunk 64, wide_v    (1,256,4,128,256) c64 (Vb < Vt)
    ragged tail                            (1,100,2,64,64) c64
    multi-batch / multi-head               (2,64,4,64,64) c32
    fp32 and bfloat16 boundary formats     both dtypes on one shape

What it measures: WALL-CLOCK per invocation with the program cache warm and
`synchronize_device` after each call — this build's real-time device profiler
is inactive (`IsProgramRealtimeProfilerActive() == False`), so per-program
device ns is unavailable here and warm host-to-host latency is the honest
proxy.  The op is ONE dispatch, so the number is dominated by device time on
every shape big enough to matter.  It is a RELATIVE instrument: compare a
refinement's table against the table in `verification_report.md`, taken on the
same machine.

MEASURED RUN-TO-RUN DRIFT IS ~3% on `min` (device clock / dispatch state), so a
perf refinement must clear ~5% on `min`, or repeat the file 3x and compare the
best of the three.  Anything smaller is not distinguishable here.

Nothing here asserts a time (that would be a flaky test).  The assertion is
only that the op still runs and returns six gradients; run with `-s` to read
the table.
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

import ttnn
from ttnn.operations.gated_delta_net_backward import gated_delta_net_backward

from eval.golden_tests.gated_delta_net_backward.helpers import make_reference_inputs

TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}

WARMUP = 3
ITERS = 20

# (B, T, H, K, V), chunk_size, dtype — the config-spanning guard set.
CASES = [
    ((1, 512, 2, 64, 64), 32, ttnn.float32),
    ((1, 128, 2, 64, 128), 64, ttnn.float32),
    ((1, 256, 4, 128, 256), 64, ttnn.float32),
    ((1, 100, 2, 64, 64), 64, ttnn.float32),
    ((2, 64, 4, 64, 64), 32, ttnn.float32),
    ((1, 256, 4, 128, 256), 64, ttnn.bfloat16),
]
CASE_IDS = [
    f"B{s[0]}_T{s[1]}_H{s[2]}_K{s[3]}_V{s[4]}_c{c}_{'fp32' if d == ttnn.float32 else 'bf16'}" for s, c, d in CASES
]


def _to_device(tensor, device, dtype):
    if tensor is None:
        return None
    return ttnn.from_torch(
        tensor.to(TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("shape,chunk_size,dtype", CASES, ids=CASE_IDS)
def test_perf_baseline(shape, chunk_size, dtype, device):
    # q/k L2-normalized by make_reference_inputs — caller contract, and the
    # forward diverges without it even though this test only times.
    ref = make_reference_inputs(shape, state_mode="with_h0_and_dht", seed=0, g_scale=0.02)
    args = [_to_device(ref[n], device, dtype) for n in ("q", "k", "v", "g", "beta", "do")]
    kwargs = dict(
        dht=_to_device(ref["dht"], device, dtype),
        initial_state=_to_device(ref["h0"], device, dtype),
        chunk_size=chunk_size,
    )

    for _ in range(WARMUP):
        out = gated_delta_net_backward(*args, **kwargs)
    ttnn.synchronize_device(device)
    assert len(out) == 6 and all(t is not None for t in out)

    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        gated_delta_net_backward(*args, **kwargs)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1e3)

    B, T, H, K, V = shape
    print(
        f"\nPERF {CASE_IDS[CASES.index((shape, chunk_size, dtype))]:34s} "
        f"median {statistics.median(samples):8.3f} ms  min {min(samples):8.3f} ms  "
        f"max {max(samples):8.3f} ms   (BH*NC = {B * H * ((T + chunk_size - 1) // chunk_size)} items)"
    )
