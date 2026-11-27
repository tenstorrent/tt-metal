# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 knob sweep (perf tooling, opt-in): set G6E_SWEEP=1 and run under
`scripts/run_safe_pytest.sh --profile --no-py-zones`. Each case sets the descriptor knobs,
drops the descriptor cache, runs G6E_SWEEP_REPS calls (default 2) and checks bit-exactness.
G6E_SWEEP_REPS ops_perf_results rows per case, in collection order.
Configs: G6E_SWEEP_CONFIGS="max_cores:eighths:stride:cpu_rows[:half_tile[:fetch_reqs[:fanout[:split]]]],..."
(max_cores 0 = whole grid; omitted trailing fields keep the descriptor defaults).
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_expand import g6_expand
import ttnn.operations.g6_expand.g6_expand_program_descriptor as pd

_DEFAULT = "0:4:1:8"
_CONFIGS = [c for c in os.environ.get("G6E_SWEEP_CONFIGS", _DEFAULT).split(",") if c]
_REPS = int(os.environ.get("G6E_SWEEP_REPS", "2"))
_WIDTHS = [int(w) for w in os.environ.get("G6E_SWEEP_WIDTHS", "512").split(",")]


@pytest.mark.skipif(not os.environ.get("G6E_SWEEP"), reason="perf sweep tooling; set G6E_SWEEP=1")
@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("cfg", _CONFIGS)
def test_g6_expand_knob_sweep(device, cfg, width):
    vals = [int(v) for v in cfg.split(":")]
    max_cores, eighths, stride, cpu_rows = vals[:4]
    saved = (
        pd.MAX_CORES,
        pd.READER_WRITE_EIGHTHS,
        pd.CORE_ORDER_STRIDE,
        pd.REPLICATE_CPU_ROWS,
        pd.HALF_TILE,
        pd.ROW_FETCH_REQUESTS,
        pd.REPLICATE_NOC_FANOUT,
        pd.REPLICATE_SPLIT,
    )
    if len(vals) > 7:
        pd.REPLICATE_SPLIT = vals[7]
    if len(vals) > 6:
        pd.REPLICATE_NOC_FANOUT = vals[6]
    if len(vals) > 4:
        pd.HALF_TILE = vals[4]
    if len(vals) > 5:
        pd.ROW_FETCH_REQUESTS = vals[5]
    pd.MAX_CORES = max_cores or None
    pd.READER_WRITE_EIGHTHS = eighths
    pd.CORE_ORDER_STRIDE = stride
    pd.REPLICATE_CPU_ROWS = cpu_rows
    pd._cache.clear()
    try:
        torch.manual_seed(0)
        x = torch.randn((1, 1, 1, width), dtype=torch.float32).to(torch.bfloat16)
        tt_x = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(_REPS):
            out = g6_expand(tt_x, [1, 1, width, width])
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(out)
        assert torch.equal(got, x.expand(1, 1, width, width)), cfg
        print(f"G6E_SWEEP cfg={cfg} width={width} plan={dict(pd._cache[next(iter(pd._cache))][1])}")
    finally:
        (
            pd.MAX_CORES,
            pd.READER_WRITE_EIGHTHS,
            pd.CORE_ORDER_STRIDE,
            pd.REPLICATE_CPU_ROWS,
            pd.HALF_TILE,
            pd.ROW_FETCH_REQUESTS,
            pd.REPLICATE_NOC_FANOUT,
            pd.REPLICATE_SPLIT,
        ) = saved
        pd._cache.clear()
