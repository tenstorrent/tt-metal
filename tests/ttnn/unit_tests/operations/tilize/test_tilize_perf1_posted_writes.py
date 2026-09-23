# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""posted_writes bake-off harness (perf_experiments/posted_writes).

TILIZE_PW="baseline,posted,..." picks the kernel-dir variants (kernels_<name>);
TILIZE_PW_CASES="focus,narrow32,..." picks the cases (default: all). Cases are the outer
parametrize so the variants of one case run back to back. Run with --profile for device ns;
a plain run is the bit-exact correctness gate. test_chain runs tilize -> add(out, out) back to
back (the consumer reads the tile writes in the very next program) with fresh random inputs and
buffer reuse, to catch a write that has not landed when the next program starts.
"""

import os
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
import ttnn.operations.tilize.tilize_program_descriptor as pd

HERE = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments/posted_writes"
VARS = os.environ.get("TILIZE_PW", "baseline").split(",")
CHAIN_ITERS = int(os.environ.get("TILIZE_PW_CHAIN", "40"))


def _hs(grid_x, grid_y, shard):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def _mc(name):
    return {
        None: None,
        "dram": ttnn.DRAM_MEMORY_CONFIG,
        "l1": ttnn.L1_MEMORY_CONFIG,
        "hs": _hs(8, 8, (32, 512)),
    }[name]


# id: (shape, in memory config, out memory config)
CASES = {
    "focus": ((1, 1, 16384, 64), "dram", None),
    "tall64": ((1, 1, 32768, 64), "dram", None),
    "narrow32": ((1, 1, 16384, 32), "dram", None),
    "wide256": ((1, 1, 8192, 256), "dram", None),
    "wide1024": ((1, 1, 2048, 1024), "dram", None),
    "tiny": ((1, 1, 128, 64), "dram", None),
    "short_wide": ((1, 1, 32, 2048), "dram", None),
    "l1_interleaved": ((1, 1, 4096, 64), "l1", "l1"),
    "sharded_to_dram": ((1, 1, 2048, 512), "hs", "dram"),
}
SEL = os.environ.get("TILIZE_PW_CASES")
CASE_IDS = SEL.split(",") if SEL else list(CASES)


def _run(device, case, seed):
    shape, in_mc, out_mc = CASES[case]
    torch.manual_seed(seed)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_mc(in_mc))
    return x, t, tilize(t, memory_config=_mc(out_mc))


@pytest.mark.parametrize("variant", VARS)
@pytest.mark.parametrize("case", CASE_IDS)
def test_pw(device, monkeypatch, case, variant):
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / f"kernels_{variant}")
    x, _, out = _run(device, case, 0)
    assert torch.equal(ttnn.to_torch(out), x)


@pytest.mark.parametrize("variant", VARS)
@pytest.mark.parametrize("case", ["focus", "tall64", "tiny", "wide1024"])
def test_chain(device, monkeypatch, case, variant):
    """tilize then an immediate device consumer of its output, many times, fresh data each time."""
    monkeypatch.setattr(pd, "KERNEL_DIR", HERE / f"kernels_{variant}")
    bad = 0
    for i in range(CHAIN_ITERS):
        x, t, out = _run(device, case, 1000 + i)
        y = ttnn.add(out, out)  # next program reads every output tile from DRAM
        z = ttnn.to_torch(y)
        if not torch.equal(z, (x.float() * 2).to(torch.bfloat16)):
            bad += 1
        ttnn.deallocate(y)
        ttnn.deallocate(out)
        ttnn.deallocate(t)
    assert bad == 0, f"{bad}/{CHAIN_ITERS} chained iterations read stale output"
