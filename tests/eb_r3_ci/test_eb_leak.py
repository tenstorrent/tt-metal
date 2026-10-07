# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, third pass: does a block-pack program slow the next program? Each probe test runs once right
after a pre test (pytest order); the probe's own program is the same under every toggle setting. CI only."""
import pytest
import torch
import ttnn

from test_eb_blk4 import _mem, _run


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


PROBES = [("add", "bfp8", "bfp8"), ("mul", "bfp8", "bfp8"), ("add", "bf16", "fp32"), ("add", "bf16", "bfp8")]
CASES = [(pre, i, role) for pre in ("bf16", "fp32", "dram") for i in range(len(PROBES)) for role in ("pre", "probe")]
if __import__("os").environ.get("EB_LEAK2"):
    # leak2: probes whose program is main's under every setting (mixed formats never take the block section), at 4 and 128
    # tiles per core; a spacer (DRAM add) between the pre and the probe; the probe twice after the pre.
    PROBES = [("add", "bf16", "bfp8", "hs8_t4"), ("add", "bf16", "bfp8", "hs8_t128"), ("add", "bf16", "bf16", "dram")]
    CASES = [(pre, i, role) for pre in ("bf16", "dram", "bf16sp") for i in range(len(PROBES)) for role in ("pre", "probe", "probe2")]


@pytest.mark.parametrize("pre, i, role", CASES, ids=[f"{p}-{'-'.join(PROBES[i])}-{r}" for p, i, r in CASES])
def test_leak(device, pre, i, role):
    """role pre: a sharded bf16 add into bf16 or fp32 (the block section with the block pack under the bp settings), or a
    DRAM add (one tile per section, never a block); role probe, the next test in pytest order: the probe op."""
    if role == "pre":
        if pre == "bf16sp":
            shape, mc = _mem("hs8_t4")
            _run(device, "add", shape, mc, "bf16", "bf16", "bf16")
            _run(device, "add", (1, 1, 256, 128), ttnn.DRAM_MEMORY_CONFIG, "bf16", "bf16", "bf16")
            return
        if pre == "dram":
            _run(device, "add", (1, 1, 256, 128), ttnn.DRAM_MEMORY_CONFIG, "bf16", "bf16", "bf16")
        else:
            shape, mc = _mem("hs8_t4")
            _run(device, "add", shape, mc, "bf16", "bf16", "bf16" if pre == "bf16" else "fp32")
        return
    if len(PROBES[i]) == 4:
        op, da, do, mem = PROBES[i]
        shape, mc = ((1, 1, 1024, 1024), ttnn.DRAM_MEMORY_CONFIG) if mem == "dram" else _mem(mem)
        _run(device, op, shape, mc, "bf16", da, do)
        return
    op, d, do = PROBES[i]
    shape, mc = _mem("hs8_t4")
    _run(device, op, shape, mc, d, d, do)
