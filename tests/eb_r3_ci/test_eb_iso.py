# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, third pass: the rows of the candidate head's A/B that read slower than main's program, alone (no
other binary_ng program before them in the process). Their program is main's under the head's rule. CI only."""
import pytest
import ttnn

from test_eb_blk4 import _mem, _run


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


ISO = [("add", "bs64_t80", "bf16", "bfp8", "bfp8", None, None), ("add", "hs8_t128", "bf16", "bfp8", "bfp8", None, None),
       ("add", "ws32_t4", "fp32", "fp32", "fp32", None, None), ("add", "hs8_t16", "bf16", "bf16", "bfp8", "relu", None),
       ("add", "hs8_t8", "bfp8", "bfp8", "bf16", "relu", None), ("add", "ws32_t4", "bf16", "bf16", "bf16", "relu", None),
       ("add", "hs8_t128", "bfp8", None, "bfp8", None, 0.375)]


@pytest.mark.parametrize("op, mem, da, db, do, act, scalar", ISO, ids=["-".join(str(x) for x in c) for c in ISO])
def test_iso(device, op, mem, da, db, do, act, scalar):
    shape, mc = _mem(mem)
    _run(device, op, shape, mc, da, db, do, act=act, scalar=scalar)
