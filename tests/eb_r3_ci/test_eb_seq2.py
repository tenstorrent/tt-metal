# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, third pass: a sharded bf16 residual add (16 or 128 tiles per core; the head's block section with
main's block pack) followed by a small sharded binary op (4 or 8 tiles per core), as a decode graph runs them. Both are
binary_ng launches, so the reduce reports the sequence's total device time per test. CI only."""
import pytest
import torch
import ttnn

from test_eb_blk4 import DT, _mem


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _t(device, shape, dtype, mc):
    return ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=DT[dtype], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)


FOLLOW = [("add", "bf16", "bf16", "bf16"), ("mul", "bf16", "bf16", "bf16"), ("add", "bf16", "bfp8", "bfp8"), ("mul", "bfp8", "bfp8", "bfp8"),
          ("add", "bfp8", "bfp8", "bfp8")]
CASES = [(pre, fm, f) for pre in ("hs8_t16", "hs8_t128", "none") for fm in ("hs8_t4", "hs8_t8") for f in range(len(FOLLOW))]


@pytest.mark.parametrize("pre, fmem, f", CASES, ids=[f"{p}-{fm}-{'-'.join(FOLLOW[f])}" for p, fm, f in CASES])
def test_seq2(device, pre, fmem, f):
    op, da, db, do = FOLLOW[f]
    fshape, fmc = _mem(fmem)
    x, y = _t(device, fshape, da, fmc), _t(device, fshape, db, fmc)
    ts = [x, y]
    if pre != "none":
        shape, mc = _mem(pre)
        a, b = _t(device, shape, "bf16", mc), _t(device, shape, "bf16", mc)
        ts += [a, b]
    ttnn.synchronize_device(device)
    if pre != "none":
        ts.append(ttnn.add(a, b, memory_config=mc))
    fn = ttnn.add if op == "add" else (lambda p, q, **kw: ttnn.multiply(p, q, fast_and_approximate_mode=True, **kw))
    ts.append(fn(x, y, dtype=DT[do], memory_config=fmc))
    ttnn.synchronize_device(device)
    for t in ts:
        ttnn.deallocate(t)
