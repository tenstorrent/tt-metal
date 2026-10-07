# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58722 review): binary_ng's sharded no-broadcast add, sub and mul, the ops whose DEST section holds
8 tiles (4 with fp32 DEST), at the sharded shapes of the HiFi2 rule. EB_R3_NO_BLOCK=1 gives the branch without the block
section (block unpack and block pack), EB_R3_NO_BLOCK_PACK=1 the block unpack alone."""
import zlib

import pytest
import torch
import ttnn
from test_eb_fid import DT, _mem


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


SHAPES = {"hs8": (1, 1, 1024, 1024), "bs64": (1, 1, 4096, 1280), "ws32": (1, 1, 32, 4096)}
OUT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
CASES = [
    (op, mem, da, db, do)
    for op in ("add", "sub", "mul")
    for mem in ("hs8", "bs64", "ws32")
    for (da, db, do) in (("bf16", "bf16", "bf16"), ("bfp8", "bfp8", "bfp8"), ("bfp8", "bfp8", "bf16"), ("bf16", "bf16", "fp32"))
] + [("mul", mem, "bf16", "bfp8", "bf16") for mem in ("hs8", "bs64")]


@pytest.mark.parametrize("op, mem, da, db, do", CASES, ids=["-".join(c) for c in CASES])
def test_block(device, op, mem, da, db, do):
    shape = SHAPES[mem]
    torch.manual_seed(zlib.crc32(f"{op}{mem}{da}{db}{do}".encode()) % 100000)
    a = torch.randn(shape, dtype=torch.bfloat16) * 3
    b = torch.randn(shape, dtype=torch.bfloat16) * 3
    mc = _mem(mem, shape)
    ta = ttnn.from_torch(a, dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, dtype=DT[db], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    fn = {"add": ttnn.add, "sub": ttnn.subtract, "mul": ttnn.multiply}[op]
    kw = dict(dtype=OUT[do], memory_config=mc)
    if op == "mul":
        kw["fast_and_approximate_mode"] = True
    for _ in range(3):
        out = fn(ta, tb, **kw)
    got = ttnn.to_torch(out).float()
    ref = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op](ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float())
    assert torch.allclose(got, ref, rtol=0.05, atol=0.5), (got - ref).abs().max()
