# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58722 review): binary_ng's sharded no-broadcast add and mul over the tiles per core (4 to 128,
height-sharded on 8 cores), a width-sharded decode residual and a block-sharded SDXL-sized tensor, for the block section's
threshold. Toggles: EB_R3_NO_BLOCK=1 (no block section), EB_R3_NO_BLOCK_PACK=1 (block unpack alone)."""
import zlib

import pytest
import torch
import ttnn
from test_eb_fid import DT


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


HS = {4: (256, 128), 8: (512, 128), 16: (1024, 128), 32: (1024, 256), 64: (1024, 512), 128: (1024, 1024)}
OUT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
DTS = [("bf16", "bf16"), ("bfp8", "bfp8"), ("bfp8", "bf16"), ("bf16", "fp32")]
CASES = [(op, f"hs8_t{t}", di, do) for op in ("add", "sub", "mul") for t in HS for (di, do) in DTS if not (do == "fp32" and t > 64)]
CASES += [(op, m, di, do) for op in ("add", "sub", "mul") for m in ("ws32_t4", "bs64_t80") for (di, do) in DTS if not (do == "fp32" and m == "bs64_t80")]


def _mem(name):
    if name.startswith("hs8"):
        h, w = HS[int(name.split("_t")[1])]
        shape = (1, 1, h, w)
        return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=2, x=4), strategy=ttnn.ShardStrategy.HEIGHT)
    if name == "ws32_t4":
        shape = (1, 1, 32, 4096)
        return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.WIDTH)
    shape = (1, 1, 4096, 1280)
    return shape, ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK)


@pytest.mark.parametrize("op, mem, di, do", CASES, ids=["-".join(c) for c in CASES])
def test_block2(device, op, mem, di, do):
    shape, mc = _mem(mem)
    torch.manual_seed(zlib.crc32(f"{op}{mem}{di}{do}".encode()) % 100000)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = torch.randn(shape, dtype=torch.bfloat16)
    ta = ttnn.from_torch(a, dtype=DT[di], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(b, dtype=DT[di], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    kw = dict(dtype=OUT[do], memory_config=mc)
    if op == "mul":
        kw["fast_and_approximate_mode"] = True
    fn = {"add": ttnn.add, "sub": ttnn.subtract, "mul": ttnn.multiply}[op]
    for _ in range(3):
        out = fn(ta, tb, **kw)
    got = ttnn.to_torch(out).float()
    ref = {"add": torch.add, "sub": torch.sub, "mul": torch.mul}[op](ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float())
    assert torch.allclose(got, ref, rtol=0.05, atol=0.1), float((got - ref).abs().max())
