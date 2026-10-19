# Scratch, not for merge. binary_ng FPU scalar ops (fast_and_approximate_mode) at interleaved and sharded shapes for bf16 and
# bfp8_b inputs; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

CASES = {  # name: (shape, sharding or memory)
    "dram_1k": ((1, 1, 1024, 1024), "dram"),
    "l1_32x4k": ((1, 1, 32, 4096), "l1"),
    "dec_w32": ((1, 1, 32, 4096), (ttnn.ShardStrategy.WIDTH, 8, 4)),  # 4 tiles per core
    "h8_16": ((1, 1, 4096, 32), (ttnn.ShardStrategy.HEIGHT, 8, 1)),  # 16
    "pre_b32": ((1, 1, 1024, 896), (ttnn.ShardStrategy.BLOCK, 4, 8)),  # 28
    "blk64": ((1, 1, 2048, 1024), (ttnn.ShardStrategy.BLOCK, 8, 8)),  # 32
}
MIX = {"bf16": (ttnn.bfloat16, ttnn.bfloat16), "bfp8": (ttnn.bfloat8_b, ttnn.bfloat8_b), "bf16-bfp8": (ttnn.bfloat16, ttnn.bfloat8_b)}
OPS = {
    "add": lambda x, **k: ttnn.add(x, 1.5, **k),
    "sub": lambda x, **k: ttnn.subtract(x, 1.5, **k),
    "mul": lambda x, **k: ttnn.multiply(x, 1.5, **k),
    "rsub": lambda x, **k: ttnn.rsub(x, 1.5, **k),
}


@pytest.mark.parametrize("mix", list(MIX))
@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("case", list(CASES))
def test_scalar(device, case, op, mix, request):
    shape, where = CASES[case]
    din, dout = MIX[mix]
    if where == "dram":
        mem = ttnn.DRAM_MEMORY_CONFIG
    elif where == "l1":
        mem = ttnn.L1_MEMORY_CONFIG
    else:
        strat, gx, gy = where
        mem = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=strat)
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(shape) * 0.5, dtype=din, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    out = None
    for _ in range(4):
        out = OPS[op](x, dtype=dout, memory_config=mem, fast_and_approximate_mode=True)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(out).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")
