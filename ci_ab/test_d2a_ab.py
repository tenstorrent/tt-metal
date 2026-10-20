# Scratch, not for merge. Ternary addcmul/addcdiv and hardswish at a few shapes (ELF controls for the dest-reuse lever); sha1 of each output to $EB_BITS_OUT.
import hashlib
import os

import pytest
import torch
import ttnn


def record(request, out):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(out).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


def dev(shape, device, dtype, seed):
    torch.manual_seed(seed)
    return ttnn.from_torch(torch.rand(shape) + 0.5, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("shape", [(1, 1, 1024, 1024), (1, 4, 2048, 512)], ids=["1k", "8k"])
@pytest.mark.parametrize("op", ["addcmul", "addcdiv", "hardswish"])
def test_d2a(device, op, shape, dtype, request):
    a, b, c = (dev(shape, device, dtype, s) for s in (1, 2, 3))
    out = None
    for _ in range(2):
        out = ttnn.hardswish(a) if op == "hardswish" else getattr(ttnn, op)(a, b, c, value=2.0)
    record(request, out)
