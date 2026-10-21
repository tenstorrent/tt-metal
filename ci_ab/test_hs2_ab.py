# Scratch, not for merge. hardswish at more dtypes and shard sizes; each output's sha1 goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


@pytest.mark.parametrize("dt", list(DT))
@pytest.mark.parametrize("shape", [(1, 1, 128, 2048), (1, 1, 256, 2048), (1, 1, 64, 512)])
def test_hs2(device, dt, shape, request):
    torch.manual_seed(0)
    cores = 8
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))])
    mc = ttnn.create_sharded_memory_config(
        shape=(shape[-2], shape[-1] // cores), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    x = ttnn.from_torch(torch.randn(*shape).bfloat16() * 4, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=mc)
    out = None
    for _ in range(2):
        out = ttnn.hardswish(x, memory_config=mc)
    ttnn.synchronize_device(device)
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")
