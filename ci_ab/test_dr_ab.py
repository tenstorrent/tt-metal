# Scratch, not for merge. Dest-reuse callers at op level: hardswish, addcmul, batch_norm (L1 sharded and DRAM); each output's sha1
# goes to $EB_BITS_OUT as "<arm> <node> <sha1>".
import hashlib
import os

import pytest
import torch
import ttnn

MEM = {"dram": None, "l1sh": "sharded"}


def shard_cfg(device, rows, cols, cores):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))])
    return ttnn.create_sharded_memory_config(
        shape=(rows, cols // cores), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)


def mk(device, shape, mem, dtype=ttnn.bfloat16):
    if mem == "dram":
        mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        mc = shard_cfg(device, shape[-2], shape[-1], 8)
    return ttnn.from_torch(torch.randn(*shape).bfloat16(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=mc), mc


def emit(out, request):
    p = os.environ.get("EB_BITS_OUT")
    if p:
        h = hashlib.sha1(ttnn.to_torch(ttnn.from_device(out)).float().numpy().tobytes()).hexdigest()
        with open(p, "a") as f:
            f.write(f"{os.environ.get('EB_ARM', '?')} {request.node.nodeid} {h}\n")


@pytest.mark.parametrize("mem", ["dram", "l1sh"])
@pytest.mark.parametrize("shape", [(1, 1, 256, 2048), (1, 1, 1024, 2048)])
def test_hardswish(device, mem, shape, request):
    torch.manual_seed(0)
    x, mc = mk(device, shape, mem)
    out = None
    for _ in range(2):
        out = ttnn.hardswish(x, memory_config=mc)
    ttnn.synchronize_device(device)
    emit(out, request)


@pytest.mark.parametrize("mem", ["dram", "l1sh"])
@pytest.mark.parametrize("shape", [(1, 1, 256, 2048)])
def test_addcmul(device, mem, shape, request):
    torch.manual_seed(0)
    a, mc = mk(device, shape, mem)
    b, _ = mk(device, shape, mem)
    c, _ = mk(device, shape, mem)
    out = None
    for _ in range(2):
        out = ttnn.addcmul(a, b, c, value=0.5, memory_config=mc)
    ttnn.synchronize_device(device)
    emit(out, request)


@pytest.mark.parametrize("shape", [(2, 32, 64, 64)])
def test_batch_norm(device, shape, request):
    torch.manual_seed(0)
    N, C, H, W = shape
    x = ttnn.from_torch(torch.randn(*shape).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    st = lambda: ttnn.from_torch(torch.rand(1, C, 1, 1).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                 device=device)
    rm, rv, w, b = st(), st(), st(), st()
    out = None
    for _ in range(2):
        out = ttnn.batch_norm(x, running_mean=rm, running_var=rv, weight=w, bias=b, eps=1e-5, training=False)
    ttnn.synchronize_device(device)
    emit(out, request)
