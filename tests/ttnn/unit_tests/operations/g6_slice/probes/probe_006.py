import torch, ttnn
from ttnn.operations.g6_slice import g6_slice
import importlib

m = importlib.import_module("ttnn.operations.g6_slice.g6_slice")

device = ttnn.open_device(device_id=0)
try:

    def mk(seed, shape):
        torch.manual_seed(seed)
        t = torch.randn(shape).to(torch.bfloat16)
        return t, ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    shape = (1, 1, 96, 256)
    s, e = [0, 0, 17, 32], [1, 1, 90, 200]
    ta, xa = mk(1, shape)
    tb, xb = mk(2, shape)
    m._PLAN_CACHE.clear()
    oa = g6_slice(xa, s, e)
    entries = len(m._PLAN_CACHE)
    ob = g6_slice(xb, s, e)
    assert len(m._PLAN_CACHE) == entries == 1
    ref_a = ta[0:1, 0:1, 17:90, 32:200]
    ref_b = tb[0:1, 0:1, 17:90, 32:200]
    ra, rb = ttnn.to_torch(oa), ttnn.to_torch(ob)
    print("addr", xa.buffer_address(), xb.buffer_address(), oa.buffer_address(), ob.buffer_address())
    print("a exact", torch.equal(ra.view(torch.int16), ref_a.view(torch.int16)))
    print("b exact", torch.equal(rb.view(torch.int16), ref_b.view(torch.int16)))
    d = m._PLAN_CACHE[next(iter(m._PLAN_CACHE))]
    print("cached reader addr", d.kernels[0].common_runtime_args[0], "writer addr", d.kernels[1].common_runtime_args[0])
finally:
    ttnn.close_device(device)
