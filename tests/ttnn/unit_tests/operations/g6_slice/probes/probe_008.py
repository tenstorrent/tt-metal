import torch, ttnn
from ttnn.operations.g6_slice import g6_slice

device = ttnn.open_device(device_id=0, trace_region_size=8 * 1024 * 1024)
try:
    for shape, s, e in [
        ((1, 1, 64, 256), [0, 0, 0, 64], [1, 1, 64, 256]),
        ((1, 1, 128, 64), [0, 0, 17, 0], [1, 1, 82, 64]),
    ]:
        t = torch.randn(shape).to(torch.bfloat16)
        x = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ref = t[tuple(slice(b, en) for b, en in zip(s, e))]
        o1 = g6_slice(x, s, e)
        n = device.num_program_cache_entries()
        o2 = g6_slice(x, s, e)
        print("entries after 2 eager", n, device.num_program_cache_entries())
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        ot = g6_slice(x, s, e)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        r = ttnn.to_torch(ot)
        print(shape, "trace exact", torch.equal(r.view(torch.int16), ref.contiguous().view(torch.int16)))
        ttnn.release_trace(device, tid)
finally:
    ttnn.close_device(device)
