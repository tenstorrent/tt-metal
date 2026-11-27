import torch, ttnn
from ttnn.operations.g6_indexed_fill import g6_indexed_fill

device = ttnn.open_device(device_id=0, trace_region_size=1 << 20)


def dev(t):
    return ttnn.from_torch(
        t.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


try:
    # stock op under the harness's L1-capture condition (program cache disabled + cleared)
    a = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    device.disable_and_clear_program_cache()
    ttnn.add(a, a)
    tid = ttnn.begin_trace_capture(device)
    try:
        ttnn.add(a, a)
        print("RESULT stock add: traced OK with program cache disabled")
    except Exception as e:
        print("RESULT stock add: FAILS with program cache disabled:", str(e).splitlines()[0][:120])
    finally:
        ttnn.end_trace_capture(device, tid)
        ttnn.release_trace(device, tid)
    # this op, same condition
    ids = dev(torch.tensor([2]))
    x = dev(torch.zeros(7, 1, 1, 32, dtype=torch.int64))
    s = dev(torch.ones(1, 1, 1, 32, dtype=torch.int64))
    g6_indexed_fill(ids, x, s)
    tid = ttnn.begin_trace_capture(device)
    try:
        g6_indexed_fill(ids, x, s)
        print("RESULT g6: traced OK with program cache disabled")
    except Exception as e:
        print("RESULT g6: FAILS with program cache disabled:", str(e).splitlines()[0][:120])
    finally:
        ttnn.end_trace_capture(device, tid)
        ttnn.release_trace(device, tid)
    # this op with the cache enabled again
    device.enable_program_cache()
    g6_indexed_fill(ids, x, s)
    tid = ttnn.begin_trace_capture(device)
    try:
        g6_indexed_fill(ids, x, s)
        print("RESULT g6: traced OK with program cache enabled")
    except Exception as e:
        print("RESULT g6: FAILS with program cache enabled:", str(e).splitlines()[0][:120])
    finally:
        ttnn.end_trace_capture(device, tid)
        ttnn.release_trace(device, tid)
finally:
    ttnn.close_device(device)
