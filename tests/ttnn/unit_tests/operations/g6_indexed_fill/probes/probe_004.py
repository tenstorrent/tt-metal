import torch, ttnn

device = ttnn.open_device(device_id=0, trace_region_size=1 << 20)
try:
    a = ttnn.from_torch(torch.ones(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    device.disable_and_clear_program_cache()
    ttnn.add(a, a)  # eager warm-up, as the golden case does
    tid = ttnn.begin_trace_capture(device)
    try:
        ttnn.add(a, a)
        print("RESULT stock ttnn.add traced fine under a disabled program cache")
    except Exception as e:
        print("RESULT stock ttnn.add fails too:", str(e).splitlines()[0][:160])
    finally:
        ttnn.end_trace_capture(device, tid)
        ttnn.release_trace(device, tid)
finally:
    ttnn.close_device(device)
