import torch, ttnn, time, threading

recs = []


def cb(batch):
    for r in batch.records:
        recs.append(((r.end_timestamp - r.start_timestamp) / r.frequency, r.core_count))


dev = ttnn.open_device(device_id=0)
try:
    h = (
        ttnn.device.RegisterProgramRealtimeProfilerCallback(cb)
        if ttnn.device.IsProgramRealtimeProfilerActive()
        else None
    )
    print("PROBE rt active", h is not None)
    for shape in [(640, 1792), (2048, 1280)]:
        x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
        for op in ("clone", "rms", "add"):
            for i in range(3):
                if op == "clone":
                    y = ttnn.clone(x)
                elif op == "add":
                    y = ttnn.add(x, 1.0)
                else:
                    y = ttnn.rms_norm(x) if hasattr(ttnn, "rms_norm") else ttnn.clone(x)
                ttnn.synchronize_device(dev)
                time.sleep(0.2)
            print("PROBE", shape, op, [(round(d / 1000, 2), c) for d, c in recs[-3:]])
finally:
    ttnn.close_device(dev)
