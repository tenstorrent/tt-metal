# R7 probe B: is the uint8 ROUND TRIP itself sound (from_torch/to_torch, RM and
# TILE)?  If a TILE uint8 readback is broken, probe A's all-zero result says
# nothing about the kernel.
import torch, ttnn

dev = ttnn.open_device(device_id=0)
try:
    t = torch.arange(32 * 64, dtype=torch.int32).remainder(251).reshape(1, 1, 32, 64)
    for dt, tdt in ((ttnn.uint8, torch.uint8), (ttnn.uint16, torch.int32), (ttnn.uint32, torch.int32)):
        for lay, name in ((ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")):
            x = ttnn.from_torch(t, dtype=dt, layout=lay, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            back = ttnn.to_torch(x).to(torch.int32)
            print(
                f"{dt} {name}: roundtrip mismatches {int((back != t).sum())}/{t.numel()}  first row {back[0,0,0,:8].tolist()}"
            )
            ttnn.deallocate(x)
    # and: does ttnn's own to_layout(TILE) on a uint8 RM tensor work on device?
    x = ttnn.from_torch(
        t, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print("uint8 RM tensor page size:", x.buffer_page_size(), "elem", x.element_size())
finally:
    ttnn.close_device(dev)
