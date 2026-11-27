import ttnn

device = ttnn.open_device(device_id=0)
try:
    for ch in range(8):
        try:
            print("bank", ch, device.dram_core_from_logical_core(ttnn.CoreCoord(ch, 0)))
        except Exception as e:
            print("err", e)
            break
    print("dram view banks", ttnn.device.GetMemoryView(device, ttnn.BufferType.DRAM).num_banks)
    print([m for m in dir(device) if "core" in m.lower() or "grid" in m.lower()])
finally:
    ttnn.close_device(device)
