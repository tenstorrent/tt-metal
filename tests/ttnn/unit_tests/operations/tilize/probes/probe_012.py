import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    num_cores = 4
    shape = (32, 256)
    ss = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))}),
        [32, shape[1] // num_cores],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ss)
    device.enable_program_cache()
    n0 = device.num_program_cache_entries()
    keep = []
    for i in range(4):
        x = torch.rand(shape, dtype=torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
        o = ttnn.tilize(t, mc)
        keep += [t, o]
        ok = torch.equal(x, ttnn.to_torch(o))
        print(
            f"iter {i}: entries_delta={device.num_program_cache_entries()-n0} equal={ok} in_addr={t.buffer_address()} out_addr={o.buffer_address()}"
        )
finally:
    ttnn.close_device(device)
