import torch, ttnn
from ttnn.operations.g6_zeros import g6_zeros_plan
from ttnn.operations.g6_zeros import g6_zeros_program_descriptor as pdm

dev = ttnn.open_device(device_id=0)
for shape, dt in [((1, 4, 256, 128), ttnn.bfloat16), ((32831, 1, 32, 256), ttnn.bfloat8_b)]:
    print("PROBE plan", shape, g6_zeros_plan(shape, dt, dev))
    # poison then free, then dispatch the op's own descriptor past validate()
    tdt = torch.bfloat16
    dirty = ttnn.from_torch(
        torch.ones(shape, dtype=tdt),
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    addr = dirty.buffer_address()
    ttnn.deallocate(dirty)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    plan = pdm.make_plan(out.buffer_num_pages(), out.buffer_aligned_page_size(), dev)
    ttnn.generic_op([out], pdm.create_program_descriptor(out, plan))
    a = ttnn.to_torch(out).to(torch.float32)
    print(
        "PROBE",
        shape,
        dt,
        "same_addr",
        out.buffer_address() == addr,
        "nonzero_bits",
        int((a.view(torch.int32) != 0).sum()),
        "of",
        a.numel(),
    )
ttnn.close_device(dev)
