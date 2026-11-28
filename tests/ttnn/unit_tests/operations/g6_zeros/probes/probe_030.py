import torch, ttnn
from ttnn.operations.g6_zeros.g6_zeros_program_descriptor import make_plan, create_program_descriptor
from ttnn.operations.g6_zeros import g6_zeros_plan

dev = ttnn.open_device(device_id=0)
for shape, dt in [
    ((1, 4, 256, 128), ttnn.bfloat16),
    ((32831, 1, 32, 256), ttnn.bfloat8_b),
    ((16, 12, 128, 128), ttnn.float32),
]:
    print("RES plan", shape, g6_zeros_plan(shape, dt, dev))
    pois = ttnn.from_torch(torch.ones(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    addr = pois.buffer_address()
    ttnn.deallocate(pois)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
    plan = make_plan(out.buffer_num_pages(), out.buffer_aligned_page_size(), dev)
    ttnn.generic_op([out], create_program_descriptor(out, plan))
    t = ttnn.to_torch(out).float()
    print("RES", shape, dt, "reused", out.buffer_address() == addr, "nonzero", int((t.view(torch.int32) != 0).sum()))
ttnn.close_device(dev)
