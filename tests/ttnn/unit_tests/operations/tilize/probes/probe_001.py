import torch, ttnn
from ttnn.operations.tilize import tilize_program_descriptor as pd
from ttnn.operations.tilize.tilize import create_program_descriptor

dev = ttnn.open_device(device_id=0)
for shape in [(1, 1, 64, 256), (1, 1, 4608, 288), (1, 1, 2048, 64), (1, 1, 32, 32), (3, 1, 64, 320)]:
    x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = create_program_descriptor(t, out, low_l1=True)
    y = ttnn.generic_op([t, out], desc)
    print("RES", shape, torch.equal(ttnn.to_torch(y), x), sum(cb.total_size for cb in desc.cbs))
ttnn.close_device(dev)
