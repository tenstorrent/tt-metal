import torch, ttnn
from ttnn.operations.tilize.tilize import create_program_descriptor

dev = ttnn.open_device(device_id=0)
D, L = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
cases = [
    ((1, 1, 64, 128), D, L),
    ((1, 1, 64, 128), L, L),
    ((1, 1, 64, 64), L, D),
    ((64, 128), D, D),
    ((2, 32, 64), D, D),
    ((1, 2, 1, 32, 64), D, D),
    ((2, 1, 1, 1, 32, 64), D, D),
    ((1, 1, 4096, 64), L, L),
    ((1, 1, 8192, 64), L, D),
    ((1, 1, 8192, 64), D, L),
]
for shape, mi, mo in cases:
    x = torch.randn(shape).bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mi)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mo)
    for low in (False, True):
        desc = create_program_descriptor(t, out, low_l1=low)
        y = ttnn.generic_op([t, out], desc)
        print("RES", shape, mi.buffer_type, mo.buffer_type, low, torch.equal(ttnn.to_torch(y), x))
ttnn.close_device(dev)
