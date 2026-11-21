import torch, ttnn
from eval import memory
from ttnn.operations.to_memory_config import to_memory_config
dev = ttnn.open_device(device_id=0)
def run(shape, layout, src, dst_name):
    t = torch.randn(shape).to(torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=dev, memory_config=src)
    if dst_name == "L1/height":
        dst = memory.place(memory.Sharded("L1","height"), tuple(shape), layout, dev)
    else:
        dst = {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}[dst_name]
    y = to_memory_config(x, dst)
    got = ttnn.to_torch(y)
    print(shape, layout, dst_name, "match" if torch.equal(got, t) else f"MISMATCH maxdiff={(got.float()-t.float()).abs().max()}", memory.memory_axes(y))
run((1,1,64,64), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, "L1")
run((1,1,17,50), ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, "L1")
run((1,1,17,50), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG, "DRAM")
run((1,1,1600,256), ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, "L1")
run((1,1,64,512), ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG, "L1/height")
run((1,4,4,64), ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, "L1/height")
ttnn.close_device(dev)
