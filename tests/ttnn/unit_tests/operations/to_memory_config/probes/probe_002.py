import torch, ttnn
from eval import memory
from ttnn.operations.to_memory_config import to_memory_config, SUPPORTED
SUPPORTED["dtype"].append(ttnn.float32)   # in-process only: probe the knob-turn claim
device = ttnn.open_device(device_id=0)
def hs(device, shape, layout):
    return memory.place(memory.build("L1","height"), shape, layout, device)
res = []
for shape in [(1,1,17,50),(2,1,100,47),(1,1,64,512),(1,4,4,64),(1,80,400),(32,17)]:
    for layout in [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]:
        for src in ["DRAM","L1","L1/height"]:
            for dst in ["DRAM","L1","L1/height"]:
                t = torch.randn(shape, dtype=torch.float32) * 1e3 + 1e-3  # values with low mantissa bits a bf16 hop would lose
                cfg = lambda w: {"DRAM": ttnn.DRAM_MEMORY_CONFIG, "L1": ttnn.L1_MEMORY_CONFIG}.get(w) or hs(device, shape, layout)
                x = ttnn.from_torch(t, dtype=ttnn.float32, layout=layout, device=device, memory_config=cfg(src))
                y = to_memory_config(x, cfg(dst))
                ok = torch.equal(ttnn.to_torch(y), t)
                res.append(ok)
                if not ok: print("FAIL", shape, layout, src, dst)
print("float32 exact:", sum(res), "/", len(res))
ttnn.close_device(device)
