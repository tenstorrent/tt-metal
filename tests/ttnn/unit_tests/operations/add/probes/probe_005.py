# Which tt-metal programs run on height-sharded tensors here (for the session test), and where add's result lands.
import torch, ttnn
from ttnn.operations.add import add
from eval.memory import build, place
device = ttnn.open_device(device_id=0)
shape = (1, 1, 64, 128)
cfg = place(build("L1", "height"), shape, ttnn.TILE_LAYOUT, device)
print("CFG", cfg)
torch.manual_seed(0)
ta, tb = torch.randn(shape, dtype=torch.bfloat16), torch.randn(shape, dtype=torch.bfloat16)
a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
out = add(a, b)
print("OUT MC", out.memory_config())
exp = (ta.float() + tb.float()).to(torch.bfloat16).float()
print("OK", torch.equal(ttnn.to_torch(out).float(), exp))
for name, fn in [
    ("to_layout RM", lambda t: ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)),
    ("to_memory_config DRAM", lambda t: ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)),
    ("sharded_to_interleaved", lambda t: ttnn.sharded_to_interleaved(t, ttnn.L1_MEMORY_CONFIG)),
]:
    try:
        r = fn(out)
        print(name, "RAN", r.memory_config().memory_layout, torch.equal(ttnn.to_torch(ttnn.from_device(r)).float(), exp))
    except Exception as e:
        print(name, "FAILED", str(e).splitlines()[0][:200])
try:
    a_i = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    a_s = ttnn.interleaved_to_sharded(a_i, cfg)
    print("interleaved_to_sharded RAN", a_s.memory_config().memory_layout, torch.equal(ttnn.to_torch(a_s), ta))
except Exception as e:
    print("interleaved_to_sharded FAILED", str(e).splitlines()[0][:200])
ttnn.close_device(device)
