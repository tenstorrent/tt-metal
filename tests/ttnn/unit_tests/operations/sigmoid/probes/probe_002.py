import torch, ttnn
from ttnn.operations.sigmoid import sigmoid
d = ttnn.open_device(device_id=0)
def ulp_err(got, ref):
    # bf16 ULP distance
    g = got.to(torch.bfloat16).view(torch.int16).int(); r = ref.to(torch.bfloat16).view(torch.int16).int()
    return (g - r).abs().max().item()
for shape, mem in [((1,1,32,64), ttnn.DRAM_MEMORY_CONFIG), ((1,80,400), ttnn.L1_MEMORY_CONFIG), ((1,1,17,50), ttnn.DRAM_MEMORY_CONFIG)]:
    torch.manual_seed(0)
    x = torch.randn(shape).to(torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem)
    out = sigmoid(t)
    y = ttnn.to_torch(out).float()
    ref = torch.sigmoid(x.double())
    print(shape, out.memory_config().buffer_type, "max abs", (y-ref).abs().max().item(), "max bf16 ulp vs rounded ref", ulp_err(y, ref.float()))
ttnn.close_device(d)
