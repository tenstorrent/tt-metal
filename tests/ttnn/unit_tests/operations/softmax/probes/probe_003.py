import torch, ttnn
from ttnn.operations.softmax import softmax
d = ttnn.open_device(device_id=0)
try:
    for shape, mem in [((1,1,32,64), ttnn.DRAM_MEMORY_CONFIG), ((1,4,16,400), ttnn.L1_MEMORY_CONFIG)]:
        torch.manual_seed(0)
        t = torch.randn(shape, dtype=torch.bfloat16)
        x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16, memory_config=mem)
        y = ttnn.to_torch(softmax(x, dim=-2)).double()
        ref = torch.softmax(t.double(), dim=-2)
        err = (y-ref).abs().max().item()
        pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0,1].item()
        print("RESULT", shape, "maxerr", err, "pcc", pcc)
finally:
    ttnn.close_device(d)
